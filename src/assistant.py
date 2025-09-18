import logging
from typing import Dict, Any
import asyncio
from datetime import datetime

import config
from agent import process_request, broadcast_command, BaseAgent, Envelope

logger = logging.getLogger(__name__)

class ManagerAgent(BaseAgent):
    """Pipeline orchestrator that routes messages through processing stages.
    
    Implements a simple state machine:
    - start -> uppercase -> reverse -> final result
    
    Includes run counter for demo purposes with safety limit.
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Route messages through the pipeline stages.
        
        State machine logic:
        1. task + start -> route to uppercase
        2. result + upper -> route to reverse  
        3. result + reverse -> send to result_list for external caller
        
        Returns modified envelope with updated routing information.
        """
        stage = env.payload.get("stage", "start")
        logger.info(f"[ManagerAgent] Processing {env.kind} at stage '{stage}' for {env.message_id}")
        logger.info(f"[ManagerAgent] Incoming: {env}")

        if env.kind == "task" and stage == "start":
            env.target_role = "uppercase"
            env.payload["stage"] = "upper"
            logger.info(f"[ManagerAgent] Sending task to uppercase")
            return env
        
        if env.kind == "result" and stage == "upper":
            env.kind = "task"
            env.target_role = "reverse"
            env.payload["stage"] = "reverse"
            logger.info(f"[ManagerAgent] Sending task to reverse")
            return env
        
        if env.kind == "result" and stage == "reverse":
            logger.info(f"[ManagerAgent] Final result ready: {env}")
            if env.result_list:
                env.target_list = env.result_list
                env.target_role = None
                env.target_agent_id = None
            return env
        
        # Unknown stage
        logger.error(f"[ManagerAgent] Unknown stage '{stage}' for {env}")
        env.payload.setdefault("errors", []).append({
            "code": "manager.stage",
            "message": f"Unknown stage '{stage}' for kind '{env.kind}'",
        })
        return env

class UppercaseAgent(BaseAgent):
    """Example worker: converts payload['text'] to uppercase and returns 'result'.
    
    Demonstrates basic text transformation agent pattern.
    Changes envelope kind from 'task' to 'result' after processing.
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Convert text to uppercase.
        
        Transforms payload['text'] to uppercase and marks envelope as 'result'.
        Logs processing details for debugging.
        """
        logger.info(f"[UppercaseAgent] Processing {env.kind} message: {env.message_id} stage: {env.payload.get('stage', 'N/A')}")
        text = env.payload.get("text", "")
        env.payload["text"] = text.upper()
        env.kind = "result"
        logger.info(f"[UppercaseAgent] Processed: '{text}' -> '{env.payload['text']}' for {env.message_id}")
        logger.info(f"[UppercaseAgent] Outgoing: {env}")
        return env

class ReverseAgent(BaseAgent):
    """Example worker: reverses payload['text'] and returns 'result'.
    
    Demonstrates basic text transformation agent pattern.
    Changes envelope kind from 'task' to 'result' after processing.
    """

    async def process(self, env: Envelope) -> Envelope:
        """Reverse the text.
        
        Transforms payload['text'] by reversing character order and marks envelope as 'result'.
        Logs processing details for debugging.
        """
        text = env.payload.get("text", "")
        env.payload["text"] = text[::-1]
        env.kind = "result"
        logger.info(f"[ReverseAgent] Processed: '{text}' -> '{env.payload['text']}' for {env.message_id}")
        logger.info(f"[ReverseAgent] Outgoing: {env}")
        return env

# Create default agent instances - they will auto-register via BaseAgent.__init__
_manager = ManagerAgent()
_upper = UppercaseAgent()
_reverse = ReverseAgent()

class AlarisAssistant:
    def __init__(self):
        self.conversations = {}  # In-memory conversation storage
        
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        logger.info(f"Processing message from user {user_id}: {message}")
        response = dict()
        payload = {"text": message, "stage": "start"}
        response["message"] = await process_request('manager', user_id, payload)
        return response
   
    async def clear_conversation_history(self, user_id: str) -> bool:
        if user_id and user_id in self.conversations:
            del self.conversations[user_id]
            logger.info(f"Cleared conversation history for user {user_id}")
            return True
        
        logger.info(f"No conversation history found for user {user_id}")
        return False

assistant = AlarisAssistant()

async def clear_conversation_history(user_id: str) -> bool:
    global assistant
    return await assistant.clear_conversation_history(user_id)

async def process_user_message(user_id: str, message: str) -> Dict[str, Any]:
    global assistant
    return await assistant.process_message(user_id, message)

if __name__ == "__main__":
    async def _demo():
        try:
            for i in range(10):
                message = f"Hello, world! [{datetime.now().isoformat()}]"
                res = await process_user_message("conv1", message)
                print("RESULT:", res)  # Expected: "!DLROW ,OLLEH"
                await asyncio.sleep(1)

            # Example: global graceful shutdown (agents finish current tasks)
            await broadcast_command("shutdown")
        finally:
            #await stop_runtime()
            pass

    asyncio.run(_demo())