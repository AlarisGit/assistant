import logging
from typing import Dict, Any
import asyncio
import time
from datetime import datetime
import json

import config
from agent import process_request, broadcast_command, BaseAgent, Envelope

logger = logging.getLogger(__name__)

class ManagerAgent(BaseAgent):
    """Pipeline orchestrator that routes messages through processing stages.
    
    Enhanced with distributed memory for conversation history and user preferences.
    
    Implements a simple state machine:
    - start -> uppercase -> reverse -> final result
    
    Demonstrates memory usage for conversation tracking.
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Route messages through the pipeline stages.
        
        Enhanced with memory usage for conversation tracking and user preferences.
        
        State machine logic:
        1. task + start -> route to uppercase (save user message to memory)
        2. result + upper -> route to reverse  
        3. result + reverse -> send to result_list (save assistant response to memory)
        
        Returns modified envelope with updated routing information.
        """
        stage = env.payload.get("stage", "start")
        logger.info(f"[ManagerAgent] Processing {env.kind} at stage '{stage}' for {env.message_id}")
        
        # Get conversation memory
        memory = await self.get_memory(env.conversation_id)

        # Check if this is a special action
        action = env.payload.get("action")
        
        if action == "clear_history":
            # Clear message history only, keep other memory intact
            await self.cleanup_message_history(env.conversation_id)
            logger.info(f"[ManagerAgent] Cleared message history for conversation {env.conversation_id}")
            
            env.payload["result"] = "Message history cleared successfully"
            env.kind = "result"
            if env.result_list:
                env.target_list = env.result_list
                env.target_role = None
                env.target_agent_id = None
            return env
        
        elif action == "clear_all":
            # Clear ALL conversation data (complete cleanup)
            await self.cleanup_memory(env.conversation_id)
            logger.info(f"[ManagerAgent] Cleared ALL data for conversation {env.conversation_id}")
            
            env.payload["result"] = "All conversation data cleared successfully"
            env.kind = "result"
            if env.result_list:
                env.target_list = env.result_list
                env.target_role = None
                env.target_agent_id = None
            return env
        
        elif action == "get_stats":
            # Get conversation statistics
            message_count = await memory.get_message_count()
            user_prefs = await memory.get("user_preferences", {})
            memory_size = await memory.get_memory_size()
            
            stats = {
                "message_count": message_count,
                "user_preferences": user_prefs,
                "memory_usage": memory_size,
                "conversation_id": env.conversation_id
            }
            
            logger.info(f"[ManagerAgent] Retrieved stats for conversation {env.conversation_id}: {stats}")
            
            env.payload["stats"] = stats
            env.kind = "result"
            if env.result_list:
                env.target_list = env.result_list
                env.target_role = None
                env.target_agent_id = None
            return env

        
        if env.kind == "task" and stage == "start":
           
            # Regular message processing
            # Save user message to conversation history
            user_text = env.payload.get("text", "")
            messages = await memory.get_messages(limit=10)
            logger.info(f"Message history: {json.dumps(messages, ensure_ascii=False, indent=4)}")
            await memory.add_message("user", user_text, {"message_id": env.message_id})
            
            # Initialize user preferences if not exists
            user_prefs = await memory.get("user_preferences", { 
                "language": "auto",
                "response_style": "detailed",
                "message_count": 0
            })
            
            # Increment message count
            user_prefs["message_count"] = user_prefs.get("message_count", 0) + 1
            await memory.set("user_preferences", user_prefs)
            
            logger.info(f"[ManagerAgent] Saved user message to memory. Total messages: {user_prefs['message_count']}")
            
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
            # Save assistant response to conversation history
            assistant_text = env.payload.get("text", "")
            await memory.add_message("assistant", assistant_text, {"message_id": env.message_id})
            
            # Get conversation stats
            message_count = await memory.get_message_count()
            user_prefs = await memory.get("user_preferences", {})
            
            logger.info(f"[ManagerAgent] Saved assistant response. Conversation has {message_count} messages")
            logger.info(f"[ManagerAgent] User preferences: {user_prefs}")
            
            # Add conversation metadata to payload for external caller
            env.payload["conversation_stats"] = {
                "message_count": message_count,
                "user_preferences": user_prefs
            }
            
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
    
    Enhanced with memory usage to track processing statistics.
    Demonstrates basic text transformation agent pattern.
    Changes envelope kind from 'task' to 'result' after processing.
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Convert text to uppercase.
        
        Enhanced with memory usage to track processing statistics per conversation.
        Transforms payload['text'] to uppercase and marks envelope as 'result'.
        """
        logger.info(f"[UppercaseAgent] Processing {env.kind} message: {env.message_id} stage: {env.payload.get('stage', 'N/A')}")
        
        # Get conversation memory
        memory = await self.get_memory(env.conversation_id)
        
        # Track processing statistics
        stats = await memory.get("uppercase_stats", {
            "processed_count": 0,
            "total_chars": 0,
            "last_processed": None
        })
        
        text = env.payload.get("text", "")
        processed_text = text.upper()
        
        # Update statistics
        stats["processed_count"] += 1
        stats["total_chars"] += len(text)
        stats["last_processed"] = time.time()
        await memory.set("uppercase_stats", stats)
        
        env.payload["text"] = processed_text
        env.kind = "result"
        
        logger.info(f"[UppercaseAgent] Processed: '{text}' -> '{processed_text}' for {env.message_id}")
        logger.info(f"[UppercaseAgent] Stats: {stats['processed_count']} messages, {stats['total_chars']} total chars")
        
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
        pass
        
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        logger.info(f"Processing message from user {user_id}: {message}")
        response = dict()
        payload = {"text": message, "stage": "start"}
        response["message"] = await process_request('manager', user_id, payload)
        return response
   
    async def clear_conversation_history(self, user_id: str) -> bool:
        """Clear conversation history for a user while keeping other memory intact."""
        try:
            # Use the distributed memory system to clear message history
            result = await process_request("manager", user_id, {
                "action": "clear_history",
                "user_id": user_id
            })
            
            if "error" not in result:
                logger.info(f"Cleared conversation history for user {user_id}")
                return True
            else:
                logger.error(f"Failed to clear history for user {user_id}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing conversation history for user {user_id}: {e}")
            return False
    
    async def clear_all_conversation_data(self, user_id: str) -> bool:
        """Clear ALL conversation data for a user (complete cleanup)."""
        try:
            # Use the distributed memory system to clear all conversation data
            result = await process_request("manager", user_id, {
                "action": "clear_all",
                "user_id": user_id
            })
            
            if "error" not in result:
                logger.info(f"Cleared all conversation data for user {user_id}")
                return True
            else:
                logger.error(f"Failed to clear all data for user {user_id}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing all conversation data for user {user_id}: {e}")
            return False
    
    async def get_conversation_stats(self, user_id: str) -> Dict[str, Any]:
        """Get conversation statistics including message count and memory usage."""
        try:
            # process_request returns a string, but we need to parse the envelope payload
            result = await process_request("manager", user_id, {
                "action": "get_stats",
                "user_id": user_id
            })
            
            # The result is a string, but we need to get the stats from the envelope
            # For now, let's use the direct memory access approach
            from agent import get_memory_manager
            
            manager = await get_memory_manager()
            memory = manager.get_memory(user_id)
            
            message_count = await memory.get_message_count()
            user_prefs = await memory.get("user_preferences", {})
            memory_size = await memory.get_memory_size()
            
            return {
                "message_count": message_count,
                "user_preferences": user_prefs,
                "memory_usage": memory_size,
                "conversation_id": user_id
            }
                
        except Exception as e:
            logger.error(f"Error getting conversation stats for user {user_id}: {e}")
            return {}

assistant = AlarisAssistant()

async def clear_conversation_history(user_id: str) -> bool:
    global assistant
    return await assistant.clear_conversation_history(user_id)

async def clear_all_conversation_data(user_id: str) -> bool:
    global assistant
    return await assistant.clear_all_conversation_data(user_id)

async def get_conversation_stats(user_id: str) -> Dict[str, Any]:
    global assistant
    return await assistant.get_conversation_stats(user_id)

async def process_user_message(user_id: str, message: str) -> Dict[str, Any]:
    global assistant
    return await assistant.process_message(user_id, message)

if __name__ == "__main__":
    async def _demo():
        try:
            print("=== Memory Cleanup Demo ===")
            
            # Send some messages to build conversation history
            print("\n1. Building conversation history...")
            for i in range(3):
                message = f"Hello message {i+1}! [{datetime.now().isoformat()}]"
                res = await process_user_message("test_user", message)
                print(f"Message {i+1} result:", res.get("message", "No response"))
                await asyncio.sleep(0.5)
            
            # Get conversation stats before cleanup
            print("\n2. Conversation stats before cleanup:")
            stats_before = await get_conversation_stats("test_user")
            print(f"   Message count: {stats_before.get('message_count', 0)}")
            print(f"   User preferences: {stats_before.get('user_preferences', {})}")
            print(f"   Memory fields: {stats_before.get('memory_usage', {}).get('fields', [])}")
            
            # Test partial cleanup (messages only)
            print("\n3. Testing partial cleanup (messages only)...")
            cleanup_result = await clear_conversation_history("test_user")
            print(f"   Cleanup successful: {cleanup_result}")
            
            # Get stats after partial cleanup
            print("\n4. Conversation stats after message cleanup:")
            stats_after = await get_conversation_stats("test_user")
            print(f"   Message count: {stats_after.get('message_count', 0)}")
            print(f"   User preferences: {stats_after.get('user_preferences', {})}")
            print(f"   Memory fields: {stats_after.get('memory_usage', {}).get('fields', [])}")
            
            # Send one more message to verify system still works
            print("\n5. Sending message after cleanup...")
            new_message = f"New message after cleanup! [{datetime.now().isoformat()}]"
            res = await process_user_message("test_user", new_message)
            print("New message result:", res.get("message", "No response"))
            
            # Final stats
            print("\n6. Final conversation stats:")
            final_stats = await get_conversation_stats("test_user")
            print(f"   Message count: {final_stats.get('message_count', 0)}")
            print(f"   User preferences: {final_stats.get('user_preferences', {})}")
            
            print("\n=== Demo Complete ===")

            # Example: global graceful shutdown (agents finish current tasks)
            await broadcast_command("shutdown")
        finally:
            #await stop_runtime()
            pass

    asyncio.run(_demo())