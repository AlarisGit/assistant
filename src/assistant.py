import logging
from typing import Dict, Any
import asyncio
import time
from datetime import datetime
import json

import config
from agent import process_request, broadcast_command, BaseAgent, Envelope
import llm
import util

logger = logging.getLogger(__name__)

class CommandAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        """Process command envelope."""
        action = env.payload.get("action")
        if action == "clear_history":
            # Clear message history only, keep other memory intact
            await self.cleanup_message_history(env.conversation_id)
            logger.info(f"[CommandAgent] Cleared message history for conversation {env.conversation_id}")
            
            env.payload["result"] = "Message history cleared successfully"
            return env.final()
        
        elif action == "clear_all":
            # Clear ALL conversation data (complete cleanup)
            await self.cleanup_memory(env.conversation_id)
            logger.info(f"[CommandAgent] Cleared ALL data for conversation {env.conversation_id}")
            
            env.payload["result"] = "All conversation data cleared successfully"
            return env.final()
        
        elif action == "get_stats":
            # Get conversation statistics
            memory = await self.get_memory(env.conversation_id)
            message_count = await memory.get_message_count()
            user_prefs = await memory.get("user_preferences", {})
            memory_size = await memory.get_memory_size()
            
            stats = {
                "message_count": message_count,
                "user_preferences": user_prefs,
                "memory_usage": memory_size,
                "conversation_id": env.conversation_id
            }
            
            logger.info(f"[CommandAgent] Retrieved stats for conversation {env.conversation_id}: {stats}")
            
            env.payload["stats"] = stats
            return env.final()
        
        # Unknown action - return error
        logger.error(f"[CommandAgent] Unknown action '{action}' for {env}")
        env.payload.setdefault("errors", []).append({
            "code": "command.action",
            "message": f"Unknown action '{action}'",
        })
        return env

class ManagerAgent(BaseAgent):
    """Pipeline orchestrator that routes messages through processing stages.
    
    Enhanced with distributed memory for conversation history and user preferences.
    
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Route messages through the pipeline stages.
        
        Enhanced with memory usage for conversation tracking and user preferences.
        
        Returns modified envelope with updated routing information.
        """
        stage = env.payload.get("stage", "start")
        logger.info(f"[ManagerAgent] Processing at stage '{stage}' for {env.message_id}")
        
        # Get conversation memory
        memory = await self.get_memory(env.conversation_id)

        # Custom logging: Log pipeline decision making
        await self.log(env.conversation_id, f"Pipeline decision: stage='{stage}'")
            
        if stage == "start":
           
            # Regular message processing
            # Save user message to conversation history
            user_text = env.payload.get("text", "")
            messages = await memory.get_messages(limit=10)
            for i, message in enumerate(messages):
                await self.log(env.conversation_id, f"History item {i}: {message}")
            logger.info(f"History: {json.dumps(messages, ensure_ascii=False, indent=4)}")
            await memory.add_message("user", user_text, {"message_id": env.message_id})
            
            # Initialize user preferences if not exists
            user_prefs = await memory.get("user_preferences", { 
                "language": "auto",
                "message_count": 0
            })
            
            # Increment message count
            user_prefs["message_count"] = user_prefs.get("message_count", 0) + 1
            await memory.set("user_preferences", user_prefs)
            
            logger.info(f"[ManagerAgent] Saved user message to memory. Total messages: {user_prefs['message_count']}")
            
            # Custom logging: Log memory state and routing decision
            await self.log(env.conversation_id, f"Memory updated: msg_count={user_prefs['message_count']} lang={user_prefs.get('language')} style={user_prefs.get('response_style')}")
            await self.log(env.conversation_id, f"Routing decision: start->detectlanguage (text_length={len(user_text)} chars)")
            
            env.target_role = "lang"
            env.payload["stage"] = "lang"
            logger.info(f"[ManagerAgent] Sending task to detectlanguage")
            return env
        
        if stage == "lang":
            # Save assistant response to conversation history
            env.payload['text'] = f"Detected language: {env.payload.get("language", "en")}"
            await memory.add_message("assistant", env.payload['text'], {"message_id": env.message_id})
            
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
            
            # Custom logging: Log final pipeline completion with stats
            await self.log(env.conversation_id, f"Pipeline complete: final_length={len(env.payload.get("text", ""))} chars total_messages={message_count}")
            await self.log(env.conversation_id, f"Conversation stats: {json.dumps(env.payload['conversation_stats'], ensure_ascii=False)}")
            
            logger.info(f"[ManagerAgent] Final result ready: {env}")
            return env.final()
        
        # Unknown stage
        logger.error(f"[ManagerAgent] Unknown stage '{stage}' for {env}")
        env.payload.setdefault("errors", []).append({
            "code": "manager.stage",
            "message": f"Unknown stage '{stage}'",
        })
        return env

class LangAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        text = env.payload.get("text", "")
        await self.log(env.conversation_id, f"Processing text: {text}")
        memory = await self.get_memory(env.conversation_id)
        lang = 'en'
        messages = await memory.get_messages(limit=6)
        for message in messages:
            await self.log(env.conversation_id, f"Message: {message}")
            if message.get("role") == "user":
                text = message.get("content", "")
                if util._contains_cyrillic(text):
                    await self.log(env.conversation_id, f"Detected Cyrillic in message: {text}")
                    lang = 'ru'
                    break
                elif util._contains_chinese(text):
                    await self.log(env.conversation_id, f"Detected Chinese in message: {text}")
                    lang = 'zh'
                    break
        await memory.set("language", lang)
        env.payload["language"] = lang
        await self.log(env.conversation_id, f"Detected language: {lang}")
 
        return env

class TranslationAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        return env
    


# Create default agent instances - they will auto-register via BaseAgent.__init__
_manager = ManagerAgent()
_command = CommandAgent()
_lang = LangAgent()
_translate = TranslationAgent()

# Direct function implementations (no proxy class needed)

async def process_user_message(user_id: str, message: str) -> Dict[str, Any]:
    """Process a user message through the agent pipeline.
    
    Args:
        user_id: User identifier (used as conversation_id)
        message: User message text to process
        
    Returns:
        Dict containing the processed message response
    """
    logger.info(f"Processing message from user {user_id}: {message}")
    response = dict()
    payload = {"text": message, "stage": "start"}
    response["message"] = await process_request('manager', user_id, payload)
    return response

async def clear_conversation_history(user_id: str) -> bool:
    """Clear conversation history for a user while keeping other memory intact.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use the distributed memory system to clear message history
        result = await process_request("command", user_id, {
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

async def clear_all_conversation_data(user_id: str) -> bool:
    """Clear ALL conversation data for a user (complete cleanup).
    
    Args:
        user_id: User identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use the distributed memory system to clear all conversation data
        result = await process_request("command", user_id, {
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

async def get_conversation_stats(user_id: str) -> Dict[str, Any]:
    """Get conversation statistics including message count and memory usage.
    
    Args:
        user_id: User identifier
        
    Returns:
        Dict containing conversation statistics
    """
    try:
        # process_request returns a string, but we need to parse the envelope payload
        result = await process_request("command", user_id, {
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