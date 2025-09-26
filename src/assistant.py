"""
================================================================================
Assistant Application Layer - Agent Implementations
================================================================================

This module contains the application-level agent implementations that define
the business logic for the conversational AI assistant. It demonstrates the
separation of concerns between system infrastructure (agent.py) and application
logic (assistant.py).

Agent Architecture:
- CommandAgent: Handles action-based commands (clear_history, clear_all)
- ManagerAgent: Orchestrates FSM-based message pipeline (start -> lang -> final)
- LangAgent: Detects language from conversation history
- TranslationAgent: Placeholder for future translation functionality

Key Features:
- Clean separation between command handling and FSM orchestration
- Distributed conversation memory with automatic cleanup
- Custom logging for agent-specific insights
- Simplified envelope handling with env.final() method
- Zero-parameter constructors with automatic registration

External API:
- process_user_message(): Main entry point for user messages
- clear_conversation_history(): Clear message history only
- clear_all_conversation_data(): Complete conversation cleanup

Usage:
    # Process user message through pipeline
    result = await process_user_message("user123", "Hello world")
    
    # Clear conversation history
    success = await clear_conversation_history("user123")
"""

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
    """Handles action-based command requests.
    
    This agent processes administrative commands that don't require FSM routing:
    - clear_history: Clear conversation message history only
    - clear_all: Complete conversation data cleanup
    
    The agent automatically routes final results back to the external caller
    using the env.final() convenience method.
    
    Role: "command" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process action-based command envelope.
        
        Args:
            env: Envelope containing action in payload
            
        Returns:
            Envelope with result or error, configured for final delivery
        """
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

        # Unknown action - return error
        logger.error(f"[CommandAgent] Unknown action '{action}' for {env}")
        env.payload.setdefault("errors", []).append({
            "code": "command.action",
            "message": f"Unknown action '{action}'",
        })
        return env

class ManagerAgent(BaseAgent):
    """FSM-based pipeline orchestrator for message processing.
    
    This agent implements a finite state machine that routes messages through
    different processing stages:
    
    State Machine:
    1. start -> lang: Route user message to language detection
    2. lang -> final: Process detected language and return result
    
    Features:
    - Distributed conversation memory for history and preferences
    - Automatic message counting and user preference tracking
    - Custom logging for pipeline decision making and statistics
    - Integration with memory cleanup and statistics reporting
    
    Role: "manager" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Route messages through the pipeline stages.
        
        Enhanced with memory usage for conversation tracking and user preferences.
        
        Returns modified envelope with updated routing information.
        """
        stage = env.payload.get("stage", "start")
        
        # Get conversation memory
        memory = await self.get_memory(env.conversation_id)

        if stage == "start":
            user_text = env.payload.get("text", "")
            messages = await memory.get_messages(limit=10)
            for i, message in enumerate(messages):
                await self.log(env.conversation_id, f"History item {i}: {message}")
            await memory.add_message("user", user_text, {"message_id": env.message_id})
            await self.log(env.conversation_id, f"Added user message to history: {user_text}")
            env.target_role = "lang"
            env.payload["stage"] = "lang"
            return env
        
        if stage == "lang":
            env.target_role = "samplellm"
            env.payload["stage"] = "response"
            return env
        
        if stage == "response":
            env.target_role = "manager"
            env.payload["stage"] = "final"
            return env
            
        if stage == 'final':
            if 'response' not in env.payload or not env.payload['response']:
                env.payload['response'] = f"No response"
            await memory.add_message("assistant", env.payload['response'], {"message_id": env.message_id})
            await self.log(env.conversation_id, f"Added assistant message to history: {env.payload['response']}")
            message_count = await memory.get_message_count()
            await self.log(env.conversation_id, f"Pipeline complete: final_length={len(env.payload.get("response", ""))} chars total_messages={message_count}")
            return env.final()
        
        # Unknown stage - return error
        env.payload.setdefault("errors", []).append({
            "code": "manager.stage",
            "message": f"Unknown stage '{stage}'",
        })
        return env.final()

class SampleAgent(BaseAgent):
    """Sample agent for testing purposes."""
    
    async def process(self, env: Envelope) -> Envelope:
        await self.log(env.conversation_id, f"Processing text: {env.payload.get("text", "")}")
        env.payload["text"] = f"Sample agent processed: {env.payload.get("text", "")}"
        return env

class SampleLLMAgent(BaseAgent):
    """Sample agent for testing purposes."""
    
    async def process(self, env: Envelope) -> Envelope:
        await self.log(env.conversation_id, f"Processing text: {env.payload.get("text", "")}")
        memory = await self.get_memory(env.conversation_id)
        history = []
        pair = {}
        for message in await memory.get_messages(limit=10):
            role = message.get("role", 'unknown')
            content = message.get("content", '')
            pair[role] = content
            if 'user' in pair and 'assistant' in pair:
                history.append((pair['user'], pair['assistant']))
                pair = {}
        # Get language from payload (set by LangAgent) or default to 'en'
        language = env.payload.get("language", "en")
        prompt_options = {'language': language}
        env.payload["response"] = llm.generate_text('sample', env.payload.get("text", ""), history, prompt_options=prompt_options)
        return env


class LangAgent(BaseAgent):
    """Language detection agent using conversation history.
    
    Analyzes recent conversation messages to detect the primary language
    being used by the user. Supports Cyrillic (Russian) and Chinese detection
    with English as the default fallback.
    
    Detection Logic:
    - Examines up to 6 recent user messages
    - Checks for Cyrillic characters (Russian)
    - Checks for Chinese characters
    - Defaults to English if no specific patterns found
    
    Role: "lang" (auto-derived from class name)
    """
    
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
    """Translation agent placeholder for future functionality.
    
    This agent is currently a placeholder that can be extended to provide
    translation services between different languages detected by LangAgent.
    
    Future capabilities:
    - Text translation between supported languages
    - Integration with translation APIs (Google Translate, etc.)
    - Conversation context-aware translation
    
    Role: "translation" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Placeholder implementation - returns envelope unchanged."""
        return env
    


# Create default agent instances - they will auto-register via BaseAgent.__init__
_manager = ManagerAgent()
_command = CommandAgent()
_lang = LangAgent()
_samplellm = SampleLLMAgent()
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
            
            # Test partial cleanup (messages only)
            print("\n2. Testing partial cleanup (messages only)...")
            cleanup_result = await clear_conversation_history("test_user")
            print(f"   Cleanup successful: {cleanup_result}")
            
            # Send one more message to verify system still works
            print("\n3. Sending message after cleanup...")
            new_message = f"New message after cleanup! [{datetime.now().isoformat()}]"
            res = await process_user_message("test_user", new_message)
            print("New message result:", res.get("message", "No response"))
            
            print("\n=== Demo Complete ===")

            # Example: global graceful shutdown (agents finish current tasks)
            await broadcast_command("shutdown")
        finally:
            #await stop_runtime()
            pass

    asyncio.run(_demo())