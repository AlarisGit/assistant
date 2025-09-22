#!/usr/bin/env python3
"""
Distributed Memory System Demonstration

This example shows how to use the distributed memory subsystem in the agentic assistant.
The memory system provides Redis-backed, conversation-scoped storage that works across
processes and hosts with a dict/list-like interface.

Key Features:
- Dict-like operations: get, set, update, keys, items
- List operations: append, extend, pop, insert, remove
- Conversation history management
- Automatic TTL and cleanup
- Memory usage statistics
- Seamless integration with BaseAgent
"""

import asyncio
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import BaseAgent, Envelope, start_runtime, stop_runtime, process_request
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryDemoAgent(BaseAgent):
    """Demonstration agent showing various memory operations."""
    
    async def process(self, env: Envelope) -> Envelope:
        """Demonstrate comprehensive memory usage patterns."""
        logger.info(f"[MemoryDemoAgent] Processing message: {env.message_id}")
        
        # Get conversation memory
        memory = await self.get_memory(env.conversation_id)
        
        # Demo 1: Basic dict-like operations
        await self._demo_dict_operations(memory, env)
        
        # Demo 2: List operations
        await self._demo_list_operations(memory, env)
        
        # Demo 3: Conversation history
        await self._demo_conversation_history(memory, env)
        
        # Demo 4: User preferences and settings
        await self._demo_user_preferences(memory, env)
        
        # Demo 5: Memory statistics
        await self._demo_memory_stats(memory, env)
        
        # Return processed envelope
        env.payload["processed_by"] = "MemoryDemoAgent"
        env.payload["memory_demo_complete"] = True
        env.kind = "result"
        
        return env
    
    async def _demo_dict_operations(self, memory, env):
        """Demonstrate dictionary-like operations."""
        logger.info("=== Dict Operations Demo ===")
        
        # Set values
        await memory.set("demo_key", "demo_value")
        await memory.set("user_id", env.conversation_id)
        await memory.set("session_start", time.time())
        
        # Get values
        demo_value = await memory.get("demo_key")
        user_id = await memory.get("user_id")
        session_start = await memory.get("session_start")
        
        logger.info(f"Retrieved values: demo_key={demo_value}, user_id={user_id}")
        
        # Update multiple values at once
        await memory.update({
            "last_activity": time.time(),
            "activity_count": 1,
            "demo_status": "active"
        })
        
        # Check if key exists
        has_demo_key = await memory.__contains__("demo_key")
        logger.info(f"Has demo_key: {has_demo_key}")
        
        # Get all keys
        all_keys = await memory.keys()
        logger.info(f"All memory keys: {all_keys}")
    
    async def _demo_list_operations(self, memory, env):
        """Demonstrate list operations."""
        logger.info("=== List Operations Demo ===")
        
        # Initialize a list
        await memory.set("demo_list", [])
        
        # Append items
        await memory.append("demo_list", "first_item")
        await memory.append("demo_list", "second_item")
        await memory.append("demo_list", {"type": "object", "value": 42})
        
        # Extend with multiple items
        await memory.extend("demo_list", ["third_item", "fourth_item"])
        
        # Get the list
        demo_list = await memory.get_list("demo_list")
        logger.info(f"Demo list: {demo_list}")
        
        # List length
        list_length = await memory.list_length("demo_list")
        logger.info(f"List length: {list_length}")
        
        # Pop an item
        popped_item = await memory.pop("demo_list")
        logger.info(f"Popped item: {popped_item}")
        
        # Insert at specific position
        await memory.insert("demo_list", 1, "inserted_item")
        
        # Final list state
        final_list = await memory.get_list("demo_list")
        logger.info(f"Final demo list: {final_list}")
    
    async def _demo_conversation_history(self, memory, env):
        """Demonstrate conversation history management."""
        logger.info("=== Conversation History Demo ===")
        
        # Add messages to conversation history
        await memory.add_message("user", "Hello, how are you?")
        await memory.add_message("assistant", "I'm doing well, thank you!")
        await memory.add_message("user", "Can you help me with something?")
        await memory.add_message("assistant", "Of course! What do you need help with?")
        
        # Get message count
        message_count = await memory.get_message_count()
        logger.info(f"Total messages in conversation: {message_count}")
        
        # Get all messages
        all_messages = await memory.get_messages()
        logger.info("Conversation history:")
        for i, msg in enumerate(all_messages):
            logger.info(f"  {i+1}. [{msg['role']}] {msg['content']} (at {msg['timestamp']})")
        
        # Get limited messages (last 2)
        recent_messages = await memory.get_messages(limit=2)
        logger.info(f"Recent messages (last 2): {len(recent_messages)} messages")
    
    async def _demo_user_preferences(self, memory, env):
        """Demonstrate user preferences and settings management."""
        logger.info("=== User Preferences Demo ===")
        
        # Initialize user preferences
        default_prefs = {
            "language": "en",
            "timezone": "UTC",
            "notification_settings": {
                "email": True,
                "sms": False,
                "push": True
            },
            "theme": "dark",
            "response_style": "detailed"
        }
        
        # Set preferences (only if not already set)
        existing_prefs = await memory.get("user_preferences", default_prefs)
        if existing_prefs == default_prefs:
            await memory.set("user_preferences", default_prefs)
            logger.info("Initialized default user preferences")
        else:
            logger.info("User preferences already exist")
        
        # Update specific preference
        prefs = await memory.get("user_preferences")
        prefs["last_login"] = time.time()
        prefs["session_count"] = prefs.get("session_count", 0) + 1
        await memory.set("user_preferences", prefs)
        
        logger.info(f"Updated user preferences: {prefs}")
        
        # Track user activity
        activity_log = await memory.get_list("activity_log", [])
        await memory.append("activity_log", {
            "action": "message_processed",
            "timestamp": time.time(),
            "message_id": env.message_id
        })
        
        activity_count = await memory.list_length("activity_log")
        logger.info(f"User activity log has {activity_count} entries")
    
    async def _demo_memory_stats(self, memory, env):
        """Demonstrate memory statistics and monitoring."""
        logger.info("=== Memory Statistics Demo ===")
        
        # Get memory size for this conversation
        memory_size = await memory.get_memory_size()
        logger.info(f"Memory usage for conversation {env.conversation_id}:")
        logger.info(f"  Total fields: {memory_size['total_fields']}")
        logger.info(f"  Total size: {memory_size['total_size_bytes']} bytes")
        logger.info(f"  Fields: {memory_size['fields']}")
        
        # Get all items in memory
        all_items = await memory.items()
        logger.info(f"All memory items ({len(all_items)} total):")
        for key, value in all_items:
            value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            logger.info(f"  {key}: {value_preview}")


class ConversationManagerAgent(BaseAgent):
    """Agent that manages conversation lifecycle and cleanup."""
    
    async def process(self, env: Envelope) -> Envelope:
        """Manage conversation lifecycle."""
        action = env.payload.get("action", "process")
        
        if action == "cleanup":
            # Clean up conversation memory
            await self.cleanup_memory(env.conversation_id)
            logger.info(f"Cleaned up memory for conversation {env.conversation_id}")
            env.payload["cleanup_complete"] = True
        
        elif action == "stats":
            # Get memory statistics
            stats = await self.get_memory_stats()
            env.payload["memory_stats"] = stats
            logger.info(f"Memory statistics: {stats}")
        
        else:
            # Regular processing
            memory = await self.get_memory(env.conversation_id)
            
            # Track conversation metadata
            await memory.set("last_processed", time.time())
            
            # Increment processing count
            process_count = await memory.get("process_count", 0)
            await memory.set("process_count", process_count + 1)
            
            logger.info(f"Processed conversation {env.conversation_id} (count: {process_count + 1})")
        
        env.kind = "result"
        return env


async def run_memory_demo():
    """Run comprehensive memory system demonstration."""
    logger.info("Starting Memory System Demonstration")
    
    # Create demo agents
    demo_agent = MemoryDemoAgent()
    manager_agent = ConversationManagerAgent()
    
    try:
        # Start the runtime
        await start_runtime()
        
        # Demo 1: Basic memory operations
        logger.info("\n" + "="*60)
        logger.info("DEMO 1: Basic Memory Operations")
        logger.info("="*60)
        
        result1 = await process_request("memorydemo", "demo_conv_1", {
            "text": "Test message for memory demo",
            "demo_type": "basic_operations"
        })
        logger.info(f"Demo 1 Result: {result1}")
        
        # Demo 2: Conversation management
        logger.info("\n" + "="*60)
        logger.info("DEMO 2: Conversation Management")
        logger.info("="*60)
        
        result2 = await process_request("conversationmanager", "demo_conv_1", {
            "action": "process",
            "text": "Managing conversation lifecycle"
        })
        logger.info(f"Demo 2 Result: {result2}")
        
        # Demo 3: Memory statistics
        logger.info("\n" + "="*60)
        logger.info("DEMO 3: Memory Statistics")
        logger.info("="*60)
        
        result3 = await process_request("conversationmanager", "demo_conv_1", {
            "action": "stats"
        })
        logger.info(f"Demo 3 Result: {result3}")
        
        # Demo 4: Multiple conversations
        logger.info("\n" + "="*60)
        logger.info("DEMO 4: Multiple Conversations")
        logger.info("="*60)
        
        # Process multiple conversations to show isolation
        for i in range(3):
            conv_id = f"demo_conv_{i+2}"
            result = await process_request("memorydemo", conv_id, {
                "text": f"Message for conversation {i+2}",
                "demo_type": "multi_conversation"
            })
            logger.info(f"Conversation {conv_id} processed")
        
        # Demo 5: Memory cleanup
        logger.info("\n" + "="*60)
        logger.info("DEMO 5: Memory Cleanup")
        logger.info("="*60)
        
        # Clean up one conversation
        cleanup_result = await process_request("conversationmanager", "demo_conv_1", {
            "action": "cleanup"
        })
        logger.info(f"Cleanup Result: {cleanup_result}")
        
        # Final statistics
        final_stats = await process_request("conversationmanager", "demo_conv_2", {
            "action": "stats"
        })
        logger.info(f"Final Statistics: {final_stats}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Stop the runtime
        await stop_runtime()
        logger.info("Memory System Demonstration Complete")


if __name__ == "__main__":
    asyncio.run(run_memory_demo())
