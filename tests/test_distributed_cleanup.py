#!/usr/bin/env python3
"""
Test Distributed Memory Cleanup

This test demonstrates that cleanup methods work across different agents
in a distributed system, even when the agent performing cleanup has never
worked with that conversation before.
"""

import asyncio
import logging
import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import BaseAgent, Envelope, start_runtime, stop_runtime, get_memory_manager
from assistant import process_user_message, clear_conversation_history, clear_all_conversation_data, get_conversation_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CleanupTestAgent(BaseAgent):
    """Agent specifically for testing distributed cleanup functionality."""
    
    async def process(self, env: Envelope) -> Envelope:
        """Test cleanup operations from an agent that never worked with the conversation."""
        action = env.payload.get("action")
        target_conversation = env.payload.get("target_conversation")
        
        logger.info(f"[CleanupTestAgent] Testing {action} on conversation {target_conversation}")
        
        if action == "test_partial_cleanup":
            # Test partial cleanup (messages only) from a different agent
            await self.cleanup_message_history(target_conversation)
            env.payload["result"] = f"Partial cleanup completed for {target_conversation}"
            
        elif action == "test_full_cleanup":
            # Test full cleanup from a different agent
            await self.cleanup_memory(target_conversation)
            env.payload["result"] = f"Full cleanup completed for {target_conversation}"
            
        elif action == "test_stats":
            # Test getting stats from a different agent
            stats = await self.get_memory_stats()
            env.payload["result"] = f"Memory stats retrieved: {len(stats.get('conversations', []))} conversations"
            
        elif action == "direct_memory_test":
            # Test direct memory access to any conversation
            memory = await self.get_memory(target_conversation)
            
            # Try to access memory that might not exist locally
            message_count = await memory.get_message_count()
            user_prefs = await memory.get("user_preferences", {})
            
            env.payload["result"] = f"Direct access: {message_count} messages, prefs: {user_prefs}"
        
        return env


async def test_distributed_cleanup():
    """Test cleanup functionality across distributed agents."""
    logger.info("=== Testing Distributed Memory Cleanup ===")
    
    try:
        # Start runtime
        await start_runtime()
        
        # Create test agent
        cleanup_agent = CleanupTestAgent()
        
        # Test 1: Create conversation data with regular agents
        logger.info("\n1. Creating conversation data with regular pipeline...")
        
        test_conversations = ["test_user_1", "test_user_2", "test_user_3"]
        
        for conv_id in test_conversations:
            # Send messages to build conversation history
            for i in range(3):
                message = f"Test message {i+1} for {conv_id}"
                result = await process_user_message(conv_id, message)
                logger.info(f"   Created message for {conv_id}: {result.get('message', 'No response')[:50]}...")
        
        # Test 2: Check initial stats
        logger.info("\n2. Initial conversation stats:")
        for conv_id in test_conversations:
            stats = await get_conversation_stats(conv_id)
            logger.info(f"   {conv_id}: {stats.get('message_count', 0)} messages, {len(stats.get('memory_usage', {}).get('fields', []))} fields")
        
        # Test 3: Test partial cleanup from different agent (CleanupTestAgent)
        logger.info("\n3. Testing partial cleanup from CleanupTestAgent (never worked with these conversations)...")
        
        from agent import process_request
        
        # Use CleanupTestAgent to clean up test_user_1 (partial)
        result = await process_request("cleanuptest", "cleanup_session", {
            "action": "test_partial_cleanup",
            "target_conversation": "test_user_1"
        })
        logger.info(f"   Partial cleanup result: {result}")
        
        # Verify partial cleanup worked
        stats_after_partial = await get_conversation_stats("test_user_1")
        logger.info(f"   After partial cleanup - Messages: {stats_after_partial.get('message_count', 0)}, Fields: {len(stats_after_partial.get('memory_usage', {}).get('fields', []))}")
        
        # Test 4: Test full cleanup from different agent
        logger.info("\n4. Testing full cleanup from CleanupTestAgent...")
        
        # Use CleanupTestAgent to clean up test_user_2 (full)
        result = await process_request("cleanuptest", "cleanup_session", {
            "action": "test_full_cleanup",
            "target_conversation": "test_user_2"
        })
        logger.info(f"   Full cleanup result: {result}")
        
        # Verify full cleanup worked
        stats_after_full = await get_conversation_stats("test_user_2")
        logger.info(f"   After full cleanup - Messages: {stats_after_full.get('message_count', 0)}, Fields: {len(stats_after_full.get('memory_usage', {}).get('fields', []))}")
        
        # Test 5: Test direct memory access from different agent
        logger.info("\n5. Testing direct memory access from CleanupTestAgent...")
        
        result = await process_request("cleanuptest", "cleanup_session", {
            "action": "direct_memory_test",
            "target_conversation": "test_user_3"
        })
        logger.info(f"   Direct memory access result: {result}")
        
        # Test 6: Test cleanup via assistant.py methods
        logger.info("\n6. Testing cleanup via assistant.py methods...")
        
        # Test partial cleanup via assistant.py
        partial_success = await clear_conversation_history("test_user_3")
        logger.info(f"   Assistant partial cleanup success: {partial_success}")
        
        # Create new conversation for full cleanup test
        await process_user_message("test_user_4", "Message for full cleanup test")
        stats_before_full = await get_conversation_stats("test_user_4")
        logger.info(f"   Before full cleanup - Messages: {stats_before_full.get('message_count', 0)}, Fields: {len(stats_before_full.get('memory_usage', {}).get('fields', []))}")
        
        # Test full cleanup via assistant.py
        full_success = await clear_all_conversation_data("test_user_4")
        logger.info(f"   Assistant full cleanup success: {full_success}")
        
        stats_after_assistant_full = await get_conversation_stats("test_user_4")
        logger.info(f"   After assistant full cleanup - Messages: {stats_after_assistant_full.get('message_count', 0)}, Fields: {len(stats_after_assistant_full.get('memory_usage', {}).get('fields', []))}")
        
        # Test 7: Verify cleanup works across agent instances
        logger.info("\n7. Testing cleanup across different agent instances...")
        
        # Get memory manager directly
        manager = await get_memory_manager()
        
        # Create memory in one agent context
        memory1 = manager.get_memory("cross_agent_test")
        await memory1.set("test_data", {"created_by": "direct_access", "timestamp": time.time()})
        await memory1.add_message("system", "Test message from direct access")
        
        logger.info("   Created memory via direct manager access")
        
        # Clean it up from CleanupTestAgent
        result = await process_request("cleanuptest", "cleanup_session", {
            "action": "test_full_cleanup",
            "target_conversation": "cross_agent_test"
        })
        logger.info(f"   Cleanup from CleanupTestAgent: {result}")
        
        # Verify it's gone
        memory2 = manager.get_memory("cross_agent_test")
        remaining_fields = await memory2.keys()
        logger.info(f"   Remaining fields after cross-agent cleanup: {remaining_fields}")
        
        logger.info("\n=== All Distributed Cleanup Tests Complete ===")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    
    finally:
        await stop_runtime()


if __name__ == "__main__":
    asyncio.run(test_distributed_cleanup())
