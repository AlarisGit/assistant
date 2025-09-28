#!/usr/bin/env python3
"""
Test script for normalized conversation history functionality.

This script demonstrates the dual storage approach where:
1. Original user messages are preserved in their original language/form
2. Normalized English versions are stored in metadata
3. Agents can choose between raw or normalized history
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import ConversationMemory, get_redis
import time


async def test_normalized_history():
    """Test the normalized history functionality."""
    print("🧪 Testing Normalized Conversation History")
    print("=" * 50)
    
    # Initialize Redis connection
    redis = await get_redis()
    
    # Create a test conversation memory
    conversation_id = "test_normalized_history"
    memory = ConversationMemory(redis, conversation_id)
    
    # Clear any existing messages
    await memory.clear_messages()
    
    print("\n📝 Step 1: Adding original user messages")
    
    # Add some test messages with different languages/quality
    test_messages = [
        {"role": "user", "content": "Привет, как дела?", "lang": "ru"},
        {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. How are you?"},
        {"role": "user", "content": "I has good day today", "lang": "en"},  # Grammar error
        {"role": "assistant", "content": "That's wonderful to hear! I'm glad you had a good day."},
        {"role": "user", "content": "你好，这是一个测试", "lang": "zh"},
        {"role": "assistant", "content": "Hello! This is indeed a test. How can I help you?"},
    ]
    
    for msg in test_messages:
        await memory.add_message(msg["role"], msg["content"])
        print(f"  Added {msg['role']}: {msg['content']}")
    
    print("\n🔄 Step 2: Simulating normalization process")
    
    # Simulate what TranslationAgent would do - add normalized versions
    normalizations = [
        {"original": "Привет, как дела?", "normalized": "Hello, how are you?", "type": "translation", "lang": "ru"},
        {"original": "I has good day today", "normalized": "I had a good day today", "type": "correction", "lang": "en"},
        {"original": "你好，这是一个测试", "normalized": "Hello, this is a test", "type": "translation", "lang": "zh"},
    ]
    
    # Get all messages and update user messages with normalized content
    messages = await memory.get_messages()
    user_msg_index = 0
    
    for i, message in enumerate(messages):
        if message.get("role") == "user":
            if user_msg_index < len(normalizations):
                norm = normalizations[user_msg_index]
                
                # Update the message with normalized content
                if "metadata" not in message:
                    message["metadata"] = {}
                
                message["metadata"]["content_norm"] = norm["normalized"]
                message["metadata"]["norm"] = {
                    "type": norm["type"],
                    "source_lang": norm["lang"],
                    "confidence": 0.95,
                    "model": "test",
                    "ts": time.time()
                }
                
                print(f"  Normalized: '{norm['original']}' → '{norm['normalized']}' ({norm['type']})")
                user_msg_index += 1
    
    # Save the updated messages
    await memory.set("messages", messages)
    
    print("\n📖 Step 3: Testing different history retrieval modes")
    
    # Test raw history
    print("\n🔤 Raw History (original messages):")
    raw_history = await memory.get_history(normalized=False)
    for i, (user_msg, assistant_msg) in enumerate(raw_history, 1):
        print(f"  {i}. User: {user_msg}")
        print(f"     Assistant: {assistant_msg}")
    
    # Test normalized history
    print("\n✨ Normalized History (English corrected/translated):")
    normalized_history = await memory.get_history(normalized=True)
    for i, (user_msg, assistant_msg) in enumerate(normalized_history, 1):
        print(f"  {i}. User: {user_msg}")
        print(f"     Assistant: {assistant_msg}")
    
    print("\n🔍 Step 4: Comparing the differences")
    
    print("\nComparison:")
    for i, ((raw_user, raw_assistant), (norm_user, norm_assistant)) in enumerate(zip(raw_history, normalized_history), 1):
        if raw_user != norm_user:
            print(f"  Message {i} - User text changed:")
            print(f"    Raw:        '{raw_user}'")
            print(f"    Normalized: '{norm_user}'")
        else:
            print(f"  Message {i} - No changes needed")
    
    print("\n✅ Step 5: Validation")
    
    # Validate that we have both versions
    assert len(raw_history) == len(normalized_history), "History lengths should match"
    assert len(raw_history) == 3, "Should have 3 conversation pairs"
    
    # Check specific transformations
    assert raw_history[0][0] == "Привет, как дела?", "Raw Russian should be preserved"
    assert normalized_history[0][0] == "Hello, how are you?", "Russian should be translated"
    
    assert raw_history[1][0] == "I has good day today", "Raw grammar error should be preserved"
    assert normalized_history[1][0] == "I had a good day today", "Grammar should be corrected"
    
    assert raw_history[2][0] == "你好，这是一个测试", "Raw Chinese should be preserved"
    assert normalized_history[2][0] == "Hello, this is a test", "Chinese should be translated"
    
    print("✅ All validations passed!")
    
    print("\n🎯 Step 6: Usage recommendations")
    print("""
Usage Guidelines:
- Language Detection: Use get_history(normalized=False) to detect original languages
- Translation: Use get_history(normalized=False) to avoid circular dependencies  
- LLM Response Generation: Use get_history(normalized=True) for coherent English context
- Debugging/Logging: Use both versions to show original vs processed text
- Auditing: Always preserve raw messages for compliance and reprocessing
    """)
    
    # Cleanup
    await memory.clear_messages()
    await redis.aclose()
    
    print("🧪 Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_normalized_history())
