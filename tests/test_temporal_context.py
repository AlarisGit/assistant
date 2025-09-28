#!/usr/bin/env python3
"""
Test script for temporal context enhancements in conversation history.

This script demonstrates:
1. Timestamped conversation history with show_timestamps=True
2. Current time context in LLM prompts via {current_time} template
3. How EssenceAgent can track conversation evolution over time
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import ConversationMemory, get_redis


async def test_temporal_context():
    """Test the temporal context functionality."""
    print("🕐 Testing Temporal Context in Conversation History")
    print("=" * 60)
    
    # Initialize Redis connection
    redis = await get_redis()
    
    # Create a test conversation memory
    conversation_id = "test_temporal_context"
    memory = ConversationMemory(redis, conversation_id)
    
    # Clear any existing messages
    await memory.clear_messages()
    
    print("\n📝 Step 1: Adding messages with time delays")
    
    # Add messages with time delays to simulate real conversation
    test_messages = [
        {"role": "user", "content": "Hello, I need help with API authentication"},
        {"role": "assistant", "content": "I'd be happy to help you with API authentication. What specific authentication method are you trying to implement?"},
        {"role": "user", "content": "I'm not sure, what options do I have?"},
        {"role": "assistant", "content": "There are several options: API keys, OAuth 2.0, JWT tokens, and basic authentication. Each has different use cases."},
        {"role": "user", "content": "Let's go with OAuth then"},
        {"role": "assistant", "content": "Great choice! OAuth 2.0 is secure and widely supported. Are you building a web application, mobile app, or server-to-server integration?"},
    ]
    
    base_time = time.time() - 300  # Start 5 minutes ago
    
    for i, msg in enumerate(test_messages):
        # Add realistic time delays between messages
        message_time = base_time + (i * 45)  # 45 seconds between messages
        await memory.add_message(msg["role"], msg["content"], {"timestamp": message_time})
        
        # Format time for display
        dt = datetime.fromtimestamp(message_time)
        time_str = dt.strftime("%Y%m%d %H%M%S")
        print(f"  [{time_str}] {msg['role']}: {msg['content']}")
        
        # Small delay to make it feel realistic
        await asyncio.sleep(0.1)
    
    print("\n🕐 Step 2: Testing different history retrieval modes")
    
    # Test regular history (no timestamps)
    print("\n📖 Regular History (no timestamps):")
    regular_history = await memory.get_history()
    for i, (user_msg, assistant_msg) in enumerate(regular_history, 1):
        print(f"  {i}. User: {user_msg}")
        print(f"     Assistant: {assistant_msg}")
    
    # Test timestamped history
    print("\n🕐 Timestamped History (show_timestamps=True):")
    timestamped_history = await memory.get_history(show_timestamps=True)
    for i, (user_msg, assistant_msg) in enumerate(timestamped_history, 1):
        print(f"  {i}. User: {user_msg}")
        print(f"     Assistant: {assistant_msg}")
    
    print("\n⏰ Step 3: Testing current time context")
    
    # Show current time formatting
    current_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d %H%M%S")
    print(f"\nCurrent time for LLM context: {current_time}")
    
    # Demonstrate how this would be used in prompt_options
    prompt_options = {
        'language': 'English',
        'current_time': current_time
    }
    print(f"Prompt options: {prompt_options}")
    
    print("\n🎯 Step 4: Temporal analysis benefits")
    
    print("""
Temporal Context Benefits:
- **Conversation Evolution**: Track how user intent changes over time
- **Urgency Detection**: Identify time-sensitive requests
- **Session Analysis**: Understand conversation pacing and flow
- **Context Awareness**: LLM knows current time vs message timestamps
- **Follow-up Timing**: Detect when users return after breaks
- **Intent Persistence**: Track how long users pursue specific topics
    """)
    
    print("\n📊 Step 5: Example temporal patterns")
    
    # Analyze the conversation timing
    messages = await memory.get_messages()
    if len(messages) >= 2:
        first_time = messages[0].get("timestamp", time.time())
        last_time = messages[-1].get("timestamp", time.time())
        duration = last_time - first_time
        
        print(f"Conversation duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
        print(f"Average time between messages: {duration/(len(messages)-1):.0f} seconds")
        
        # Show time gaps
        print("\nMessage timing analysis:")
        for i in range(1, len(messages)):
            prev_time = messages[i-1].get("timestamp", time.time())
            curr_time = messages[i].get("timestamp", time.time())
            gap = curr_time - prev_time
            
            prev_dt = datetime.fromtimestamp(prev_time)
            curr_dt = datetime.fromtimestamp(curr_time)
            
            print(f"  Gap {i}: {gap:.0f}s between {prev_dt.strftime('%H:%M:%S')} and {curr_dt.strftime('%H:%M:%S')}")
    
    print("\n✅ Step 6: EssenceAgent integration")
    
    print("""
EssenceAgent Temporal Features:
- Receives timestamped conversation history for context
- Gets current_time in prompt for temporal awareness
- Can detect conversation patterns and timing
- Understands when topics were discussed
- Can identify stale vs fresh conversation threads
- Helps with intent tracking over time
    """)
    
    # Cleanup
    await memory.clear_messages()
    await redis.aclose()
    
    print("🕐 Temporal context testing completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_temporal_context())
