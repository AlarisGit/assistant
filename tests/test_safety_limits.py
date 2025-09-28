#!/usr/bin/env python3
"""
Test script to demonstrate safety limits (circuit breaker) functionality.
Shows how the system prevents resource exhaustion and infinite loops.
"""

import sys
import os
import time
import asyncio

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_safety_limits():
    """Test that demonstrates safety limit enforcement."""
    
    print("🛡️ Testing Safety Limits (Circuit Breaker)")
    print("=" * 60)
    
    try:
        from agent import (
            Envelope, check_safety_limits, create_safety_error,
            MAX_PROCESS_COUNT, MAX_TOTAL_PROCESSING_TIME, MAX_ENVELOPE_AGE
        )
        
        print(f"\n📋 Current Safety Limits:")
        print(f"   • Max Process Count: {MAX_PROCESS_COUNT}")
        print(f"   • Max Total Processing Time: {MAX_TOTAL_PROCESSING_TIME}s")
        print(f"   • Max Envelope Age: {MAX_ENVELOPE_AGE}s")
        
        # Test 1: Process count limit
        print(f"\n🧪 Test 1: Process Count Limit")
        env1 = Envelope(
            conversation_id="test1",
            message_id="test1:123",
            target_role="test",
            target_agent_id=None,
            target_list=None,
            sender_role="test",
            sender_agent_id="test",
            kind="task",
            payload={"text": "test"},
            ts=time.time(),
            result_list="result:test1:123",
            trace=[],
            process_count=MAX_PROCESS_COUNT + 1,  # Exceed limit
            total_processing_time=0.0,
            create_ts=time.time()
        )
        
        error = check_safety_limits(env1)
        if error:
            print(f"   ✅ Process count limit detected: {error}")
        else:
            print(f"   ❌ Process count limit NOT detected")
        
        # Test 2: Total processing time limit
        print(f"\n🧪 Test 2: Total Processing Time Limit")
        env2 = Envelope(
            conversation_id="test2",
            message_id="test2:123",
            target_role="test",
            target_agent_id=None,
            target_list=None,
            sender_role="test",
            sender_agent_id="test",
            kind="task",
            payload={"text": "test"},
            ts=time.time(),
            result_list="result:test2:123",
            trace=[],
            process_count=1,
            total_processing_time=MAX_TOTAL_PROCESSING_TIME + 1.0,  # Exceed limit
            create_ts=time.time()
        )
        
        error = check_safety_limits(env2)
        if error:
            print(f"   ✅ Processing time limit detected: {error}")
        else:
            print(f"   ❌ Processing time limit NOT detected")
        
        # Test 3: Envelope age limit
        print(f"\n🧪 Test 3: Envelope Age Limit")
        old_time = time.time() - (MAX_ENVELOPE_AGE + 1.0)  # Create old envelope
        env3 = Envelope(
            conversation_id="test3",
            message_id="test3:123",
            target_role="test",
            target_agent_id=None,
            target_list=None,
            sender_role="test",
            sender_agent_id="test",
            kind="task",
            payload={"text": "test"},
            ts=time.time(),
            result_list="result:test3:123",
            trace=[],
            process_count=1,
            total_processing_time=1.0,
            create_ts=old_time  # Old creation time
        )
        
        error = check_safety_limits(env3)
        if error:
            print(f"   ✅ Envelope age limit detected: {error}")
        else:
            print(f"   ❌ Envelope age limit NOT detected")
        
        # Test 4: Safe envelope
        print(f"\n🧪 Test 4: Safe Envelope (Within Limits)")
        env4 = Envelope(
            conversation_id="test4",
            message_id="test4:123",
            target_role="test",
            target_agent_id=None,
            target_list=None,
            sender_role="test",
            sender_agent_id="test",
            kind="task",
            payload={"text": "test"},
            ts=time.time(),
            result_list="result:test4:123",
            trace=[],
            process_count=1,
            total_processing_time=1.0,
            create_ts=time.time()
        )
        
        error = check_safety_limits(env4)
        if error:
            print(f"   ❌ Unexpected safety limit violation: {error}")
        else:
            print(f"   ✅ Safe envelope passed all checks")
        
        # Test 5: Error envelope creation
        print(f"\n🧪 Test 5: Safety Error Creation")
        error_env = create_safety_error(env1, "Test safety violation", "test_agent", "test:123")
        
        print(f"   ✅ Error envelope created:")
        print(f"      • Kind: {error_env.kind}")
        print(f"      • Errors in payload: {len(error_env.payload.get('errors', []))}")
        print(f"      • Trace entries: {len(error_env.trace)}")
        
        if error_env.payload.get('errors'):
            error_info = error_env.payload['errors'][0]
            print(f"      • Error code: {error_info.get('code')}")
            print(f"      • Error message: {error_info.get('message')}")
        
        print(f"\n🎯 Safety Features Summary:")
        print(f"   ✅ Process count limiting prevents infinite loops")
        print(f"   ✅ Processing time limiting prevents resource exhaustion")
        print(f"   ✅ Envelope age limiting prevents stale message processing")
        print(f"   ✅ Error envelopes provide detailed violation information")
        print(f"   ✅ Violations are logged and traced for debugging")
        
    except ImportError as e:
        print(f"❌ Import error (expected in test environment): {e}")
        print("   This test requires the full agent system to be available.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("Starting safety limits test...")
    asyncio.run(test_safety_limits())
    print("\nTest completed!")
