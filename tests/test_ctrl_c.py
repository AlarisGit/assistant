#!/usr/bin/env python3
"""
Test script to demonstrate improved Ctrl+C handling.
Run this and press Ctrl+C during processing to see immediate response.
"""

import sys
import os
import asyncio

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_ctrl_c_handling():
    """Test that demonstrates responsive Ctrl+C handling."""
    
    print("🧪 Testing Ctrl+C Handling")
    print("=" * 50)
    
    print("\n1. Starting runtime...")
    
    # Import here to avoid the dotenv issue
    try:
        from agent import start_runtime, process_request
        
        await start_runtime()
        print("✅ Runtime started successfully")
        
        print("\n2. Starting a request that will take some time...")
        print("   💡 Try pressing Ctrl+C during processing!")
        print("   The system should respond immediately now.")
        
        # Start a request
        payload = {"text": "Hello, world! This is a test message.", "stage": "start"}
        
        print("\n3. Processing request...")
        result = await process_request("manager", "test_conv", payload)
        
        print(f"\n4. ✅ Result: {result}")
        
    except KeyboardInterrupt:
        print("\n🎯 Ctrl+C detected - this should now be handled gracefully!")
        print("   The system should exit cleanly without hanging.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   This might be due to missing dependencies in the test environment.")

if __name__ == "__main__":
    print("Starting Ctrl+C handling test...")
    print("Press Ctrl+C during processing to test immediate response!")
    print()
    
    try:
        asyncio.run(test_ctrl_c_handling())
    except KeyboardInterrupt:
        print("\n🎯 Ctrl+C caught at top level - system should exit cleanly!")
    
    print("\nTest completed!")
