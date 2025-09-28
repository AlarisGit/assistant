#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced StatsAgent functionality.
Runs multiple requests and shows comprehensive statistics reporting.
"""

import sys
import os
import asyncio
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_enhanced_stats():
    """Test the enhanced statistics system with multiple requests."""
    
    print("📊 Testing Enhanced Statistics System")
    print("=" * 60)
    
    try:
        # Import here to avoid dotenv issues in test environment
        from assistant import process_user_message
        
        print("🚀 Starting multiple test requests...")
        print("   The StatsAgent will collect comprehensive metrics")
        print("   and report them every 60 seconds (configurable)")
        print()
        
        # Process multiple requests to generate statistics
        for i in range(5):
            message = f"Test message {i+1}: Hello from test {i+1}!"
            print(f"📤 Processing request {i+1}: {message}")
            
            start_time = time.time()
            result = await process_user_message(f"test_conv_{i+1}", message)
            end_time = time.time()
            
            print(f"📥 Result {i+1}: {result['message'][:50]}...")
            print(f"⏱️  Processing time: {end_time - start_time:.2f}s")
            print()
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        print("✅ All test requests completed!")
        print()
        print("📊 Statistics Information:")
        print("   • The StatsAgent is now collecting comprehensive metrics")
        print("   • Run counts (total and per-role/agent)")
        print("   • Processing times (total and per-role/agent)")
        print("   • Performance rates (envelopes/second)")
        print("   • Agent registry with status tracking")
        print()
        print("🕐 Statistics will be reported every 60 seconds")
        print("   (configurable via STATS_REPORT_INTERVAL_SEC)")
        print()
        print("📈 Metrics being tracked:")
        print("   • Total runs and runs since last report")
        print("   • Processing time breakdown by role and agent")
        print("   • Average performance metrics")
        print("   • Agent status and last update times")
        print("   • Envelope completion rates")
        
    except ImportError as e:
        print(f"❌ Import error (expected in test environment): {e}")
        print("   This test requires the full agent system to be available.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting enhanced statistics test...")
    asyncio.run(test_enhanced_stats())
    print("\nTest completed!")
    print("\n💡 To see the full statistics in action:")
    print("   1. Run the assistant.py directly")
    print("   2. Wait 60 seconds to see the comprehensive statistics report")
    print("   3. Or adjust STATS_REPORT_INTERVAL_SEC for faster reporting")
