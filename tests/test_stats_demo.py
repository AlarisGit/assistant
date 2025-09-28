#!/usr/bin/env python3
"""
Demo script to show enhanced statistics in action.
Runs assistant.py and processes a few requests to generate statistics.
"""

import sys
import os
import asyncio
import time
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def demo_enhanced_stats():
    """Demo the enhanced statistics system."""
    
    print("📊 ENHANCED STATISTICS SYSTEM DEMO")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("📈 Statistics will be reported every 15 seconds")
    print("🔄 Processing multiple requests to generate metrics...")
    print()
    
    try:
        # Import and start the system
        # Import assistant to register the application agents
        import assistant  # This registers ManagerAgent, UppercaseAgent, ReverseAgent
        from agent import start_runtime, process_request
        
        # Start the runtime (this will start all agents including StatsAgent)
        await start_runtime()
        print("✅ Runtime started with enhanced StatsAgent")
        print()
        
        # Process several requests to generate statistics
        for i in range(3):
            message = f"Demo message {i+1}: Testing enhanced statistics at {datetime.now().strftime('%H:%M:%S')}"
            print(f"📤 Processing request {i+1}...")
            
            start_time = time.time()
            result = await process_request("manager", f"demo_conv_{i+1}", {"text": message, "stage": "start"})
            end_time = time.time()
            
            print(f"📥 Result {i+1}: {result[:50]}...")
            print(f"⏱️  Processing time: {end_time - start_time:.2f}s")
            print()
            
            # Wait a bit between requests
            await asyncio.sleep(2)
        
        print("🎯 All demo requests completed!")
        print("📊 The StatsAgent is now collecting and will report comprehensive metrics")
        print("⏳ Waiting for statistics report (every 15 seconds)...")
        print()
        
        # Wait for statistics to be reported
        await asyncio.sleep(20)  # Wait 20 seconds to see at least one report
        
        print("✅ Demo completed! Check the logs above for comprehensive statistics.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting enhanced statistics demo...")
    asyncio.run(demo_enhanced_stats())
    print("\n🎉 Demo completed!")
