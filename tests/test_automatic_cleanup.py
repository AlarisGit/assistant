#!/usr/bin/env python3
"""
Test script to demonstrate automatic cleanup functionality.
This script intentionally "forgets" to call stop_runtime() to show
that the atexit handler will clean up automatically.
"""

import sys
import os
import asyncio

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_automatic_cleanup():
    """Test that demonstrates automatic cleanup on exit."""
    from agent import start_runtime, get_registered_agents
    
    print("🧪 Testing Automatic Cleanup System")
    print("=" * 50)
    
    print("\n1. Starting runtime...")
    await start_runtime()
    
    agents = get_registered_agents()
    print(f"✅ Started {len(agents)} agents:")
    for agent in agents:
        print(f"   - {agent.role} agent {agent.agent_id}")
    
    print("\n2. Simulating normal application work...")
    await asyncio.sleep(1)  # Simulate some work
    
    print("\n3. 🚨 INTENTIONALLY NOT CALLING stop_runtime()!")
    print("   This simulates a developer forgetting to clean up.")
    print("   The atexit handler should automatically clean up when the process exits.")
    
    print("\n4. Process will now exit...")
    print("   Watch for automatic cleanup messages in the logs!")
    
    # Process exits here without calling stop_runtime()
    # The atexit handler should kick in and clean up automatically

if __name__ == "__main__":
    print("Starting test that demonstrates automatic cleanup...")
    asyncio.run(test_automatic_cleanup())
    print("Test completed - process exiting (cleanup should happen automatically)")
