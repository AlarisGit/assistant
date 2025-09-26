# Concurrent Processing Architecture Guide

## Overview

This document provides a comprehensive guide to the concurrent processing capabilities of the Alaris Assistant, explaining how the system handles multiple users simultaneously while maintaining enterprise-level reliability and performance.

## Concurrent Processing Architecture

### Non-Blocking Telegram Bot

The Telegram bot (`src/tg.py`) implements a non-blocking architecture that processes multiple users simultaneously:

#### Before: Sequential Processing (Blocking)
```python
async def handle_message(self, update, context):
    thinking_message = await update.message.reply_text("🤔 Thinking...")
    
    # 🚨 BLOCKING: Waits for complete processing (3-10+ seconds)
    response = await assistant.process_user_message(user_id, message_text)
    
    # Only after this completes can next user be processed
    await thinking_message.edit_text(response["message"])
```

**Problem:** User 2 must wait for User 1's complete processing before seeing "Thinking..." message.

#### After: Concurrent Processing (Non-Blocking)
```python
async def handle_message(self, update, context):
    thinking_message = await update.message.reply_text("🤔 Thinking...")
    
    # 🚀 NON-BLOCKING: Start background task and return immediately
    asyncio.create_task(self._process_message_async(user_id, message_text, thinking_message))
    # Handler returns immediately, allowing next user to be processed

async def _process_message_async(self, user_id, message_text, thinking_message):
    # This runs in background for each user independently
    response = await assistant.process_user_message(user_id, message_text)
    await thinking_message.edit_text(response["message"])
```

**Solution:** All users see "Thinking..." immediately, processing happens in parallel background tasks.

### Parallel LLM Processing

The agent framework supports multiple instances of LLM-processing agents for concurrent API calls:

#### Agent Configuration
```python
# Production configuration (src/assistant.py)
_manager = ManagerAgent()        # 1 instance - lightweight routing
_command = CommandAgent()        # 1 instance - administrative commands  
_lang = LangAgent()             # 1 instance - fast language detection
_samplellm1 = SampleLLMAgent()  # 1st concurrent LLM processor
_samplellm2 = SampleLLMAgent()  # 2nd concurrent LLM processor
_translate = TranslationAgent() # 1 instance - future services
```

#### Load Balancing via Redis Consumer Groups

Redis Streams automatically distribute messages across available agent instances:

```python
# When ManagerAgent routes to "samplellm" role:
env.target_role = "samplellm"

# Redis consumer group automatically selects available agent:
# - If samplellm1 is busy → message goes to samplellm2
# - If samplellm2 is busy → message goes to samplellm1  
# - If both busy → message waits in queue for next available
```

### Performance Comparison

#### Scenario: 2 Users Send Messages Simultaneously

**With Concurrent Processing (Current):**
```
T+0.0s: User1 → "Hello" → "Thinking..." appears instantly
T+0.1s: User2 → "Hi"    → "Thinking..." appears instantly
T+0.2s: Both messages enter agent pipeline simultaneously
T+3.0s: Both LLM calls complete, both users get responses
```
**Total time for User2: 3 seconds**

**Without Concurrent Processing (Previous):**
```
T+0.0s: User1 → "Hello" → "Thinking..." appears
T+0.1s: User2 → "Hi"    → waits for User1 to complete
T+3.0s: User1 gets response
T+3.1s: User2 → "Thinking..." finally appears  
T+6.0s: User2 gets response
```
**Total time for User2: 6 seconds (100% slower!)**

## Scaling Configuration

### LLM Provider Limits

Configure agent instances based on your LLM provider's concurrent request limits:

#### OpenAI Configuration
```python
# OpenAI typically allows 3-5 concurrent requests
_samplellm1 = SampleLLMAgent()
_samplellm2 = SampleLLMAgent() 
_samplellm3 = SampleLLMAgent()  # Optional 3rd instance
```

#### Google Gemini Configuration
```python
# Google may have different limits
_samplellm1 = SampleLLMAgent()
_samplellm2 = SampleLLMAgent()
# Check Google's documentation for exact limits
```

#### Ollama (Local) Configuration
```python
# Local processing - depends on hardware
_samplellm1 = SampleLLMAgent()  # May be sufficient for local models
# Add more instances based on GPU/CPU capacity
```

### Redis Connection Scaling

Ensure Redis connection pool can handle all agent instances:

```python
# config.py - Scale connection pool with agent count
REDIS_MAX_CONNECTIONS = 50  # 6 agents × 4 connections + headroom

# Connection usage calculation:
# - Each agent uses ~4 connections (role stream, direct stream, heartbeat, broadcast)
# - 6 agents × 4 = 24 connections used
# - 50 pool size = 48% utilization (healthy)
# - 26 connections remaining for future scaling
```

## Monitoring & Debugging

### Conversation-Specific Logs

Each conversation gets isolated log files for debugging concurrent processing:

```
log/conversations/user123/2025.09.26/22:30:15.log
log/conversations/user456/2025.09.26/22:30:16.log
```

**Log Format:**
```
[2025-09-26 22:30:15.123] [  0.000s] [manager:M4:12345:1] RECEIVED: user123:1727380215.123
[2025-09-26 22:30:15.125] [  0.002s] [manager:M4:12345:1] SENT: to role stream samplellm
[2025-09-26 22:30:15.130] [  0.007s] [samplellm:M4:12345:2] RECEIVED: user123:1727380215.123
[2025-09-26 22:30:18.145] [  3.022s] [samplellm:M4:12345:2] PROCESSING_COMPLETE: LLM response ready
```

### Performance Metrics

Monitor concurrent processing performance:

```python
# Key metrics to track:
# 1. Average response time per user
# 2. Concurrent user count
# 3. LLM agent utilization
# 4. Redis connection pool usage
# 5. Queue depth in Redis streams
```

## Best Practices

### 1. Agent Instance Planning
- **Lightweight agents**: 1 instance (Manager, Command, Lang)
- **Heavy processing agents**: Multiple instances based on provider limits
- **Future services**: 1 instance initially, scale as needed

### 2. Error Handling
- Each background task has independent error handling
- Failed requests don't affect other concurrent users
- Graceful degradation with fallback responses

### 3. Resource Management
- Monitor Redis connection pool utilization
- Scale agent instances based on actual usage patterns
- Use conversation logs to debug specific user issues

### 4. Testing Concurrent Load
```python
# Test script for concurrent processing
import asyncio
import aiohttp

async def test_concurrent_users():
    tasks = []
    for i in range(5):  # Simulate 5 concurrent users
        task = asyncio.create_task(send_message(f"user{i}", f"Hello {i}"))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    print(f"All {len(results)} users processed concurrently")

asyncio.run(test_concurrent_users())
```

## Production Deployment

### Environment Configuration
```bash
# Essential for concurrent processing
export REDIS_MAX_CONNECTIONS=50
export REDIS_SOCKET_TIMEOUT=70.0

# LLM provider configuration
export OPENAI_API_KEY=your_key
export GEN_MODEL=gpt-4@openai

# Telegram bot
export TELEGRAM_BOT_TOKEN=your_token
```

### Deployment Checklist
- ✅ Redis connection pool sized appropriately
- ✅ Multiple LLM agent instances configured
- ✅ Telegram bot using non-blocking architecture
- ✅ Conversation logging enabled for debugging
- ✅ Error handling tested under concurrent load
- ✅ Performance monitoring in place

The concurrent processing architecture transforms the Alaris Assistant from a sequential system into a truly scalable, production-ready platform capable of handling multiple users simultaneously with enterprise-level performance and reliability.
