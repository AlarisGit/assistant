# Redis Streams-Based Agentic Architecture

## Overview

This document defines the Redis Streams-based architecture for the SMS Platform Assistant, implementing a unified agent system where all conversation state flows through Envelope messages. This approach eliminates complex context merging while providing natural distributed processing capabilities.

## Core Architecture Principles

### 1. Unified Agent Model
- All components inherit from `BaseAgent` with role-specific `process()` methods
- Agents differentiated by `role` rather than class hierarchy
- Stateless design with all conversation data in Envelope payload
- Horizontal scaling through multiple instances per role

### 2. Redis Streams Communication
- **Point-to-Role**: `stream:role:{role}` with consumer groups for load balancing
- **Point-to-Agent**: `stream:agent:{agent_id}` for direct routing
- **Broadcast Control**: Pub/Sub channels for system commands
- **Reply Lists**: Per-request Redis lists for external response delivery

### 3. Envelope-Based State Management
- Single message type containing all conversation data
- No separate Context object - everything in payload
- Complete trace system for debugging and observability
- Manager-controlled history cleanup and state management

## Current Implementation

### Demo Pipeline

**Current PoC Pipeline:**
```
manager → uppercase → manager → reverse → manager → result_list
```

#### Implemented Agents
```python
class ManagerAgent(BaseAgent):
    """Pipeline orchestrator with state machine routing"""
    role = "manager"
    
class UppercaseAgent(BaseAgent):
    """Example worker: converts text to uppercase"""
    role = "uppercase"
    
class ReverseAgent(BaseAgent):
    """Example worker: reverses text character order"""
    role = "reverse"
    
class StatsAgent(BaseAgent):
    """Monitoring agent for lifecycle events"""
    role = "stat"
```

### Future SMS Assistant Agents

**Planned Production Pipeline:**
```
manager → translate → clarify → search_docs → summarize → quality_control → result_list
```

#### Future Core Agents
```python
class TranslationAgent(BaseAgent):
    """Detects language and translates to English"""
    role = "translate"
    
class ClarificationAgent(BaseAgent):
    """Determines if query needs clarification"""
    role = "clarify"
    
class SearchAgent(BaseAgent):
    """RAG-based document search using Qdrant"""
    role = "search_docs"
    
class SummarizationAgent(BaseAgent):
    """LLM-powered response synthesis"""
    role = "summarize"
    
class QualityControlAgent(BaseAgent):
    """Response evaluation and retry triggering"""
    role = "quality_control"
```

### Envelope Structure

```python
@dataclass
class Envelope:
    conversation_id: str              # Conversation correlation key
    message_id: str                   # Unique message identifier
    target_role: Optional[str]        # Target role for routing
    target_agent_id: Optional[str]    # Direct agent targeting
    target_list: Optional[str]        # Target Redis list for external responses
    
    sender_role: str                  # Sender role
    sender_agent_id: str              # Sender agent ID
    
    kind: str                         # "task" | "result" | "control" | "status"
    payload: Dict[str, Any]           # All conversation data and results
    ts: float                         # Timestamp
    
    result_list: str                  # Final response destination list
    trace: List[Dict[str, Any]]       # Complete execution audit trail
```

### Current Demo Payload Schema

```python
payload = {
    "text": "Hello, world!",           # Text being processed
    "stage": "start",                 # Pipeline stage marker
    "errors": [],                     # Accumulated errors
    "__agent_timeout_sec": 30.0       # Per-message timeout override
}
```

### Future SMS Assistant Payload Schema

```python
payload = {
    # Conversation State
    "messages": [                     # Full conversation history
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."}
    ],
    "user_preferences": {             # User settings
        "language": "auto",
        "response_style": "detailed",
        "timezone": "UTC"
    },
    "user_query": str,                # Current user input
    
    # Agent Results (accumulated through pipeline)
    "translation": {
        "detected_language": "spanish",
        "translated_query": "How to configure SMS settings?",
        "confidence": 0.95
    },
    "clarification": {
        "certainty_level": 0.8,
        "clarifying_questions": ["Which SMS feature specifically?"],
        "needs_clarification": False
    },
    "search_results": {
        "chunks": [{"content": "...", "source": "...", "score": 0.9}],
        "sources": ["docs/sms-config.html", "docs/sms-api.html"],
        "relevance_scores": [0.9, 0.85]
    },
    "summary": {
        "response": "To configure SMS settings...",
        "confidence": 0.92,
        "sources_used": ["docs/sms-config.html"]
    },
    
    # Pipeline Control
    "stage": "after_translate",       # Current pipeline position
    "retry_count": 0,                 # Retry attempts
    "errors": [],                     # Accumulated errors
    "__agent_timeout_sec": 30.0       # Per-message timeout override
}
```

## Redis Communication Patterns

### Stream Keys and Consumer Groups
```
# Role streams (load balanced)
stream:role:manager          → cg:role:manager
stream:role:translate        → cg:role:translate
stream:role:clarify          → cg:role:clarify
stream:role:search_docs      → cg:role:search_docs
stream:role:summarize        → cg:role:summarize

# Agent streams (direct routing)
stream:agent:{agent_id}      → cg:agent:{agent_id}

# Broadcast channels
broadcast:all                # System-wide commands
broadcast:role:stat          # Lifecycle events
broadcast:role:{role}        # Role-specific commands

# Result lists
result:{message_id}          # External response delivery
```

### Message Flow Example
```python
# 1. External request
await process_request("conv_123", "Hello, world!")

# 2. Initial envelope to manager
XADD "stream:role:manager" {
    "envelope": json({
        "conversation_id": "conv_123",
        "message_id": "conv_123:1726339057.123",
        "target_role": "manager",
        "sender_role": "external",
        "kind": "task",
        "payload": {"text": "Hello, world!", "stage": "start"},
        "result_list": "result:conv_123:1726339057.123"
    })
}

# 3. Manager routes to uppercase
XADD "stream:role:uppercase" {
    "envelope": json({...payload: {"stage": "upper"}...})
}

# 4. Uppercase agent processes and returns to manager
# BaseAgent automatically sets target_role = env.sender_role ("manager")

# 5. Manager routes to reverse
XADD "stream:role:reverse" {
    "envelope": json({...payload: {"stage": "reverse"}...})
}

# 6. Final response delivery to Redis list
LPUSH "result:conv_123:1726339057.123" json({final_envelope})
```

## Integration with Existing Infrastructure

### LLM Interface Integration
```python
class TranslationAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        import llm  # Use existing provider-agnostic interface
        
        response = await llm.generate(
            model=config.TRANSLATION_MODEL,  # From config
            prompt=f"Detect language and translate: {env.payload['user_query']}",
            max_tokens=512
        )
        
        env.payload["translation"] = {
            "detected_language": "auto-detected",
            "translated_query": response.content,
            "confidence": 0.95
        }
        env.kind = "result"
        return env
```

### RAG Pipeline Integration
```python
class SearchAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        import qdrant  # Use existing RAG infrastructure
        
        query = env.payload["translation"]["translated_query"]
        
        # Embedding generation hidden at low level
        chunks = await qdrant.search_similar(
            query=query,
            limit=5,
            score_threshold=0.7
        )
        
        env.payload["search_results"] = {
            "chunks": chunks,
            "sources": [chunk["source"] for chunk in chunks],
            "relevance_scores": [chunk["score"] for chunk in chunks]
        }
        env.kind = "result"
        return env
```

### Telegram Bot Integration
```python
# Direct integration in tg.py
from agent import process_request

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    conversation_id = f"tg_{update.effective_chat.id}"
    
    try:
        response = await process_request(conversation_id, user_message)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")
```

## Concurrency and Scaling

### Horizontal Scaling
```
# Multiple agents per role across hosts
Host A: ManagerAgent, TranslationAgent
Host B: ManagerAgent, SearchAgent  
Host C: SummarizationAgent, QualityControlAgent

# Redis automatically load balances within consumer groups
```

### Backpressure Management
```python
# Agents pause consumption when busy
if self._busy.is_set():
    await asyncio.sleep(0.05)
    continue

# Ensures fair distribution across agent instances
```

### State Management
```python
# All state in Envelope - no external storage needed
env.payload["messages"] = conversation_history
env.payload["user_preferences"] = user_settings

# Manager handles history cleanup
if len(env.payload["messages"]) > MAX_HISTORY:
    env.payload["messages"] = env.payload["messages"][-MAX_HISTORY:]
```

## Reliability and Error Handling

### Automatic Return Routing
```python
# BaseAgent automatically routes responses back to sender
env.target_role = env.sender_role  # Send back to sender by default
env.target_agent_id = None
env.target_list = None

# Agents can override this behavior in their process() method
```

### Safety Timeouts
```python
# Default and per-message timeouts
DEFAULT_TASK_TIMEOUT_SEC = 60.0
timeout = float(env.payload.get("__agent_timeout_sec", self.task_timeout_sec))
env2 = await asyncio.wait_for(self.process(env), timeout=timeout)
```

### Error Accumulation and Trace System
```python
# Execution trace with timing and error information
trace_item = {
    'start_ts': time.time(),
    'role': self.role,
    'agent_id': self.agent_id,
    'end_ts': time.time(),
    'duration': duration,
    'exception': exception_info  # If error occurred
}
env.trace.append(trace_item)

# Errors added to payload for manager routing decisions
env.payload.setdefault("errors", []).append({
    "code": "manager.stage",
    "message": f"Unknown stage '{stage}' for kind '{env.kind}'"
})
```

### Quality Control Flow
```python
# QC agent evaluates and triggers retries
if quality_score < QUALITY_THRESHOLD:
    env.payload["retry_count"] += 1
    env.reply_role = "clarify"  # Return to clarification
else:
    env.reply_list = original_reply_list  # Send to user
```

## Observability

### Complete Trace System
```python
# Each agent adds detailed trace entry with timing
trace_item = {
    'start_ts': time.time(),
    'role': self.role,
    'agent_id': self.agent_id,
    'end_ts': time.time(),
    'duration': end_ts - start_ts,
    'exception': exception_info  # If error occurred
}
env.trace.append(trace_item)

# Final envelope contains complete pipeline execution history with precise timing
```

### Lifecycle Monitoring
```python
# StatsAgent receives all lifecycle events
{
    "kind": "status",
    "event": "heartbeat",  # init | heartbeat | reload | exit
    "role": self.role,
    "agent_id": self.agent_id,
    "busy": self._busy.is_set(),
    "ts": time.time()
}
```

## Development and Testing

### Adding New Agents
```python
# 1. Create agent class
class NewFeatureAgent(BaseAgent):
    def __init__(self, redis: Redis):
        super().__init__(redis, role="new_feature")
    
    async def process(self, env: Envelope) -> Envelope:
        # Process and update payload
        env.payload["new_feature_result"] = {...}
        env.kind = "result"
        return env

# 2. Update ManagerAgent routing logic
# 3. Add to runtime startup
# 4. Test with isolated envelopes
```

### Testing Strategy
```python
# Unit test individual agents
async def test_uppercase_agent():
    env = Envelope(
        conversation_id="test",
        message_id="test_1",
        target_role="uppercase",
        target_agent_id=None,
        target_list=None,
        sender_role="external",
        sender_agent_id="test",
        kind="task",
        payload={"text": "hello world", "stage": "upper"},
        ts=time.time(),
        result_list="result:test_1",
        trace=[]
    )
    
    agent = UppercaseAgent(mock_redis)
    result = await agent.process(env)
    
    assert result.payload["text"] == "HELLO WORLD"
    assert result.kind == "result"
```

## Future Extensions

### Planned Optimizations
1. **Agent Specialization**: Split into domain-specific agents after PoC
2. **Priority Streams**: `stream:role:{role}:prio:{level}` for urgent requests
3. **Result Caching**: Cache expensive operations (search, LLM calls)
4. **DLQ Implementation**: Dead letter queues for failed messages
5. **Conversation Sharding**: For strict ordering requirements
6. **Advanced Routing**: ML-based workflow selection

### Performance Enhancements
```python
# Future: Priority-based routing
stream:role:manager:prio:high     # Urgent user requests
stream:role:manager:prio:normal   # Standard queries
stream:role:manager:prio:low      # Background processing

# Future: Intelligent caching
class CachedSearchAgent(SearchAgent):
    async def process(self, env: Envelope) -> Envelope:
        cache_key = hash(env.payload["translation"]["translated_query"])
        if cached := await self.redis.get(f"search_cache:{cache_key}"):
            env.payload["search_results"] = json.loads(cached)
        else:
            # Perform search and cache result
            pass
```

## Security Considerations

### Data Isolation
- Each conversation isolated by `conversation_id`
- No shared state between conversations
- All data contained within Envelope payload

### Access Control
```python
# Redis security for production
REDIS_URL = "rediss://user:pass@redis.example.com:6380/0"  # TLS + auth

# Input validation in agents
async def validate_envelope(self, env: Envelope) -> bool:
    # Validate conversation_id format
    # Check payload size limits
    # Sanitize user inputs
    return True
```

### Audit Trail
Complete execution history preserved in `env.trace` for security auditing and debugging.

---

## Summary

This Redis Streams-based architecture provides:

✅ **Simplified State Management**: No complex context merging  
✅ **Natural Distribution**: Horizontal scaling across hosts  
✅ **Complete Observability**: Full trace system and lifecycle monitoring  
✅ **Flexible Routing**: Point-to-role and point-to-agent messaging  
✅ **Reliability**: Timeouts, error handling, and retry mechanisms  
✅ **Integration Ready**: Works with existing LLM and RAG infrastructure  

The design starts with a simple proof-of-concept pipeline and provides clear extension points for future optimization and specialization.
