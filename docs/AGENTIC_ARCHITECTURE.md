# Production-Grade Redis Streams Agentic Architecture

## Overview

This document defines the production-grade Redis Streams-based architecture for the SMS Platform Assistant, implementing a unified agent system with comprehensive reliability, monitoring, and connection management features. The system provides enterprise-level robustness while maintaining developer simplicity through automatic cleanup, connection pooling, retry logic, and comprehensive statistics reporting.

## Architecture Evolution

### Version 2.0 - Production Features
- ✅ **Redis Connection Pool**: Eliminates bottlenecks with 20-connection pool
- ✅ **Automatic Retry Logic**: Exponential backoff for connection failures
- ✅ **Health Monitoring**: Continuous Redis connection health checks
- ✅ **Safety Limits**: Circuit breaker system prevents resource exhaustion
- ✅ **File-Based Statistics**: Clean reporting without logger noise
- ✅ **Automatic Cleanup**: Graceful shutdown on exit/signals
- ✅ **Dynamic Agent Registry**: Runtime agent creation and management
- ✅ **Zero-Config Agents**: Automatic role derivation from class names

## Core Architecture Principles

### 1. Unified Agent Model
- All components inherit from `BaseAgent` with role-specific `process()` methods
- Agents differentiated by `role` rather than class hierarchy
- Stateless design with all conversation data in Envelope payload
- Horizontal scaling through multiple instances per role

### 2. Production-Grade Redis Management
- **Connection Pool**: 20-connection pool with automatic scaling
- **Retry Logic**: Exponential backoff for temporary failures
- **Health Monitoring**: 30-second ping checks with performance tracking
- **Graceful Recovery**: Automatic reconnection and failure detection

### 3. Redis Streams Communication
- **Point-to-Role**: `stream:role:{role}` with consumer groups for load balancing
- **Point-to-Agent**: `stream:agent:{agent_id}` for direct routing
- **Broadcast Control**: Pub/Sub channels for system commands
- **Reply Lists**: Per-request Redis lists for external response delivery

### 4. Envelope-Based State Management
- Single message type containing all conversation data
- No separate Context object - everything in payload
- Complete trace system for debugging and observability
- Manager-controlled history cleanup and state management

## Production System Architecture

### High-Level System Overview

```mermaid
graph TB
    subgraph "External Interfaces"
        TG[Telegram Bot]
        API[REST API]
        CLI[CLI Interface]
    end
    
    subgraph "Redis Connection Pool"
        POOL["Connection Pool<br/>Max: 20 connections<br/>Retry: Exponential Backoff<br/>Health: 30s checks"]
    end
    
    subgraph "Agent Runtime"
        REG["Dynamic Agent Registry<br/>Auto-registration<br/>Runtime creation"]
        STATS["Statistics Agent<br/>File-based reporting<br/>Performance monitoring"]
    end
    
    subgraph "Redis Infrastructure"
        STREAMS["Redis Streams<br/>Role & Agent routing"]
        PUBSUB["Pub/Sub Channels<br/>Broadcast control"]
        LISTS["Result Lists<br/>External responses"]
    end
    
    subgraph "Safety Systems"
        CIRCUIT["Circuit Breaker<br/>Process limits<br/>Time limits<br/>Age limits"]
        CLEANUP["Automatic Cleanup<br/>Signal handlers<br/>Exit handlers"]
    end
    
    TG --> POOL
    API --> POOL
    CLI --> POOL
    POOL --> STREAMS
    POOL --> PUBSUB
    POOL --> LISTS
    REG --> STREAMS
    STATS --> STREAMS
    CIRCUIT --> STREAMS
    CLEANUP --> REG
```

### Current Demo Pipeline

**Current PoC Pipeline:**
```mermaid
sequenceDiagram
    participant E as External Client
    participant M as Manager Agent
    participant U as Uppercase Agent
    participant R as Reverse Agent
    participant L as Result List
    
    E->>M: "Hello, world!" (stage: start)
    M->>U: Forward to uppercase (stage: upper)
    U->>M: "HELLO, WORLD!" (stage: upper)
    M->>R: Forward to reverse (stage: reverse)
    R->>M: "!DLROW ,OLLEH" (stage: reverse)
    M->>L: Final result
    L->>E: "!DLROW ,OLLEH"
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

### Production Envelope Structure

```python
@dataclass
class Envelope:
    # Message Identification
    conversation_id: str              # Conversation correlation key
    message_id: str                   # Unique message identifier
    
    # Routing Information
    target_role: Optional[str]        # Target role for routing
    target_agent_id: Optional[str]    # Direct agent targeting
    target_list: Optional[str]        # Target Redis list for external responses
    
    # Sender Information
    sender_role: str                  # Sender role
    sender_agent_id: str              # Sender agent ID
    
    # Message Type and Data
    kind: str                         # "task" | "result" | "control" | "status"
    payload: Dict[str, Any]           # All conversation data and results
    update_ts: float                  # Last update timestamp
    
    # Response Routing
    result_list: str                  # Final response destination list
    
    # Observability and Safety
    trace: List[Dict[str, Any]]       # Complete execution audit trail
    process_count: int = 0            # Safety: Processing step counter
    total_processing_time: float = 0.0  # Safety: Cumulative processing time
    create_ts: float = 0.0            # Safety: Envelope creation timestamp
```

### Trace Entry Structure

```python
# Each agent adds detailed trace entry
trace_entry = {
    "start_ts": float,               # Processing start time
    "role": str,                     # Agent role
    "agent_id": str,                 # Specific agent instance
    "end_ts": float,                 # Processing end time
    "duration": float,               # Processing duration (end_ts - start_ts)
    "error": Optional[str],          # Error message if processing failed
    "safety_violation": Optional[bool]  # True if safety limits triggered
}
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

## Production Features

### Redis Connection Pool Management

```mermaid
graph LR
    subgraph "Connection Pool Architecture"
        A1[Agent 1] --> POOL["Redis Connection Pool<br/>Max: 20 connections<br/>Health checks: 30s"]
        A2[Agent 2] --> POOL
        A3[Agent 3] --> POOL
        A4[Agent N] --> POOL
        
        POOL --> R1[Redis Connection 1]
        POOL --> R2[Redis Connection 2]
        POOL --> R3[Redis Connection ...]
        POOL --> RN[Redis Connection 20]
        
        R1 --> REDIS[Redis Server]
        R2 --> REDIS
        R3 --> REDIS
        RN --> REDIS
    end
```

#### Connection Pool Features
```python
class RedisConnectionManager:
    """
    Production-grade Redis connection management:
    - Connection pooling (20 max connections)
    - Exponential backoff retry logic
    - Health monitoring with ping checks
    - Automatic failure detection and recovery
    - Graceful connection cleanup
    """
    
    async def initialize(self) -> None:
        retry_policy = Retry(
            backoff=ExponentialBackoff(),
            retries=3,  # Configurable via REDIS_RETRY_ATTEMPTS
            supported_errors=(ConnectionError, TimeoutError, ResponseError)
        )
        
        self.pool = ConnectionPool.from_url(
            REDIS_URL,
            max_connections=20,  # Configurable via REDIS_MAX_CONNECTIONS
            retry=retry_policy,
            socket_timeout=5.0,  # Configurable via REDIS_SOCKET_TIMEOUT
            health_check_interval=30  # Configurable via REDIS_HEALTH_CHECK_INTERVAL
        )
```

### Safety Limits (Circuit Breaker System)

```mermaid
flowchart TD
    START[Envelope Received] --> CHECK{Safety Check}
    
    CHECK -->|Process Count >= 50| STOP1["Circuit Breaker:<br/>Too Many Steps"]
    CHECK -->|Total Time >= 300s| STOP2["Circuit Breaker:<br/>Processing Timeout"]
    CHECK -->|Age >= 600s| STOP3["Circuit Breaker:<br/>Stale Message"]
    CHECK -->|All Limits OK| PROCESS[Process Envelope]
    
    STOP1 --> ERROR["Create Safety Error<br/>Add to Trace<br/>Return to Caller"]
    STOP2 --> ERROR
    STOP3 --> ERROR
    
    PROCESS --> UPDATE["Increment process_count<br/>Add processing_time<br/>Update trace"]
    UPDATE --> SEND[Send to Next Agent]
```

#### Safety Configuration
```python
# Configurable safety limits
MAX_PROCESS_COUNT = 50          # Maximum processing steps
MAX_TOTAL_PROCESSING_TIME = 300.0  # Maximum 5 minutes processing
MAX_ENVELOPE_AGE = 600.0        # Maximum 10 minutes envelope age

# Safety check before each agent processes envelope
def check_safety_limits(env: Envelope) -> Optional[str]:
    if env.process_count >= MAX_PROCESS_COUNT:
        return f"Process count limit exceeded: {env.process_count} >= {MAX_PROCESS_COUNT}"
    
    if env.total_processing_time >= MAX_TOTAL_PROCESSING_TIME:
        return f"Total processing time limit exceeded: {env.total_processing_time:.2f}s >= {MAX_TOTAL_PROCESSING_TIME}s"
    
    if env.create_ts > 0:
        envelope_age = time.time() - env.create_ts
        if envelope_age >= MAX_ENVELOPE_AGE:
            return f"Envelope age limit exceeded: {envelope_age:.2f}s >= {MAX_ENVELOPE_AGE}s"
    
    return None
```

### File-Based Statistics System

```mermaid
graph TD
    STATS[Statistics Agent] --> COLLECT["Collect Metrics:<br/>- Agent performance<br/>- Envelope metrics<br/>- Processing times<br/>- Error rates"]
    
    COLLECT --> LIVE["agent_statistics.txt<br/>(Live reporting every 60s)"]
    COLLECT --> FINAL["agent_final_statistics.txt<br/>(On system shutdown)"]
    
    LIVE --> CLEAN1["Clean Format:<br/>- No logger timestamps<br/>- No icons/emojis<br/>- Professional layout"]
    
    FINAL --> CLEAN2["Session Summary:<br/>- Total uptime<br/>- Agent performance<br/>- System efficiency"]
```

#### Statistics Output Example
```
================================================================================
COMPREHENSIVE AGENT PERFORMANCE STATISTICS
================================================================================

System Uptime: 60.0s | Report Interval: 60.0s
Active Agents: 3

RUN STATISTICS:
   Total Runs: 50 | Since Last Report: 50
   By Role (Total | Since Last):
      manager: 30 | 30
      reverse: 10 | 10
      uppercase: 10 | 10

ENVELOPE METRICS:
   Total Envelopes: 10 | Since Last Report: 10
   Average Envelope Age: 1.00s | Recent (last 10): 1.00s
   Average Processing Time: 1.00s | Recent: 1.00s
   Average Process Count: 1.0 steps | Recent: 1.0 steps

AGENT REGISTRY:
   Agent ID             Role         Status       Last Update  Runs     Avg Time  
   -------------------- ------------ ------------ ------------ -------- ----------
   M4:14278:2           manager      idle         0.9s ago     30       1.002s
   M4:14278:3           uppercase    idle         3.9s ago     10       1.002s
   M4:14278:4           reverse      idle         1.9s ago     10       1.002s
================================================================================
```

### Automatic Cleanup System

```mermaid
flowchart TD
    subgraph "Cleanup Triggers"
        EXIT["Normal Exit<br/>(atexit handler)"]
        SIGINT["Ctrl+C<br/>(SIGINT)"]
        SIGTERM["Process Kill<br/>(SIGTERM)"]
    end
    
    EXIT --> DETECT{"Event Loop<br/>State?"}
    SIGINT --> DETECT
    SIGTERM --> DETECT
    
    DETECT -->|Running| SCHEDULE["Schedule Cleanup Task<br/>loop.create_task()"]
    DETECT -->|Stopped| ASYNC["Run Async Cleanup<br/>asyncio.run()"]
    DETECT -->|Missing| SYNC["Fallback Sync Cleanup<br/>_sync_cleanup()"]
    
    SCHEDULE --> CLEANUP["Emergency Stop Runtime<br/>- Stop agents (2s timeout)<br/>- Close Redis pool (1s timeout)<br/>- Generate final stats"]
    ASYNC --> CLEANUP
    SYNC --> CLEANUP
    
    CLEANUP --> SUCCESS["Graceful Shutdown<br/>✅ No hanging processes<br/>✅ No leaked connections"]
```

### Dynamic Agent Registry

```mermaid
classDiagram
    class BaseAgent {
        +__init__()
        +start()
        +stop()
        +process(env)
        -_register_agent()
    }
    
    class AgentRegistry {
        -_agent_registry: List[BaseAgent]
        -_runtime_started: bool
        +register_agent(agent)
        +start_runtime()
        +stop_runtime()
    }
    
    class ManagerAgent {
        +role: "manager"
        +process(env)
    }
    
    class StatsAgent {
        +role: "stats"
        +process(env)
        +generate_statistics()
    }
    
    BaseAgent <|-- ManagerAgent
    BaseAgent <|-- StatsAgent
    BaseAgent --> AgentRegistry : auto-registers
    AgentRegistry --> BaseAgent : manages lifecycle
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

## Production Configuration

### Environment Variables

```bash
# Redis Connection Pool Settings
REDIS_URL="redis://localhost:6379/0"
REDIS_MAX_CONNECTIONS=20              # Connection pool size
REDIS_RETRY_ATTEMPTS=3                # Retry count for failed operations
REDIS_SOCKET_TIMEOUT=5.0              # Socket operation timeout
REDIS_SOCKET_CONNECT_TIMEOUT=5.0      # Connection establishment timeout
REDIS_HEALTH_CHECK_INTERVAL=30        # Health check frequency (seconds)

# Safety Limits
MAX_PROCESS_COUNT=50                  # Maximum processing steps per envelope
MAX_TOTAL_PROCESSING_TIME=300.0       # Maximum processing time (seconds)
MAX_ENVELOPE_AGE=600.0                # Maximum envelope age (seconds)

# Statistics Reporting
REPORT_INTERVAL_SECONDS=60            # Statistics reporting interval

# Logging
LOG_LEVEL=INFO                        # Logging level
LOG_DIR="./log"                       # Log directory
DEBUG_LOG_FILE="debug.log"            # Debug log file
INFO_LOG_FILE="info.log"              # Info log file
```

### Production Deployment Checklist

- ✅ **Redis Security**: Use TLS and authentication in production
- ✅ **Connection Limits**: Tune `REDIS_MAX_CONNECTIONS` based on load
- ✅ **Safety Limits**: Adjust timeouts based on expected processing times
- ✅ **Monitoring**: Set up log aggregation for statistics files
- ✅ **Health Checks**: Monitor Redis connection health metrics
- ✅ **Resource Limits**: Set appropriate memory and CPU limits
- ✅ **Signal Handling**: Ensure graceful shutdown in container environments

## Development and Testing

### Zero-Config Agent Creation

```python
# Modern agent creation - automatic role derivation and registration
class TranslationAgent(BaseAgent):  # Role automatically becomes "translation"
    def __init__(self):
        super().__init__()  # Zero parameters needed!
    
    async def process(self, env: Envelope) -> Envelope:
        # Business logic here
        env.payload["translation_result"] = await translate(env.payload["text"])
        env.kind = "result"
        return env

# Agent automatically registers and starts when runtime is active
translation_agent = TranslationAgent()  # That's it!
```

### Development Workflow

```mermaid
flowchart TD
    CREATE["1. Create Agent Class<br/>class MyAgent(BaseAgent)"] --> IMPLEMENT["2. Implement process()<br/>Business logic only"]
    IMPLEMENT --> INSTANTIATE["3. Create Instance<br/>agent = MyAgent()"]
    INSTANTIATE --> AUTO["4. Automatic Features<br/>✅ Role derived from class name<br/>✅ Auto-registered in registry<br/>✅ Redis connection managed<br/>✅ Safety limits applied<br/>✅ Statistics collected"]
    AUTO --> TEST["5. Test with Envelopes<br/>Unit test process() method"]
    TEST --> DEPLOY["6. Production Ready<br/>✅ Automatic cleanup<br/>✅ Health monitoring<br/>✅ Error handling"]
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

This production-grade Redis Streams-based architecture provides:

### Core Features
✅ **Simplified State Management**: No complex context merging - everything in Envelope payload  
✅ **Natural Distribution**: Horizontal scaling across hosts with Redis Streams load balancing  
✅ **Complete Observability**: Full trace system with precise timing and error tracking  
✅ **Flexible Routing**: Point-to-role and point-to-agent messaging patterns  
✅ **Integration Ready**: Works seamlessly with existing LLM and RAG infrastructure  

### Production Features
✅ **Redis Connection Pool**: 20-connection pool eliminates bottlenecks and improves performance  
✅ **Automatic Retry Logic**: Exponential backoff handles temporary Redis connection failures  
✅ **Health Monitoring**: Continuous Redis ping checks with performance tracking  
✅ **Safety Limits**: Circuit breaker system prevents infinite loops and resource exhaustion  
✅ **File-Based Statistics**: Clean professional reporting without logger noise  
✅ **Automatic Cleanup**: Graceful shutdown on exit/signals with multiple fallback layers  
✅ **Dynamic Agent Registry**: Runtime agent creation and management  
✅ **Zero-Config Development**: Automatic role derivation and Redis connection management  

### Reliability & Operations
✅ **Comprehensive Error Handling**: Safety violations, timeouts, and connection failures  
✅ **Production Monitoring**: Real-time statistics and performance metrics  
✅ **Graceful Degradation**: System continues operating during minor issues  
✅ **Developer Experience**: Ultra-simple agent creation with automatic infrastructure management  
✅ **Enterprise Ready**: Configurable limits, security features, and deployment checklist  

### Architecture Benefits

```mermaid
graph TB
    subgraph "Developer Experience"
        SIMPLE["Ultra-Simple<br/>Zero-config agents<br/>Automatic registration<br/>Focus on business logic"]
    end
    
    subgraph "Production Reliability"
        ROBUST["Enterprise-Grade<br/>Connection pooling<br/>Retry logic<br/>Safety limits<br/>Health monitoring"]
    end
    
    subgraph "Operational Excellence"
        OPS["Operations-Friendly<br/>File-based statistics<br/>Automatic cleanup<br/>Graceful shutdown<br/>Comprehensive logging"]
    end
    
    SIMPLE --> REDIS["Redis Streams<br/>Distributed Processing"]
    ROBUST --> REDIS
    OPS --> REDIS
    
    REDIS --> SCALE["Horizontal Scaling<br/>High Availability<br/>Production Ready"]
```

The architecture successfully balances **developer simplicity** with **production robustness**, providing enterprise-grade reliability while maintaining an ultra-simple development experience. The system starts with a working proof-of-concept and scales seamlessly to production workloads.
