"""
================================================================================
Unified async agent architecture on Redis (Streams + Pub/Sub).
All participants (client, manager, workers, stats) inherit from BaseAgent and
differ only by their 'role' and overridden process() method.

This module is intentionally self-contained and heavily documented to minimize
onboarding time for new contributors. Keep comments and docstrings in English.

--------------------------------------------------------------------------------
TABLE OF CONTENTS
--------------------------------------------------------------------------------
1) High-Level Overview
2) Terminology and Roles
3) Message Model (Envelope)
4) Addressing & Routing
5) Redis Keys and Consumer Groups
6) Lifecycle, Heartbeats, and Control
7) Concurrency & Backpressure
8) Timeouts & Reliability (ACKs, DLQ, Idempotency)
9) Scaling Patterns (Horizontal, Sharding, Priorities)
10) Observability (Stats role, what to log/emit)
11) Security Notes
12) Usage Patterns (recipes and examples)
13) Known Limitations
14) Extensibility Guide

--------------------------------------------------------------------------------
1) HIGH-LEVEL OVERVIEW
--------------------------------------------------------------------------------
- This module implements a unified agent runtime on top of Redis 5+.
- All components are "agents" derived from BaseAgent:
  * Client-facing "requester" code creates an initial Envelope and pushes it into
    a role stream (e.g., to 'manager').
  * ManagerAgent is just another agent with role='manager'; it routes messages
    between roles/agents according to a scenario (stages/pipeline).
  * Worker agents perform actual tasks (LLM call, parsing, enrichment, etc.).
  * StatsAgent listens to lifecycle/heartbeat broadcast events for monitoring.

- Communication channels:
  * Point-to-role messages: Redis Streams "stream:role:{role}" with a consumer
    group "cg:role:{role}". Exactly one agent of that role processes a message.
  * Point-to-agent messages: Redis Streams "stream:agent:{agent_id}" with
    "cg:agent:{agent_id}". Use this to address a specific agent instance.
  * Broadcast commands: Redis Pub/Sub channels:
      "broadcast:all" and "broadcast:role:{role}"
    Used for system-wide or role-scoped control (shutdown/reload, etc.).

- Backpressure & fairness:
  * An agent pauses role/direct stream consumption while it is busy processing
    a task, but continues handling broadcast commands. This ensures fair
    distribution across same-role agents (a busy agent won't keep pulling more).

--------------------------------------------------------------------------------
2) TERMINOLOGY AND ROLES
--------------------------------------------------------------------------------
- role: a logical function an agent fulfills (e.g., "manager", "uppercase",
  "reverse", "stat", "vectorize", "llm", etc.).
- agent_id: a unique identifier of a concrete agent instance:
    "{HOSTNAME}:{PID}:{COUNTER}"
  This enables multiple agents of the same role in a single process or across
  multiple hosts.

- Predefined roles in this example:
  * "manager": decides routing between worker roles.
  * "uppercase": transforms text to uppercase.
  * "reverse": reverses text.
  * "stat": listens to lifecycle/heartbeat events over Pub/Sub.

--------------------------------------------------------------------------------
3) MESSAGE MODEL (ENVELOPE)
--------------------------------------------------------------------------------
- Envelope is the only message type in Streams; it carries:
  * Addressing: target_role, target_agent_id
  * Sender: sender_role, sender_agent_id
  * Payload: arbitrary JSON (domain data + control flags)
  * Trace: a list of breadcrumbs for debugging
  * Timestamps: ts (float)

- Final responses for external callers:
  * The "result_list" is a per-request Redis LIST where the final Envelope is
    pushed (LPUSH). The caller BRPOP's from it to get the result.

--------------------------------------------------------------------------------
4) ADDRESSING & ROUTING
--------------------------------------------------------------------------------
- Role-routed task:
    XADD "stream:role:{role}" { envelope=json(...) }
  Exactly one agent of that role processes it (via XREADGROUP), then ACKs.

- Direct message to a specific agent:
    XADD "stream:agent:{agent_id}" { envelope=json(...) }
  Use this when you must route to a particular instance (sticky session, local
  cache, device/adapter bound to that instance, etc.).

- Broadcast:
    PUBLISH "broadcast:all" | "broadcast:role:{role}" json({"cmd": ...})
  All subscribers receive and handle it (e.g., "shutdown", "reload").


--------------------------------------------------------------------------------
5) REDIS KEYS AND CONSUMER GROUPS
--------------------------------------------------------------------------------
- Role streams:     stream:role:{role}
- Agent streams:    stream:agent:{agent_id}
- Consumer groups:  cg:role:{role}, cg:agent:{agent_id}
- Broadcast:        broadcast:all, broadcast:role:{role}

- Ensure groups exist with MKSTREAM (see ensure_group()).
- We use XACK after processing.

--------------------------------------------------------------------------------
6) LIFECYCLE, HEARTBEATS, AND CONTROL
--------------------------------------------------------------------------------
- On start(): emits status "init" to broadcast:role:stat
- Heartbeats: every HEARTBEAT_INTERVAL_SEC emit "heartbeat" with busy state.
- On stop(): emits status "exit".
- Broadcast control:
  * {"cmd": "shutdown"} → agent completes current task (if any) and stops.
  * {"cmd": "reload"}   → placeholder for hot-reload config.

--------------------------------------------------------------------------------
7) CONCURRENCY & BACKPRESSURE
--------------------------------------------------------------------------------
- Each agent has two stream loops:
    _role_loop()   reads from its role stream
    _direct_loop() reads from its private agent stream
  Both loops are paused while the agent is "busy" to avoid over-pulling.

- Broadcast loop is independent and always processed when possible.

- This model enables running multiple instances per role across processes/hosts.

--------------------------------------------------------------------------------
8) TIMEOUTS & RELIABILITY (ACKs, DLQ, IDEMPOTENCY)
--------------------------------------------------------------------------------
- Safety timeout:
  * DEFAULT_TASK_TIMEOUT_SEC = 60.0 (configurable per agent).
  * Per-message override via payload["__agent_timeout_sec"] (float seconds).
  * If elapsed → we wrap the Envelope with an error and route as "result".

- ACKs:
  * We XACK after processing each message.
  * If the agent crashes mid-task, pending entries remain in the group pending
    list and can be claimed by another consumer (add claim logic if needed).

- DLQ (not implemented here, but recommended):
  * On repeated failures, XADD to "stream:role:{role}:dlq" with attempts count.

- Idempotency:
  * Upstream systems should include a stable message_id per logical task.
  * Agents/manager can store short-lived dedup markers (e.g., SETEX) keyed by
    message_id, if reprocessing must be avoided.

--------------------------------------------------------------------------------
9) SCALING PATTERNS (HORIZONTAL, SHARDING, PRIORITIES)
--------------------------------------------------------------------------------
- Horizontal: start more agents per role on more hosts; they share the same
  "stream:role:{role}" consumer group.
- Sharding (optional if strict ordering is required):
  * Use N sub-streams "stream:role:{role}:shard:{i}" and map by hash(conversation_id).
- Priorities:
  * Maintain multiple role streams like "stream:role:{role}:prio:{L}"
    and poll in descending order.

--------------------------------------------------------------------------------
10) OBSERVABILITY
--------------------------------------------------------------------------------
- StatsAgent (role="stat") can listen to broadcast:role:stat for:
    {"event":"init|heartbeat|reload|exit", ...}
- Recommended:
  * Persist heartbeats and liveness in a time-series DB (with TTL).
  * Track task_start/task_finish in trace (already appended).
  * Export counters/histograms (success/fail/timeout by role).

--------------------------------------------------------------------------------
11) SECURITY NOTES
--------------------------------------------------------------------------------
- Streams/PubSub aren't authenticated/authorized by themselves.
- If deploying across trust boundaries, secure Redis (auth/TLS, network ACLs).
- Sanitize payloads; do not trust external inputs for unsafe operations.

--------------------------------------------------------------------------------
12) USAGE PATTERNS
--------------------------------------------------------------------------------
A) External caller (see process_request()):
   - Build message_id and result_list
   - Send an initial Envelope to role 'manager'
   - BRPOP result_list for the final result

B) Broadcast commands:
   - await broadcast_command("shutdown")                  # all agents
   - await broadcast_command("reload", role="uppercase")  # only uppercase role

C) Direct messaging to an agent (example snippet):
   - Suppose you have a specific agent_id to target (e.g., sticky session).
     Use send_direct_task() helper to push to stream:agent:{agent_id}.

D) Adding a new worker:
   - Subclass BaseAgent(role="newrole"), override process(env), and return env.
   - Have ManagerAgent route to "newrole" based on scenario.

--------------------------------------------------------------------------------
13) KNOWN LIMITATIONS
--------------------------------------------------------------------------------
- No strict per-conversation ordering: multiple messages of the same conversation
  may be processed concurrently by multiple agents.
- No retry/claim logic implemented for pending entries (add if needed).
- No DLQ stream yet (template suggested above).
- In-process instance counter resets on restart (agent_id remains unique enough
  when combined with HOSTNAME and PID).

--------------------------------------------------------------------------------
14) EXTENSIBILITY GUIDE
--------------------------------------------------------------------------------
- Extend Envelope with domain fields or standardize error schema.
- Implement DLQ and retry backoff in BaseAgent._emit_error() or in Manager.
- Implement shard-aware streams if you need ordering.
- Add a ConfigAgent that listens to "reload" and updates a shared config key.
- Replace the example workers with real tasks (e.g., LLM/RAG, I/O, CPU work).
- Replace ManagerAgent.process() logic with a rule engine or state machine.

================================================================================
"""

import asyncio
import atexit
import json
import os
import signal
import socket
import time
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path

from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ResponseError, ConnectionError, TimeoutError
from redis.retry import Retry
from redis.backoff import ExponentialBackoff

import config
import logging
from util import get_hostname, get_pid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -------------------------- Environment / constants --------------------------

REDIS_URL = os.getenv("REDIS_URL", config.REDIS_URL)

def role_stream_key(role: str) -> str:
    """Redis stream key for role-addressed messages."""
    return f"stream:role:{role}"

def agent_stream_key(agent_id: str) -> str:
    """Redis stream key for direct agent-addressed messages."""
    return f"stream:agent:{agent_id}"

def role_group_name(role: str) -> str:
    """Consumer group name for a role stream."""
    return f"cg:role:{role}"

def agent_group_name(agent_id: str) -> str:
    """Consumer group name for a dedicated agent stream."""
    return f"cg:agent:{agent_id}"

def role_broadcast_channel(role: str) -> str:
    """Pub/Sub channel name for a specific role broadcasts."""
    return f"broadcast:role:{role}"

BROADCAST_ALL = "broadcast:all"
STATUS_ROLE = "stats"  # lifecycle/heartbeat broadcast role (matches StatsAgent role)

HEARTBEAT_INTERVAL_SEC = 5.0
DEFAULT_TASK_TIMEOUT_SEC = 60.0  # safety timeout per agent (can be overridden per-envelope)
STATS_REPORT_INTERVAL_SEC = 60.0  # how often to print comprehensive statistics

# -------------------------- Safety Limits (Circuit Breaker) --------------------------

# Maximum number of processing steps before circuit breaker trips
MAX_PROCESS_COUNT = 50

# Maximum total processing time in seconds before circuit breaker trips
# This is cumulative time spent in agent.process() methods (sum of trace item durations)
MAX_TOTAL_PROCESSING_TIME = 300.0  # 5 minutes

# Maximum age of envelope in seconds before circuit breaker trips
# This is time since envelope creation (create_ts), not individual processing step time
MAX_ENVELOPE_AGE = 600.0  # 10 minutes

# -------------------------- Utilities (host, pid, counters) --------------------------

_instance_counter = 0
def get_next_counter() -> int:
    """Monotonic in-process counter (not persisted)."""
    global _instance_counter
    _instance_counter += 1
    return _instance_counter

def build_agent_id() -> str:
    """AGENT_ID = {HOSTNAME}:{PID}:{COUNTER}."""
    return f"{get_hostname()}:{get_pid()}:{get_next_counter()}"

def derive_role_from_class_name(class_name: str) -> str:
    """Derive agent role from class name.
    
    Removes 'Agent' suffix and converts to lowercase.
    Examples:
    - 'ManagerAgent' -> 'manager'
    - 'TranslationAgent' -> 'translation'
    - 'ClarificationAgent' -> 'clarification'
    - 'StatsAgent' -> 'stats'
    
    This ensures perfect consistency between class names and roles,
    eliminating the possibility of typos or mismatched role names.
    """
    if class_name.endswith('Agent'):
        role = class_name[:-5]  # Remove 'Agent' suffix
    else:
        role = class_name
    return role.lower()

# -------------------------- Envelope --------------------------

@dataclass
class Envelope:
    """
    Canonical payload for Streams messages.

    Fields:
      conversation_id: logical request correlation key (cross-message)
      message_id:      unique per-message identifier (e.g., f"{conv}:{time.time()}")
      target_role:     desired role to handle this message; None for "any role" (rare)
      target_agent_id: direct agent id; when set, message is for stream:agent:{id}
      target_list:     direct list id; when set, message is for list:{id}

      sender_role:     role of the sender (for audit/debug)
      sender_agent_id: concrete agent id of the sender

      payload:arbitrary JSON (domain inputs/outputs, control flags, errors)
      update_ts: unix timestamp (last update time)

      trace: Breadcrumbs to trace the path (agent id, role, ts, notes)
    """
    conversation_id: str
    message_id: str
    target_role: Optional[str]
    target_agent_id: Optional[str]
    target_list: Optional[str]

    sender_role: str
    sender_agent_id: str

    payload: Dict[str, Any]
    update_ts: float

    result_list: str

    trace: List[Dict[str, Any]]  # Individual processing step records with start_ts/end_ts/duration
    
    # Safety/Circuit Breaker attributes
    process_count: int = 0  # Number of times this envelope has been processed by agents
    total_processing_time: float = 0.0  # Cumulative time spent in agent.process() methods (sum of trace durations)
    create_ts: float = 0.0  # Envelope creation ts (for measuring overall envelope age, not individual steps)

    def to_stream_fields(self) -> Dict[str, str]:
        """Serialize to field/value pairs for XADD."""
        return {"envelope": json.dumps(asdict(self), ensure_ascii=False)}

    @staticmethod
    def from_stream_fields(fields: Dict[bytes, bytes]) -> "Envelope":
        """Deserialize from XREAD/XREADGROUP raw fields."""
        env = json.loads(fields[b"envelope"].decode("utf-8"))
        return Envelope(**env)
    
    def __str__(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

# -------------------------- Distributed Memory Subsystem --------------------------

class ConversationMemory:
    """
    Distributed conversation memory with dict/list-like interface backed by Redis.
    
    Provides seamless access to conversation-scoped data across processes and hosts.
    All operations are atomic and support JSON-serializable data types.
    
    Usage:
        memory = ConversationMemory(redis, "conv_123")
        
        # Dict-like operations
        memory["user_preferences"] = {"language": "en", "style": "detailed"}
        prefs = memory["user_preferences"]
        
        # List operations for conversation history
        memory.append("messages", {"role": "user", "content": "Hello"})
        history = memory.get_list("messages")
        
        # Automatic cleanup with TTL
        memory.set_ttl(3600)  # 1 hour
    """
    
    def __init__(self, redis: Redis, conversation_id: str, default_ttl: int = 3600):
        self.redis = redis
        self.conversation_id = conversation_id
        self.default_ttl = default_ttl
        self._prefix = f"memory:conv:{conversation_id}"
    
    def _key(self, field: str) -> str:
        """Generate Redis key for a memory field."""
        return f"{self._prefix}:{field}"
    
    async def __getitem__(self, field: str) -> Any:
        """Get a value from memory (dict-like access)."""
        key = self._key(field)
        value = await self.redis.get(key)
        if value is None:
            raise KeyError(f"Memory field '{field}' not found for conversation {self.conversation_id}")
        return json.loads(value.decode('utf-8'))
    
    async def __setitem__(self, field: str, value: Any) -> None:
        """Set a value in memory (dict-like access)."""
        key = self._key(field)
        serialized = json.dumps(value, ensure_ascii=False)
        await self.redis.setex(key, self.default_ttl, serialized)
    
    async def __delitem__(self, field: str) -> None:
        """Delete a field from memory."""
        key = self._key(field)
        deleted = await self.redis.delete(key)
        if deleted == 0:
            raise KeyError(f"Memory field '{field}' not found for conversation {self.conversation_id}")
    
    async def __contains__(self, field: str) -> bool:
        """Check if a field exists in memory."""
        key = self._key(field)
        return await self.redis.exists(key) > 0
    
    async def get(self, field: str, default: Any = None) -> Any:
        """Get a value with default fallback (dict-like)."""
        try:
            return await self[field]
        except KeyError:
            return default
    
    async def delete(self, field: str) -> bool:
        """Delete a field from memory.
        
        Args:
            field: The field name to delete
            
        Returns:
            bool: True if field was deleted, False if it didn't exist
        """
        key = self._key(field)
        deleted = await self.redis.delete(key)
        return deleted > 0
    
    async def set(self, field: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value with optional custom TTL."""
        key = self._key(field)
        serialized = json.dumps(value, ensure_ascii=False)
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        await self.redis.setex(key, ttl_to_use, serialized)
    
    async def update(self, data: Dict[str, Any]) -> None:
        """Update multiple fields at once (dict-like)."""
        pipe = self.redis.pipeline()
        for field, value in data.items():
            key = self._key(field)
            serialized = json.dumps(value, ensure_ascii=False)
            pipe.setex(key, self.default_ttl, serialized)
        await pipe.execute()
    
    async def keys(self) -> List[str]:
        """Get all field names in this conversation memory."""
        pattern = f"{self._prefix}:*"
        keys = await self.redis.keys(pattern)
        # Extract field names by removing prefix
        prefix_len = len(self._prefix) + 1  # +1 for the colon
        return [key.decode('utf-8')[prefix_len:] for key in keys]
    
    async def items(self) -> List[tuple]:
        """Get all field-value pairs (dict-like)."""
        fields = await self.keys()
        items = []
        for field in fields:
            try:
                value = await self[field]
                items.append((field, value))
            except KeyError:
                # Field might have been deleted between keys() and get
                continue
        return items
    
    # List operations for conversation history and similar use cases
    
    async def append(self, field: str, item: Any) -> None:
        """Append an item to a list field."""
        current_list = await self.get_list(field)
        current_list.append(item)
        await self.set(field, current_list)
    
    async def extend(self, field: str, items: List[Any]) -> None:
        """Extend a list field with multiple items."""
        current_list = await self.get_list(field)
        current_list.extend(items)
        await self.set(field, current_list)
    
    async def get_list(self, field: str, default: Optional[List[Any]] = None) -> List[Any]:
        """Get a list field, returning empty list if not found."""
        if default is None:
            default = []
        try:
            value = await self[field]
            return value if isinstance(value, list) else default
        except KeyError:
            return default
    
    async def pop(self, field: str, index: int = -1) -> Any:
        """Remove and return an item from a list field."""
        current_list = await self.get_list(field)
        if not current_list:
            raise IndexError(f"pop from empty list in field '{field}'")
        item = current_list.pop(index)
        await self.set(field, current_list)
        return item
    
    async def insert(self, field: str, index: int, item: Any) -> None:
        """Insert an item at a specific position in a list field."""
        current_list = await self.get_list(field)
        current_list.insert(index, item)
        await self.set(field, current_list)
    
    async def remove(self, field: str, item: Any) -> None:
        """Remove first occurrence of an item from a list field."""
        current_list = await self.get_list(field)
        current_list.remove(item)  # Raises ValueError if not found
        await self.set(field, current_list)
    
    async def list_length(self, field: str) -> int:
        """Get the length of a list field."""
        current_list = await self.get_list(field)
        return len(current_list)
    
    # Conversation history specific methods
    
    async def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        await self.append("messages", message)
    
    async def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation messages, optionally limited to recent N messages."""
        messages = await self.get_list("messages")
        if limit is not None and len(messages) > limit:
            return messages[-limit:]
        return messages
    
    async def clear_messages(self) -> None:
        """Clear all conversation messages."""
        await self.set("messages", [])
    
    async def get_message_count(self) -> int:
        """Get the number of messages in conversation history."""
        return await self.list_length("messages")
    
    # Utility methods
    
    async def set_ttl(self, ttl: int) -> None:
        """Update TTL for all fields in this conversation."""
        fields = await self.keys()
        pipe = self.redis.pipeline()
        for field in fields:
            key = self._key(field)
            pipe.expire(key, ttl)
        await pipe.execute()
    
    async def clear_all(self) -> None:
        """Delete all memory for this conversation."""
        pattern = f"{self._prefix}:*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
    
    async def get_memory_size(self) -> Dict[str, Any]:
        """Get memory usage statistics for this conversation."""
        fields = await self.keys()
        total_keys = len(fields)
        total_size = 0
        
        for field in fields:
            key = self._key(field)
            try:
                size = await self.redis.memory_usage(key)
                if size:
                    total_size += size
            except Exception:
                # memory_usage might not be available in all Redis versions
                pass
        
        return {
            "conversation_id": self.conversation_id,
            "total_fields": total_keys,
            "total_size_bytes": total_size,
            "fields": fields
        }


class MemoryManager:
    """
    Global memory manager for conversation-scoped distributed memory.
    
    Provides factory methods and cleanup utilities for ConversationMemory instances.
    Integrates with BaseAgent to provide seamless memory access.
    """
    
    def __init__(self, redis: Redis, default_ttl: int = 3600):
        self.redis = redis
        self.default_ttl = default_ttl
        self._memory_cache: Dict[str, ConversationMemory] = {}
    
    def get_memory(self, conversation_id: str, ttl: Optional[int] = None) -> ConversationMemory:
        """Get or create ConversationMemory for a conversation."""
        if conversation_id not in self._memory_cache:
            memory_ttl = ttl if ttl is not None else self.default_ttl
            self._memory_cache[conversation_id] = ConversationMemory(
                self.redis, conversation_id, memory_ttl
            )
        return self._memory_cache[conversation_id]
    
    async def cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up memory for a specific conversation.
        
        Works across distributed agents - any agent can clean up any conversation.
        """
        # Get memory instance (creates if not in cache) - works for any agent
        memory = self.get_memory(conversation_id)
        await memory.clear_all()
        
        # Remove from local cache if present
        if conversation_id in self._memory_cache:
            del self._memory_cache[conversation_id]
            
    async def cleanup_messages(self, conversation_id: str) -> None:
        """Clean up message history for a specific conversation."""
        # Get memory instance (creates if not in cache)
        memory = self.get_memory(conversation_id)
        await memory.clear_messages()
    
    async def cleanup_expired_conversations(self, max_age_seconds: int = 86400) -> int:
        """Clean up conversations older than max_age_seconds. Returns count of cleaned conversations."""
        pattern = "memory:conv:*"
        keys = await self.redis.keys(pattern)
        
        current_time = time.time()
        cleaned_count = 0
        
        # Group keys by conversation_id
        conversations = {}
        for key in keys:
            key_str = key.decode('utf-8')
            # Extract conversation_id from key like "memory:conv:conv_123:field_name"
            parts = key_str.split(':')
            if len(parts) >= 3:
                conv_id = parts[2]
                if conv_id not in conversations:
                    conversations[conv_id] = []
                conversations[conv_id].append(key)
        
        # Check each conversation for expiry
        for conv_id, conv_keys in conversations.items():
            # Check if any key in this conversation is expired
            should_cleanup = False
            for key in conv_keys:
                ttl = await self.redis.ttl(key)
                if ttl == -2:  # Key doesn't exist (expired)
                    should_cleanup = True
                    break
                elif ttl > 0:
                    # Calculate age from TTL
                    age = self.default_ttl - ttl
                    if age > max_age_seconds:
                        should_cleanup = True
                        break
            
            if should_cleanup:
                await self.cleanup_conversation(conv_id)
                cleaned_count += 1
        
        return cleaned_count
    
    async def get_all_conversations(self) -> List[str]:
        """Get list of all conversation IDs with active memory."""
        pattern = "memory:conv:*"
        keys = await self.redis.keys(pattern)
        
        conversations = set()
        for key in keys:
            key_str = key.decode('utf-8')
            parts = key_str.split(':')
            if len(parts) >= 3:
                conversations.add(parts[2])
        
        return list(conversations)
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get overall memory usage statistics."""
        conversations = await self.get_all_conversations()
        total_conversations = len(conversations)
        
        stats = {
            "total_conversations": total_conversations,
            "conversations": []
        }
        
        for conv_id in conversations:
            memory = self.get_memory(conv_id)
            conv_stats = await memory.get_memory_size()
            stats["conversations"].append(conv_stats)
        
        return stats


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None

async def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        redis = await get_redis()
        _memory_manager = MemoryManager(redis)
    return _memory_manager

# -------------------------- Safety Functions --------------------------

def check_safety_limits(env: Envelope) -> Optional[str]:
    """Check if envelope violates any safety limits.
    
    Checks three distinct safety metrics:
    1. Process count: Number of processing steps (prevents infinite loops)
    2. Total processing time: Cumulative time spent in agent.process() methods
    3. Envelope age: Time since envelope creation (prevents stale message processing)
    
    Note: Envelope age (create_ts) is different from individual step timing (trace items).
    Envelope age measures overall lifetime, while trace items measure individual processing steps.
    
    Returns:
        None if safe to continue
        Error message string if safety limit violated
    """
    current_time = time.time()
    
    # Check process count limit (prevents infinite loops)
    if env.process_count >= MAX_PROCESS_COUNT:
        return f"Process count limit exceeded: {env.process_count} >= {MAX_PROCESS_COUNT}"
    
    # Check total processing time limit (cumulative time across all processing steps)
    if env.total_processing_time >= MAX_TOTAL_PROCESSING_TIME:
        return f"Total processing time limit exceeded: {env.total_processing_time:.2f}s >= {MAX_TOTAL_PROCESSING_TIME}s"
    
    # Check envelope age limit (time since envelope creation, not individual step time)
    if env.create_ts > 0:  # Only check if create_ts was set
        envelope_age = current_time - env.create_ts
        if envelope_age >= MAX_ENVELOPE_AGE:
            return f"Envelope age limit exceeded: {envelope_age:.2f}s >= {MAX_ENVELOPE_AGE}s"
    
    return None

def create_safety_error(env: Envelope, error_message: str, agent_role: str, agent_id: str) -> Envelope:
    """Create an error envelope for safety limit violations.
    
    Adds error to payload and trace, marks as result for return to caller.
    """
    # Add error to payload
    env.payload.setdefault("errors", []).append({
        "code": "safety.limit_exceeded",
        "message": error_message,
        "agent_role": agent_role,
        "agent_id": agent_id,
        "ts": time.time()
    })
    
    # Add error trace entry
    env.trace.append({
        "start_ts": time.time(),
        "role": agent_role,
        "agent_id": agent_id,
        "end_ts": time.time(),
        "duration": 0.0,
        "error": error_message,
        "safety_violation": True
    })
    
    # Error envelope created for safety violation
    
    return env

# -------------------------- Redis helpers --------------------------

async def ensure_group(redis: Redis, stream: str, group: str) -> None:
    """
    Ensure a consumer group exists (MKSTREAM).
    This is safe to call repeatedly at startup.
    """
    try:
        await redis.xgroup_create(stream, group, id="$", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" in str(e):
            return
        raise

async def xadd(redis: Redis, stream: str, fields: Dict[str, str]) -> str:
    """Light wrapper for XADD."""
    return await redis.xadd(stream, fields)

# -------------------------- BaseAgent --------------------------

class BaseAgent:
    """
    Unified async agent base:
      - Consumes from:
          stream:role:{role}  via group cg:role:{role}
          stream:agent:{id}   via group cg:agent:{id}
      - Subscribes to:
          broadcast:all, broadcast:role:{role}
      - Emits lifecycle/status to:
          broadcast:role:stat  (init, heartbeat, reload, exit)
      - Applies a safety timeout per task (default 60s; override per-envelope by
        payload["__agent_timeout_sec"]).
      - Pauses role/direct consumption while busy, but keeps handling broadcasts.

    Override process(env) in subclasses:
      - Do work (I/O, CPU, etc.).
      - Return env (modified as needed).
        env.result_list (final).
    """

    def __init__(
        self,
        task_timeout_sec: float = DEFAULT_TASK_TIMEOUT_SEC,
    ) -> None:
        self.redis = None  # Will be initialized in start()
        self.role = derive_role_from_class_name(self.__class__.__name__)
        self.agent_id = build_agent_id()
        self.task_timeout_sec = task_timeout_sec

        # Streams / groups
        self._role_stream = role_stream_key(self.role)
        self._role_group = role_group_name(self.role)
        self._agent_stream = agent_stream_key(self.agent_id)
        self._agent_group = agent_group_name(self.agent_id)

        # State
        self._stop = asyncio.Event()
        self._busy = asyncio.Event()
        self._busy.clear()
        self._shutdown_requested = False

        # Async tasks
        self._t_role: Optional[asyncio.Task] = None
        self._t_direct: Optional[asyncio.Task] = None
        self._t_broadcast: Optional[asyncio.Task] = None
        self._t_heartbeat: Optional[asyncio.Task] = None
        
        # Auto-register this agent in the global registry
        _register_agent(self)

    # ---- lifecycle ----

    async def start(self) -> None:
        """Create groups/streams, emit 'init', and start loops."""
        # Initialize Redis client
        self.redis = await get_redis()
        
        await ensure_group(self.redis, self._role_stream, self._role_group)
        await ensure_group(self.redis, self._agent_stream, self._agent_group)
        logger.info(f"Started {self.role} agent {self.agent_id}")
        await self._publish_status("init")
        self._t_role = asyncio.create_task(self._role_loop())
        self._t_direct = asyncio.create_task(self._direct_loop())
        self._t_broadcast = asyncio.create_task(self._broadcast_loop())
        self._t_heartbeat = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Request graceful stop and emit 'exit'."""
        self._shutdown_requested = True
        self._stop.set()
        for t in (self._t_role, self._t_direct, self._t_broadcast, self._t_heartbeat):
            if t:
                t.cancel()
        await self._publish_status("exit")
        logger.info(f"Stopped {self.role} agent {self.agent_id}")

    # ---- abstract worker ----

    async def process(self, env: Envelope) -> Envelope:
        """Override in subclasses. Return updated envelope."""
        raise NotImplementedError("BaseAgent.process must be overridden")

    # ---- distributed memory interface ----

    async def get_memory(self, conversation_id: str) -> ConversationMemory:
        """Get conversation memory for the given conversation ID.
        
        Provides dict/list-like interface for storing conversation-scoped data
        that persists across agents and processes.
        
        Usage:
            memory = await self.get_memory(env.conversation_id)
            
            # Dict-like operations
            await memory.set("user_preferences", {"language": "en"})
            prefs = await memory.get("user_preferences", {})
            
            # List operations for conversation history
            await memory.add_message("user", "Hello")
            messages = await memory.get_messages(limit=10)
        """
        manager = await get_memory_manager()
        return manager.get_memory(conversation_id)
    
    async def cleanup_memory(self, conversation_id: str) -> None:
        """Clean up all memory for a conversation.
        
        Use this when a conversation need to be completely reset including history and any other context items.
        """
        manager = await get_memory_manager()
        await manager.cleanup_conversation(conversation_id)

    async def cleanup_message_history(self, conversation_id: str) -> None:
        """Clean up only message history for a conversation.
        
        Use this when a conversation is complete or needs to be reset.
        The method only wipes message history keeping all other context items intact.
        """
        manager = await get_memory_manager()
        await manager.cleanup_messages(conversation_id)

    
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics across all conversations."""
        manager = await get_memory_manager()
        return await manager.get_memory_stats()
    
    # ---- conversation-specific logging ----

    async def _get_start_ts(self, conversation_id: str) -> float:
        """Get the start timestamp for a conversation from distributed memory."""
        memory = await self.get_memory(conversation_id)
        start_ts = await memory.get("start_ts")
        if start_ts is None:
            # This is the initial request - store the start timestamp
            start_ts = time.time()
            await memory.set("start_ts", start_ts)
        return start_ts
        
    async def _clear_start_ts(self, conversation_id: str) -> None:
        """Clear the start timestamp for a conversation from distributed memory."""
        memory = await self.get_memory(conversation_id)
        # Use proper delete method - returns True if deleted, False if didn't exist
        deleted = await memory.delete("start_ts")
        logger.debug(f"Cleared start_ts for conversation {conversation_id}: deleted={deleted}")
    
    async def _get_conversation_log_path(self, conversation_id: str) -> tuple[Path, float]:
        """Get the path for conversation-specific log file and start timestamp.
        
        Returns:
            tuple: (log_path, start_ts) where start_ts is retrieved from or stored in memory
        """
        # Get or create conversation start timestamp from distributed memory
        start_ts = await self._get_start_ts(conversation_id)
        
        # Format: CONFIG.LOG_DIR/conversations/conversation_id/start_timestamp.log
        conversations_dir = Path(config.LOG_DIR) / "conversations" / conversation_id
        conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Format timestamp as YYYY-MM-DD_HH-MM-SS for filename
        start_datetime = datetime.fromtimestamp(start_ts)
        timestamp_str = start_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        
        return conversations_dir / f"{timestamp_str}.log", start_ts
    
    async def log(self, conversation_id: str, message: str) -> None:
        """Log a message to the conversation-specific log file.
        
        Args:
            conversation_id: The conversation identifier
            message: The message to log
        """
        try:
            current_ts = time.time()
            
            # Get log path and start timestamp from distributed memory
            log_path, start_ts = await self._get_conversation_log_path(conversation_id)
            relative_ts = current_ts - start_ts
            
            # Format timestamps
            absolute_time = datetime.fromtimestamp(current_ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            relative_time = f"{relative_ts:.3f}"
            
            # Format log entry with improved alignment
            log_entry = f"[{absolute_time}] {relative_time:>7} {self.agent_id:<10} {self.role:<10} {message}\n"
            
            # Append to log file
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            # Don't let logging errors break the main flow
            logger.debug(f"Error writing to conversation log: {e}")
    
    async def log_envelope(self, env: Envelope, action: str, details: str = "") -> None:
        """Log envelope processing with standardized format.
        
        Args:
            env: The envelope being processed
            action: The action being performed (e.g., "received", "processed", "sent")
            details: Optional additional details
        """
        # Create message from envelope attributes
        message_parts = [
            f"{action.upper()}: message {env.message_id}",
            f"stage={env.payload.get('stage', 'N/A')}",
            f"target_role={env.target_role}",
            f"process_count={env.process_count}",
            f"total_time={env.total_processing_time:.3f}s"
        ]
        
        if details:
            message_parts.append(details)
            
        message = " | ".join(message_parts)
        
        # Use unified log method for consistent formatting
        await self.log(env.conversation_id, message)

    # ---- loops ----

    async def _role_loop(self) -> None:
        """Consume role-addressed tasks while not busy."""
        consumer = f"{self.role}@{self.agent_id}"
        try:
            while not self._stop.is_set():
                if self._busy.is_set():
                    await asyncio.sleep(0.05)
                    continue

                resp = await self.redis.xreadgroup(
                    groupname=self._role_group,
                    consumername=consumer,
                    streams={self._role_stream: ">"},
                    count=1,
                    block=1000,
                )
                if not resp:
                    continue

                for _stream, messages in resp:
                    for msg_id, fields in messages:
                        try:
                            env = Envelope.from_stream_fields(fields)

                            # Accept tasks for this role or unspecified role; ignore direct-targets here
                            if env.target_role not in (None, self.role):
                                logger.info(f"[{self.role}:{self.agent_id}] Skipping message for role {env.target_role}: {env.message_id}")
                                await self.redis.xack(self._role_stream, self._role_group, msg_id)
                                continue
                            if env.target_agent_id is not None:
                                logger.info(f"[{self.role}:{self.agent_id}] Skipping direct message for agent {env.target_agent_id}: {env.message_id}")
                                await self.redis.xack(self._role_stream, self._role_group, msg_id)
                                continue

                            logger.info(f"[{self.role}:{self.agent_id}] Processing message: {env.message_id} stage: {env.payload.get('stage', 'N/A')}")
                            
                            # Log envelope reception
                            await self.log_envelope(env, "received", f"from role stream {self._role_stream}")

                            await self._handle_envelope(env)
                        finally:
                            await self.redis.xack(self._role_stream, self._role_group, msg_id)
        except asyncio.CancelledError:
            pass

    async def _direct_loop(self) -> None:
        """Consume direct messages for this specific agent while not busy."""
        consumer = f"direct@{self.agent_id}"
        try:
            while not self._stop.is_set():
                if self._busy.is_set():
                    await asyncio.sleep(0.05)
                    continue

                resp = await self.redis.xreadgroup(
                    groupname=self._agent_group,
                    consumername=consumer,
                    streams={self._agent_stream: ">"},
                    count=1,
                    block=1000,
                )
                if not resp:
                    continue

                for _stream, messages in resp:
                    for msg_id, fields in messages:
                        try:
                            env = Envelope.from_stream_fields(fields)

                            # Accept direct messages for this specific agent only
                            if env.target_agent_id != self.agent_id:
                                await self.redis.xack(self._agent_stream, self._agent_group, msg_id)
                                continue

                            # Log envelope reception
                            await self.log_envelope(env, "received", f"from direct stream {self._agent_stream}")

                            await self._handle_envelope(env)
                        finally:
                            await self.redis.xack(self._agent_stream, self._agent_group, msg_id)
        except asyncio.CancelledError:
            pass
    
    async def process_data(self, data: dict) -> None:
        logger.debug(f"[{self.role}:{self.agent_id}] Received broadcast: {data}")

    async def _broadcast_loop(self) -> None:
        """Handle global/role broadcast control commands."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(BROADCAST_ALL, role_broadcast_channel(self.role))
        try:
            while not self._stop.is_set():
                try:
                    # Use get_message with timeout to avoid blocking indefinitely
                    msg = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=1.0)
                    if msg is None or msg.get("type") != "message":
                        continue
                    
                    try:
                        data = json.loads(msg["data"].decode("utf-8"))
                        await self.process_data(data)
                    except Exception:
                        continue

                    cmd = data.get("cmd")
                    if cmd == "shutdown":
                        self._shutdown_requested = True
                        self._stop.set()
                    elif cmd == "reload":
                        await self._publish_status("reload")
                        
                except asyncio.TimeoutError:
                    # Timeout is expected - just check stop flag and continue
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"Error in broadcast loop: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe(BROADCAST_ALL, role_broadcast_channel(self.role))
                await pubsub.aclose()
            except Exception as e:
                logger.debug(f"Error closing pubsub in broadcast loop: {e}")

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to role 'stat'."""
        try:
            while not self._stop.is_set():
                await self._publish_status("heartbeat")
                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
        except asyncio.CancelledError:
            pass

    # ---- envelope handling ----

    async def _handle_envelope(self, env: Envelope) -> None:
        """Process an envelope with timeout and error handling.
        
        Automatically sets up return routing to sender by default.
        Agents can override this behavior in their process() method.
        Adds execution trace with timing and error information.
        Includes safety checks to prevent resource exhaustion.
        """
        # Process all envelopes (no kind filtering needed)
        # Check safety limits before processing
        safety_error = check_safety_limits(env)
        if safety_error:
            logger.warning(f"[{self.role}:{self.agent_id}] Safety limit violated: {safety_error}")
            await self.log_envelope(env, "safety_violation", safety_error)
            error_env = create_safety_error(env, safety_error, self.role, self.agent_id)
            
            # Set target_role and target_agent_id to None to drop the envelope
            error_env.target_role = None
            error_env.target_agent_id = None
            
            # Send error back to result_list if specified, otherwise drop
            if error_env.result_list:
                logger.info(f"[{self.role}:{self.agent_id}] Sending safety error to result_list: {error_env.result_list}")
                # Set target_list so _send() routes it correctly
                error_env.target_list = error_env.result_list
                await self._send(error_env)
            else:
                logger.info(f"[{self.role}:{self.agent_id}] Dropping envelope due to safety violation (no result_list)")
                await self.log_envelope(env, "dropped", "safety violation, no result_list")
            return
        
        # All agents should set busy flag for proper load balancing
        self._busy.set()
        logger.info(f"Incoming: {env}")
        
        # Increment process count
        env.process_count += 1

        trace_item = {}
        process_start_time = time.time()
        trace_item['start_ts'] = process_start_time
        trace_item['role'] = self.role
        trace_item['agent_id'] = self.agent_id

        #By default envelope after process() will be sent back to sender
        #More complex agent implementation can override this behavior
        env.target_role = env.sender_role
        env.target_agent_id = None
        env.target_list = None

        try:
            timeout = float(env.payload.get("__agent_timeout_sec", self.task_timeout_sec))
            logger.info(f"Timeout: {timeout}")
            await self.log_envelope(env, "processing_start", f"timeout={timeout}s")
            await self._publish_status("process")
            env2 = await asyncio.wait_for(self.process(env), timeout=timeout)
            logger.info(f"After process: {env2}")
            if env2 is not None:
                env = env2

            exception = None
            await self.log_envelope(env, "processing_complete", f"success")
        except asyncio.TimeoutError:
            exception = f"Task exceeded safety timeout: {self.task_timeout_sec}"
            logger.error(exception)
            await self.log_envelope(env, "processing_error", f"timeout after {self.task_timeout_sec}s")
        except Exception as e:
            logger.error(f"[BaseAgent] Exception in process: {e}")
            exception = repr(e)
            await self.log_envelope(env, "processing_error", f"exception: {exception}")
            
        trace_item['end_ts'] = time.time()
        trace_item['duration'] = trace_item['end_ts'] - trace_item['start_ts']
        
        # Update total processing time
        env.total_processing_time += trace_item['duration']
        
        if exception:
            trace_item['exception'] = exception
        
        env.trace.append(trace_item)
        
        # Report envelope completion metrics to StatsAgent
        await self._report_envelope_completion(env, trace_item['duration'])
        
        logger.info(f"Outgoing: {env}")
        
        # Check for self-loop (agent targeting itself) - immediate infinite loop prevention
        if env.target_role == self.role and env.target_agent_id is None:
            error_message = f"Self-loop detected: agent '{self.role}' targeting itself (infinite loop prevention)"
            logger.error(f"[{self.role}:{self.agent_id}] {error_message}")
            await self.log_envelope(env, "self_loop_error", error_message)
            
            # Create error envelope
            error_env = create_safety_error(env, error_message, self.role, self.agent_id)
            error_env.target_role = None
            error_env.target_agent_id = None
            
            # Send error back to result_list if specified, otherwise drop
            if error_env.result_list:
                logger.info(f"[{self.role}:{self.agent_id}] Sending self-loop error to result_list: {error_env.result_list}")
                error_env.target_list = error_env.result_list
                await self._send(error_env)
            else:
                logger.info(f"[{self.role}:{self.agent_id}] Dropping envelope due to self-loop (no result_list)")
                await self.log_envelope(env, "dropped", "self-loop detected, no result_list")
            
            await self._publish_status("idle")
            self._busy.clear()
            return
        
        # Log before sending
        await self.log_envelope(env, "sending", f"to target_role={env.target_role} target_agent_id={env.target_agent_id} target_list={env.target_list}")

        await self._send(env)

        await self._publish_status("idle")
        self._busy.clear()
    
    async def _report_envelope_completion(self, env: Envelope, step_duration: float) -> None:
        """Report comprehensive envelope completion metrics to StatsAgent via broadcast."""
        try:
            current_time = time.time()
            
            # Calculate envelope age (time since creation)
            envelope_age = current_time - env.create_ts if env.create_ts > 0 else 0.0
            
            payload = {
                "type": "envelope_completed",
                "agent_id": self.agent_id,
                "role": self.role,
                "step_duration": step_duration,  # This processing step duration
                "envelope_total_processing_time": env.total_processing_time,  # Total processing time
                "envelope_process_count": env.process_count,  # Number of processing steps
                "envelope_age": envelope_age,  # Time since envelope creation
                "envelope_id": env.message_id,  # Unique envelope identifier
                "conversation_id": env.conversation_id,  # Conversation identifier
                "ts": current_time
            }
            channel = role_broadcast_channel(STATUS_ROLE)
            await self.redis.publish(channel, json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"Failed to report envelope completion: {e}")
    

    async def _send(self, env: Envelope) -> None:
        """Send envelope to target destination.
        
        Routes based on target fields:
        - target_role: sends to role stream (load balanced)
        - target_agent_id: sends to specific agent stream
        - target_list: pushes raw JSON to Redis list for external consumers
        
        Updates sender information and update_ts before sending.
        """
        env.sender_role = self.role
        env.sender_agent_id = self.agent_id
        env.update_ts = time.time()

        if env.target_role:
            logger.info(f"[{self.role}:{self.agent_id}] Sending to role {env.target_role}: {env.message_id}")
            await xadd(self.redis, role_stream_key(env.target_role), env.to_stream_fields())
            await self.log_envelope(env, "sent", f"to role stream {role_stream_key(env.target_role)}")
        elif env.target_agent_id:
            logger.info(f"[{self.role}:{self.agent_id}] Sending to agent {env.target_agent_id}: {env.message_id}")
            await xadd(self.redis, agent_stream_key(env.target_agent_id), env.to_stream_fields())
            await self.log_envelope(env, "sent", f"to agent stream {agent_stream_key(env.target_agent_id)}")
        elif env.target_list:
            logger.info(f"[{self.role}:{self.agent_id}] Sending to list {env.target_list}: {env.message_id}")
            await self.redis.lpush(env.target_list, json.dumps(asdict(env), ensure_ascii=False))
            await self.log_envelope(env, "sent", f"to result list {env.target_list}")
            # Clear start timestamp when conversation completes (result sent to external list)
            await self._clear_start_ts(env.conversation_id)

    async def _publish_status(self, event: str) -> None:
        """Publish lifecycle/status to role 'stat' broadcast channel."""
        payload = {
            "type": event,            # For StatsAgent compatibility
            "event": event,           # init | heartbeat | process | idle | reload | exit
            "status": event,          # For StatsAgent compatibility
            "role": self.role,
            "agent_id": self.agent_id,
            "host": get_hostname(),
            "pid": get_pid(),
            "busy": self._busy.is_set(),
            "ts": time.time(),
        }
        channel = role_broadcast_channel(STATUS_ROLE)
        logger.debug(f"[{self.role}:{self.agent_id}] Publishing status to {channel}: {payload}")
        await self.redis.publish(channel, json.dumps(payload, ensure_ascii=False))

# -------------------------- Concrete agents --------------------------

class StatsAgent(BaseAgent):
    """Comprehensive monitoring agent that collects and reports performance metrics.
    
    Tracks:
    - Run counts (total and per-role/agent)
    - Processing times (total and per-role/agent)
    - Performance metrics (envelopes/second)
    - Agent registry with status tracking
    - Detailed performance analytics
    
    Reports comprehensive statistics every STATS_REPORT_INTERVAL_SEC seconds.
    """

    def __init__(self):
        super().__init__()
        
        # Metrics storage
        self.start_time = time.time()
        self.last_report_time = time.time()
        
        # Run counting
        self.total_runs = 0
        self.runs_since_last_report = 0
        self.runs_by_role = {}  # role -> count
        self.runs_by_agent = {}  # agent_id -> count
        self.runs_by_role_since_last = {}  # role -> count since last report
        self.runs_by_agent_since_last = {}  # agent_id -> count since last report
        
        # Processing time tracking
        self.total_processing_time = 0.0
        self.processing_time_since_last = 0.0
        self.processing_time_by_role = {}  # role -> total_time
        self.processing_time_by_agent = {}  # agent_id -> total_time
        self.processing_time_by_role_since_last = {}  # role -> time since last report
        self.processing_time_by_agent_since_last = {}  # agent_id -> time since last report
        
        # Agent registry and status tracking
        self.agent_registry = {}  # agent_id -> {role, last_status, last_update, total_runs, total_time}
        
        # Performance tracking
        self.envelope_timestamps = []  # List of envelope completion ts for rate calculation
        
        # Envelope-level metrics
        self.total_envelopes_processed = 0
        self.envelopes_since_last_report = 0
        self.envelope_ages = []  # List of envelope ages for average calculation
        self.envelope_processing_times = []  # List of total envelope processing times
        self.envelope_process_counts = []  # List of envelope process counts
        self.completed_envelopes = {}  # envelope_id -> {age, total_time, process_count, ts}
        
        # Reporting task
        self._stats_task = None

    async def start(self) -> None:
        # Initialize Redis client
        self.redis = await get_redis()
        
        await self._publish_status("init")
        self._t_broadcast = asyncio.create_task(self._broadcast_loop())
        self._t_heartbeat = asyncio.create_task(self._heartbeat_loop())
        self._stats_task = asyncio.create_task(self._stats_reporting_loop())

    async def stop(self) -> None:
        """Stop the stats agent and cancel reporting task."""
        if self._stats_task:
            self._stats_task.cancel()
        await super().stop()

    async def _stats_reporting_loop(self) -> None:
        """Periodically report comprehensive statistics."""
        try:
            while not self._stop.is_set():
                await asyncio.sleep(STATS_REPORT_INTERVAL_SEC)
                if not self._stop.is_set():
                    await self._report_comprehensive_stats()
        except asyncio.CancelledError:
            pass

    async def process_data(self, data: dict) -> None:
        """Process broadcast status messages and envelope completion events.
        
        Collects metrics from agent status updates and envelope processing events.
        """
        try:
            # Handle different types of status messages
            msg_type = data.get('type', 'unknown')
            agent_id = data.get('agent_id')
            role = data.get('role')
            
            if msg_type == 'envelope_completed' and agent_id and role:
                # Track envelope completion
                self._track_envelope_completion(data)
            elif msg_type in ('init', 'process', 'idle', 'shutdown') and agent_id and role:
                # Track agent status updates
                self._track_agent_status(data)
                
        except Exception as e:
            logger.debug(f"StatsAgent error processing data: {e}")
        
        return data
    
    def _track_envelope_completion(self, data: dict) -> None:
        """Track metrics from completed envelope processing."""
        agent_id = data.get('agent_id')
        role = data.get('role')
        step_duration = data.get('step_duration', 0.0)  # Individual step duration
        envelope_total_time = data.get('envelope_total_processing_time', 0.0)
        envelope_process_count = data.get('envelope_process_count', 0)
        envelope_age = data.get('envelope_age', 0.0)
        envelope_id = data.get('envelope_id')
        conversation_id = data.get('conversation_id')
        ts = data.get('ts', time.time())
        
        # Update run counts
        self.total_runs += 1
        self.runs_since_last_report += 1
        
        self.runs_by_role[role] = self.runs_by_role.get(role, 0) + 1
        self.runs_by_agent[agent_id] = self.runs_by_agent.get(agent_id, 0) + 1
        
        self.runs_by_role_since_last[role] = self.runs_by_role_since_last.get(role, 0) + 1
        self.runs_by_agent_since_last[agent_id] = self.runs_by_agent_since_last.get(agent_id, 0) + 1
        
        # Update processing times (using step duration for agent-level metrics)
        self.total_processing_time += step_duration
        self.processing_time_since_last += step_duration
        
        self.processing_time_by_role[role] = self.processing_time_by_role.get(role, 0.0) + step_duration
        self.processing_time_by_agent[agent_id] = self.processing_time_by_agent.get(agent_id, 0.0) + step_duration
        
        self.processing_time_by_role_since_last[role] = self.processing_time_by_role_since_last.get(role, 0.0) + step_duration
        self.processing_time_by_agent_since_last[agent_id] = self.processing_time_by_agent_since_last.get(agent_id, 0.0) + step_duration
        
        # Track envelope-level metrics (only count each envelope once)
        if envelope_id and envelope_id not in self.completed_envelopes:
            self.total_envelopes_processed += 1
            self.envelopes_since_last_report += 1
            
            # Store envelope metrics for averaging
            self.envelope_ages.append(envelope_age)
            self.envelope_processing_times.append(envelope_total_time)
            self.envelope_process_counts.append(envelope_process_count)
            
            # Keep track of completed envelopes to avoid double counting
            self.completed_envelopes[envelope_id] = {
                'age': envelope_age,
                'total_time': envelope_total_time,
                'process_count': envelope_process_count,
                'ts': ts,
                'conversation_id': conversation_id
            }
            
            # Clean up old envelope records (keep last 1000 for memory management)
            if len(self.completed_envelopes) > 1000:
                # Remove oldest 100 entries
                oldest_keys = sorted(self.completed_envelopes.keys())[:100]
                for key in oldest_keys:
                    del self.completed_envelopes[key]
        
        # Track envelope completion ts for rate calculation
        self.envelope_timestamps.append(ts)
        
        # Keep only recent ts (last 5 minutes for rate calculation)
        cutoff_time = ts - 300  # 5 minutes
        self.envelope_timestamps = [ts for ts in self.envelope_timestamps if ts > cutoff_time]
        
        # Update agent registry
        if agent_id not in self.agent_registry:
            self.agent_registry[agent_id] = {
                'role': role,
                'last_status': 'processing',
                'last_update': ts,
                'total_runs': 0,
                'total_time': 0.0
            }
        
        self.agent_registry[agent_id]['total_runs'] += 1
        self.agent_registry[agent_id]['total_time'] += step_duration
        self.agent_registry[agent_id]['last_update'] = ts
    
    def _track_agent_status(self, data: dict) -> None:
        """Track agent status updates."""
        agent_id = data.get('agent_id')
        role = data.get('role')
        status = data.get('status', 'unknown')
        ts = data.get('ts', time.time())
        
        if agent_id not in self.agent_registry:
            self.agent_registry[agent_id] = {
                'role': role,
                'last_status': status,
                'last_update': ts,
                'total_runs': 0,
                'total_time': 0.0
            }
        else:
            self.agent_registry[agent_id]['last_status'] = status
            self.agent_registry[agent_id]['last_update'] = ts
    
    async def _report_comprehensive_stats(self) -> None:
        """Generate comprehensive performance statistics and write to file."""
        current_time = time.time()
        uptime = current_time - self.start_time
        time_since_last_report = current_time - self.last_report_time
        
        # Generate report content
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE AGENT PERFORMANCE STATISTICS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # System overview
        report_lines.append(f"System Uptime: {uptime:.1f}s | Report Interval: {time_since_last_report:.1f}s")
        report_lines.append(f"Active Agents: {len(self.agent_registry)}")
        report_lines.append("")
        
        # Run statistics
        self._add_run_statistics(report_lines, time_since_last_report)
        
        # Processing time statistics
        self._add_processing_time_statistics(report_lines, time_since_last_report)
        
        # Performance metrics
        self._add_performance_metrics(report_lines, uptime, time_since_last_report)
        
        # Envelope metrics
        self._add_envelope_metrics(report_lines, time_since_last_report)
        
        # Agent registry
        self._add_agent_registry(report_lines, current_time)
        
        report_lines.append("=" * 80)
        
        # Write report to file
        self._write_stats_report(report_lines)
        
        # Reset since-last-report counters
        self._reset_since_last_report_counters()
        self.last_report_time = current_time
    
    def _write_stats_report(self, report_lines: list) -> None:
        """Write statistics report to file."""
        try:
            with open('agent_statistics.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
        except Exception as e:
            logger.debug(f"Failed to write statistics report: {e}")
    
    def _add_run_statistics(self, report_lines: list, time_since_last_report: float) -> None:
        """Add run count statistics to report."""
        report_lines.append("RUN STATISTICS:")
        report_lines.append(f"   Total Runs: {self.total_runs} | Since Last Report: {self.runs_since_last_report}")
        
        if self.runs_by_role:
            report_lines.append("   By Role (Total | Since Last):")
            for role in sorted(self.runs_by_role.keys()):
                total = self.runs_by_role[role]
                since_last = self.runs_by_role_since_last.get(role, 0)
                report_lines.append(f"      {role}: {total} | {since_last}")
        
        if self.runs_by_agent:
            report_lines.append("   By Agent (Total | Since Last):")
            for agent_id in sorted(self.runs_by_agent.keys()):
                total = self.runs_by_agent[agent_id]
                since_last = self.runs_by_agent_since_last.get(agent_id, 0)
                report_lines.append(f"      {agent_id}: {total} | {since_last}")
        
        report_lines.append("")
    
    def _add_processing_time_statistics(self, report_lines: list, time_since_last_report: float) -> None:
        """Add processing time statistics to report."""
        report_lines.append("PROCESSING TIME STATISTICS:")
        report_lines.append(f"   Total Time: {self.total_processing_time:.2f}s | Since Last Report: {self.processing_time_since_last:.2f}s")
        
        if self.processing_time_by_role:
            report_lines.append("   By Role (Total | Since Last):")
            for role in sorted(self.processing_time_by_role.keys()):
                total = self.processing_time_by_role[role]
                since_last = self.processing_time_by_role_since_last.get(role, 0.0)
                report_lines.append(f"      {role}: {total:.2f}s | {since_last:.2f}s")
        
        if self.processing_time_by_agent:
            report_lines.append("   By Agent (Total | Since Last):")
            for agent_id in sorted(self.processing_time_by_agent.keys()):
                total = self.processing_time_by_agent[agent_id]
                since_last = self.processing_time_by_agent_since_last.get(agent_id, 0.0)
                report_lines.append(f"      {agent_id}: {total:.2f}s | {since_last:.2f}s")
        
        report_lines.append("")
    
    def _add_performance_metrics(self, report_lines: list, uptime: float, time_since_last_report: float) -> None:
        """Add performance metrics to report."""
        report_lines.append("PERFORMANCE METRICS:")
        
        # Overall rates
        overall_rate = self.total_runs / uptime if uptime > 0 else 0
        recent_rate = self.runs_since_last_report / time_since_last_report if time_since_last_report > 0 else 0
        report_lines.append(f"   Overall Rate: {overall_rate:.2f} env/s | Recent Rate: {recent_rate:.2f} env/s")
        
        # Rate by role
        if self.runs_by_role:
            report_lines.append("   Rate by Role (Overall | Recent):")
            for role in sorted(self.runs_by_role.keys()):
                total_runs = self.runs_by_role[role]
                recent_runs = self.runs_by_role_since_last.get(role, 0)
                overall_rate_role = total_runs / uptime if uptime > 0 else 0
                recent_rate_role = recent_runs / time_since_last_report if time_since_last_report > 0 else 0
                report_lines.append(f"      {role}: {overall_rate_role:.2f} env/s | {recent_rate_role:.2f} env/s")
        
        # Rate by agent
        if self.runs_by_agent:
            report_lines.append("   Rate by Agent (Overall | Recent):")
            for agent_id in sorted(self.runs_by_agent.keys()):
                total_runs = self.runs_by_agent[agent_id]
                recent_runs = self.runs_by_agent_since_last.get(agent_id, 0)
                overall_rate_agent = total_runs / uptime if uptime > 0 else 0
                recent_rate_agent = recent_runs / time_since_last_report if time_since_last_report > 0 else 0
                report_lines.append(f"      {agent_id}: {overall_rate_agent:.2f} env/s | {recent_rate_agent:.2f} env/s")
        
        report_lines.append("")
    
    def _add_envelope_metrics(self, report_lines: list, time_since_last_report: float) -> None:
        """Add envelope-level metrics to report."""
        report_lines.append("ENVELOPE METRICS:")
        report_lines.append(f"   Total Envelopes: {self.total_envelopes_processed} | Since Last Report: {self.envelopes_since_last_report}")
        
        if self.envelope_ages:
            # Calculate averages
            avg_age = sum(self.envelope_ages) / len(self.envelope_ages)
            avg_processing_time = sum(self.envelope_processing_times) / len(self.envelope_processing_times)
            avg_process_count = sum(self.envelope_process_counts) / len(self.envelope_process_counts)
            
            # Calculate recent averages (last 100 envelopes or all if less)
            recent_count = min(100, len(self.envelope_ages))
            recent_avg_age = sum(self.envelope_ages[-recent_count:]) / recent_count if recent_count > 0 else 0
            recent_avg_processing_time = sum(self.envelope_processing_times[-recent_count:]) / recent_count if recent_count > 0 else 0
            recent_avg_process_count = sum(self.envelope_process_counts[-recent_count:]) / recent_count if recent_count > 0 else 0
            
            report_lines.append(f"   Average Envelope Age: {avg_age:.2f}s | Recent (last {recent_count}): {recent_avg_age:.2f}s")
            report_lines.append(f"   Average Processing Time: {avg_processing_time:.2f}s | Recent: {recent_avg_processing_time:.2f}s")
            report_lines.append(f"   Average Process Count: {avg_process_count:.1f} steps | Recent: {recent_avg_process_count:.1f} steps")
            
            # Min/Max statistics
            min_age, max_age = min(self.envelope_ages), max(self.envelope_ages)
            min_time, max_time = min(self.envelope_processing_times), max(self.envelope_processing_times)
            min_count, max_count = min(self.envelope_process_counts), max(self.envelope_process_counts)
            
            report_lines.append(f"   Age Range: {min_age:.2f}s - {max_age:.2f}s")
            report_lines.append(f"   Processing Time Range: {min_time:.2f}s - {max_time:.2f}s")
            report_lines.append(f"   Process Count Range: {min_count} - {max_count} steps")
        else:
            report_lines.append("   No envelope data available yet")
        
        report_lines.append("")
    
    def _add_agent_registry(self, report_lines: list, current_time: float) -> None:
        """Add agent registry with status and timing information to report."""
        report_lines.append("AGENT REGISTRY:")
        
        if not self.agent_registry:
            report_lines.append("   No agents registered yet")
            return
        
        report_lines.append(f"   {'Agent ID':<20} {'Role':<12} {'Status':<12} {'Last Update':<12} {'Runs':<8} {'Avg Time':<10}")
        report_lines.append(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")
        
        for agent_id, info in sorted(self.agent_registry.items()):
            role = info['role']
            status = info['last_status']
            last_update = info['last_update']
            total_runs = info['total_runs']
            total_time = info['total_time']
            
            time_since_update = current_time - last_update
            avg_time = total_time / total_runs if total_runs > 0 else 0
            
            report_lines.append(f"   {agent_id:<20} {role:<12} {status:<12} {time_since_update:<8.1f}s ago {total_runs:<8} {avg_time:<8.3f}s")
    
    
    def _reset_since_last_report_counters(self) -> None:
        """Reset all 'since last report' counters."""
        self.runs_since_last_report = 0
        self.processing_time_since_last = 0.0
        self.runs_by_role_since_last.clear()
        self.runs_by_agent_since_last.clear()
        self.processing_time_by_role_since_last.clear()
        self.processing_time_by_agent_since_last.clear()
        
        # Reset envelope counters
        self.envelopes_since_last_report = 0
    
    def generate_final_statistics(self) -> None:
        """Generate final statistics report on system shutdown and write to file."""
        current_time = time.time()
        total_uptime = current_time - self.start_time
        
        # Generate final report content
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FINAL SYSTEM STATISTICS (SESSION SUMMARY)")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Session overview
        report_lines.append(f"Total Session Time: {total_uptime:.1f}s ({total_uptime/60:.1f} minutes)")
        report_lines.append(f"Total Agents: {len(self.agent_registry)}")
        report_lines.append("")
        
        # Final run statistics
        report_lines.append("FINAL RUN STATISTICS:")
        report_lines.append(f"   Total Operations: {self.total_runs}")
        if total_uptime > 0:
            overall_rate = self.total_runs / total_uptime
            report_lines.append(f"   Average Rate: {overall_rate:.2f} operations/second")
        
        if self.runs_by_role:
            report_lines.append("   Operations by Role:")
            for role in sorted(self.runs_by_role.keys()):
                count = self.runs_by_role[role]
                percentage = (count / self.total_runs * 100) if self.total_runs > 0 else 0
                report_lines.append(f"      {role}: {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Final processing time statistics
        report_lines.append("FINAL PROCESSING TIME STATISTICS:")
        report_lines.append(f"   Total Processing Time: {self.total_processing_time:.2f}s")
        if self.total_runs > 0:
            avg_processing_time = self.total_processing_time / self.total_runs
            report_lines.append(f"   Average per Operation: {avg_processing_time:.3f}s")
        
        if total_uptime > 0:
            cpu_utilization = (self.total_processing_time / total_uptime) * 100
            report_lines.append(f"   CPU Utilization: {cpu_utilization:.1f}%")
        report_lines.append("")
        
        # Final envelope statistics
        if self.total_envelopes_processed > 0:
            report_lines.append("FINAL ENVELOPE STATISTICS:")
            report_lines.append(f"   Total Envelopes Processed: {self.total_envelopes_processed}")
            
            if self.envelope_ages:
                avg_age = sum(self.envelope_ages) / len(self.envelope_ages)
                min_age, max_age = min(self.envelope_ages), max(self.envelope_ages)
                report_lines.append(f"   Average Envelope Age: {avg_age:.2f}s (range: {min_age:.2f}s - {max_age:.2f}s)")
            
            if self.envelope_processing_times:
                avg_proc_time = sum(self.envelope_processing_times) / len(self.envelope_processing_times)
                min_proc_time, max_proc_time = min(self.envelope_processing_times), max(self.envelope_processing_times)
                report_lines.append(f"   Average Processing Time: {avg_proc_time:.2f}s (range: {min_proc_time:.2f}s - {max_proc_time:.2f}s)")
            
            if self.envelope_process_counts:
                avg_steps = sum(self.envelope_process_counts) / len(self.envelope_process_counts)
                min_steps, max_steps = min(self.envelope_process_counts), max(self.envelope_process_counts)
                report_lines.append(f"   Average Process Steps: {avg_steps:.1f} (range: {min_steps} - {max_steps})")
            report_lines.append("")
        
        # Agent performance summary
        if self.agent_registry:
            report_lines.append("AGENT PERFORMANCE SUMMARY:")
            report_lines.append(f"   {'Agent ID':<20} {'Role':<12} {'Operations':<12} {'Total Time':<12} {'Avg Time':<10}")
            report_lines.append(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
            
            for agent_id, info in sorted(self.agent_registry.items()):
                role = info['role']
                total_runs = info['total_runs']
                total_time = info['total_time']
                avg_time = total_time / total_runs if total_runs > 0 else 0
                
                report_lines.append(f"   {agent_id:<20} {role:<12} {total_runs:<12} {total_time:<8.2f}s {avg_time:<8.3f}s")
            report_lines.append("")
        
        # System efficiency metrics
        report_lines.append("SYSTEM EFFICIENCY:")
        if total_uptime > 0 and self.total_envelopes_processed > 0:
            envelope_rate = self.total_envelopes_processed / total_uptime
            report_lines.append(f"   Envelope Throughput: {envelope_rate:.2f} envelopes/second")
            
            if self.envelope_processing_times:
                total_envelope_time = sum(self.envelope_processing_times)
                efficiency = (total_envelope_time / total_uptime) * 100
                report_lines.append(f"   Processing Efficiency: {efficiency:.1f}% (time spent in actual processing)")
        
        report_lines.append("=" * 80)
        report_lines.append("Session completed successfully!")
        report_lines.append("=" * 80)
        
        # Write final report to file
        try:
            with open('agent_final_statistics.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
        except Exception as e:
            logger.debug(f"Failed to write final statistics report: {e}")


# Global Redis connection pool and manager
_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[Redis] = None
_redis_health_task: Optional[asyncio.Task] = None

class RedisConnectionManager:
    """Manages Redis connection pool with retry logic and health monitoring."""
    
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[Redis] = None
        self.health_task: Optional[asyncio.Task] = None
        self.is_healthy = True
        self.last_health_check = 0.0
        
    async def initialize(self) -> None:
        """Initialize Redis connection pool with retry configuration."""
        if self.pool is not None:
            return  # Already initialized
            
        try:
            # Create retry configuration
            retry_policy = Retry(
                backoff=ExponentialBackoff(),
                retries=config.REDIS_RETRY_ATTEMPTS,
                supported_errors=(
                    ConnectionError,
                    TimeoutError,
                    ResponseError,
                )
            )
            
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                REDIS_URL,
                max_connections=config.REDIS_MAX_CONNECTIONS,
                retry=retry_policy,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=config.REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=False,
                health_check_interval=config.REDIS_HEALTH_CHECK_INTERVAL
            )
            
            # Create Redis client with the pool
            self.client = Redis(connection_pool=self.pool)
            
            # Test the connection
            await self.client.ping()
            logger.info(f"Redis connection pool initialized: {config.REDIS_MAX_CONNECTIONS} max connections")
            
            # Start health monitoring
            self.health_task = asyncio.create_task(self._health_monitor())
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise
    
    async def get_client(self) -> Redis:
        """Get Redis client, initializing if necessary."""
        if self.client is None:
            await self.initialize()
        return self.client
    
    async def _health_monitor(self) -> None:
        """Monitor Redis connection health and log issues."""
        while True:
            try:
                await asyncio.sleep(config.REDIS_HEALTH_CHECK_INTERVAL)
                
                if self.client:
                    start_time = time.time()
                    await self.client.ping()
                    ping_time = (time.time() - start_time) * 1000  # ms
                    
                    if not self.is_healthy:
                        logger.info(f"Redis connection restored (ping: {ping_time:.1f}ms)")
                        self.is_healthy = True
                    
                    self.last_health_check = time.time()
                    
                    # Log slow pings
                    if ping_time > 100:  # > 100ms
                        logger.warning(f"Slow Redis ping: {ping_time:.1f}ms")
                        
            except Exception as e:
                if self.is_healthy:
                    logger.error(f"Redis health check failed: {e}")
                    self.is_healthy = False
                    
    async def close(self) -> None:
        """Close Redis connections and cleanup."""
        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except asyncio.CancelledError:
                pass
                
        if self.client:
            await self.client.aclose()
            self.client = None
            
        if self.pool:
            await self.pool.aclose()
            self.pool = None
            
        logger.info("Redis connection pool closed")

# Global Redis manager instance
_redis_manager = RedisConnectionManager()

async def get_redis() -> Redis:
    """Get Redis client from connection pool with automatic retry logic.
    
    This provides a robust Redis client with:
    - Connection pooling for better performance
    - Automatic retry on connection failures
    - Health monitoring and logging
    """
    return await _redis_manager.get_client()

async def init_redis() -> None:
    """Initialize Redis connection pool. Call this at startup."""
    await _redis_manager.initialize()

async def close_redis() -> None:
    """Close Redis connection pool. Call this at shutdown."""
    await _redis_manager.close()

# -------------------------- Automatic Cleanup System --------------------------

def _cleanup_on_exit() -> None:
    """Emergency cleanup function called on process exit.
    
    This ensures agents are gracefully stopped even if stop_runtime() 
    is not called explicitly. Handles interpreter shutdown gracefully.
    """
    global _runtime_started
    
    if not _runtime_started:
        return
        
    logger.info("Process exiting - performing automatic agent cleanup")
    
    # During normal program termination, asyncio is often unavailable
    # This is expected behavior, not an error - use sync cleanup
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # Event loop closed - normal during shutdown
            logger.debug("Event loop closed during shutdown - using sync cleanup")
            _sync_cleanup()
            return
        
        # Only try async cleanup if loop is available and not shutting down
        if not loop.is_running():
            # Try a quick async cleanup with very short timeout
            try:
                asyncio.run(_emergency_stop_runtime())
                logger.info("Async cleanup completed successfully")
                return
            except Exception as e:
                logger.debug(f"Async cleanup failed during shutdown: {e} - using sync cleanup")
        else:
            # Loop is running but we're in atexit - likely shutting down
            logger.debug("Event loop running during atexit - using sync cleanup")
            
    except (RuntimeError, AttributeError):
        # No event loop available - normal during interpreter shutdown
        logger.debug("No event loop available during shutdown - using sync cleanup")
    
    # Fall back to synchronous cleanup (most reliable during shutdown)
    _sync_cleanup()

async def _emergency_stop_runtime() -> None:
    """Emergency async version of stop_runtime for cleanup."""
    global _runtime_started
    
    if not _runtime_started:
        return
    
    logger.info(f"Emergency stopping runtime with {len(_agent_registry)} registered agents")
    
    # Generate final statistics before emergency shutdown
    try:
        _generate_exit_statistics()
    except Exception as e:
        logger.debug(f"Error generating exit statistics during emergency cleanup: {e}")
    
    # Stop all agents with shorter timeout for emergency cleanup
    for agent in _agent_registry:
        try:
            logger.info(f"Emergency stopping {agent.role} agent {agent.agent_id}")
            await asyncio.wait_for(agent.stop(), timeout=2.0)  # Shorter timeout
        except asyncio.TimeoutError:
            logger.warning(f"Timeout stopping {agent.role} agent {agent.agent_id}")
        except Exception as e:
            logger.error(f"Error stopping {agent.role} agent {agent.agent_id}: {e}")
    
    # Close Redis connection pool
    try:
        await asyncio.wait_for(close_redis(), timeout=1.0)
    except Exception as e:
        logger.error(f"Error closing Redis connection pool: {e}")
    
    _runtime_started = False
    logger.info("Emergency runtime cleanup completed")

def _sync_cleanup() -> None:
    """Synchronous cleanup for interpreter shutdown scenarios.
    
    This is the normal cleanup path during program termination when
    asyncio is no longer available. This is expected behavior.
    """
    global _runtime_started
    
    logger.info("Performing synchronous cleanup (normal during program exit)")
    
    # Generate final statistics before sync cleanup
    try:
        _generate_exit_statistics()
    except Exception as e:
        logger.debug(f"Error generating exit statistics during sync cleanup: {e}")
    
    # Set shutdown flags on all agents (safe during interpreter shutdown)
    for agent in _agent_registry:
        try:
            # Set shutdown flags without using asyncio
            agent._shutdown_requested = True
            
            # Try to set the stop event if it exists and is accessible
            if hasattr(agent, '_stop') and agent._stop is not None:
                try:
                    agent._stop.set()
                except Exception:
                    # Event might be from a closed loop, ignore
                    pass
            
            logger.info(f"Set shutdown flag for {agent.role} agent {agent.agent_id}")
        except Exception as e:
            # During interpreter shutdown, some operations may fail - that's OK
            logger.debug(f"Minor error setting shutdown flag for {agent.role} agent: {e}")
    
    # Try to close Redis connection pool synchronously if possible
    try:
        # Don't try to close Redis during interpreter shutdown as it may hang
        # The OS will clean up the connections anyway
        logger.info("Skipping Redis pool cleanup during interpreter shutdown (OS will handle)")
    except Exception as e:
        logger.debug(f"Redis pool cleanup skipped: {e}")
    
    _runtime_started = False
    logger.info("Synchronous cleanup completed successfully")

def _signal_handler(signum, frame) -> None:
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    
    # Set the shutdown event to interrupt any waiting operations
    try:
        # Try to set the shutdown event if event loop exists
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            # Schedule setting the event in the event loop
            if _shutdown_event is not None:
                loop.call_soon_threadsafe(_shutdown_event.set)
    except RuntimeError:
        # No event loop, that's OK - cleanup will handle it
        pass
    
    # Perform cleanup
    _cleanup_on_exit()
    
# Register cleanup handlers
atexit.register(_cleanup_on_exit)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

logger.info("Automatic cleanup system initialized")

# Dynamic agent registry
_agent_registry: List[BaseAgent] = []
_runtime_started = False
_shutdown_event: Optional[asyncio.Event] = None  # Global shutdown signal

def _get_shutdown_event() -> asyncio.Event:
    """Get or create the global shutdown event."""
    global _shutdown_event
    if _shutdown_event is None:
        _shutdown_event = asyncio.Event()
    return _shutdown_event

def _register_agent(agent: BaseAgent) -> None:
    """Register an agent in the global registry.
    
    If runtime is already started, automatically start the agent.
    This enables dynamic agent creation after system initialization.
    """
    global _agent_registry, _runtime_started
    
    _agent_registry.append(agent)
    logger.info(f"Registered {agent.role} agent {agent.agent_id} in registry")
    
    # If runtime is already started, start this agent immediately
    if _runtime_started:
        logger.info(f"Runtime already started, auto-starting {agent.role} agent {agent.agent_id}")
        # We need to schedule this as a task since we can't await in a sync function
        asyncio.create_task(agent.start())

def get_registered_agents() -> List[BaseAgent]:
    """Get a copy of the current agent registry."""
    return _agent_registry.copy()

# Create default agent instances - they will auto-register via BaseAgent.__init__
_stats = StatsAgent()

async def start_runtime() -> None:
    """
    Start all registered agents exactly once (idempotent).
    Call this during application boot, or let process_request() call it lazily.
    """
    global _runtime_started
    if _runtime_started:
        return
    
    # Initialize Redis connection pool first
    logger.info("Initializing Redis connection pool")
    await init_redis()
    
    logger.info(f"Starting runtime with {len(_agent_registry)} registered agents")
    for agent in _agent_registry:
        logger.info(f"Starting {agent.role} agent {agent.agent_id}")
        await agent.start()
    
    _runtime_started = True
    logger.info("Runtime started successfully")

def _generate_exit_statistics() -> None:
    """Generate final statistics from StatsAgent if available."""
    try:
        # Find the StatsAgent in the registry
        stats_agent = None
        for agent in _agent_registry:
            if isinstance(agent, StatsAgent):
                stats_agent = agent
                break
        
        if stats_agent:
            stats_agent.generate_final_statistics()
        else:
            logger.info("No StatsAgent found - skipping final statistics")
    except Exception as e:
        logger.debug(f"Error generating final statistics: {e}")

async def stop_runtime() -> None:
    """Graceful shutdown of all registered agents and Redis connection."""
    global _runtime_started
    if not _runtime_started:
        return
    
    logger.info(f"Stopping runtime with {len(_agent_registry)} registered agents")
    
    # Generate final statistics before stopping agents
    _generate_exit_statistics()
    
    for agent in _agent_registry:
        logger.info(f"Stopping {agent.role} agent {agent.agent_id}")
        await agent.stop()
    
    # Close Redis connection pool
    await close_redis()
    _runtime_started = False
    logger.info("Runtime stopped successfully")

# -------------------------- Broadcast helpers --------------------------

async def broadcast_command(cmd: str, role: Optional[str] = None, **kwargs: Any) -> None:
    """
    Publish a control command via Pub/Sub.
      cmd: e.g., "shutdown", "reload"
      role: if provided, publish to broadcast:role:{role}, else to broadcast:all
      kwargs: extra fields merged into the JSON payload
    """
    payload = {"cmd": cmd, "ts": time.time(), **kwargs}
    channel = role_broadcast_channel(role) if role else BROADCAST_ALL
    redis = await get_redis()
    await redis.publish(channel, json.dumps(payload, ensure_ascii=False))


async def process_request(role: str, conversation_id: str, payload: dict) -> str:
    """External entry point to process a request through the agent pipeline.
    
    Creates initial envelope, sends to manager, and waits for final result.
    Uses Redis list with BRPOP for synchronous result retrieval.
    
    Args:
        conversation_id: Logical grouping identifier for related messages
        message: Text content to process
        
    Returns:
        Processed text result or error message
    """
    await start_runtime()
    
    message_id = f"{conversation_id}:{time.time()}"
    result_list = f"result:{message_id}"
    current_time = time.time()
    
    env = Envelope(
        conversation_id=conversation_id,
        message_id=message_id,
        target_role=role,
        target_agent_id=None,
        target_list=None,
        sender_role="ext",
        sender_agent_id="proc_req",
        payload=payload,
        update_ts=current_time,
        result_list=result_list,
        trace=[],
        # Initialize safety attributes
        process_count=0,
        total_processing_time=0.0,
        create_ts=current_time,
    )
    
    logger.info(f"[process_request] Starting request {message_id}")
    
    # Send initial task to manager
    redis = await get_redis()
    await xadd(redis, role_stream_key("manager"), env.to_stream_fields())
    logger.info(f"[process_request] Sent initial task to manager stream for {message_id}")
    
    # Wait for final result with shutdown awareness
    logger.info(f"[process_request] Waiting for result on {result_list}")
    
    try:
        # Create tasks for both result waiting and shutdown detection
        result_task = asyncio.create_task(redis.brpop(result_list, timeout=config.ASSISTANT_TIMEOUT))
        shutdown_event = _get_shutdown_event()
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        
        # Wait for either result or shutdown signal
        done, pending = await asyncio.wait(
            [result_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Check which task completed
        if shutdown_task in done:
            logger.info(f"[process_request] Shutdown signal received, cancelling wait for {message_id}")
            return "Request cancelled due to shutdown"
        
        # Get the result from the completed result task
        br = result_task.result()
        
        if not br:
            logger.error(f"[process_request] Timeout waiting for result on {result_list}")
            result = 'Timeout waiting for final result'
        else:
            _, result_json = br
            result_env = Envelope(**json.loads(result_json.decode('utf-8')))
            logger.info(f"[process_request] Received final result for {message_id}: '{result_env.payload.get('text', 'No result')}'")
            logger.info(result_env)
            result = result_env.payload.get("text", "No result received")
            
    except asyncio.CancelledError:
        logger.info(f"[process_request] Request {message_id} cancelled")
        result = "Request cancelled"
    except Exception as e:
        logger.error(f"[process_request] Error waiting for result: {e}")
        result = f"Error: {e}"
        
    return result

# -------------------------- Local demo --------------------------

if __name__ == "__main__":
    pass