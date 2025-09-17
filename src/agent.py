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
  * Semantics: kind ∈ {"task", "result", "control", "status"}
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
    {"kind":"status","event":"init|heartbeat|reload|exit", ...}
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
   - Subclass BaseAgent(role="newrole"), override process(env), and set env.kind="result".
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

from redis.asyncio import Redis
from redis.exceptions import ResponseError

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
STATUS_ROLE = "stat"  # lifecycle/heartbeat broadcast role

HEARTBEAT_INTERVAL_SEC = 5.0
DEFAULT_TASK_TIMEOUT_SEC = 60.0  # safety timeout per agent (can be overridden per-envelope)

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

      kind:   "task" | "result" | "control" | "status"
      payload:arbitrary JSON (domain inputs/outputs, control flags, errors)
      ts:     unix timestamp

      trace: Breadcrumbs to trace the path (agent id, role, ts, notes)
    """
    conversation_id: str
    message_id: str
    target_role: Optional[str]
    target_agent_id: Optional[str]
    target_list: Optional[str]

    sender_role: str
    sender_agent_id: str

    kind: str
    payload: Dict[str, Any]
    ts: float

    result_list: str

    trace: List[Dict[str, Any]]

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
      - Set env.kind to "result" (or "task" if forwarding).
        env.result_list (final).
    """

    def __init__(
        self,
        task_timeout_sec: float = DEFAULT_TASK_TIMEOUT_SEC,
    ) -> None:
        self.redis = get_redis()
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

                            logger.info(f"[{self.role}:{self.agent_id}] Processing {env.kind} message: {env.message_id} stage: {env.payload.get('stage', 'N/A')}")

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
            async for msg in pubsub.listen():
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
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(BROADCAST_ALL, role_broadcast_channel(self.role))
            await pubsub.aclose()

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
        """
        if env.kind == "control":
            return

        if env.kind in ("task", "result"):
            # All agents should set busy flag for proper load balancing
            self._busy.set()
            logger.info(f"Incoming: {env}")

            trace_item = {}
            trace_item['start_ts'] = time.time()
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
                await self._publish_status("process")
                await asyncio.sleep(1)
                env2 = await asyncio.wait_for(self.process(env), timeout=timeout)
                logger.info(f"After process: {env2}")
                if env2 is not None:
                    env = env2

                exception = None
            except asyncio.TimeoutError:
                exception = f"Task exceeded safety timeout: {self.task_timeout_sec}"
                logger.error(exception)
            except Exception as e:
                logger.error(f"[BaseAgent] Exception in process: {e}")
                exception = repr(e)
                
            trace_item['end_ts'] = time.time()
            trace_item['duration'] = trace_item['end_ts'] - trace_item['start_ts']
            if exception:
                trace_item['exception'] = exception
            
            env.trace.append(trace_item)
            
            logger.info(f"Outgoing: {env}")

            await self._send(env)

            await self._publish_status("idle")
            self._busy.clear()

    async def _send(self, env: Envelope) -> None:
        """Send envelope to target destination.
        
        Routes based on target fields:
        - target_role: sends to role stream (load balanced)
        - target_agent_id: sends to specific agent stream
        - target_list: pushes raw JSON to Redis list for external consumers
        
        Updates sender information and timestamp before sending.
        """
        env.sender_role = self.role
        env.sender_agent_id = self.agent_id
        env.ts = time.time()

        if env.target_role:
            logger.info(f"[{self.role}:{self.agent_id}] Sending to role {env.target_role}: {env.message_id}")
            await xadd(self.redis, role_stream_key(env.target_role), env.to_stream_fields())
        elif env.target_agent_id:
            logger.info(f"[{self.role}:{self.agent_id}] Sending to agent {env.target_agent_id}: {env.message_id}")
            await xadd(self.redis, agent_stream_key(env.target_agent_id), env.to_stream_fields())
        elif env.target_list:
            logger.info(f"[{self.role}:{self.agent_id}] Sending to list {env.target_list}: {env.message_id}")
            await self.redis.lpush(env.target_list, json.dumps(asdict(env), ensure_ascii=False))

    async def _publish_status(self, event: str) -> None:
        """Publish lifecycle/status to role 'stat' broadcast channel."""
        payload = {
            "kind": "status",
            "event": event,           # init | heartbeat | process | idle | reload | exit
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

class ManagerAgent(BaseAgent):
    """Pipeline orchestrator that routes messages through processing stages.
    
    Implements a simple state machine:
    - start -> uppercase -> reverse -> final result
    
    Includes run counter for demo purposes with safety limit.
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Route messages through the pipeline stages.
        
        State machine logic:
        1. task + start -> route to uppercase
        2. result + upper -> route to reverse  
        3. result + reverse -> send to result_list for external caller
        
        Returns modified envelope with updated routing information.
        """
        stage = env.payload.get("stage", "start")
        logger.info(f"[ManagerAgent] Processing {env.kind} at stage '{stage}' for {env.message_id}")
        logger.info(f"[ManagerAgent] Incoming: {env}")

        if env.kind == "task" and stage == "start":
            env.target_role = "uppercase"
            env.payload["stage"] = "upper"
            logger.info(f"[ManagerAgent] Sending task to uppercase")
            return env
        
        if env.kind == "result" and stage == "upper":
            env.kind = "task"
            env.target_role = "reverse"
            env.payload["stage"] = "reverse"
            logger.info(f"[ManagerAgent] Sending task to reverse")
            return env
        
        if env.kind == "result" and stage == "reverse":
            logger.info(f"[ManagerAgent] Final result ready: {env}")
            if env.result_list:
                env.target_list = env.result_list
                env.target_role = None
                env.target_agent_id = None
            return env
        
        # Unknown stage
        logger.error(f"[ManagerAgent] Unknown stage '{stage}' for {env}")
        env.payload.setdefault("errors", []).append({
            "code": "manager.stage",
            "message": f"Unknown stage '{stage}' for kind '{env.kind}'",
        })
        return env


class UppercaseAgent(BaseAgent):
    """Example worker: converts payload['text'] to uppercase and returns 'result'.
    
    Demonstrates basic text transformation agent pattern.
    Changes envelope kind from 'task' to 'result' after processing.
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Convert text to uppercase.
        
        Transforms payload['text'] to uppercase and marks envelope as 'result'.
        Logs processing details for debugging.
        """
        logger.info(f"[UppercaseAgent] Processing {env.kind} message: {env.message_id} stage: {env.payload.get('stage', 'N/A')}")
        text = env.payload.get("text", "")
        env.payload["text"] = text.upper()
        env.kind = "result"
        logger.info(f"[UppercaseAgent] Processed: '{text}' -> '{env.payload['text']}' for {env.message_id}")
        logger.info(f"[UppercaseAgent] Outgoing: {env}")
        return env


class ReverseAgent(BaseAgent):
    """Example worker: reverses payload['text'] and returns 'result'.
    
    Demonstrates basic text transformation agent pattern.
    Changes envelope kind from 'task' to 'result' after processing.
    """

    async def process(self, env: Envelope) -> Envelope:
        """Reverse the text.
        
        Transforms payload['text'] by reversing character order and marks envelope as 'result'.
        Logs processing details for debugging.
        """
        text = env.payload.get("text", "")
        env.payload["text"] = text[::-1]
        env.kind = "result"
        logger.info(f"[ReverseAgent] Processed: '{text}' -> '{env.payload['text']}' for {env.message_id}")
        logger.info(f"[ReverseAgent] Outgoing: {env}")
        return env


class StatsAgent(BaseAgent):
    """Monitoring agent that collects lifecycle and heartbeat events.
    
    Only subscribes to broadcast channels, doesn't consume from streams.
    Receives status events from all other agents for observability.
    Extend this class to persist metrics to time-series databases.
    """


    async def start(self) -> None:
        await self._publish_status("init")
        self._t_broadcast = asyncio.create_task(self._broadcast_loop())
        self._t_heartbeat = asyncio.create_task(self._heartbeat_loop())

    async def process_data(self, data: dict) -> None:
        """Process broadcast status messages from other agents.
        
        Override this method to implement metrics collection,
        alerting, or other monitoring functionality.
        """
        logger.debug(f"StatsAgent received: {data}")
        # Not used in this minimal example.
        return data


_redis: Redis = Redis.from_url(REDIS_URL, decode_responses=False)

def get_redis() -> Redis:
    """Get the global Redis connection instance.
    
    This provides a centralized Redis connection that all agents can use,
    eliminating the need to pass Redis connections as constructor arguments.
    """
    return _redis

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
    
    # During interpreter shutdown, asyncio is often unavailable
    # Go directly to synchronous cleanup which is more reliable
    try:
        # Try to check if we can still use asyncio
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        
        # Only try async cleanup if loop is available and not shutting down
        if not loop.is_running():
            # Try a quick async cleanup with very short timeout
            try:
                asyncio.run(_emergency_stop_runtime())
                return
            except Exception as e:
                logger.warning(f"Async cleanup failed during shutdown: {e}")
        else:
            # Loop is running but we're in atexit - likely shutting down
            logger.info("Event loop running during atexit - using sync cleanup")
            
    except (RuntimeError, AttributeError) as e:
        logger.info(f"Asyncio unavailable during shutdown ({e}) - using sync cleanup")
    
    # Fall back to synchronous cleanup (most reliable during shutdown)
    _sync_cleanup()

async def _emergency_stop_runtime() -> None:
    """Emergency async version of stop_runtime for cleanup."""
    global _runtime_started
    
    if not _runtime_started:
        return
    
    logger.info(f"Emergency stopping runtime with {len(_agent_registry)} registered agents")
    
    # Stop all agents with shorter timeout for emergency cleanup
    for agent in _agent_registry:
        try:
            logger.info(f"Emergency stopping {agent.role} agent {agent.agent_id}")
            await asyncio.wait_for(agent.stop(), timeout=2.0)  # Shorter timeout
        except asyncio.TimeoutError:
            logger.warning(f"Timeout stopping {agent.role} agent {agent.agent_id}")
        except Exception as e:
            logger.error(f"Error stopping {agent.role} agent {agent.agent_id}: {e}")
    
    # Close Redis connection
    try:
        await asyncio.wait_for(_redis.aclose(), timeout=1.0)
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")
    
    _runtime_started = False
    logger.info("Emergency runtime cleanup completed")

def _sync_cleanup() -> None:
    """Synchronous cleanup for interpreter shutdown scenarios."""
    global _runtime_started
    
    logger.info("Performing synchronous cleanup")
    
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
    
    # Try to close Redis connection synchronously if possible
    try:
        # Don't try to close Redis during interpreter shutdown as it may hang
        # The OS will clean up the connection anyway
        logger.info("Skipping Redis cleanup during interpreter shutdown (OS will handle)")
    except Exception as e:
        logger.debug(f"Redis cleanup skipped: {e}")
    
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
_manager = ManagerAgent()
_upper = UppercaseAgent()
_reverse = ReverseAgent()
_stats = StatsAgent()

async def start_runtime() -> None:
    """
    Start all registered agents exactly once (idempotent).
    Call this during application boot, or let process_request() call it lazily.
    """
    global _runtime_started
    if _runtime_started:
        return
    
    logger.info(f"Starting runtime with {len(_agent_registry)} registered agents")
    for agent in _agent_registry:
        logger.info(f"Starting {agent.role} agent {agent.agent_id}")
        await agent.start()
    
    _runtime_started = True
    logger.info("Runtime started successfully")

async def stop_runtime() -> None:
    """Graceful shutdown of all registered agents and Redis connection."""
    global _runtime_started
    if not _runtime_started:
        return
    
    logger.info(f"Stopping runtime with {len(_agent_registry)} registered agents")
    for agent in _agent_registry:
        logger.info(f"Stopping {agent.role} agent {agent.agent_id}")
        await agent.stop()
    
    await _redis.aclose()
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
    await _redis.publish(channel, json.dumps(payload, ensure_ascii=False))


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
    
    env = Envelope(
        conversation_id=conversation_id,
        message_id=message_id,
        target_role=role,
        target_agent_id=None,
        target_list=None,
        sender_role="external",
        sender_agent_id="process_request",
        kind="task",
        payload=payload,
        ts=time.time(),
        result_list=result_list,
        trace=[],
    )
    
    logger.info(f"[process_request] Starting request {message_id}")
    
    # Send initial task to manager
    await xadd(_redis, role_stream_key("manager"), env.to_stream_fields())
    logger.info(f"[process_request] Sent initial task to manager stream for {message_id}")
    
    # Wait for final result with shutdown awareness
    logger.info(f"[process_request] Waiting for result on {result_list}")
    
    try:
        # Create tasks for both result waiting and shutdown detection
        result_task = asyncio.create_task(_redis.brpop(result_list, timeout=10))
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
    async def _demo():
        print("Starting unified demo with documentation...")
        try:
            await start_runtime()
            payload = {"text": f"Hello, world! [{datetime.now().isoformat()}]", "stage": "start"}
            res = await process_request("manager", "conv1", payload)
            print("RESULT:", res)  # Expected: "!DLROW ,OLLEH"

            # Example: broadcast a reload to uppercase agents
            await broadcast_command("reload", role="uppercase")

            # Example: global graceful shutdown (agents finish current tasks)
            await broadcast_command("shutdown")
        finally:
            #await stop_runtime()
            pass

    asyncio.run(_demo())
