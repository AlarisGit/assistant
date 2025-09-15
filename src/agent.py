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
  * Reply routing: reply_role | reply_agent_id | reply_list
  * Trace: a list of breadcrumbs for debugging
  * Timestamps: ts (float)

- Final responses for external callers:
  * The "reply_list" is a per-request Redis LIST where the final Envelope is
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

- Default routing policy after process():
  * If reply_role is set → route to that role stream (next hop).
  * Else if reply_agent_id is set → route to that agent's stream.
  * Else if reply_list is set → push to LIST (final).
  * Else → drop (no route defined).

--------------------------------------------------------------------------------
5) REDIS KEYS AND CONSUMER GROUPS
--------------------------------------------------------------------------------
- Role streams:     stream:role:{role}
- Agent streams:    stream:agent:{agent_id}
- Consumer groups:  cg:role:{role}, cg:agent:{agent_id}
- Broadcast:        broadcast:all, broadcast:role:{role}
- Reply lists:      reply:{message_id}

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
   - Build message_id and reply_list
   - Send an initial Envelope to role 'manager'
   - BRPOP reply_list for the final result

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
import json
import os
import socket
import time
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Optional, List

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

# -------------------------- Envelope --------------------------

@dataclass
class Envelope:
    """
    Canonical payload for Streams messages.

    Fields:
      conversation_id: logical request correlation key (cross-message)
      message_id:     unique per-message identifier (e.g., f"{conv}:{time.time()}")
      target_role:    desired role to handle this message; None for "any role" (rare)
      target_agent_id:direct agent id; when set, message is for stream:agent:{id}

      sender_role:     role of the sender (for audit/debug)
      sender_agent_id: concrete agent id of the sender

      kind:   "task" | "result" | "control" | "status"
      payload:arbitrary JSON (domain inputs/outputs, control flags, errors)
      ts:     unix timestamp

      reply_role:      next-hop role (pipeline continuation)
      reply_agent_id:  next-hop direct agent id (sticky routing)
      reply_list:      final sink for external responses (BRPOP by caller)

      trace: Breadcrumbs to trace the path (agent id, role, ts, notes)
    """
    conversation_id: str
    message_id: str
    target_role: Optional[str]
    target_agent_id: Optional[str]

    sender_role: str
    sender_agent_id: str

    kind: str
    payload: Dict[str, Any]
    ts: float

    reply_role: Optional[str]
    reply_agent_id: Optional[str]
    reply_list: Optional[str]

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
      - Decide next hop by setting env.reply_role / env.reply_agent_id /
        env.reply_list (final).
    """

    def __init__(
        self,
        redis: Redis,
        role: str,
        task_timeout_sec: float = DEFAULT_TASK_TIMEOUT_SEC,
    ) -> None:
        self.redis = redis
        self.role = role
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
                                logger.debug(f"[{self.role}:{self.agent_id}] Skipping message for role {env.target_role}: {env.message_id}")
                                await self.redis.xack(self._role_stream, self._role_group, msg_id)
                                continue
                            if env.target_agent_id is not None:
                                logger.debug(f"[{self.role}:{self.agent_id}] Skipping direct message for agent {env.target_agent_id}: {env.message_id}")
                                await self.redis.xack(self._role_stream, self._role_group, msg_id)
                                continue

                            logger.info(f"[{self.role}:{self.agent_id}] Processing {env.kind} message: {env.message_id} stage: {env.payload.get('stage', 'N/A')} reply_list: {env.reply_list}")

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
        """Process an envelope with timeout and error handling."""
        if env.kind == "control":
            return

        if env.kind in ("task", "result"):
            # All agents should set busy flag for proper load balancing
            self._busy.set()
            try:
                env.trace.append({
                    "agent_id": self.agent_id,
                    "role": self.role,
                    "ts": time.time(),
                    "action": "start_process",
                })
                timeout = float(env.payload.get("__agent_timeout_sec", self.task_timeout_sec))
                await self._publish_status("process")
                env2 = await asyncio.wait_for(self.process(env), timeout=timeout)
                if env2 is not None:
                    env2.trace.append({
                        "agent_id": self.agent_id,
                        "role": self.role,
                        "ts": time.time(),
                        "action": "end_process",
                    })
                    await self._default_route_after_process(env2)
            except asyncio.TimeoutError:
                await self._emit_error(env, "timeout", "Task exceeded safety timeout")
            except Exception as e:
                await self._emit_error(env, "exception", repr(e))
            finally:
                await self._publish_status("idle")
                self._busy.clear()

    async def _default_route_after_process(self, env: Envelope) -> None:
        """Route envelope after processing to reply destination."""
        if env.reply_agent_id:
            await self._send_to_agent(env.reply_agent_id, env)
        elif env.reply_role:
            await self._send_to_role(env.reply_role, env)
        elif env.reply_list:
            await self._send_to_list(env.reply_list, env)
        else:
            logger.warning(f"[{self.role}:{self.agent_id}] No reply destination for envelope {env.message_id}")

    async def _emit_error(self, env: Envelope, code: str, message: str) -> None:
        """Append error into payload, mark as 'result', and default-route."""
        env.payload.setdefault("errors", []).append({"code": code, "message": message})
        env.kind = "result"
        await self._default_route_after_process(env)

    # ---- send helpers ----

    async def _send_to_role(self, role: str, env: Envelope) -> None:
        role_stream = role_stream_key(role)
        env2 = replace(env,
            target_role=role,
            target_agent_id=None,
            sender_role=self.role,
            sender_agent_id=self.agent_id,
            reply_agent_id=None,
            reply_role=None,
            reply_list=None,
            ts=time.time(),
        )
        logger.info(f"[{self.role}:{self.agent_id}] Sending to role {role}: {env.message_id}")
        await xadd(self.redis, role_stream, env2.to_stream_fields())

    async def _send_to_agent(self, agent_id: str, env: Envelope) -> None:
        agent_stream = agent_stream_key(agent_id)
        env2 = replace(env,
            target_role=env.target_role,  # may be preserved as a hint
            target_agent_id=agent_id,
            sender_role=self.role,
            sender_agent_id=self.agent_id,
            reply_agent_id=None,
            reply_role=None,
            reply_list=None,
            ts=time.time(),
        )
        logger.info(f"[{self.role}:{self.agent_id}] Sending to agent {agent_id}: {env.message_id}")
        await xadd(self.redis, agent_stream, env2.to_stream_fields())
    
    async def _send_to_list(self, list_name: str, env: Envelope) -> None:
        env2 = replace(env,
            target_role=env.target_role,  # may be preserved as a hint
            target_agent_id=env.target_agent_id,
            sender_role=self.role,
            sender_agent_id=self.agent_id,
            reply_agent_id=None,
            reply_role=None,
            reply_list=None,
            ts=time.time(),
        )
        logger.info(f"[{self.role}:{self.agent_id}] Sending to list {list_name}: {env.message_id}")
        await xadd(self.redis, list_name, env2.to_stream_fields())

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
    """
    Dynamic router example:
      client(task 'start') -> manager -> uppercase (result) -> manager -> reverse (result) -> manager -> reply_list

    The manager determines the next hop based on env.payload['stage'] and sets
    reply_role back to 'manager' until the final stage, where it leaves reply_list
    set and clears reply_role (so BaseAgent routes to the per-request list).
    """

    def __init__(self, redis: Redis):
        super().__init__(redis, role="manager")

    async def process(self, env: Envelope) -> Envelope:
        """Route messages through the pipeline stages."""
        stage = env.payload.get("stage", "start")
        logger.info(f"[ManagerAgent] Processing {env.kind} at stage '{stage}' for {env.message_id}")
        
        if env.kind == "task" and stage == "start":
            # Send task to uppercase agent, result should come back to manager
            new_env = Envelope(
                conversation_id=env.conversation_id,
                message_id=env.message_id,
                target_role="uppercase",
                target_agent_id=None,
                sender_role=self.role,
                sender_agent_id=self.agent_id,
                kind="task",
                payload={**env.payload, "stage": "after_upper"},
                ts=time.time(),
                reply_role="manager",  # Result should come back to manager
                reply_agent_id=None,
                reply_list=None,
                trace=env.trace + [{"agent_id": self.agent_id, "role": self.role, "action": "route_to_uppercase"}],
            )
            logger.info(f"[ManagerAgent] Sending task to uppercase: {env.message_id}")
            self._send_to_role("uppercase", new_env)
            return None
        
        if env.kind == "result" and stage == "after_upper":
            # Send task to reverse agent, result should come back to manager
            new_env = Envelope(
                conversation_id=env.conversation_id,
                message_id=env.message_id,
                target_role="reverse",
                target_agent_id=None,
                sender_role=self.role,
                sender_agent_id=self.agent_id,
                kind="task",
                payload={**env.payload, "stage": "after_reverse"},
                ts=time.time(),
                reply_role="manager",  # Result should come back to manager
                reply_agent_id=None,
                reply_list=None,
                trace=env.trace + [{"agent_id": self.agent_id, "role": self.role, "action": "route_to_reverse"}],
            )
            logger.info(f"[ManagerAgent] Sending task to reverse: {env.message_id}")
            await self._send_to_role("reverse", new_env)
            return None
        
        if env.kind == "result" and stage == "after_reverse":
            # Final result - send to reply list
            logger.info(f"[ManagerAgent] Final result ready: {env.message_id}")
            if env.reply_list:
                await self._send_to_list(env.result_list, env)
            return env
        
        # Unknown stage
        logger.error(f"[ManagerAgent] Unknown stage '{stage}' for {env.kind}: {env.message_id}")
        env.payload.setdefault("errors", []).append({
            "code": "manager.stage",
            "message": f"Unknown stage '{stage}' for kind '{env.kind}'",
        })
        return env


class UppercaseAgent(BaseAgent):
    """Example worker: converts payload['text'] to uppercase and returns 'result'."""
    def __init__(self, redis: Redis):
        super().__init__(redis, role="uppercase")

    async def process(self, env: Envelope) -> Envelope:
        """Convert text to uppercase."""
        text = env.payload.get("text", "")
        env.payload["text"] = text.upper()
        env.payload["stage"] = "after_upper"
        env.kind = "result"
        # Don't modify reply_* - just process and return result to whoever sent this
        logger.info(f"[UppercaseAgent] Processed: '{text}' -> '{env.payload['text']}' for {env.message_id}")
        return env


class ReverseAgent(BaseAgent):
    """Example worker: reverses payload['text'] and returns 'result'."""
    def __init__(self, redis: Redis):
        super().__init__(redis, role="reverse")

    async def process(self, env: Envelope) -> Envelope:
        """Reverse the text."""
        text = env.payload.get("text", "")
        env.payload["text"] = text[::-1]
        env.payload["stage"] = "after_reverse"
        env.kind = "result"
        # Don't modify reply_* - just process and return result to whoever sent this
        logger.info(f"[ReverseAgent] Processed: '{text}' -> '{env.payload['text']}' for {env.message_id}")
        return env


class StatsAgent(BaseAgent):
    """
    Example 'stat' role agent that only listens to lifecycle broadcasts
    (no stream consumption). Extend this class to persist metrics.
    """

    def __init__(self, redis: Redis):
        super().__init__(redis, role=STATUS_ROLE)

    async def start(self) -> None:
        await self._publish_status("init")
        self._t_broadcast = asyncio.create_task(self._broadcast_loop())
        self._t_heartbeat = asyncio.create_task(self._heartbeat_loop())

    async def process_data(self, data: dict) -> None:
        logger.debug(f"StatsAgent received: {data}")
        # Not used in this minimal example.
        return data

# -------------------------- Global runtime (no orchestrator class) --------------------------

_redis: Redis = Redis.from_url(REDIS_URL, decode_responses=False)

_manager = ManagerAgent(_redis)
_upper = UppercaseAgent(_redis)
_reverse = ReverseAgent(_redis)
_stats = StatsAgent(_redis)

_started = False

async def start_runtime() -> None:
    """
    Start all agents exactly once (idempotent).
    Call this during application boot, or let process_request() call it lazily.
    """
    global _started
    if _started:
        return
    await _manager.start()
    await _upper.start()
    await _reverse.start()
    await _stats.start()
    _started = True

async def stop_runtime() -> None:
    """Graceful shutdown of all agents and Redis connection."""
    global _started
    if not _started:
        return
    for a in (_manager, _upper, _reverse, _stats):
        await a.stop()
    await _redis.aclose()
    _started = False

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

# -------------------------- Direct messaging helper (optional) --------------------------

async def send_direct_task(
    agent_id: str,
    conversation_id: str,
    message_id: str,
    payload: Dict[str, Any],
    reply_role: Optional[str] = None,
    reply_agent_id: Optional[str] = None,
    reply_list: Optional[str] = None,
    role_hint: Optional[str] = None,
) -> None:
    """
    Send a direct message to a specific agent.
      - role_hint (optional) remains in env.target_role as a hint for logging.
      - Set reply_* to define the next hop.
    """
    env = Envelope(
        conversation_id=conversation_id,
        message_id=message_id,
        target_role=role_hint,
        target_agent_id=agent_id,
        sender_role="external",
        sender_agent_id="send_to_agent",
        kind="task",
        payload=payload,
        ts=time.time(),
        reply_role=reply_role,
        reply_agent_id=reply_agent_id,
        reply_list=reply_list,
        trace=[],
    )
    await xadd(_redis, agent_stream_key(agent_id), env.to_stream_fields())

# -------------------------- External entry point --------------------------

async def process_request(conversation_id: str, message: str) -> str:
    """External entry point to process a request."""
    await start_runtime()
    
    message_id = f"{conversation_id}:{time.time()}"
    reply_list = f"reply:{message_id}"
    
    env = Envelope(
        conversation_id=conversation_id,
        message_id=message_id,
        target_role="manager",
        target_agent_id=None,
        sender_role="external",
        sender_agent_id="process_request",
        kind="task",
        payload={"text": message, "stage": "start"},
        ts=time.time(),
        reply_role=None,
        reply_agent_id=None,
        reply_list=None,
        result_list=reply_list,
        trace=[],
    )
    
    logger.info(f"[process_request] Starting request {message_id} with text: '{message}'")
    
    # Send initial task to manager
    await xadd(_redis, role_stream_key("manager"), env.to_stream_fields())
    logger.info(f"[process_request] Sent initial task to manager stream for {message_id}")
    
    # Wait for final result
    logger.info(f"[process_request] Waiting for result on {reply_list}")
    br = await _redis.brpop(reply_list, timeout=30)
    result = 'No result'
    if not br:
        logger.error(f"[process_request] Timeout waiting for result on {reply_list}")
        result = 'Timeout waiting for final result'
    else:
        _, result_json = br
        result_env = Envelope.from_json(result_json)
        logger.info(f"[process_request] Received final result for {message_id}: '{result_env.payload.get('text', 'No result')}'")
        result = result_env.payload.get("text", "No result received")
    return result

# -------------------------- Local demo --------------------------

if __name__ == "__main__":
    async def _demo():
        print("Starting unified demo with documentation...")
        try:
            await start_runtime()
            res = await process_request("conv1", "Hello, world!")
            print("RESULT:", res)  # Expected: "!DLROW ,OLLEH"

            # Example: broadcast a reload to uppercase agents
            await broadcast_command("reload", role="uppercase")

            # Example: global graceful shutdown (agents finish current tasks)
            await broadcast_command("shutdown")
        finally:
            await stop_runtime()

    asyncio.run(_demo())
