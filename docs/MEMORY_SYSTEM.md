# Distributed Memory Subsystem

The distributed memory subsystem provides Redis-backed, conversation-scoped storage that enables stateless agents to share context across processes and hosts. It offers a familiar dict/list-like interface while handling serialization, TTL management, and cleanup automatically.

## Overview

### Key Features

- **Distributed**: Works across processes and hosts via Redis backend
- **Conversation-scoped**: Automatic isolation by conversation ID
- **Dict/List Interface**: Familiar Python operations (`memory["key"]`, `memory.append()`)
- **Automatic TTL**: Configurable time-to-live with cleanup
- **Type Safety**: JSON serialization for all Python data types
- **Integrated**: Seamless integration with BaseAgent class
- **Statistics**: Memory usage monitoring and reporting

### Architecture

```
BaseAgent
├── get_memory(conversation_id) -> ConversationMemory
├── cleanup_memory(conversation_id)
└── get_memory_stats()

ConversationMemory (Redis-backed)
├── Dict operations: get, set, update, keys, items
├── List operations: append, extend, pop, insert, remove
├── Conversation history: add_message, get_messages
└── Utilities: set_ttl, clear_all, get_memory_size

MemoryManager
├── Factory for ConversationMemory instances
├── Cleanup and TTL management
└── Global statistics and monitoring
```

## Usage Guide

### Basic Dict Operations

```python
class MyAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        # Get conversation memory
        memory = await self.get_memory(env.conversation_id)
        
        # Set values
        await memory.set("user_id", "user_123")
        await memory.set("preferences", {"language": "en", "theme": "dark"})
        
        # Get values with defaults
        user_id = await memory.get("user_id")
        prefs = await memory.get("preferences", {})
        
        # Update multiple values
        await memory.update({
            "last_activity": time.time(),
            "session_count": 1
        })
        
        # Check existence
        if await memory.__contains__("user_preferences"):
            # Key exists
            pass
        
        # Get all keys
        all_keys = await memory.keys()
        
        # Get all items
        items = await memory.items()
        
        return env
```

### List Operations

```python
class ListDemoAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        memory = await self.get_memory(env.conversation_id)
        
        # Initialize list
        await memory.set("activity_log", [])
        
        # Append items
        await memory.append("activity_log", {
            "action": "login",
            "timestamp": time.time()
        })
        
        # Extend with multiple items
        await memory.extend("activity_log", [
            {"action": "view_page", "page": "home"},
            {"action": "click_button", "button": "submit"}
        ])
        
        # Get list
        activities = await memory.get_list("activity_log")
        
        # List operations
        length = await memory.list_length("activity_log")
        last_item = await memory.pop("activity_log")  # Remove last
        await memory.insert("activity_log", 0, {"action": "session_start"})
        
        return env
```

### Conversation History Management

```python
class ChatAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        memory = await self.get_memory(env.conversation_id)
        
        # Add messages to conversation
        user_message = env.payload.get("text", "")
        await memory.add_message("user", user_message, {
            "message_id": env.message_id,
            "timestamp": time.time()
        })
        
        # Process and respond
        response = await self.generate_response(user_message)
        await memory.add_message("assistant", response)
        
        # Get conversation history
        recent_messages = await memory.get_messages(limit=10)
        all_messages = await memory.get_messages()
        message_count = await memory.get_message_count()
        
        # Use history for context
        context = self.build_context(recent_messages)
        
        env.payload["text"] = response
        return env
```

### User Preferences and Settings

```python
class PreferencesAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        memory = await self.get_memory(env.conversation_id)
        
        # Initialize default preferences
        default_prefs = {
            "language": None,
            "response_style": "detailed",
            "notifications": True,
            "timezone": "UTC"
        }
        
        # Get existing or set defaults
        prefs = await memory.get("user_preferences", default_prefs)
        
        # Update preferences based on user input
        if "language" in env.payload:
            prefs["language"] = env.payload["language"]
            await memory.set("user_preferences", prefs)
        
        # Track user activity
        await memory.append("user_actions", {
            "action": env.payload.get("action", "unknown"),
            "timestamp": time.time(),
            "preferences_at_time": prefs.copy()
        })
        
        return env
```

### Memory Management and Cleanup

```python
class MemoryManagerAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        action = env.payload.get("action")
        
        if action == "cleanup_conversation":
            # Clean up specific conversation
            conv_id = env.payload.get("conversation_id")
            await self.cleanup_memory(conv_id)
            
        elif action == "get_stats":
            # Get memory statistics
            stats = await self.get_memory_stats()
            env.payload["stats"] = stats
            
        elif action == "set_ttl":
            # Update TTL for conversation
            memory = await self.get_memory(env.conversation_id)
            ttl = env.payload.get("ttl", 3600)
            await memory.set_ttl(ttl)
            
        elif action == "get_size":
            # Get memory size for conversation
            memory = await self.get_memory(env.conversation_id)
            size_info = await memory.get_memory_size()
            env.payload["size_info"] = size_info
        
        return env
```

## Configuration

### TTL Settings

```python
# Default TTL for all memory fields (1 hour)
DEFAULT_TTL = 3600

# Custom TTL per conversation
memory = await self.get_memory(conversation_id)
await memory.set("key", "value", ttl=7200)  # 2 hours

# Update TTL for all fields in conversation
await memory.set_ttl(1800)  # 30 minutes
```

### Memory Cleanup

```python
# Automatic cleanup of expired conversations
manager = await get_memory_manager()
cleaned_count = await manager.cleanup_expired_conversations(max_age_seconds=86400)

# Manual cleanup
await manager.cleanup_conversation("conv_123")

# Clear all memory for a conversation
memory = await self.get_memory("conv_123")
await memory.clear_all()
```

## Redis Key Structure

The memory system uses a hierarchical Redis key structure:

```
memory:conv:{conversation_id}:{field_name}
```

Examples:
```
memory:conv:user_123:user_preferences
memory:conv:user_123:messages
memory:conv:user_123:activity_log
memory:conv:session_456:uppercase_stats
```

## Performance Considerations

### Best Practices

1. **Use appropriate TTL**: Set reasonable TTL values to prevent memory bloat
2. **Batch operations**: Use `update()` for multiple field updates
3. **Limit list sizes**: Implement list rotation for large datasets
4. **Monitor usage**: Regular cleanup of expired conversations
5. **Efficient serialization**: Avoid storing large binary data

### Memory Monitoring

```python
# Get memory statistics
stats = await self.get_memory_stats()
print(f"Total conversations: {stats['total_conversations']}")

for conv_stats in stats['conversations']:
    print(f"Conversation {conv_stats['conversation_id']}:")
    print(f"  Fields: {conv_stats['total_fields']}")
    print(f"  Size: {conv_stats['total_size_bytes']} bytes")
```

### Cleanup Strategies

```python
# Periodic cleanup task
async def cleanup_task():
    while True:
        manager = await get_memory_manager()
        cleaned = await manager.cleanup_expired_conversations(max_age_seconds=86400)
        logger.info(f"Cleaned up {cleaned} expired conversations")
        await asyncio.sleep(3600)  # Run every hour

# Conversation-specific cleanup
async def end_conversation(conversation_id: str):
    await self.cleanup_memory(conversation_id)
    logger.info(f"Conversation {conversation_id} ended and cleaned up")
```

## Error Handling

```python
class RobustAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        try:
            memory = await self.get_memory(env.conversation_id)
            
            # Safe get with default
            value = await memory.get("key", "default_value")
            
            # Handle missing keys
            try:
                required_value = await memory["required_key"]
            except KeyError:
                # Initialize required value
                await memory.set("required_key", "initial_value")
                required_value = "initial_value"
            
            # Safe list operations
            try:
                item = await memory.pop("my_list")
            except IndexError:
                # List is empty
                item = None
            
        except Exception as e:
            logger.error(f"Memory operation failed: {e}")
            # Handle gracefully
            
        return env
```

## Integration Examples

### SMS Assistant with Memory

```python
class SMSAssistantAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        memory = await self.get_memory(env.conversation_id)
        
        # Track conversation context
        await memory.add_message("user", env.payload["text"])
        
        # Get user preferences
        prefs = await memory.get("user_preferences", {
            "language": "auto",
            "response_length": "medium"
        })
        
        # Get conversation history for context
        history = await memory.get_messages(limit=5)
        
        # Generate response using LLM with context
        response = await self.generate_contextual_response(
            env.payload["text"], 
            history, 
            prefs
        )
        
        # Save assistant response
        await memory.add_message("assistant", response)
        
        # Update conversation stats
        stats = await memory.get("conversation_stats", {"message_count": 0})
        stats["message_count"] += 1
        stats["last_activity"] = time.time()
        await memory.set("conversation_stats", stats)
        
        env.payload["text"] = response
        return env
```

This distributed memory system provides the foundation for building sophisticated, stateful conversations while maintaining the benefits of stateless agent design. It enables rich context sharing, user personalization, and conversation continuity across the entire agent pipeline.
