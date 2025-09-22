# AI Assistant Project Context - Alaris Assistant

## Project Overview
**Alaris Assistant** is a production-grade agentic system built on Redis Streams with distributed memory management for sophisticated conversation handling across processes and hosts.

## Current Project State

### Architecture Completed ✅
- **BaseAgent Framework**: Universal base class with automatic role derivation
- **Redis Streams Communication**: Point-to-role and point-to-agent messaging
- **Distributed Memory System**: Conversation-scoped storage with dict/list interface
- **Safety Systems**: Circuit breakers, process limits, comprehensive monitoring
- **Connection Management**: Production-grade Redis connection pool with retry logic

### Recent Developments ✅
- **Memory Cleanup Methods**: Both partial (messages only) and full cleanup
- **Cross-Agent Cleanup**: Fixed distributed cleanup to work across any agent
- **Project Organization**: Clean directory structure with proper .gitignore
- **Documentation**: Comprehensive guides and examples

### Key Agreements & Rules

#### 🚨 CRITICAL SECURITY RULES
- **NEVER access .env files** - Contains secrets, API keys, passwords
- **Always respect privacy** and data security boundaries
- **Use Python script workaround** for gitignored files when needed
- **Clean up temporary files** after use

#### 📁 Project Structure Standards
```
assistant/
├── src/           # Production source code
├── tests/         # Test suites (test_*.py)
├── examples/      # Demos (*_demo.py)
├── docs/          # Documentation (*.md)
├── .windsurf/     # Project configuration (this directory)
└── [gitignored]   # cache/, log/, data/, venv3.13/
```

#### 🔧 Development Patterns
- **Stateless agent design** with distributed memory context
- **Conversation-scoped memory** with automatic TTL management
- **BaseAgent inheritance** for all agent implementations
- **Comprehensive error handling** and trace logging
- **Async/await patterns** throughout the codebase

### Current Focus Areas
1. **SMS Assistant Pipeline**: Building production SMS conversation agents
2. **Memory System Optimization**: Performance and cleanup improvements  
3. **LLM Integration**: Multi-provider support (OpenAI, Google, Ollama)
4. **RAG Capabilities**: Qdrant vector database integration

### Agent Pipeline Architecture
```
External → ManagerAgent → UppercaseAgent → ReverseAgent → Result
         ↓
    Memory System (Redis-backed, conversation-scoped)
         ↓
    Statistics & Monitoring
```

### Memory System Capabilities
- **Dict operations**: get, set, update, keys, items
- **List operations**: append, extend, pop, insert, remove  
- **Conversation history**: add_message, get_messages, get_message_count
- **Cleanup methods**: cleanup_memory() (full), cleanup_message_history() (partial)
- **Statistics**: get_memory_stats(), get_memory_size()

### Integration Points
- **Redis**: Primary communication and storage backend
- **LLM Providers**: OpenAI, Google, Ollama with caching
- **Vector DB**: Qdrant for RAG capabilities
- **Monitoring**: Comprehensive statistics and health checks

### Next Development Priorities
1. SMS assistant agent implementations
2. LLM integration for conversation processing
3. RAG document ingestion pipeline
4. Production deployment configuration

## Multi-Computer Sync Strategy
This file and project.json are committed to Git to ensure consistent AI assistant context across all development environments. Update these files as project agreements evolve.
