# Alaris Assistant - Production-Grade Agentic Framework

This directory contains comprehensive documentation for the Alaris Assistant production-grade agentic system built on Redis Streams with enterprise-level reliability and concurrent processing capabilities.

## Documentation Index

### Core Architecture
- **[AGENTIC_ARCHITECTURE.md](AGENTIC_ARCHITECTURE.md)** - Complete system architecture and design patterns
- **[CONCURRENT_PROCESSING.md](CONCURRENT_PROCESSING.md)** - Concurrent processing and scaling guide
- **[MEMORY_SYSTEM.md](MEMORY_SYSTEM.md)** - Distributed memory subsystem with conversation isolation
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development guidelines and best practices

### Quick Start
1. **Architecture Overview**: Read [Agentic Architecture](AGENTIC_ARCHITECTURE.md) for system design principles
2. **Concurrent Processing**: Review [Concurrent Processing Guide](CONCURRENT_PROCESSING.md) for scaling patterns
3. **Memory Management**: Check [Memory System](MEMORY_SYSTEM.md) for conversation context patterns  
4. **Development Setup**: Follow [Development Guide](DEVELOPMENT.md) for local setup
5. **Live Demo**: Run `python src/assistant.py` for interactive demonstration
6. **Production Bot**: Configure `TELEGRAM_BOT_TOKEN` for concurrent user handling

### System Components

#### Core Modules (`src/`)
- `agent.py` - Foundation framework with BaseAgent class and Redis Streams communication
- `assistant.py` - Main assistant implementation with pipeline orchestration
- `llm.py` - Multi-provider LLM integration (OpenAI, Google, Ollama)
- `qdrant.py` - Vector database integration for RAG capabilities
- `config.py` - Configuration management with environment variables

#### Key Features
- **🚀 Concurrent Processing** - Multiple users handled simultaneously with non-blocking Telegram bot
- **⚡ Scalable LLM Integration** - Multiple agent instances for parallel API calls (2+ concurrent requests)
- **🛡️ Production-grade Safety** - Circuit breakers, self-loop detection, automatic error recovery
- **🔄 Redis Streams Architecture** - Enterprise connection pooling (50 connections) with health monitoring
- **💾 Distributed Memory System** - Conversation isolation across processes/hosts with automatic cleanup
- **📊 Zero-config Agents** - Automatic role derivation and registration with intelligent defaults
- **🔧 Horizontal Scaling** - Load balancing and backpressure management for production workloads

### Development

#### Project Structure
```
assistant/
├── src/           # Production source code
├── examples/      # Demonstration scripts
├── tests/         # Test suites
├── docs/          # Documentation (this directory)
├── prompts/       # LLM prompts and templates
├── data/          # Data files
├── cache/         # Cache directory
└── log/           # Log files
```

#### Running the System
```bash
# Interactive assistant demo with concurrent processing
python src/assistant.py

# Telegram bot with production-grade concurrent handling
python src/tg.py  # Requires TELEGRAM_BOT_TOKEN in environment

# Memory system demonstration
python examples/memory_system_demo.py
```

#### Configuration
```bash
# Essential environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_MAX_CONNECTIONS=50

# LLM provider configuration
export OPENAI_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
export GEN_MODEL=gpt-4@openai  # or gemini-2.5-flash@google

# Telegram bot (optional)
export TELEGRAM_BOT_TOKEN=your_bot_token
```

### Production Architecture Highlights

The Alaris Assistant is an enterprise-grade agentic system featuring:

#### 🏗️ **Concurrent Processing Architecture**
- **Non-blocking Telegram Bot**: Multiple users processed simultaneously
- **Parallel LLM Calls**: 2+ concurrent API requests with automatic load balancing
- **Redis Streams Foundation**: Enterprise connection pooling with 50+ connections
- **Horizontal Scaling**: Multiple agent instances per role across processes/hosts

#### 🛡️ **Production Safety & Reliability**
- **Circuit Breakers**: Automatic protection against infinite loops and resource exhaustion
- **Self-Loop Detection**: Immediate prevention of routing errors
- **Graceful Error Recovery**: Comprehensive fallback mechanisms with structured error handling
- **Automatic Cleanup**: Signal-based shutdown handling for all termination scenarios

#### 💾 **Distributed Memory & Logging**
- **Conversation Isolation**: Each conversation has dedicated memory space and log files
- **Cross-Process Context**: Memory accessible across all agent instances and hosts
- **Hierarchical Logging**: Day-based organization with conversation-specific debug files
- **Automatic TTL Management**: Memory cleanup with configurable expiration policies

#### 🚀 **Developer Experience**
- **Zero-Config Agents**: Automatic role derivation from class names (ManagerAgent → "manager")
- **Simplified APIs**: `env.final()` method encapsulates complex routing logic
- **Hot-Pluggable Agents**: Dynamic registration with runtime agent creation
- **Comprehensive Documentation**: Production-ready with detailed architecture guides

**Ready for production SMS assistant deployments with sophisticated conversation management, concurrent user handling, and enterprise-level reliability.**
