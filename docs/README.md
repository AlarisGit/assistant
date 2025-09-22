# Alaris Assistant - Documentation

This directory contains all documentation for the Alaris Assistant agentic system.

## Documentation Index

### Core Architecture
- **[AGENTIC_ARCHITECTURE.md](AGENTIC_ARCHITECTURE.md)** - Comprehensive system architecture overview
- **[MEMORY_SYSTEM.md](MEMORY_SYSTEM.md)** - Distributed memory subsystem documentation

### Quick Start
1. Review the [Agentic Architecture](AGENTIC_ARCHITECTURE.md) to understand the system design
2. Check the [Memory System](MEMORY_SYSTEM.md) for context management patterns
3. Explore the [examples/](../examples/) directory for working demonstrations
4. Run tests from the [tests/](../tests/) directory to verify functionality

### System Components

#### Core Modules (`src/`)
- `agent.py` - Foundation framework with BaseAgent class and Redis Streams communication
- `assistant.py` - Main assistant implementation with pipeline orchestration
- `llm.py` - Multi-provider LLM integration (OpenAI, Google, Ollama)
- `qdrant.py` - Vector database integration for RAG capabilities
- `config.py` - Configuration management with environment variables

#### Key Features
- **Production-grade Redis Streams** communication with connection pooling
- **Distributed memory system** for conversation context across processes/hosts
- **Safety systems** with circuit breakers and comprehensive monitoring
- **Zero-config agents** with automatic role derivation
- **Horizontal scaling** with load balancing and backpressure management

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

#### Running Examples
```bash
# Memory system demonstration
python examples/memory_system_demo.py

# Main assistant demo
python src/assistant.py
```

#### Running Tests
```bash
# Distributed cleanup test
python tests/test_distributed_cleanup.py
```

### Architecture Highlights

The Alaris Assistant is a production-grade agentic system built on Redis Streams with:

- **Enterprise-level reliability** - Connection pooling, health monitoring, automatic cleanup
- **Stateless agent design** - All context stored in distributed memory system
- **Horizontal scalability** - Multiple agents per role across hosts
- **Comprehensive monitoring** - Detailed statistics and performance tracking
- **Developer-friendly** - Zero-config agents with intuitive interfaces

Ready for production SMS assistant deployments with sophisticated conversation management and context sharing capabilities.
