# Alaris Assistant

A production-grade agentic system built on Redis Streams with distributed memory management for sophisticated conversation handling.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main assistant demo
python src/assistant.py

# Run memory system demonstration
python examples/memory_system_demo.py

# Run distributed cleanup tests
python tests/test_distributed_cleanup.py
```

## Project Structure

```
assistant/
├── src/           # Production source code
│   ├── agent.py           # Core agent framework
│   ├── assistant.py       # Main assistant implementation
│   ├── llm.py            # LLM integration
│   ├── qdrant.py         # Vector database integration
│   └── config.py         # Configuration management
├── examples/      # Demonstration scripts
├── tests/         # Test suites
├── docs/          # Documentation
├── prompts/       # LLM prompts and templates
├── data/          # Data files
├── cache/         # Cache directory
└── log/           # Log files
```

## Key Features

- **🚀 Production-Ready**: Enterprise-grade Redis connection pooling, health monitoring, automatic cleanup
- **🔄 Distributed Memory**: Conversation-scoped context sharing across processes and hosts
- **⚡ Horizontal Scaling**: Multiple agents per role with load balancing
- **🛡️ Safety Systems**: Circuit breakers, process limits, comprehensive error handling
- **📊 Monitoring**: Detailed statistics and performance tracking
- **🎯 Zero-Config**: Automatic agent registration and role derivation

## Architecture

The system uses Redis Streams for inter-agent communication with a unified BaseAgent class that provides:

- **Stateless Design**: All conversation context stored in distributed memory
- **Pipeline Processing**: Messages flow through specialized agents (Manager → Uppercase → Reverse)
- **Safety Limits**: Process count, time limits, and envelope age restrictions
- **Comprehensive Tracing**: Complete audit trail for debugging and monitoring

## Documentation

See [docs/README.md](docs/README.md) for complete documentation including:

- [Agentic Architecture](docs/AGENTIC_ARCHITECTURE.md) - System design overview
- [Memory System](docs/MEMORY_SYSTEM.md) - Distributed memory patterns

## Development

The project follows clean organization standards:
- **Source code** in `src/`
- **Examples** in `examples/`
- **Tests** in `tests/`
- **Documentation** in `docs/`

Ready for production SMS assistant deployments with sophisticated conversation management capabilities.
