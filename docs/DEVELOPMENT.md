# Development Guidelines - Alaris Assistant

## Multi-Computer Development Workflow

### Setup on New Computer
1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd assistant
   ```

2. **Environment Setup**
   ```bash
   python3.13 -m venv venv3.13
   source venv3.13/bin/activate  # or venv3.13\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Load Project Context**
   - Windsurf will automatically read `.windsurf/project.json`
   - AI assistant loads context from `.windsurf/ai_context.md`
   - Review `docs/DEVELOPMENT.md` (this file) for guidelines

### Project Structure Standards

#### Directory Organization
```
assistant/
├── src/                    # Production source code
│   ├── agent.py           # Core agent framework
│   ├── assistant.py       # Main assistant implementation
│   ├── llm.py            # LLM integration
│   ├── qdrant.py         # Vector database
│   └── config.py         # Configuration management
├── tests/                 # Test suites
│   └── test_*.py         # Test files with test_ prefix
├── examples/              # Demonstration scripts
│   └── *_demo.py         # Demo files with _demo suffix
├── docs/                  # Documentation
│   ├── README.md         # Documentation index
│   ├── AGENTIC_ARCHITECTURE.md  # System architecture
│   ├── MEMORY_SYSTEM.md  # Memory system guide
│   └── DEVELOPMENT.md    # This file
├── .windsurf/            # Project configuration
│   ├── project.json      # Project settings and standards
│   └── ai_context.md     # AI assistant context
└── [gitignored dirs]     # cache/, log/, data/, venv3.13/
```

### Coding Standards

#### Python Style
- **Python Version**: 3.13+
- **Line Length**: 88 characters (Black formatter)
- **Import Style**: Absolute imports preferred
- **Docstring Style**: Google format
- **Async Patterns**: Use async/await throughout

#### Agent Development
- **Base Class**: All agents inherit from `BaseAgent`
- **Role Derivation**: Automatic from class name (e.g., `ManagerAgent` → `manager`)
- **Memory Access**: Use `await self.get_memory(conversation_id)`
- **Error Handling**: Comprehensive try/catch with logging

#### Memory System Patterns
```python
# Standard memory usage pattern
class MyAgent(BaseAgent):
    async def process(self, env: Envelope) -> Envelope:
        memory = await self.get_memory(env.conversation_id)
        
        # Dict operations
        await memory.set("key", "value")
        value = await memory.get("key", default_value)
        
        # Conversation history
        await memory.add_message("user", user_text)
        messages = await memory.get_messages(limit=10)
        
        return env
```

### Git Workflow

#### Branch Strategy
- **main**: Production-ready code
- **feature/***: Feature development branches
- **fix/***: Bug fix branches

#### Commit Style
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`
- Include scope when relevant: `feat(memory): add cleanup methods`
- Keep commits atomic and descriptive

#### Sync Strategy
- **Project config files** (`.windsurf/`, `docs/`) are committed to Git
- **Personal settings** stay local to each computer
- **AI context** syncs automatically via Git

### Security Guidelines

#### Critical Rules
- ❌ **NEVER** commit `.env` files or secrets
- ❌ **NEVER** access `.env` via scripts or workarounds
- ✅ **ALWAYS** use `.env.example` for configuration templates
- ✅ **ALWAYS** clean up temporary files

#### Gitignore Workaround
When AI assistant needs access to gitignored files:
1. Create temporary Python script
2. Execute via command line
3. Get content from output
4. Remove temporary script
5. **Never use for sensitive files**

### Testing Standards

#### Test Organization
- **Location**: `tests/` directory
- **Naming**: `test_*.py` files
- **Structure**: One test file per module
- **Coverage**: Aim for comprehensive coverage of agent functionality

#### Test Patterns
```python
# tests/test_memory_system.py
import asyncio
import pytest
from src.agent import BaseAgent, start_runtime, stop_runtime

class TestAgent(BaseAgent):
    async def process(self, env):
        # Test implementation
        return env

async def test_memory_operations():
    await start_runtime()
    try:
        # Test logic here
        pass
    finally:
        await stop_runtime()
```

### Documentation Standards

#### File Organization
- **README.md**: Project overview and quick start
- **Architecture docs**: Detailed system design
- **API docs**: Module and class documentation
- **Examples**: Working code demonstrations

#### Writing Style
- Clear, concise explanations
- Code examples for complex concepts
- Step-by-step instructions for setup
- Regular updates as system evolves

### AI Assistant Integration

#### Context Synchronization
- Project context stored in `.windsurf/ai_context.md`
- Updated as agreements and architecture evolve
- Committed to Git for multi-computer sync
- AI assistant loads context automatically

#### Development Assistance
- AI understands project architecture and patterns
- Follows security rules (never accesses .env)
- Uses established coding standards
- Maintains project organization principles

### Environment Variables

#### Required Variables (.env)
```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password

# LLM API Keys (choose your providers)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
OLLAMA_BASE_URL=http://localhost:11434

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_key
```

#### Configuration Template
Use `.env.example` for sharing configuration structure without secrets.

### Troubleshooting

#### Common Issues
1. **Import Errors**: Ensure `src/` is in Python path
2. **Redis Connection**: Check Redis server is running
3. **Memory Access**: Use proper async/await patterns
4. **Agent Registration**: Ensure agents inherit from BaseAgent

#### Debug Tools
- **Logs**: Check `log/info.log` for runtime information
- **Statistics**: Review `agent_statistics.txt` for performance
- **Memory Stats**: Use `get_memory_stats()` for memory analysis

This development guide ensures consistent practices across all computers and team members working on the Alaris Assistant project.
