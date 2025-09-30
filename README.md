# Alaris SMS Platform Assistant

A production-grade RAG-based documentation assistant built on a sophisticated agentic architecture with enterprise Redis Streams, distributed memory, and intelligent conversation management. Designed to help company users with the Alaris SMS telecom billing and routing platform.

## 🎯 Project Overview

The Alaris Assistant is a **two-stage system** that combines data preparation with intelligent user assistance:

### 📊 **Stage 1: Data Preparation Pipeline**
Transforms source documentation into RAG-ready artifacts:
- **`crawl.py`**: Web crawler with caching, robots.txt compliance, and metadata extraction
- **`prepare.py`**: Markdown processor with LLM-powered image descriptions, summarization, and smart chunking

### 🤖 **Stage 2: User Assistance System**
Production-grade agentic pipeline for user queries:
- **Redis Streams Architecture**: Enterprise-level concurrent processing with automatic load balancing
- **Intelligent Agent Pipeline**: Multi-stage processing from language detection to response generation
- **RAG Integration**: Vector search with Qdrant for documentation-grounded responses
- **Telegram Bot**: Non-blocking concurrent user handling with professional UX

## ✨ Key Features

### 🏗️ **Production Architecture**
- **Concurrent Processing**: Multiple users handled simultaneously with non-blocking Telegram bot
- **Enterprise Redis Streams**: Connection pooling (50+ connections), health monitoring, automatic cleanup
- **Distributed Memory**: Conversation-scoped storage across processes and hosts with automatic TTL management
- **Safety Systems**: Circuit breakers, self-loop detection, envelope aging, comprehensive error handling
- **Zero-Config Agents**: Automatic role derivation from class names (ManagerAgent → "manager")

### 🧠 **Intelligent Processing Pipeline**
- **Language Detection**: Automatic language detection from conversation history (Russian, Chinese, English, etc.)
- **Translation & Normalization**: Mandatory translation and spell-checking for all queries
- **Context-Aware Questions**: Canonical question extraction with temporal awareness and pronoun resolution
- **Security Guardrails**: LLM-powered content filtering with prompt injection protection
- **Professional Clarification**: Viktor Aleksandrovich persona for polite user guidance

### 🔍 **RAG Capabilities**
- **Multi-Provider LLM Integration**: OpenAI, Google Gemini, Ollama support with unified interface
- **Vector Search**: Qdrant integration for semantic document retrieval
- **Smart Chunking**: Sentence-aware chunking with overlap and token limits
- **Image Augmentation**: LLM-generated image descriptions for vision-aware content enrichment
- **Metadata Extraction**: Source URLs, breadcrumbs, descriptions, summaries, keypoints

### 📊 **Enterprise Features**
- **Hierarchical Logging**: Day-based conversation-specific logs with isolated debugging
- **Performance Monitoring**: Rate limiting, metrics tracking, statistics reporting
- **Automatic Cleanup**: Signal-based shutdown handling for all termination scenarios
- **Horizontal Scaling**: Multiple agent instances per role with Redis consumer groups

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.13+ with virtual environment
python3.13 -m venv venv3.13
source venv3.13/bin/activate  # or venv3.13\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy .env.example to .env and update)
export OPENAI_API_KEY=your_key
export GOOGLE_API_KEY=your_key
export REDIS_HOST=localhost
export REDIS_PORT=6379
export QDRANT_URL=http://localhost:6333
export TELEGRAM_BOT_TOKEN=your_token  # Optional for Telegram bot
```

### Running the System

```bash
# Data Preparation (Stage 1)
python src/crawl.py    # Crawl documentation
python src/prepare.py  # Process and index to Qdrant

# User Assistance (Stage 2)
python src/assistant.py  # Interactive console demo
python src/tg.py        # Telegram bot (concurrent processing)

# Testing
python tests/test_distributed_cleanup.py
python examples/memory_system_demo.py
```

## 📁 Project Structure

```
assistant/
├── src/                    # Production source code
│   ├── agent.py           # Core agentic framework with BaseAgent
│   ├── assistant.py       # Agent implementations (Manager, Lang, Translation, Essence, etc.)
│   ├── llm.py            # Multi-provider LLM integration (OpenAI, Google, Ollama)
│   ├── qdrant.py         # Vector database integration
│   ├── crawl.py          # Web crawler with caching
│   ├── prepare.py        # Markdown processor with RAG preparation
│   ├── tg.py             # Telegram bot with concurrent processing
│   ├── config.py         # Configuration management
│   ├── util.py           # Utility functions (language detection, etc.)
│   ├── ratelimit_metrics.py  # Rate limiting and performance metrics
│   ├── stream_cleanup.py # Redis stream cleanup manager
│   └── sample.py         # Configuration testing utility
├── docs/                  # Comprehensive documentation
│   ├── README.md         # Documentation index
│   ├── AGENTIC_ARCHITECTURE.md  # System design and agent framework
│   ├── PIPELINE_ARCHITECTURE.md # RAG pipeline and agent definitions
│   ├── MEMORY_SYSTEM.md  # Distributed memory guide
│   ├── RAG_DATA_PREPARATION.md  # Data preparation pipeline
│   ├── CONCURRENT_PROCESSING.md # Concurrent processing guide
│   └── DEVELOPMENT.md    # Development guidelines
├── prompts/              # LLM prompt templates
│   ├── usr_rsp.md       # Response generation (Viktor Aleksandrovich)
│   ├── usr_essence.md   # Canonical question extraction
│   ├── usr_translate.md # Translation and normalization
│   ├── usr_guardrails.md # Safety and content filtering
│   ├── usr_clarify.md   # Clarification message composition
│   ├── usr_sum.md       # Document summarization
│   ├── usr_vsn.md       # Image description (vision)
│   └── usr_sample.md    # Sample response generation
├── tests/                # Test suites
├── examples/             # Demonstration scripts
├── data/                 # Data directory (gitignored)
│   ├── html/            # Raw HTML snapshots
│   ├── md/              # Markdown exports
│   ├── images/          # Downloaded images
│   └── docs/            # RAG-ready JSON documents
├── cache/                # Cache directory (gitignored)
├── log/                  # Log directory (gitignored)
├── venv3.13/            # Virtual environment (gitignored)
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration (gitignored)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🤖 Agent Architecture

### Core Agents (Implemented)

1. **CommandAgent** (1 instance) - Administrative commands
   - `clear_history`: Clear conversation message history
   - `clear_all`: Complete conversation data cleanup
   - `set_language`: User language preferences

2. **ManagerAgent** (1 instance) - FSM-based pipeline orchestration
   - Centralized routing authority with intelligent stage management
   - Unified clarification handling across all agents
   - Conversation memory integration

3. **LangAgent** (1 instance) - Language detection
   - Analyzes conversation history for language patterns
   - Supports Cyrillic (Russian), Chinese, English, and more
   - User preference tracking

4. **TranslationAgent** (2 instances) - Translation and normalization
   - Mandatory processing for all queries
   - Translation for non-English + spell-checking for English
   - Prompt injection immunity

5. **EssenceAgent** (2 instances) - Canonical question extraction
   - Context-aware question formulation with temporal awareness
   - Pronoun resolution and topic tracking
   - RAG-ready self-contained questions

6. **GuardrailsAgent** (2 instances) - Safety and content filtering
   - LLM-powered abuse detection
   - Harmful content, prompt injection, system manipulation detection
   - Professional refusal handling

7. **ResponseAgent** (2 instances) - LLM response generation
   - Flagship LLM integration (GPT-4, Gemini, etc.)
   - Documentation-grounded responses
   - Uncertainty detection with clarification requests

8. **ClarificationAgent** (2 instances) - User clarification
   - Viktor Aleksandrovich persona (wise librarian)
   - Context-aware, localized, professional guidance
   - Universal clarification system

### Planned Agents (RAG Enhancement)

- **SearchAgent**: Multi-level RAG search (keyword, semantic, hierarchical)
- **AugmentationAgent**: Enhanced prompt crafting with context
- **QualityAgent**: Response validation with retry capability

See [docs/PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md) for complete agent definitions.

## 📚 Documentation

### Essential Reading
1. **[docs/README.md](docs/README.md)** - Documentation index and quick start
2. **[docs/PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md)** - RAG pipeline and agent definitions
3. **[docs/AGENTIC_ARCHITECTURE.md](docs/AGENTIC_ARCHITECTURE.md)** - System design and agent framework
4. **[docs/MEMORY_SYSTEM.md](docs/MEMORY_SYSTEM.md)** - Distributed memory guide

### Additional Documentation
- **[docs/RAG_DATA_PREPARATION.md](docs/RAG_DATA_PREPARATION.md)** - Data preparation pipeline (crawl.py, prepare.py)
- **[docs/CONCURRENT_PROCESSING.md](docs/CONCURRENT_PROCESSING.md)** - Concurrent processing and scaling
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development guidelines and standards

## 🔧 Configuration

### Environment Variables

```bash
# === Version ===
VERSION=2.4.0

# === Redis Configuration ===
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=70.0
REDIS_HEALTH_CHECK_INTERVAL=30

# === LLM Provider Configuration ===
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434

# === Model Selection ===
GEN_MODEL=gpt-5-nano@openai
SUM_MODEL=gpt-oss-20b@ollama
EMB_MODEL=text-embedding-3-large@openai
EMB_DIM=3072
VSN_MODEL=gemma3:27b@ollama

# === Qdrant Configuration ===
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=assistant_text-embedding-3-large

# === Telegram Bot (Optional) ===
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_INVITE_CODE=your_invite_code

# === Documentation Source ===
DOC_BASE_URL=https://your-docs-domain.com

# === Directory Configuration ===
DATA_DIR=./data
CACHE_DIR=./cache
LOG_DIR=./log
PROMPTS_DIR=./prompts

# === Processing Configuration ===
ASSISTANT_TIMEOUT=300
ASSISTANT_HISTORY_LIMIT=10
MIN_CHUNK_TOKENS=50
MAX_CHUNK_TOKENS=512
OVERLAP_CHUNK_TOKENS=75

# === Logging ===
LOG_LEVEL=INFO
DEBUG_LOG_FILE=debug.log
INFO_LOG_FILE=info.log
```

## 🎯 Use Cases

### For Company Users
- **Documentation Search**: Find answers in large SMS platform manuals
- **Multilingual Support**: Ask questions in Russian, Chinese, English, etc.
- **Context-Aware Answers**: Follow-up questions understand conversation history
- **Professional Guidance**: Viktor Aleksandrovich clarifies ambiguous requests
- **Source References**: All answers include documentation links

### For Developers
- **Production-Ready Framework**: Enterprise-grade agentic system
- **Distributed Memory**: Conversation state across processes
- **RAG Pipeline**: Complete data preparation to response generation
- **Multi-Provider LLM**: Flexible provider integration
- **Concurrent Processing**: Handle multiple users simultaneously

## 🔒 Security Features

- **Prompt Injection Protection**: TranslationAgent immune to embedded instructions
- **Content Filtering**: GuardrailsAgent with LLM-powered analysis
- **Multi-Layer Security**: Translation → Essence → Guardrails → Response
- **Professional Refusal**: Inappropriate requests handled with polite clarification
- **Audit Trail**: Complete envelope tracing with timing information

## 📊 Performance

- **Concurrent Users**: Multiple users processed simultaneously
- **Redis Connection Pool**: 50 connections for high throughput
- **LLM Instance Scaling**: 2+ instances per heavy agent for parallelism
- **Response Time**: 3-10 seconds typical (varies by LLM provider)
- **Memory Efficiency**: Automatic TTL and cleanup

## 🛠️ Development

### Adding New Agents

```python
class MyNewAgent(BaseAgent):  # Role automatically becomes "mynew"
    def __init__(self):
        super().__init__()  # Zero parameters needed!
    
    async def process(self, env: Envelope) -> Envelope:
        # Your business logic here
        env.payload["result"] = await process_data(env.payload["input"])
        return env.final()  # Simplified final delivery

# Agent automatically registers when instantiated
_mynew = MyNewAgent()
```

### Testing

```bash
# Run specific tests
python tests/test_distributed_cleanup.py

# Run examples
python examples/memory_system_demo.py

# Interactive testing
python src/assistant.py
```

## 📦 Dependencies

Key dependencies (see [requirements.txt](requirements.txt) for complete list):
- **Python 3.13+**: Core language
- **Redis 6.4.0+**: Streams and distributed memory
- **python-telegram-bot 20.0+**: Telegram integration
- **openai 1.0.0+**: OpenAI API
- **google-generativeai 0.8.0+**: Google Gemini API
- **qdrant-client**: Vector database
- **beautifulsoup4, markdownify**: Web crawling and HTML processing
- **tiktoken**: Token counting
- **langdetect**: Language detection

## 🚀 Deployment

### Docker Deployment (Recommended)

```bash
# Build image
docker build -t alaris-assistant .

# Run with docker-compose
docker-compose up -d
```

### Manual Deployment

```bash
# Ensure Redis and Qdrant are running
redis-server &
qdrant-server &

# Run data preparation
python src/crawl.py
python src/prepare.py

# Run assistant
python src/tg.py  # Telegram bot
# or
python src/assistant.py  # Interactive console
```

## 📝 License

[Your License Here]

## 👥 Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for development guidelines.

## 📧 Contact

[Your Contact Information]

---

**Ready for production SMS assistant deployments with sophisticated conversation management, concurrent user handling, and enterprise-level reliability.**
