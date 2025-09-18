import os
import logging
from dotenv import load_dotenv
import sys # Import sys for printing to stderr
import inspect

# Determine the path to the .env file (assuming it's in the project root)
# If the script is run from DeskGuard/src/, the root is one level up.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load environment variables from .env file if it exists
if os.path.exists(dotenv_path):
    #print(f"DEBUG: Found .env file at: {dotenv_path}", file=sys.stderr)
    load_dotenv(dotenv_path=dotenv_path, override=True) # Use override=True to ensure .env takes precedence over existing env vars if loaded
    #print(f"DEBUG: load_dotenv finished.", file=sys.stderr)
else:
    print(f"DEBUG: Warning: .env file not found at {dotenv_path}. Using system environment variables or defaults.", file=sys.stderr)
    sys.exit(1)

# --- Logging ---
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT = '%(asctime)s - %(name)-10s : %(lineno)3s - %(levelname)-5s - %(message)s'
# Ensure log directory exists if file logging is used later
LOG_DIR = os.getenv("LOG_DIR", os.path.join(project_root, 'log'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow all log messages to flow to handlers

logging.basicConfig(
    format=LOG_FORMAT,
    level=LOG_LEVEL
)
# After your basicConfig call
logging.getLogger().handlers[0].setLevel(LOG_LEVEL)  # Explicitly set the root handler level

# Set up optional rotating file handlers for debug and info logs
DEBUG_LOG_FILE = os.getenv("DEBUG_LOG_FILE", 'debug.log')
INFO_LOG_FILE = os.getenv("INFO_LOG_FILE", 'info.log')

# Check if LOG_DIR exists and is writable
if os.path.exists(LOG_DIR) and os.access(LOG_DIR, os.W_OK):
    try:
        # Configure rotating file handler for debug logs
        if DEBUG_LOG_FILE and DEBUG_LOG_FILE.strip():
            from logging.handlers import RotatingFileHandler
            debug_log_path = os.path.join(LOG_DIR, DEBUG_LOG_FILE)
            debug_handler = RotatingFileHandler(
                debug_log_path,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logging.getLogger().addHandler(debug_handler)
            logger.debug(f"Debug log file handler initialized: {debug_log_path}")
        
        # Configure rotating file handler for info logs
        if INFO_LOG_FILE and INFO_LOG_FILE.strip():
            from logging.handlers import RotatingFileHandler
            info_log_path = os.path.join(LOG_DIR, INFO_LOG_FILE)
            info_handler = RotatingFileHandler(
                info_log_path,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            info_handler.setLevel(logging.INFO)
            info_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logging.getLogger().addHandler(info_handler)
            logger.debug(f"Info log file handler initialized: {info_log_path}")
    except Exception as e:
        logger.error(f"Failed to initialize log file handlers: {str(e)}")
else:
    logger.warning(f"Log directory {LOG_DIR} does not exist or is not writable. File logging disabled.")

DATA_DIR = os.getenv("DATA_DIR", os.path.join(project_root, 'data'))
os.makedirs(DATA_DIR, exist_ok=True)
if os.path.exists(DATA_DIR) and os.access(DATA_DIR, os.W_OK):
    logger.info(f"Data dir: {DATA_DIR}")
else:
    logger.warning(f"Data directory {DATA_DIR} does not exist or is not writable.")
    sys.exit(1)

CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(project_root, 'cache'))
os.makedirs(CACHE_DIR, exist_ok=True)
if os.path.exists(CACHE_DIR) and os.access(CACHE_DIR, os.W_OK):
    logger.info(f"Cache dir: {CACHE_DIR}")
else:
    logger.warning(f"Cache directory {CACHE_DIR} does not exist or is not writable.")
    sys.exit(1)

PROMPTS_DIR = os.getenv("PROMPTS_DIR", os.path.join(project_root, 'prompts'))
if os.path.exists(PROMPTS_DIR):
    logger.info(f"Prompts dir: {PROMPTS_DIR}")
else:
    logger.warning(f"Prompts directory {PROMPTS_DIR} does not exist.")
    sys.exit(1)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

#SUM_MODEL = os.getenv("SUM_MODEL", "gemini-2.5-flash@google")
SUM_MODEL = os.getenv("SUM_MODEL", "gpt-oss-20b@ollama")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-large@openai")
EMB_DIM = int(os.getenv("EMB_DIM", 1024 if EMB_MODEL.endswith("@ollama") else 3072))
#VSN_MODEL = os.getenv("VSN_MODEL", "llava:7b@ollama")
VSN_MODEL = os.getenv("VSN_MODEL", "gemma3:27b@ollama")
RSP_MODEL = os.getenv("RSP_MODEL", "gpt-5-nano@openai")

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", f"assistant_{EMB_MODEL.split('@')[0]}")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

# Redis Connection Pool Configuration
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))  # Pool size
REDIS_RETRY_ON_TIMEOUT = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
REDIS_RETRY_ON_ERROR = os.getenv("REDIS_RETRY_ON_ERROR", "true").lower() == "true"
REDIS_RETRY_ATTEMPTS = int(os.getenv("REDIS_RETRY_ATTEMPTS", "3"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))  # seconds
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))  # seconds
REDIS_HEALTH_CHECK_INTERVAL = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))  # seconds

MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", 50))
OVERLAP_CHUNK_TOKENS = int(os.getenv("OVERLAP_CHUNK_TOKENS", 75))
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", 512))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_INVITE_CODE = os.getenv("TELEGRAM_INVITE_CODE", "")

DOC_BASE_URL = os.getenv("DOC_BASE_URL", "")

VERSION = os.getenv("VERSION", "1.0.1")

logger.info("Configuration loaded.")

# --- Rate limiting and metrics ---
# Interval reporting in seconds. If set to 0, interval metrics are disabled.
REPORT_INTERVAL_SECONDS = int(os.getenv("REPORT_INTERVAL_SECONDS", "60"))

# Per-model+provider limits. If a model key is absent, no limits are applied for it.
# Key format: "<model>@<provider>" (e.g., "gpt-5-nano@openai").
# Values: {"rpm": int or None, "tpm": int or None}
RATE_LIMITS = {
    # Example:
    # "gpt-5-nano@openai": {"rpm": 3000, "tpm": 180000},
    # "gemini-2.5-flash@google": {"rpm": 60, "tpm": 60000},
}

def print_config():
    config_vars = {}
    for name, value in inspect.getmembers(sys.modules[__name__]):
        if name.isupper():  # Check if the name is in uppercase (convention for constants)
            config_vars[name] = value

    # Sort the variables alphabetically by name
    sorted_config_vars = dict(sorted(config_vars.items()))

    for name, value in sorted_config_vars.items():
        logger.debug(f"{name} = {value}")
