import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# File paths
KNOWLEDGE_BASE_JSON = str(DATA_DIR / "knowledge_base.json")
LOG_FILE = str(LOGS_DIR / "qa_log.txt")

# Vector DB paths - Support multiple policy types
GRATUITY_DB_PATH = str(BASE_DIR / "document_chroma_store_gratuity")
LEAVE_DB_PATH = str(BASE_DIR / "document_chroma_store_leave")
UPKILLING_DB_PATH = str(BASE_DIR / "document_chroma_store_upskilling")
HARASSMENT_DB_PATH = str(BASE_DIR / "document_chroma_store_harassment")

# Policy type mappings
POLICY_TYPES = {
    "gratuity": {
        "db_path": GRATUITY_DB_PATH,
        "keywords": ["gratuity", "retirement", "benefits", "compensation", "salary"]
    },
    "leave": {
        "db_path": LEAVE_DB_PATH,
        "keywords": ["leave", "vacation", "sick", "maternity", "paternity", "holiday"]
    },
    "upskilling": {
        "db_path": UPKILLING_DB_PATH,
        "keywords": ["training", "learning", "development", "skill", "course", "certification"]
    },
    "harassment": {
        "db_path": HARASSMENT_DB_PATH,
        "keywords": ["harassment", "discrimination", "workplace", "conduct", "policy"]
    },
    "general": {
        "db_path": str(BASE_DIR / "document_chroma_store_general"),
        "keywords": ["policy", "employee", "company", "organization", "guidelines"]
    }
}

# Text splitting settings
CHUNK_SIZE = 500  # Smaller chunks for better context
CHUNK_OVERLAP = 100  # Overlap to maintain context between chunks

# Model settings
LLM_MODEL = "gemma2-9b-it"  # Using Gemma model for chat
TEMPERATURE = 0.7  # Add temperature for more natural responses
MAX_TOKENS = 2048  # Maximum response length

# Retrieval settings
TOP_K_MATCHES = 3  # Number of similar documents to retrieve
SIMILARITY_THRESHOLD = 0.4  # Minimum similarity score for matches (reduced from 0.6 for better results)

# Table extraction settings
TABLE_EXTRACTION_ENABLED = True
TABLE_MIN_ROWS = 2
TABLE_MIN_COLS = 2

# Logging settings
LOG_LEVEL = "INFO"
ENABLE_DEBUG = False

# API Keys (from environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
