import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent.absolute()
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # File paths
    KNOWLEDGE_BASE_JSON: str = str(DATA_DIR / "knowledge_base.json")
    LOG_FILE: str = str(LOGS_DIR / "qa_log.txt")
    
    # Vector DB paths
    GRATUITY_DB_PATH: str = str(BASE_DIR / "document_chroma_store_gratuity")
    LEAVE_DB_PATH: str = str(BASE_DIR / "document_chroma_store_leave")
    UPSKILLING_DB_PATH: str = str(BASE_DIR / "document_chroma_store_upskilling")
    
    # Text splitting settings
    CHUNK_SIZE: int = 500  # Smaller chunks for better context
    CHUNK_OVERLAP: int = 100  # Overlap to maintain context between chunks
    
    # Model settings
    LLM_MODEL: str = "gemma2-9b-it"  # Using Gemma model for chat
    TEMPERATURE: float = 0.7  # Controls randomness in generation
    MAX_TOKENS: int = 2048  # Maximum response length
    
    # Retrieval settings
    TOP_K_MATCHES: int = 3  # Number of similar documents to retrieve
    SIMILARITY_THRESHOLD: float = -1.0  # Minimum similarity score for matches
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    ENABLE_DEBUG: bool = False
    
    # API Keys (from environment)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    class Config:
        case_sensitive = True

# Create settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.LOGS_DIR, exist_ok=True)