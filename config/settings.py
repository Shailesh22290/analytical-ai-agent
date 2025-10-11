"""
Configuration settings for the Analytical AI Agent
"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Global settings for the agent"""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model names
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GENERATIVE_MODEL: str = "gemini-2.0-flash-exp"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INPUT_DIR: Path = DATA_DIR / "input"
    VECTOR_DIR: Path = DATA_DIR / "vectors"
    
    # Vector DB settings
    VECTOR_DIMENSION: int = 3072  # Gemini embedding dimension
    FAISS_INDEX_TYPE: str = "Flat"  # or "IVFFlat" for larger datasets
    
    # Supported intents
    SUPPORTED_INTENTS: List[str] = [
        "compare_averages",
        "filter_threshold",
        "sort",
        "top_n",
        "compare_top",
        "explain_row"
    ]
    
    # LLM settings
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.1  # Low temperature for deterministic parsing
    
    # Pandas display settings
    MAX_ROWS_DISPLAY: int = 100
    
    @classmethod
    def validate(cls) -> bool:
        """Validate settings"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in environment")
        
        # Create directories if they don't exist
        cls.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def get_vector_db_path(cls, file_id: str) -> Path:
        """Get path for vector DB file"""
        return cls.VECTOR_DIR / f"{file_id}.faiss"
    
    @classmethod
    def get_metadata_path(cls, file_id: str) -> Path:
        """Get path for metadata file"""
        return cls.VECTOR_DIR / f"{file_id}_metadata.pkl"


# Create settings instance
settings = Settings()