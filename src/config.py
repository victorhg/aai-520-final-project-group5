"""
Configuration and main entry point for the Investment Research Agent.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Agent configuration
AGENT_CONFIG = {
    "model": os.getenv("DEFAULT_MODEL", "gpt-4"),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "memory_file": os.getenv("MEMORY_FILE", "./data/investment_agent_memory.json"),
    "max_iterations": int(os.getenv("MAX_ITERATIONS", "3")),
    "target_quality_score": float(os.getenv("TARGET_QUALITY_SCORE", "8.0"))
}

# API configuration
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "news_api": os.getenv("NEWS_API_KEY")
}

# Data configuration
DATA_CONFIG = {
    "data_dir": os.getenv("DATA_DIR", "./data"),
    "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
    "cache_duration_hours": int(os.getenv("CACHE_DURATION_HOURS", "24"))
}

def validate_configuration():
    """Validate that required configuration is present."""
    issues = []
    
    if not API_KEYS["openai"]:
        issues.append("OPENAI_API_KEY is required")
    
    if not os.path.exists(DATA_CONFIG["data_dir"]):
        try:
            os.makedirs(DATA_CONFIG["data_dir"])
        except Exception as e:
            issues.append(f"Cannot create data directory: {e}")
    
    return issues

def get_agent_config():
    """Get validated agent configuration."""
    issues = validate_configuration()
    if issues:
        raise ValueError(f"Configuration issues: {'; '.join(issues)}")
    
    return AGENT_CONFIG, API_KEYS, DATA_CONFIG