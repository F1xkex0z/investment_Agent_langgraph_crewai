"""
Configuration module for CrewAI system
"""

import os
from typing import Dict, Any

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有dotenv，尝试手动加载
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

class Config:
    """Configuration class"""

    # Version information
    VERSION = "1.0.0"
    PROJECT_NAME = "CrewAI Investment Analysis System"

    # System configuration
    MAX_WORKERS = 4
    DEFAULT_TIMEOUT = 30
    ENABLE_CACHING = True
    CACHE_ENABLED = True

    def __init__(self):
        # LLM Configuration - ensure all values are strings
        self.gemini_api_key = str(os.getenv('GEMINI_API_KEY', ''))
        self.gemini_model = str(os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'))
        self.openai_compatible_api_key = str(os.getenv('OPENAI_COMPATIBLE_API_KEY', ''))
        self.openai_compatible_base_url = str(os.getenv('OPENAI_COMPATIBLE_BASE_URL', ''))
        self.openai_compatible_model = str(os.getenv('OPENAI_COMPATIBLE_MODEL', 'gpt-4o'))

        # Data source configuration
        self.data_source_timeout = 30
        self.enable_cache = True
        self.cache_ttl = 300

        # Agent configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.enable_reasoning = False

        # Logging configuration
        self.log_level = 'INFO'
        self.enable_detailed_logging = False

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            'gemini_api_key': self.gemini_api_key,
            'gemini_model': self.gemini_model,
            'openai_compatible_api_key': self.openai_compatible_api_key,
            'openai_compatible_base_url': self.openai_compatible_base_url,
            'openai_compatible_model': self.openai_compatible_model
        }

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return {
            'timeout': self.data_source_timeout,
            'enable_cache': self.enable_cache,
            'cache_ttl': self.cache_ttl
        }

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return {
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'enable_reasoning': self.enable_reasoning
        }

    def validate_config(self) -> bool:
        """Validate configuration"""
        # Basic validation
        if not self.gemini_api_key and not self.openai_compatible_api_key:
            print("警告: 未配置API密钥，将使用模拟模式")
            return True

        return True

# Global config instance
config = Config()