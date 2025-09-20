"""
CrewAI System Settings

This module contains system-wide settings and constants.
"""

import os
from typing import Dict, Any, List


class CrewAISettings:
    """System-wide settings for CrewAI investment system"""

    # Data Source Settings
    DATA_SOURCES = {
        "akshare": {
            "enabled": True,
            "timeout": 30,
            "retry_count": 3
        },
        "news_api": {
            "enabled": True,
            "timeout": 60,
            "retry_count": 3,
            "max_articles": 100
        }
    }

    # Analysis Settings
    ANALYSIS_CONFIG = {
        "technical": {
            "default_period": "1y",
            "indicators": ["ma", "rsi", "macd", "bollinger", "volume"]
        },
        "fundamental": {
            "metrics": [
                "pe_ratio", "pb_ratio", "roe", "roa", "debt_ratio",
                "current_ratio", "revenue_growth", "earnings_growth"
            ]
        },
        "sentiment": {
            "default_news_count": 5,
            "max_news_count": 100,
            "sentiment_threshold": 0.2
        },
        "valuation": {
            "methods": ["dcf", "comparable", "multiples"],
            "discount_rate": 0.08,
            "growth_years": 5
        }
    }

    # Risk Management Settings
    RISK_CONFIG = {
        "max_position_size": 0.2,  # 20% of portfolio
        "stop_loss_threshold": 0.1,  # 10% loss
        "take_profit_threshold": 0.2,  # 20% profit
        "volatility_threshold": 0.3,  # 30% annual volatility
        "beta_threshold": 1.5  # 1.5 beta
    }

    # CrewAI Process Settings
    PROCESS_CONFIG = {
        "default_process": "hierarchical",
        "enable_parallel_analysis": True,
        "enable_async_execution": False,
        "max_execution_time": 1800,  # 30 minutes
        "max_rpm": 60  # Requests per minute
    }

    # Logging and Monitoring
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_path": "logs/crewai.log",
        "max_file_size": "10MB",
        "backup_count": 5
    }

    # Cache Settings
    CACHE_CONFIG = {
        "enabled": True,
        "ttl": 3600,  # 1 hour
        "max_size": 1000,
        "storage_type": "memory"  # "memory", "file", "redis"
    }

    # API Settings
    API_CONFIG = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "reload": True,
        "log_level": "info"
    }

    @classmethod
    def get_data_source_config(cls, source: str) -> Dict[str, Any]:
        """Get configuration for specific data source"""
        return cls.DATA_SOURCES.get(source, {})

    @classmethod
    def get_analysis_config(cls, analysis_type: str) -> Dict[str, Any]:
        """Get configuration for specific analysis type"""
        return cls.ANALYSIS_CONFIG.get(analysis_type, {})

    @classmethod
    def get_risk_config(cls) -> Dict[str, Any]:
        """Get risk management configuration"""
        return cls.RISK_CONFIG

    @classmethod
    def get_process_config(cls) -> Dict[str, Any]:
        """Get process configuration"""
        return cls.PROCESS_CONFIG

    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration"""
        return cls.LOGGING_CONFIG

    @classmethod
    def get_cache_config(cls) -> Dict[str, Any]:
        """Get cache configuration"""
        return cls.CACHE_CONFIG

    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration"""
        return cls.API_CONFIG

    @classmethod
    def from_environment(cls) -> 'CrewAISettings':
        """Create settings from environment variables"""
        settings = cls()

        # Override with environment variables if they exist
        if os.getenv('CREWAI_LOG_LEVEL'):
            settings.LOGGING_CONFIG['level'] = os.getenv('CREWAI_LOG_LEVEL')

        if os.getenv('CREWAI_MAX_RPM'):
            settings.PROCESS_CONFIG['max_rpm'] = int(os.getenv('CREWAI_MAX_RPM'))

        if os.getenv('CREWAI_MAX_EXECUTION_TIME'):
            settings.PROCESS_CONFIG['max_execution_time'] = int(os.getenv('CREWAI_MAX_EXECUTION_TIME'))

        if os.getenv('API_HOST'):
            settings.API_CONFIG['host'] = os.getenv('API_HOST')

        if os.getenv('API_PORT'):
            settings.API_CONFIG['port'] = int(os.getenv('API_PORT'))

        return settings


def get_settings() -> CrewAISettings:
    """Get global settings instance"""
    return CrewAISettings.from_environment()


# Environment-specific settings
def get_environment_settings() -> Dict[str, Any]:
    """Get settings based on current environment"""
    env = os.getenv('ENVIRONMENT', 'development')

    if env == 'production':
        return {
            "log_level": "WARNING",
            "cache_enabled": True,
            "max_rpm": 30,
            "async_execution": True,
            "monitoring_enabled": True
        }
    elif env == 'testing':
        return {
            "log_level": "DEBUG",
            "cache_enabled": False,
            "max_rpm": 100,
            "async_execution": False,
            "monitoring_enabled": False
        }
    else:  # development
        return {
            "log_level": "INFO",
            "cache_enabled": True,
            "max_rpm": 60,
            "async_execution": False,
            "monitoring_enabled": True
        }