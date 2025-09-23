"""
CrewAI系统配置模块
集中管理所有配置项
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """系统配置类"""

    # 项目基础配置
    PROJECT_NAME: str = "CrewAI Investment Analysis System"
    VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = "logs/crewai_system.log"

    # LLM配置 - 确保所有URL都是字符串类型
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_COMPATIBLE_API_KEY: Optional[str] = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    OPENAI_COMPATIBLE_BASE_URL: Optional[str] = str(os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""))
    OPENAI_COMPATIBLE_MODEL: Optional[str] = os.getenv("OPENAI_COMPATIBLE_MODEL")

    # 系统配置
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))

    # 数据源配置
    AKSHARE_TIMEOUT: int = int(os.getenv("AKSHARE_TIMEOUT", "30"))
    AKSHARE_RETRY_COUNT: int = int(os.getenv("AKSHARE_RETRY_COUNT", "3"))

    # API配置
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8001"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "True").lower() == "true"

    # 安全配置
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # 路径配置
    BASE_DIR: Path = Path(__file__).parent
    LOG_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = BASE_DIR / "cache"

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        cls.LOG_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)

    @classmethod
    def validate_config(cls):
        """验证配置是否有效"""
        required_vars = [
            "GEMINI_API_KEY",
        ]

        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"缺少必要的环境变量: {', '.join(missing_vars)}")

        return True

# 创建全局配置实例
config = Config()

# 创建必要目录
config.create_directories()

# 验证配置
try:
    config.validate_config()
except ValueError as e:
    print(f"配置验证失败: {e}")
    print("请检查 .env 文件中的环境变量配置")

# 为了方便访问，将常用属性提升到模块级别
PROJECT_NAME = config.PROJECT_NAME
VERSION = config.VERSION
DEBUG = config.DEBUG