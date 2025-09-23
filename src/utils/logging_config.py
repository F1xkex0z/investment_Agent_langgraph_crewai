import time
import os
import logging
from typing import Optional

# 尝试导入统一日志系统，如果失败则使用备用配置
try:
    from .unified_logging import get_unified_logger, unified_logger
    USE_UNIFIED_LOGGING = True
except ImportError:
    USE_UNIFIED_LOGGING = False


SUCCESS_ICON = "✅"
ERROR_ICON = "❌"
WAIT_ICON = "🔄"


def setup_logger(name: str, log_dir: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """设置统一的日志配置

    Args:
        name: logger的名称
        log_dir: 日志文件目录，如果为None则使用默认的logs目录
        log_file: 日志文件名，如果为None则使用name作为文件名

    Returns:
        配置好的logger实例
    """
    # 优先使用统一日志系统
    if USE_UNIFIED_LOGGING:
        # 根据名称映射到合适的日志类别
        category_map = {
            'api': 'api',
            'market_data_agent': 'agents',
            'technical_analyst_agent': 'agents',
            'fundamentals_agent': 'agents',
            'valuation_agent': 'agents',
            'portfolio_management_agent': 'agents',
            'sentiment_agent': 'agents',
            'macro_analyst_agent': 'agents',
            'macro_news_agent': 'agents',
            'news_crawler': 'data',
            'llm_clients': 'api',
            'main_workflow': 'main',
            'agent_state': 'debug',
            'structured_terminal': 'main',
            'data_source_adapter': 'data'
        }

        category = category_map.get(name, 'main')
        return get_unified_logger(category)

    # 备用：使用原有的日志配置
    return _setup_legacy_logger(name, log_dir, log_file)


def _setup_legacy_logger(name: str, log_dir: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """备用日志配置（当统一日志系统不可用时使用）"""

    # 设置 root logger 的级别为 DEBUG
    logging.getLogger().setLevel(logging.DEBUG)

    # 获取或创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # logger本身记录DEBUG级别及以上
    logger.propagate = False  # 防止日志消息传播到父级logger

    # 如果已经有处理器，不再添加
    if logger.handlers:
        return logger

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 控制台显示DEBUG级别及以上

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # 创建文件处理器
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        log_file = f"{name}.log"
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG级别及以上的日志
    file_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 预定义的图标
