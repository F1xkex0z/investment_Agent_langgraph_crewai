"""
简单统一的日志配置系统
提供清晰、简洁的日志记录功能
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# 尝试导入统一日志系统，如果失败则使用备用配置
try:
    import sys
    import os
    # 添加项目根目录到Python路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.utils.unified_logging import get_unified_logger, setup_crewai_logging
    USE_UNIFIED_LOGGING = True

    # 立即配置CrewAI日志重定向
    setup_crewai_logging()
except ImportError:
    USE_UNIFIED_LOGGING = False

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保日志目录存在
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# 日志图标常量
SUCCESS_ICON = "✅"
ERROR_ICON = "❌"
WAIT_ICON = "🔄"


def setup_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    设置简单的日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件名（可选）
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    # 优先使用统一日志系统
    if USE_UNIFIED_LOGGING:
        # 根据名称映射到合适的日志类别
        category_map = {
            'main': 'main',
            'agent': 'agents',
            'task': 'agents',
            'api': 'api',
            'error': 'errors',
            'performance': 'performance',
            'data_source_adapter': 'data',
            'data_tool': 'data'
        }

        category = category_map.get(name, 'main')
        return get_unified_logger(category)

    # 备用：使用原有的日志配置
    return _setup_legacy_logger(name, log_file, level)


def _setup_legacy_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """备用日志配置（当统一日志系统不可用时使用）"""
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建格式化器 - 包含更详细的调试信息
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # 改为DEBUG级别
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 确保文件记录DEBUG级别
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取通用的日志记录器
    """
    log_file = f"{name}.log"
    return setup_logger(name, log_file, "DEBUG")


# 预定义的常用日志记录器 - 全部设置为DEBUG级别
system_logger = setup_logger("system", "system.log", "DEBUG")
agent_logger = setup_logger("agent", "agent.log", "DEBUG")
task_logger = setup_logger("task", "task.log", "DEBUG")
api_logger = setup_logger("api", "api.log", "DEBUG")
error_logger = setup_logger("error", "error.log", "DEBUG")
performance_logger = setup_logger("performance", "performance.log", "DEBUG")
data_logger = setup_logger("data", "data.log", "DEBUG")
market_logger = setup_logger("market", "market.log", "DEBUG")
crew_logger = setup_logger("crew", "crew.log", "DEBUG")


def log_system_info(message: str, level: str = "INFO"):
    """记录系统信息"""
    log_method = getattr(system_logger, level.lower())
    log_method(message)


def log_agent_info(agent_name: str, message: str, level: str = "INFO"):
    """记录智能体信息"""
    log_method = getattr(agent_logger, level.lower())
    log_method(f"[{agent_name}] {message}")


def log_task_info(task_name: str, message: str, level: str = "INFO"):
    """记录任务信息"""
    log_method = getattr(task_logger, level.lower())
    log_method(f"[{task_name}] {message}")


def log_api_info(endpoint: str, message: str, level: str = "INFO"):
    """记录API信息"""
    log_method = getattr(api_logger, level.lower())
    log_method(f"[{endpoint}] {message}")


def log_error(error_msg: str, component: str = None):
    """记录错误信息"""
    if component:
        error_logger.error(f"[{component}] {error_msg}")
    else:
        error_logger.error(error_msg)


def log_performance(operation: str, execution_time: float, details: str = ""):
    """记录性能信息"""
    message = f"{operation} - 耗时: {execution_time:.2f}秒"
    if details:
        message += f" - {details}"
    performance_logger.info(message)


# 简单的日志记录函数
def log_info(message: str):
    """记录信息级别的日志"""
    system_logger.info(message)


def log_debug(message: str):
    """记录调试级别的日志"""
    system_logger.debug(message)


def log_warning(message: str):
    """记录警告级别的日志"""
    system_logger.warning(message)


def log_error_simple(message: str):
    """记录错误级别的日志"""
    system_logger.error(message)


def log_success(message: str):
    """记录成功信息"""
    system_logger.info(f"{SUCCESS_ICON} {message}")


def log_failure(message: str):
    """记录失败信息"""
    system_logger.error(f"{ERROR_ICON} {message}")


def log_waiting(message: str):
    """记录等待信息"""
    system_logger.info(f"{WAIT_ICON} {message}")


def log_data_collection(source: str, data_type: str, count: int, execution_time: float = None, details: str = ""):
    """记录数据收集信息"""
    message = f"数据收集 - 来源: {source}, 类型: {data_type}, 数量: {count}"
    if execution_time:
        message += f", 耗时: {execution_time:.2f}秒"
    if details:
        message += f", 详情: {details}"
    data_logger.info(message)


def log_api_call(endpoint: str, method: str = "GET", params: dict = None, response_time: float = None, status: str = "SUCCESS"):
    """记录API调用信息"""
    message = f"API调用 - {method} {endpoint}"
    if params:
        message += f" - 参数: {params}"
    if response_time:
        message += f" - 响应时间: {response_time:.2f}秒"
    message += f" - 状态: {status}"
    api_logger.info(message)


def log_market_data(ticker: str, data_source: str, data_points: int, price_range: tuple = None):
    """记录市场数据信息"""
    message = f"市场数据 - 股票: {ticker}, 来源: {data_source}, 数据点: {data_points}"
    if price_range:
        message += f", 价格区间: {price_range[0]} - {price_range[1]}"
    market_logger.info(message)


def log_crew_activity(component: str, action: str, details: str = ""):
    """记录CrewAI活动信息"""
    message = f"CrewAI活动 - 组件: {component}, 动作: {action}"
    if details:
        message += f", 详情: {details}"
    crew_logger.info(message)