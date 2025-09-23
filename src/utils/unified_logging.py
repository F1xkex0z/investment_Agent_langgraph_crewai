"""
统一日志配置系统
整合所有日志到少数几个文件中，提供更完整的调试信息
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

# 项目根目录和日志目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
CREWAI_LOGS_DIR = PROJECT_ROOT / "crewai_system" / "logs"

# 确保日志目录存在
LOGS_DIR.mkdir(exist_ok=True)
CREWAI_LOGS_DIR.mkdir(exist_ok=True)

# 对于CrewAI系统，使用专门的logs目录
LOGS_DIR = CREWAI_LOGS_DIR

# 日志图标常量
SUCCESS_ICON = "✅"
ERROR_ICON = "❌"
WAIT_ICON = "🔄"
DEBUG_ICON = "🔍"
INFO_ICON = "ℹ️"
WARN_ICON = "⚠️"


class UnifiedLoggingSystem:
    """统一日志管理系统"""

    def __init__(self):
        self.loggers = {}
        self.log_files = {
            'main': 'investment_system.log',
            'api': 'api_calls.log',
            'agents': 'agents.log',
            'data': 'data_processing.log',
            'performance': 'performance.log',
            'errors': 'errors.log',
            'debug': 'debug.log'
        }
        self._setup_loggers()

    def _setup_loggers(self):
        """设置所有日志记录器"""

        # 主系统日志 - 记录主要流程
        self.loggers['main'] = self._create_logger(
            'main',
            self.log_files['main'],
            logging.DEBUG
        )

        # API调用日志 - 记录所有API调用
        self.loggers['api'] = self._create_logger(
            'api',
            self.log_files['api'],
            logging.DEBUG
        )

        # 智能体日志 - 记录所有agent活动
        self.loggers['agents'] = self._create_logger(
            'agents',
            self.log_files['agents'],
            logging.DEBUG
        )

        # 数据处理日志 - 记录数据获取和处理
        self.loggers['data'] = self._create_logger(
            'data',
            self.log_files['data'],
            logging.DEBUG
        )

        # 性能日志 - 记录执行时间和性能指标
        self.loggers['performance'] = self._create_logger(
            'performance',
            self.log_files['performance'],
            logging.DEBUG
        )

        # 错误日志 - 只记录错误
        self.loggers['errors'] = self._create_logger(
            'errors',
            self.log_files['errors'],
            logging.ERROR
        )

        # 调试日志 - 详细的调试信息
        self.loggers['debug'] = self._create_logger(
            'debug',
            self.log_files['debug'],
            logging.DEBUG
        )

    def _create_logger(self, name: str, filename: str, level: int) -> logging.Logger:
        """创建配置好的日志记录器"""

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # 避免重复添加处理器
        if logger.handlers:
            return logger

        # 创建更详细的格式化器
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 简单格式化器用于控制台
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        log_path = LOGS_DIR / filename
        print(f"创建日志文件: {log_path}")  # 调试信息
        file_handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)  # 确保文件处理器记录所有DEBUG级别信息
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        return logger

    def get_logger(self, category: str = 'main') -> logging.Logger:
        """获取指定类别的日志记录器"""
        return self.loggers.get(category, self.loggers['main'])

    def log_system_event(self, message: str, level: str = 'INFO', **kwargs):
        """记录系统事件"""
        logger = self.get_logger('main')
        log_method = getattr(logger, level.lower(), logger.info)

        extra_info = ""
        if kwargs:
            extra_info = f" - {json.dumps(kwargs, ensure_ascii=False, default=str)}"

        log_method(f"{message}{extra_info}")

    def log_api_call(self, endpoint: str, method: str = 'GET', params: Dict = None,
                    response: Any = None, execution_time: float = None, error: str = None):
        """记录API调用"""
        logger = self.get_logger('api')

        status = "SUCCESS" if error is None else "ERROR"
        icon = SUCCESS_ICON if error is None else ERROR_ICON

        log_message = f"{icon} {method} {endpoint}"
        if params:
            log_message += f" - Params: {json.dumps(params, ensure_ascii=False, default=str)[:200]}"
        if execution_time:
            log_message += f" - Time: {execution_time:.3f}s"
        if response:
            response_size = len(str(response)) if response else 0
            log_message += f" - Size: {response_size} chars"
        if error:
            log_message += f" - Error: {error}"

        if error:
            logger.error(log_message)
        else:
            logger.info(log_message)

    def log_agent_activity(self, agent_name: str, action: str, details: Dict = None,
                          execution_time: float = None):
        """记录智能体活动"""
        logger = self.get_logger('agents')

        icon = INFO_ICON
        log_message = f"{icon} [{agent_name}] {action}"

        if details:
            details_str = json.dumps(details, ensure_ascii=False, default=str)
            log_message += f" - {details_str[:300]}"  # 限制长度
        if execution_time:
            log_message += f" - Time: {execution_time:.3f}s"

        logger.info(log_message)

        # 同时记录到调试日志
        debug_logger = self.get_logger('debug')
        debug_logger.debug(f"DETAILED: {log_message}")

    def log_data_operation(self, operation: str, data_type: str, identifier: str,
                          data_size: int = 0, execution_time: float = None,
                          success: bool = True, error: str = None):
        """记录数据操作"""
        logger = self.get_logger('data')

        status = "SUCCESS" if success else "FAILED"
        icon = SUCCESS_ICON if success else ERROR_ICON

        log_message = f"{icon} {operation} - {data_type}:{identifier}"
        log_message += f" - Status: {status}"
        log_message += f" - Size: {data_size}"

        if execution_time:
            log_message += f" - Time: {execution_time:.3f}s"
        if error:
            log_message += f" - Error: {error}"

        if success:
            logger.info(log_message)
        else:
            logger.error(log_message)

    def log_performance(self, operation: str, execution_time: float, **metrics):
        """记录性能指标"""
        logger = self.get_logger('performance')

        metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
        log_message = f"⏱️ {operation} - Time: {execution_time:.3f}s"
        if metrics_str:
            log_message += f" | {metrics_str}"

        logger.info(log_message)

    def log_error(self, error: Exception, context: Dict = None, component: str = None):
        """记录错误信息"""
        logger = self.get_logger('errors')

        error_type = type(error).__name__
        error_msg = str(error)

        log_message = f"{ERROR_ICON} ERROR in {component or 'Unknown'}"
        log_message += f" - Type: {error_type}"
        log_message += f" - Message: {error_msg}"

        if context:
            context_str = json.dumps(context, ensure_ascii=False, default=str)
            log_message += f" - Context: {context_str[:500]}"

        logger.error(log_message, exc_info=True)

    def log_debug(self, message: str, **kwargs):
        """记录调试信息"""
        logger = self.get_logger('debug')

        extra_info = ""
        if kwargs:
            extra_info = f" - {json.dumps(kwargs, ensure_ascii=False, default=str)}"

        logger.debug(f"{DEBUG_ICON} {message}{extra_info}")


# 全局统一日志系统实例
unified_logger = UnifiedLoggingSystem()


def get_unified_logger(category: str = 'main') -> logging.Logger:
    """获取统一日志系统的记录器"""
    return unified_logger.get_logger(category)


# 兼容性别名函数
def log_system_event(message: str, level: str = 'INFO', **kwargs):
    """记录系统事件"""
    unified_logger.log_system_event(message, level, **kwargs)


def log_api_call(endpoint: str, method: str = 'GET', params: Dict = None,
                response: Any = None, execution_time: float = None, error: str = None):
    """记录API调用"""
    unified_logger.log_api_call(endpoint, method, params, response, execution_time, error)


def log_agent_activity(agent_name: str, action: str, details: Dict = None,
                      execution_time: float = None):
    """记录智能体活动"""
    unified_logger.log_agent_activity(agent_name, action, details, execution_time)


def log_data_operation(operation: str, data_type: str, identifier: str,
                      data_size: int = 0, execution_time: float = None,
                      success: bool = True, error: str = None):
    """记录数据操作"""
    unified_logger.log_data_operation(operation, data_type, identifier, data_size,
                                    execution_time, success, error)


def log_performance(operation: str, execution_time: float, **metrics):
    """记录性能指标"""
    unified_logger.log_performance(operation, execution_time, **metrics)


def log_error(error: Exception, context: Dict = None, component: str = None):
    """记录错误信息"""
    unified_logger.log_error(error, context, component)


def log_debug(message: str, **kwargs):
    """记录调试信息"""
    unified_logger.log_debug(message, **kwargs)


# 向后兼容的简单函数
def log_info(message: str):
    """记录信息"""
    unified_logger.log_system_event(message)


def log_warning(message: str):
    """记录警告"""
    unified_logger.log_system_event(message, 'WARNING')


def log_success(message: str):
    """记录成功"""
    unified_logger.log_system_event(f"{SUCCESS_ICON} {message}")


def log_failure(message: str):
    """记录失败"""
    unified_logger.log_system_event(f"{ERROR_ICON} {message}", 'ERROR')


def setup_crewai_logging():
    """设置CrewAI相关的日志配置"""
    # 配置所有CrewAI相关的日志重定向到统一日志系统
    crewai_patterns = [
        'crewai',
        'crewai_agent',
        'crewai_task',
        'crewai_system',
        'httpx',  # CrewAI使用的HTTP客户端
        'openai',  # CrewAI可能使用的OpenAI客户端
    ]

    for pattern in crewai_patterns:
        crewai_logger = logging.getLogger(pattern)
        crewai_logger.handlers = []  # 清除现有处理器
        crewai_logger.propagate = False  # 防止传播到根logger

        # 根据pattern选择合适的目标日志文件
        if pattern in ['crewai_agent', 'crewai_task']:
            target_category = 'agents'
        elif pattern in ['httpx', 'openai']:
            target_category = 'api'
        else:
            target_category = 'debug'

        # 添加处理器到统一日志文件
        for handler in _get_file_handlers(target_category):
            crewai_logger.addHandler(handler)

        # 配置日志级别
        crewai_logger.setLevel(logging.INFO)

    print("✅ CrewAI日志重定向配置完成")


def _get_file_handlers(category: str):
    """获取指定类别的文件处理器"""
    handlers = []

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 获取对应的日志文件路径
    log_filename = unified_logger.log_files.get(category, f'{category}.log')
    log_path = LOGS_DIR / log_filename

    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    return handlers


def cleanup_old_logs(days_to_keep: int = 30):
    """清理旧的日志文件"""
    from datetime import timedelta

    cutoff_date = datetime.now() - timedelta(days=days_to_keep)

    for log_file in LOGS_DIR.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_date.timestamp():
            log_file.unlink()
            print(f"Deleted old log file: {log_file}")


if __name__ == "__main__":
    # 测试日志系统
    log_system_event("系统启动", level="INFO")
    log_api_call("stock_zh_a_spot_em", params={"symbol": "600519"}, execution_time=0.5)
    log_agent_activity("MarketDataAgent", "收集市场数据", {"ticker": "600519", "records": 244})
    log_data_operation("获取价格历史", "股票", "600519", data_size=244, execution_time=1.2)
    log_performance("完整分析流程", 126.35, agents_count=7, data_sources=3)
    log_debug("测试调试信息", test_value=123)

    print("✅ 统一日志系统测试完成")
    print(f"📁 日志文件位置: {LOGS_DIR}")
    print("📋 生成的日志文件:")
    for name, filename in unified_logger.log_files.items():
        log_path = LOGS_DIR / filename
        if log_path.exists():
            size = log_path.stat().st_size
            print(f"   - {filename}: {size} bytes")