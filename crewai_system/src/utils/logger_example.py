"""
日志系统使用示例
展示如何在crewai系统中使用简化的日志功能
"""

import time
from crewai_system.src.utils.logging_config import (
    get_logger, log_info, log_debug, log_warning, log_error_simple,
    log_success, log_failure, log_waiting,
    log_agent_info, log_task_info, log_api_info, log_error, log_performance
)


def example_usage():
    """演示日志系统的使用方法"""

    # 1. 基础日志记录
    log_info("系统启动")
    log_debug("调试信息：正在加载配置")
    log_warning("警告：检测到配置文件可能过期")
    log_error_simple("发生了一个错误")
    log_success("操作成功完成")
    log_failure("操作失败")
    log_waiting("正在等待响应...")

    # 2. 分类日志记录
    log_agent_info("MarketDataAgent", "开始获取市场数据")
    log_task_info("数据分析", "正在处理技术指标")
    log_api_info("/api/analyze", "收到分析请求")
    log_error("API调用失败", "ExternalAPI")

    # 3. 性能日志
    start_time = time.time()
    time.sleep(0.1)  # 模拟操作
    end_time = time.time()
    log_performance("数据处理", end_time - start_time, "处理了1000条记录")

    # 4. 使用自定义日志记录器
    custom_logger = get_logger("my_component")
    custom_logger.info("这是自定义组件的日志")

    # 5. 智能体日志示例
    class ExampleAgent:
        def __init__(self):
            self.agent_name = "ExampleAgent"
            self.logger = get_logger(f"agent.{self.agent_name}")

        def do_work(self):
            self.logger.info(f"{self.agent_name} 开始工作")
            time.sleep(0.05)
            self.logger.info(f"{self.agent_name} 工作完成")

    agent = ExampleAgent()
    agent.do_work()


if __name__ == "__main__":
    example_usage()
    print("日志示例运行完成，请查看logs目录中的日志文件")