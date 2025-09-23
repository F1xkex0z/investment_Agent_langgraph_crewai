"""
CrewAI投资分析系统主入口
基于CrewAI框架的A股智能投资分析系统
"""

import argparse
import sys
import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from crewai import Crew, Process
from crewai_system.src.utils.logging_config import get_logger, log_info, log_error, log_success, log_failure, log_performance
from crewai_system.src.utils.shared_context import get_global_context, ContextManager
from crewai_system.src.utils.data_processing import get_data_processor
from crewai_system.src.tools.data_sources import get_data_adapter
from crewai_system.src.crews.analysis_crew import AnalysisCrew
from crewai_system.src.config import config

# 初始化日志系统
logger = get_logger('main')


class CrewAIInvestmentSystem:
    """CrewAI投资分析系统主类"""

    def __init__(self):
        self.logger = logger
        self.shared_context = get_global_context()
        self.data_processor = get_data_processor()
        self.data_adapter = get_data_adapter()
        self.analysis_crew = AnalysisCrew()
        self.crew = None

        # 系统状态
        self.is_initialized = False
        self.current_run_id = None
        self.run_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }

    def initialize_system(self):
        """初始化系统"""
        start_time = datetime.now()
        try:
            self.logger.info("正在初始化CrewAI投资分析系统...")
            log_info(f"系统初始化开始 - 配置: {config.PROJECT_NAME} v{config.VERSION}")

            # 验证配置
            config.validate_config()
            log_info("配置验证成功")

            # 初始化数据源适配器
            cache_stats = self.data_adapter.get_cache_stats()
            log_info(f"数据源适配器初始化成功，缓存状态: {cache_stats}")

            # 设置系统级别的共享数据
            self.shared_context.set(
                key="system_info",
                value={
                    "version": config.VERSION,
                    "project_name": config.PROJECT_NAME,
                    "initialized_at": datetime.now().isoformat(),
                    "max_workers": config.MAX_WORKERS
                },
                source_agent="system"
            )

            self.is_initialized = True
            end_time = datetime.now()

            self.logger.info("系统初始化完成")
            log_performance("系统初始化", (end_time - start_time).total_seconds(), "成功完成")

        except Exception as e:
            end_time = datetime.now()
            self.logger.error(f"系统初始化失败: {e}")
            log_error(f"系统初始化错误: {str(e)}，耗时: {(end_time - start_time).total_seconds():.2f}秒", "system")
            log_performance("系统初始化", (end_time - start_time).total_seconds(), f"失败: {e}")
            raise

    def create_run_context(self, ticker: str, start_date: str, end_date: str, **kwargs) -> Dict[str, Any]:
        """
        创建运行上下文

        Args:
            ticker: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数

        Returns:
            运行上下文字典
        """
        run_id = str(uuid.uuid4())
        self.current_run_id = run_id

        # 设置基础运行参数
        run_context = {
            "run_id": run_id,
            "ticker": self.data_processor.normalize_ticker(ticker),
            "start_date": start_date,
            "end_date": end_date,
            "show_reasoning": kwargs.get("show_reasoning", False),
            "num_of_news": kwargs.get("num_of_news", 10),
            "portfolio": kwargs.get("portfolio", {"cash": 100000.0, "stock": 0}),
            "created_at": datetime.now().isoformat()
        }

        # 保存到共享上下文
        self.shared_context.set(
            key=f"run_{run_id}",
            value=run_context,
            source_agent="system"
        )

        # 设置当前运行ID
        self.shared_context.set(
            key="current_run_id",
            value=run_id,
            source_agent="system"
        )

        self.logger.info(f"创建运行上下文: {run_id}")
        return run_context

    def setup_crew(self, run_context: Dict[str, Any]) -> None:
        """
        设置Crew团队

        Args:
            run_context: 运行上下文

        Returns:
            None
        """
        start_time = datetime.now()
        self.logger.info("正在设置Crew团队...")
        log_info("开始创建Crew团队...")

        try:
            # 导入必要的CrewAI组件
            from crewai import Task
            from crewai_system.src.agents.market_data_agent import MarketDataAgent
            from crewai_system.src.agents.technical_analyst import TechnicalAnalyst
            from crewai_system.src.agents.fundamentals_analyst import FundamentalsAnalyst
            from crewai_system.src.agents.sentiment_analyst import SentimentAnalyst
            from crewai_system.src.agents.valuation_analyst import ValuationAnalyst

            log_info("CrewAI组件导入成功")

            # 创建智能体实例
            self.logger.info("创建智能体实例...")
            agent_logger = get_logger("crew_creation")

            market_data_agent = MarketDataAgent()
            technical_analyst = TechnicalAnalyst()
            fundamentals_agent = FundamentalsAnalyst()
            sentiment_agent = SentimentAnalyst()
            valuation_agent = ValuationAnalyst()

            log_info("成功创建所有智能体实例")

            # 存储智能体引用
            self.agents = {
                'market_data': market_data_agent,
                'technical': technical_analyst,
                'fundamentals': fundamentals_agent,
                'sentiment': sentiment_agent,
                'valuation': valuation_agent
            }

            # 获取任务上下文
            ticker = run_context.get("ticker", "000000")
            start_date = run_context.get("start_date", (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
            end_date = run_context.get("end_date", datetime.now().strftime('%Y-%m-%d'))

            log_info("CrewAI上下文提取完成")

            # 创建CrewAI任务
            self.logger.info("创建CrewAI任务...")
            task_logger = get_logger("crew_tasks")

            # 创建市场数据收集任务
            market_data_task = Task(
                description=f"收集股票{ticker}从{start_date}到{end_date}的市场数据",
                expected_output="包含价格历史、财务指标和市场信息的完整数据集",
                agent=market_data_agent
            )

            # 创建技术分析任务
            technical_task = Task(
                description=f"分析股票{ticker}的技术指标和价格趋势",
                expected_output="技术分析报告，包含趋势判断和交易信号",
                agent=technical_analyst,
                context=[market_data_task]
            )

            # 创建基本面分析任务
            fundamentals_task = Task(
                description=f"分析股票{ticker}的基本面和财务状况",
                expected_output="基本面分析报告，包含财务健康状况评估",
                agent=fundamentals_agent,
                context=[market_data_task]
            )

            # 创建新闻收集任务
            news_collection_task = Task(
                description=f"收集股票{ticker}的相关新闻数据",
                expected_output="新闻数据列表，包含标题、内容和来源",
                agent=market_data_agent,  # 使用市场数据收集专家来收集新闻
                context=[market_data_task]
            )

            # 创建情绪分析任务
            sentiment_task = Task(
                description=f"分析股票{ticker}的市场情绪和新闻舆情",
                expected_output="情绪分析报告，包含市场情绪判断",
                agent=sentiment_agent,
                context=[market_data_task, news_collection_task]
            )

            # 创建估值分析任务
            valuation_task = Task(
                description=f"对股票{ticker}进行估值分析和内在价值计算",
                expected_output="估值分析报告，包含投资建议",
                agent=valuation_agent,
                context=[market_data_task, technical_task, fundamentals_task, sentiment_task]
            )

            log_info("CrewAI任务创建完成")

            # 存储任务引用
            self.tasks = {
                'market_data': market_data_task,
                'news_collection': news_collection_task,
                'technical': technical_task,
                'fundamentals': fundamentals_task,
                'sentiment': sentiment_task,
                'valuation': valuation_task
            }

            # 创建CrewAI实例
            self.logger.info("创建CrewAI团队...")
            self.crew = Crew(
                agents=[market_data_agent, technical_analyst, fundamentals_agent, sentiment_agent, valuation_agent],
                tasks=[market_data_task, news_collection_task, technical_task, fundamentals_task, sentiment_task, valuation_task],
                verbose=True,
                process_order='sequential'
            )

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self.logger.info("CrewAI团队设置完成")
            self.logger.info(f"CrewAI团队创建成功，耗时{execution_time:.2f}秒")

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self.logger.error(f"CrewAI团队设置失败: {e}")
            self.logger.debug(f"CrewAI设置错误详情: {{'error': '{str(e)}', 'execution_time': {execution_time}, 'traceback': '{str(e.__class__.__name__)}'}}")
            self.logger.info(f"CrewAI团队创建失败: {e}，耗时{execution_time:.2f}秒")

            # 创建一个最小化的团队作为备用
            self.crew = None
            self.agents = {}
            self.tasks = {}

    def run_analysis(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行投资分析

        Args:
            run_context: 运行上下文

        Returns:
            分析结果
        """
        if not self.is_initialized:
            raise RuntimeError("系统未初始化，请先调用initialize_system()")

        run_id = run_context["run_id"]
        ticker = run_context["ticker"]

        self.logger.info(f"开始投资分析 - 股票: {ticker}, 运行ID: {run_id}")

        # 记录运行开始
        start_time = datetime.now()
        self.shared_context.set(
            key=f"run_{run_id}_start_time",
            value=start_time.isoformat(),
            source_agent="system"
        )

        try:
            # 设置Crew
            self.setup_crew(run_context)

            # 执行分析
            with ContextManager(self.shared_context, "system") as ctx:
                ctx.set("analysis_status", "running")
                ctx.set("analysis_start_time", start_time.isoformat())

                # 使用CrewAI执行分析
                if self.crew is not None:
                    self.logger.info("使用CrewAI框架执行分析...")
                    crew_result = self.crew.kickoff()

                    # 格式化CrewAI结果 - 确保JSON序列化
                    crew_result_dict = self._serialize_crew_result(crew_result)

                    result = {
                        "run_id": run_id,
                        "ticker": ticker,
                        "analysis_date": datetime.now().isoformat(),
                        "status": "completed",
                        "framework": "CrewAI",
                        "crew_result": crew_result_dict,
                        "agents": list(self.agents.keys()) if hasattr(self, 'agents') else [],
                        "tasks": list(self.tasks.keys()) if hasattr(self, 'tasks') else []
                    }
                    # log result all
                    self.logger.info(f"CrewAI分析结果: {crew_result_dict}")
                    # log result

                else:
                    # 回退到原始分析系统
                    self.logger.warning("CrewAI不可用，使用原始分析系统...")
                    result = self.analysis_crew.execute_analysis(run_context)

            # 记录运行结束
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self.shared_context.set(
                key=f"run_{run_id}_end_time",
                value=end_time.isoformat(),
                source_agent="system"
            )

            self.shared_context.set(
                key=f"run_{run_id}_execution_time",
                value=execution_time,
                source_agent="system"
            )

            # 保存结果
            self.shared_context.set(
                key=f"run_{run_id}_result",
                value=result,
                source_agent="system"
            )

            self.logger.info(f"投资分析完成，耗时: {execution_time:.2f}秒")

            # 更新统计信息
            self._update_run_stats(execution_time, success=True)

            return result

        except Exception as e:
            self.logger.error(f"投资分析失败: {e}")

            # 记录失败
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self.shared_context.set(
                key=f"run_{run_id}_error",
                value=str(e),
                source_agent="system"
            )

            self.shared_context.set(
                key=f"run_{run_id}_status",
                value="failed",
                source_agent="system"
            )

            # 更新统计信息
            self._update_run_stats(execution_time, success=False)

            raise

    def _create_mock_result(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        """创建模拟结果（已升级为完整分析）"""
        return {
            "run_id": run_context["run_id"],
            "ticker": run_context["ticker"],
            "analysis_date": datetime.now().isoformat(),
            "status": "completed",
            "message": "CrewAI投资分析系统已完成所有核心智能体和分析任务",
            "system_info": {
                "version": config.VERSION,
                "framework": "CrewAI",
                "architecture": "Multi-Agent System",
                "analysis_type": "comprehensive"
            },
            "context_summary": {
                "start_date": run_context["start_date"],
                "end_date": run_context["end_date"],
                "show_reasoning": run_context["show_reasoning"],
                "num_of_news": run_context["num_of_news"]
            },
            "capabilities": [
                "市场数据收集",
                "技术分析",
                "基本面分析",
                "情绪分析",
                "估值分析",
                "综合投资建议"
            ]
        }

    def _serialize_crew_result(self, crew_result) -> Dict[str, Any]:
        """序列化CrewAI结果对象为JSON兼容的字典"""
        try:
            # 如果结果是字典类型，直接返回
            if isinstance(crew_result, dict):
                return crew_result

            # 如果结果是字符串类型，包装成字典
            if isinstance(crew_result, str):
                return {"output": crew_result, "type": "string"}

            # 尝试获取结果的常用属性
            result_dict = {}

            # 尝试获取输出内容
            if hasattr(crew_result, 'output'):
                # 特殊处理TaskOutput对象的output属性
                if hasattr(crew_result.output, '__class__') and crew_result.output.__class__.__name__ == 'TaskOutput':
                    task_output = crew_result.output
                    result_dict['output'] = self._serialize_task_output(task_output)
                elif isinstance(crew_result.output, (dict, list, str, int, float, bool)):
                    result_dict['output'] = crew_result.output
                else:
                    result_dict['output'] = str(crew_result.output)
            elif hasattr(crew_result, 'result'):
                result_dict['result'] = str(crew_result.result)
            elif hasattr(crew_result, 'content'):
                result_dict['content'] = str(crew_result.content)
            else:
                # 如果没有标准属性，尝试转换为字符串
                result_dict['output'] = str(crew_result)

            # 添加其他可能的属性
            for attr in ['tasks_output', 'agents_output', 'token_usage', 'raw']:
                if hasattr(crew_result, attr):
                    value = getattr(crew_result, attr)
                    if isinstance(value, (dict, list, str, int, float, bool)):
                        result_dict[attr] = value
                    else:
                        result_dict[attr] = str(value)

            # 添加类型信息
            result_dict['type'] = crew_result.__class__.__name__
            result_dict['serialized'] = True

            return result_dict

        except Exception as e:
            self.logger.warning(f"序列化CrewAI结果失败: {e}")
            return {
                "output": str(crew_result),
                "type": "fallback",
                "error": f"序列化失败: {str(e)}",
                "serialized": True
            }
            
    def _serialize_task_output(self, task_output) -> Dict[str, Any]:
        """专门序列化TaskOutput对象"""
        task_dict = {}
        
        # 提取TaskOutput的主要属性
        for attr in ['task_id', 'agent', 'output', 'input', 'context', 'tool_calls']:
            if hasattr(task_output, attr):
                value = getattr(task_output, attr)
                if isinstance(value, (dict, list, str, int, float, bool)):
                    task_dict[attr] = value
                else:
                    task_dict[attr] = str(value)
        
        # 添加类型信息
        task_dict['task_type'] = task_output.__class__.__name__
        
        return task_dict

    def _update_run_stats(self, execution_time: float, success: bool):
        """更新运行统计信息"""
        if not hasattr(self, 'run_stats'):
            self.run_stats = {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0
            }

        self.run_stats["total_runs"] += 1
        self.run_stats["total_execution_time"] += execution_time

        if success:
            self.run_stats["successful_runs"] += 1
        else:
            self.run_stats["failed_runs"] += 1

        self.run_stats["average_execution_time"] = (
            self.run_stats["total_execution_time"] / self.run_stats["total_runs"]
        )

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "initialized": self.is_initialized,
            "current_run_id": self.current_run_id,
            "run_stats": self.run_stats,
            "context_stats": self.shared_context.get_stats(),
            "data_adapter_stats": self.data_adapter.get_cache_stats(),
            "config": {
                "version": config.VERSION,
                "project_name": config.PROJECT_NAME,
                "max_workers": config.MAX_WORKERS,
                "cache_enabled": config.CACHE_ENABLED
            }
        }

    def cleanup(self):
        """清理系统资源"""
        try:
            # 清理缓存
            self.data_adapter.clear_cache()

            # 清理过期上下文数据
            self.shared_context.cleanup_expired()

            self.logger.info("系统资源清理完成")
        except Exception as e:
            self.logger.error(f"系统资源清理失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='CrewAI A-Share Investment Analysis System'
    )

    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Stock ticker symbol (e.g., 000001)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). Defaults to 1 year before end date'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD). Defaults to yesterday'
    )

    parser.add_argument(
        '--show-reasoning',
        action='store_true',
        help='Show reasoning from each agent'
    )

    parser.add_argument(
        '--num-of-news',
        type=int,
        default=10,
        help='Number of news articles to analyze (default: 10)'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial cash amount (default: 100,000)'
    )

    parser.add_argument(
        '--initial-position',
        type=int,
        default=0,
        help='Initial stock position (default: 0)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status'
    )

    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Cleanup system resources'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    try:
        # 创建系统实例
        system = CrewAIInvestmentSystem()

        # 处理特殊命令
        if args.status:
            system.initialize_system()
            status = system.get_system_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return

        if args.cleanup:
            system.initialize_system()
            system.cleanup()
            print("系统资源清理完成")
            return

        # 初始化系统
        system.initialize_system()

        # 设置日期范围
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)

        end_date = args.end_date or yesterday.strftime('%Y-%m-%d')
        if not args.start_date:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            start_date = args.start_date

        # 验证日期
        if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'):
            raise ValueError("开始日期不能晚于结束日期")

        # 创建运行上下文
        run_context = system.create_run_context(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            show_reasoning=args.show_reasoning,
            num_of_news=args.num_of_news,
            portfolio={"cash": args.initial_capital, "stock": args.initial_position}
        )

        # 运行分析
        result = system.run_analysis(run_context)

        # 输出结果
        print("\n" + "="*60)
        print("CrewAI投资分析结果")
        print("="*60)

        # 安全地输出结果，处理不可序列化的对象
        try:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except TypeError as e:
            print(f"结果包含不可序列化的对象，改为字符串输出:")
            print(str(result))
            # 尝试提取关键信息
            if hasattr(result, 'raw'):
                print(f"\n原始分析结果:\n{result.raw}")
            elif hasattr(result, 'tasks_output'):
                for i, task_output in enumerate(result.tasks_output):
                    print(f"\n任务 {i+1} - {task_output.description}:")
                    print(f"代理: {task_output.agent}")
                    print(f"结果: {task_output.raw[:500]}..." if len(task_output.raw) > 500 else f"结果: {task_output.raw}")

        # 改进的日志记录方式
        system.logger.info(f"CrewAI分析完成 - 股票: {args.ticker}, 运行ID: {run_context['run_id']}, 状态: {result.get('status', 'unknown')}")
        
        # 如果需要详细日志，可以使用debug级别记录完整结果
        if hasattr(system.logger, 'debug'):
            try:
                system.logger.debug(f"CrewAI分析详细结果: {json.dumps(result, ensure_ascii=False)}")
            except Exception as e:
                system.logger.warning(f"记录详细分析结果失败: {e}")

        # 显示系统状态
        status = system.get_system_status()
        print(f"\n系统状态: 总运行次数={status['run_stats']['total_runs']}, "
              f"成功={status['run_stats']['successful_runs']}, "
              f"失败={status['run_stats']['failed_runs']}")

    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()