"""
数据收集任务
负责收集市场数据、财务数据和新闻数据
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from crewai_system.src.tasks.base_task import BaseTask
from crewai_system.src.agents.market_data_agent import MarketDataAgent
from crewai_system.src.agents.base_agent import BaseAgent
from crewai_system.src.utils.shared_context import get_global_context
from crewai_system.src.utils.data_flow_manager import data_flow_manager


class DataCollectionTask(BaseTask):
    """数据收集任务"""

    def __init__(self):
        # Initialize parent first to avoid attribute filtering
        super().__init__(
            description="收集股票市场数据、财务指标和相关信息",
            expected_output="完整的市场数据包，包含价格历史、财务指标和市场信息",
            agent=None  # Will set after parent initialization
        )
        # Set attributes after parent initialization
        self._market_data_agent = MarketDataAgent()
        self._shared_context = get_global_context()
        # Update agent reference
        self._agent = self._market_data_agent

    @property
    def market_data_agent(self):
        """获取市场数据智能体"""
        return self._market_data_agent

    @property
    def shared_context(self):
        """获取共享上下文"""
        return self._shared_context

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务（为了兼容analysis_crew.py中的调用）"""
        return self.execute_with_retry(context)

    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据收集任务

        Args:
            context: 任务上下文

        Returns:
            收集的数据
        """
        try:
            ticker = context["ticker"]
            start_date = context["start_date"]
            end_date = context["end_date"]
            show_reasoning = context.get("show_reasoning", False)

            self.log_task_start(f"开始收集{ticker}的数据")

            # 检查数据依赖（数据收集任务无依赖）
            if not data_flow_manager.ensure_data_dependencies("market_data", ticker):
                self.debug_logger.info("数据收集任务无依赖，直接执行")

            # 执行数据收集
            result = self.market_data_agent.process_task({
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "show_reasoning": show_reasoning
            })

            # 将数据保存到数据流管理器
            success = data_flow_manager.store_data(
                data_type="market_data",
                ticker=ticker,
                data=result.get("content", {}),
                source_agent="DataCollectionTask",
                metadata={
                    "collection_time": datetime.now().isoformat(),
                    "data_quality": self._assess_data_quality(result.get("content", {}))
                }
            )

            if not success:
                self.debug_logger.warning("市场数据存储失败")

            self.log_task_complete(f"完成{ticker}的数据收集")
            return result

        except Exception as e:
            self.log_task_error(e, "数据收集任务失败")
            raise

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker", "start_date", "end_date"]

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """验证输入数据"""
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in context or context[field] is None:
                self.debug_logger.error(f"缺少必需字段: {field}")
                return False
        return True

    def estimate_execution_time(self, context: Dict[str, Any]) -> float:
        """估算执行时间（秒）"""
        # 数据收集通常需要10-30秒
        return 15.0

    def get_task_priority(self) -> int:
        """获取任务优先级（越高优先级越高）"""
        return 100  # 数据收集是最高优先级

    def can_retry(self, error: Exception) -> bool:
        """判断是否可以重试"""
        # 网络错误和数据源错误可以重试
        error_type = type(error).__name__
        retryable_errors = [
            "ConnectionError", "TimeoutError", "HTTPError",
            "RequestException", "DataNotAvailableError"
        ]
        return error_type in retryable_errors

    def get_retry_delay(self, attempt: int) -> float:
        """获取重试延迟时间"""
        # 指数退避策略
        return min(2 ** attempt, 60)  # 最多60秒

    def _assess_data_quality(self, data: Dict[str, Any]) -> str:
        """
        评估数据质量

        Args:
            data: 数据字典

        Returns:
            质量评估结果
        """
        quality_score = 0

        # 评估价格数据质量
        if "prices" in data and data["prices"]:
            price_count = len(data["prices"])
            if price_count > 250:
                quality_score += 3
            elif price_count > 100:
                quality_score += 2
            elif price_count > 20:
                quality_score += 1

        # 评估财务数据质量
        if "financial_metrics" in data and data["financial_metrics"]:
            metrics_count = len(data["financial_metrics"])
            if metrics_count > 15:
                quality_score += 3
            elif metrics_count > 8:
                quality_score += 2
            elif metrics_count > 3:
                quality_score += 1

        # 评估市场信息质量
        if "market_info" in data and data["market_info"].get("market_cap", 0) > 0:
            quality_score += 2

        # 根据总分评估质量
        if quality_score >= 8:
            return "excellent"
        elif quality_score >= 6:
            return "good"
        elif quality_score >= 4:
            return "fair"
        else:
            return "poor"


class NewsCollectionTask(BaseTask):
    """新闻收集任务"""

    def __init__(self):
        from agents.market_data_agent import MarketDataAgent
        self._news_agent = MarketDataAgent()
        super().__init__(
            description="收集股票相关新闻和市场情绪数据",
            expected_output="新闻数据包，包含标题、内容和情感分析",
            agent=self._news_agent
        )
        self._shared_context = get_global_context()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务（为了兼容analysis_crew.py中的调用）"""
        return self.execute_with_retry(context)

    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行新闻收集任务

        Args:
            context: 任务上下文

        Returns:
            收集的新闻数据
        """
        self.debug_logger.info(f"🔍 [DEBUG] ===== NewsCollectionTask 开始执行 =====")
        self.debug_logger.info(f"🔍 [DEBUG] context keys: {list(context.keys())}")

        try:
            ticker = context["ticker"]
            num_of_news = context.get("num_of_news", 10)
            show_reasoning = context.get("show_reasoning", False)

            self.debug_logger.info(f"🔍 [DEBUG] 参数 - ticker: {ticker}, num_of_news: {num_of_news}, show_reasoning: {show_reasoning}")
            self.log_task_start(f"开始收集{ticker}的新闻数据")

            # 使用真实API收集新闻数据
            self.debug_logger.info(f"🔍 [DEBUG] ===== 开始调用新闻搜索API =====")
            from crewai_system.src.tools.data_sources import get_data_adapter
            data_adapter = get_data_adapter()
            news_data = data_adapter.search_financial_news_safe(ticker, num_of_news)
            self.debug_logger.info(f"🔍 [DEBUG] API调用完成，获取到新闻数据大小: {len(news_data) if news_data else 0}")

            # 将数据保存到数据流管理器
            self.debug_logger.info(f"🔍 [DEBUG] ===== 开始存储新闻数据到数据流管理器 =====")
            success = data_flow_manager.store_data(
                data_type="news_data",
                ticker=ticker,
                data=news_data,
                source_agent="NewsCollectionTask",
                metadata={
                    "news_count": len(news_data),
                    "collection_time": datetime.now().isoformat()
                }
            )
            self.debug_logger.info(f"🔍 [DEBUG] 数据流管理器存储结果: {success}")

            if not success:
                self.debug_logger.warning("🔍 [DEBUG] 新闻数据存储失败")
            else:
                self.debug_logger.info(f"🔍 [DEBUG] 新闻数据存储成功")

            result = {
                "content": news_data,
                "signal": "data_collected",
                "confidence": 0.8,
                "reasoning": f"成功收集{len(news_data)}条新闻",
                "metadata": {
                    "ticker": ticker,
                    "news_count": len(news_data),
                    "collection_time": datetime.now().isoformat()
                }
            }

            self.debug_logger.info(f"🔍 [DEBUG] NewsCollectionTask 执行完成，返回新闻数量: {len(news_data) if news_data else 0}")
            self.log_task_complete(f"完成{ticker}的新闻收集")
            return result

        except Exception as e:
            self.debug_logger.error(f"🔍 [DEBUG] NewsCollectionTask 执行失败: {e}")
            import traceback
            self.debug_logger.error(f"🔍 [DEBUG] 详细错误堆栈: {traceback.format_exc()}")
            self.log_task_error(e, "新闻收集任务失败")
            raise

  
    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker"]

    def estimate_execution_time(self, context: Dict[str, Any]) -> float:
        """估算执行时间（秒）"""
        num_of_news = context.get("num_of_news", 10)
        return 5.0 + num_of_news * 0.5  # 每条新闻约0.5秒

    def get_task_priority(self) -> int:
        """获取任务优先级"""
        return 90  # 新闻收集优先级较高

    def can_retry(self, error: Exception) -> bool:
        """判断是否可以重试"""
        return True  # 新闻收集通常可以重试


class TechnicalAnalysisTask(BaseTask):
    """技术分析任务"""

    def __init__(self):
        # Initialize parent first to avoid attribute filtering
        super().__init__(
            description="执行技术分析，计算各种技术指标",
            expected_output="技术分析结果，包含趋势、指标和交易信号",
            agent=None  # Will set after parent initialization
        )
        # Set attributes after parent initialization
        # 延迟导入避免循环依赖
        from agents.technical_analyst import TechnicalAnalyst
        self._technical_analyst = TechnicalAnalyst()
        self._shared_context = get_global_context()
        # Update agent reference
        self._agent = self._technical_analyst
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务（为了兼容analysis_crew.py中的调用）"""
        return self.execute_with_retry(context)

    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行技术分析任务

        Args:
            context: 任务上下文

        Returns:
            技术分析结果
        """
        try:
            ticker = context["ticker"]
            show_reasoning = context.get("show_reasoning", False)

            self.log_task_start(f"开始{ticker}的技术分析")

            # 检查数据依赖
            if not data_flow_manager.ensure_data_dependencies("technical_analysis", ticker):
                missing_deps = data_flow_manager.get_missing_dependencies("technical_analysis", ticker)
                raise ValueError(f"技术分析缺少依赖数据: {missing_deps}")

            # 从数据流管理器获取价格数据
            market_data_entry = data_flow_manager.retrieve_data("market_data", ticker)
            if market_data_entry is None:
                raise ValueError(f"缺少{ticker}的市场数据")

            market_data = market_data_entry.get("content", {})
            prices_data = market_data.get("prices", [])

            if not prices_data:
                raise ValueError(f"缺少{ticker}的价格数据")

            # 执行技术分析
            result = self._technical_analyst.process_task({
                "ticker": ticker,
                "prices": prices_data,
                "show_reasoning": show_reasoning
            })

            # 将结果保存到数据流管理器
            success = data_flow_manager.store_data(
                data_type="technical_analysis",
                ticker=ticker,
                data=result.get("content", {}),
                source_agent="TechnicalAnalysisTask",
                metadata={
                    "analysis_time": datetime.now().isoformat(),
                    "signal_strength": result.get("confidence", 0)
                }
            )

            if not success:
                self.debug_logger.warning("技术分析结果存储失败")

            self.log_task_complete(f"完成{ticker}的技术分析")
            return result

        except Exception as e:
            self.log_task_error(e, "技术分析任务失败")
            raise

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker"]

    def estimate_execution_time(self, context: Dict[str, Any]) -> float:
        """估算执行时间（秒）"""
        return 3.0  # 技术分析相对较快

    def get_task_priority(self) -> int:
        """获取任务优先级"""
        return 80  # 技术分析优先级中等

    def can_retry(self, error: Exception) -> bool:
        """判断是否可以重试"""
        return True


class FundamentalsAnalysisTask(BaseTask):
    """基本面分析任务"""

    def __init__(self):
        # Initialize parent first to avoid attribute filtering
        super().__init__(
            description="执行基本面分析，评估公司财务状况",
            expected_output="基本面分析结果，包含盈利能力、财务健康等指标",
            agent=None  # Will set after parent initialization
        )
        # Set attributes after parent initialization
        from agents.fundamentals_analyst import FundamentalsAnalyst
        self._fundamentals_analyst = FundamentalsAnalyst()
        self._shared_context = get_global_context()
        # Update agent reference
        self._agent = self._fundamentals_analyst
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务（为了兼容analysis_crew.py中的调用）"""
        return self.execute_with_retry(context)

    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行基本面分析任务

        Args:
            context: 任务上下文

        Returns:
            基本面分析结果
        """
        try:
            ticker = context["ticker"]
            show_reasoning = context.get("show_reasoning", False)

            self.log_task_start(f"开始{ticker}的基本面分析")

            # 检查数据依赖
            if not data_flow_manager.ensure_data_dependencies("fundamentals_analysis", ticker):
                missing_deps = data_flow_manager.get_missing_dependencies("fundamentals_analysis", ticker)
                raise ValueError(f"基本面分析缺少依赖数据: {missing_deps}")

            # 从数据流管理器获取市场数据
            market_data_entry = data_flow_manager.retrieve_data("market_data", ticker)
            if market_data_entry is None:
                raise ValueError(f"缺少{ticker}的市场数据")

            market_data_content = market_data_entry.get("content", {})
            content = market_data_content
            financial_metrics = content.get("financial_metrics", {})
            market_info = content.get("market_info", {})

            if not financial_metrics:
                raise ValueError(f"缺少{ticker}的财务数据")

            # 执行基本面分析
            result = self._fundamentals_analyst.process_task({
                "ticker": ticker,
                "financial_metrics": financial_metrics,
                "market_info": market_info,
                "show_reasoning": show_reasoning
            })

            # 将结果保存到数据流管理器
            success = data_flow_manager.store_data(
                data_type="fundamentals_analysis",
                ticker=ticker,
                data=result.get("content", {}),
                source_agent="FundamentalsAnalysisTask",
                metadata={
                    "analysis_time": datetime.now().isoformat(),
                    "quality_score": result.get("content", {}).get("analysis_result", {}).get("quality_score", 0)
                }
            )

            if not success:
                self.debug_logger.warning("基本面分析结果存储失败")

            self.log_task_complete(f"完成{ticker}的基本面分析")
            return result

        except Exception as e:
            self.log_task_error(e, "基本面分析任务失败")
            raise

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker"]

    def estimate_execution_time(self, context: Dict[str, Any]) -> float:
        """估算执行时间（秒）"""
        return 5.0  # 基本面分析需要一定时间

    def get_task_priority(self) -> int:
        """获取任务优先级"""
        return 80  # 基本面分析优先级中等

    def can_retry(self, error: Exception) -> bool:
        """判断是否可以重试"""
        return True


class SentimentAnalysisTask(BaseTask):
    """情绪分析任务"""

    def __init__(self):
        # Initialize parent first to avoid attribute filtering
        super().__init__(
            description="执行情绪分析，评估市场情绪和情感",
            expected_output="情绪分析结果，包含情绪指标和情感分析",
            agent=None  # Will set after parent initialization
        )
        # Set attributes after parent initialization
        from agents.sentiment_analyst import SentimentAnalyst
        self._sentiment_analyst = SentimentAnalyst()
        self._shared_context = get_global_context()
        # Update agent reference
        self._agent = self._sentiment_analyst
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务（为了兼容analysis_crew.py中的调用）"""
        return self.execute_with_retry(context)

    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行情绪分析任务

        Args:
            context: 任务上下文

        Returns:
            情绪分析结果
        """
        self.debug_logger.info(f"🔍 [DEBUG] ===== SentimentAnalysisTask 开始执行 =====")
        self.debug_logger.info(f"🔍 [DEBUG] context keys: {list(context.keys())}")

        try:
            ticker = context["ticker"]
            num_of_news = context.get("num_of_news", 10)
            show_reasoning = context.get("show_reasoning", False)

            self.debug_logger.info(f"🔍 [DEBUG] 参数 - ticker: {ticker}, num_of_news: {num_of_news}, show_reasoning: {show_reasoning}")
            self.log_task_start(f"开始{ticker}的情绪分析")

            # 检查数据依赖
            self.debug_logger.info(f"🔍 [DEBUG] ===== 检查数据依赖 =====")
            if not data_flow_manager.ensure_data_dependencies("sentiment_analysis", ticker):
                missing_deps = data_flow_manager.get_missing_dependencies("sentiment_analysis", ticker)
                self.debug_logger.warning(f"🔍 [DEBUG] 情绪分析缺少部分依赖数据: {missing_deps}")
                # 继续执行，但可能会影响分析质量
            else:
                self.debug_logger.info(f"🔍 [DEBUG] 数据依赖检查通过")

            # 从数据流管理器获取数据
            self.debug_logger.info(f"🔍 [DEBUG] ===== 从数据流管理器获取数据 =====")
            news_data_entry = data_flow_manager.retrieve_data("news_data", ticker)
            market_data_entry = data_flow_manager.retrieve_data("market_data", ticker)

            self.debug_logger.info(f"🔍 [DEBUG] news_data_entry: {'存在' if news_data_entry else '不存在'}")
            self.debug_logger.info(f"🔍 [DEBUG] market_data_entry: {'存在' if market_data_entry else '不存在'}")

            if news_data_entry is None:
                self.debug_logger.error(f"🔍 [DEBUG] 缺少{ticker}的新闻数据")
                raise ValueError(f"缺少{ticker}的新闻数据")

            news_content = news_data_entry.get("content", [])
            self.debug_logger.info(f"🔍 [DEBUG] 获取到的新闻数据大小: {len(news_content)}")

            if market_data_entry is None:
                self.debug_logger.warning(f"🔍 [DEBUG] 缺少{ticker}的市场数据，情绪分析可能不完整")
                market_content = {}
            else:
                market_content = market_data_entry.get("content", {})
                self.debug_logger.info(f"🔍 [DEBUG] 获取到的市场数据大小: {len(market_content)}")

            market_info = market_content.get("market_info", {})
            self.debug_logger.info(f"🔍 [DEBUG] market_info keys: {list(market_info.keys()) if market_info else 'None'}")

            if not news_content:
                self.debug_logger.error(f"🔍 [DEBUG] 新闻内容为空: {ticker}")
                raise ValueError(f"缺少{ticker}的新闻数据")

            # 执行情绪分析
            self.debug_logger.info(f"🔍 [DEBUG] ===== 开始执行情绪分析 =====")
            self.debug_logger.info(f"🔍 [DEBUG] 传递给SentimentAnalyst的news_content大小: {len(news_content)}")
            result = self._sentiment_analyst.process_task({
                "ticker": ticker,
                "news_data": news_content,
                "market_data": market_info,
                "num_of_news": num_of_news,
                "show_reasoning": show_reasoning
            })
            self.debug_logger.info(f"🔍 [DEBUG] 情绪分析完成，结果signal: {result.get('signal', 'unknown')}")

            # 将结果保存到数据流管理器
            self.debug_logger.info(f"🔍 [DEBUG] ===== 存储情绪分析结果 =====")
            success = data_flow_manager.store_data(
                data_type="sentiment_analysis",
                ticker=ticker,
                data=result.get("content", {}),
                source_agent="SentimentAnalysisTask",
                metadata={
                    "analysis_time": datetime.now().isoformat(),
                    "sentiment_score": result.get("content", {}).get("analysis_result", {}).get("overall_sentiment", 0)
                }
            )
            self.debug_logger.info(f"🔍 [DEBUG] 情绪分析结果存储: {'成功' if success else '失败'}")

            if not success:
                self.debug_logger.warning("🔍 [DEBUG] 情绪分析结果存储失败")

            self.debug_logger.info(f"🔍 [DEBUG] SentimentAnalysisTask 执行完成")
            self.log_task_complete(f"完成{ticker}的情绪分析")
            return result

        except Exception as e:
            self.debug_logger.error(f"🔍 [DEBUG] SentimentAnalysisTask 执行失败: {e}")
            import traceback
            self.debug_logger.error(f"🔍 [DEBUG] 详细错误堆栈: {traceback.format_exc()}")
            self.log_task_error(e, "情绪分析任务失败")
            raise

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker"]

    def estimate_execution_time(self, context: Dict[str, Any]) -> float:
        """估算执行时间（秒）"""
        return 4.0  # 情绪分析时间适中

    def get_task_priority(self) -> int:
        """获取任务优先级"""
        return 70  # 情绪分析优先级中等偏低

    def can_retry(self, error: Exception) -> bool:
        """判断是否可以重试"""
        return True


class ValuationAnalysisTask(BaseTask):
    """估值分析任务"""

    def __init__(self):
        # Initialize parent first to avoid attribute filtering
        super().__init__(
            description="执行估值分析，计算公司内在价值",
            expected_output="估值分析结果，包含各种估值方法和投资建议",
            agent=None  # Will set after parent initialization
        )
        # Set attributes after parent initialization
        from agents.valuation_analyst import ValuationAnalyst
        self._valuation_analyst = ValuationAnalyst()
        self._shared_context = get_global_context()
        # Update agent reference
        self._agent = self._valuation_analyst
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务（为了兼容analysis_crew.py中的调用）"""
        return self.execute_with_retry(context)

    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行估值分析任务

        Args:
            context: 任务上下文

        Returns:
            估值分析结果
        """
        try:
            ticker = context["ticker"]
            show_reasoning = context.get("show_reasoning", False)

            self.log_task_start(f"开始{ticker}的估值分析")

            # 检查数据依赖
            if not data_flow_manager.ensure_data_dependencies("valuation_analysis", ticker):
                missing_deps = data_flow_manager.get_missing_dependencies("valuation_analysis", ticker)
                raise ValueError(f"估值分析缺少依赖数据: {missing_deps}")

            # 从数据流管理器获取数据
            market_data_entry = data_flow_manager.retrieve_data("market_data", ticker)
            if market_data_entry is None:
                raise ValueError(f"缺少{ticker}的市场数据")

            market_data_content = market_data_entry.get("content", {})
            financial_metrics = market_data_content.get("financial_metrics", {})
            market_info = market_data_content.get("market_info", {})

            # 为估值分析添加当前价格
            market_info["current_price"] = market_info.get("market_cap", 0) / 100000000  # 假设1亿股

            if not financial_metrics:
                raise ValueError(f"缺少{ticker}的财务数据")

            # 执行估值分析
            result = self._valuation_analyst.process_task({
                "ticker": ticker,
                "financial_metrics": financial_metrics,
                "market_info": market_info,
                "show_reasoning": show_reasoning
            })

            # 将结果保存到数据流管理器
            success = data_flow_manager.store_data(
                data_type="valuation_analysis",
                ticker=ticker,
                data=result.get("content", {}),
                source_agent="ValuationAnalysisTask",
                metadata={
                    "analysis_time": datetime.now().isoformat(),
                    "valuation_grade": result.get("content", {}).get("analysis_result", {}).get("investment_metrics", {}).get("valuation_grade", "unknown")
                }
            )

            if not success:
                self.debug_logger.warning("估值分析结果存储失败")

            self.log_task_complete(f"完成{ticker}的估值分析")
            return result

        except Exception as e:
            self.log_task_error(e, "估值分析任务失败")
            raise

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker"]

    def estimate_execution_time(self, context: Dict[str, Any]) -> float:
        """估算执行时间（秒）"""
        return 8.0  # 估值分析时间较长

    def get_task_priority(self) -> int:
        """获取任务优先级"""
        return 75  # 估值分析优先级中等偏高

    def can_retry(self, error: Exception) -> bool:
        """判断是否可以重试"""
        return True