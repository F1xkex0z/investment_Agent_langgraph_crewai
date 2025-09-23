"""
市场数据智能体
负责收集和预处理市场数据，包括股价历史、财务指标和市场信息
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from crewai.tools import BaseTool

from crewai_system.src.agents.base_agent import BaseAgent
from crewai_system.src.tools.data_sources import (
    get_data_adapter,
    PriceHistoryTool,
    FinancialMetricsTool,
    MarketDataTool,
    NewsSearchTool
)
from crewai_system.src.utils.data_processing import get_data_processor


class MarketDataAgent(BaseAgent):
    """市场数据智能体"""

    def __init__(self):
        super().__init__(
            role="市场数据收集专家",
            goal="收集和预处理股票市场数据，为投资分析提供全面的数据支持",
            backstory="""你是一位专业的市场数据收集专家，拥有多年的金融市场数据经验。
            你精通各种数据源的API调用，能够高效地获取股价历史、财务指标、市场数据等信息，
            并对数据进行清洗和预处理，确保数据的准确性和完整性。
            你的工作是为后续的分析智能体提供高质量的数据基础。""",
            tools=[
                PriceHistoryTool(get_data_adapter()),
                FinancialMetricsTool(get_data_adapter()),
                MarketDataTool(get_data_adapter()),
                NewsSearchTool(get_data_adapter())
            ],
            agent_name="MarketDataAgent"
        )

        self._data_processor = get_data_processor()
        self._data_adapter = get_data_adapter()

    # 注意：请直接使用实例变量 self._data_processor 和 self._data_adapter
    # 不再通过属性访问器访问这些实例变量

    def process_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理市场数据收集任务

        Args:
            task_context: 任务上下文，包含股票代码、日期范围等信息

        Returns:
            处理结果，包含收集到的市场数据
        """
        ticker = task_context.get("ticker")
        start_date = task_context.get("start_date")
        end_date = task_context.get("end_date")
        show_reasoning = task_context.get("show_reasoning", False)

        self.log_execution_start(f"收集{ticker}的市场数据")

        try:
            # 验证输入参数
            required_fields = ["ticker", "start_date", "end_date"]
            if not self.validate_input(task_context, required_fields):
                raise ValueError(f"缺少必需字段: {required_fields}")

            # 标准化股票代码
            normalized_ticker = self._data_processor.normalize_ticker(ticker)

            # 设置默认日期
            if not start_date or not end_date:
                start_date, end_date = self._data_processor.get_date_range(365)

            # 收集各种数据
            market_data = self._collect_market_data(normalized_ticker, start_date, end_date)

            # 数据验证和清洗
            validated_data = self._validate_and_clean_data(market_data)

            # 生成数据摘要
            data_summary = self._generate_data_summary(validated_data, normalized_ticker, start_date, end_date)

            # 记录推理过程
            if show_reasoning:
                self.log_reasoning(data_summary, "市场数据收集推理过程")

            result = self.format_agent_output(
                content=validated_data,
                signal="data_collected",
                confidence=0.95,
                reasoning=f"成功收集{normalized_ticker}从{start_date}到{end_date}的市场数据",
                metadata={
                    "ticker": normalized_ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data_quality": self._assess_data_quality(validated_data)
                }
            )

            self.log_execution_complete(f"成功收集{normalized_ticker}的市场数据")
            return result

        except Exception as e:
            self.log_execution_error(e, f"收集{ticker}市场数据失败")
            raise

    def _collect_market_data(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        收集市场数据

        Args:
            ticker: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            收集到的市场数据
        """
        self.logger.info(f"开始收集{ticker}的市场数据，时间范围: {start_date} 至 {end_date}")

        market_data = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "collection_time": datetime.now().isoformat()
        }

        # 收集价格历史数据
        try:
            prices_df = self._data_adapter.get_price_history_safe(ticker, start_date, end_date)
            market_data["prices"] = self._data_processor.pandas_to_dict_records(prices_df)
            self.logger.debug(f"成功获取价格数据，包含{len(market_data['prices'])}条记录")
        except Exception as e:
            self.logger.error(f"获取价格数据失败: {e}")
            market_data["prices"] = []
            market_data["prices_error"] = str(e)

        # 收集财务指标
        try:
            financial_metrics = self._data_adapter.get_financial_metrics_safe(ticker)
            market_data["financial_metrics"] = financial_metrics
            self.logger.debug(f"成功获取财务指标，包含{len(financial_metrics)}个指标")
        except Exception as e:
            self.logger.error(f"获取财务指标失败: {e}")
            market_data["financial_metrics"] = {}
            market_data["financial_metrics_error"] = str(e)

        # 收集市场数据
        try:
            market_info = self._data_adapter.get_market_data_safe(ticker)
            market_data["market_info"] = market_info
            self.logger.debug(f"成功获取市场信息，市值: {market_info.get('market_cap', '未知')}")
        except Exception as e:
            self.logger.error(f"获取市场信息失败: {e}")
            market_data["market_info"] = {"market_cap": 0}
            market_data["market_info_error"] = str(e)

        # 计算基本统计信息
        if market_data["prices"]:
            market_data["statistics"] = self._calculate_price_statistics(market_data["prices"])

        return market_data

    def _validate_and_clean_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证和清洗数据

        Args:
            market_data: 原始市场数据

        Returns:
            清洗后的数据
        """
        validated_data = market_data.copy()

        # 验证价格数据
        if "prices" in validated_data and validated_data["prices"]:
            cleaned_prices = []
            for price_record in validated_data["prices"]:
                # 清理数值字段
                cleaned_record = self._data_processor.clean_numeric_data(price_record)
                # 验证必要字段
                if self._validate_price_record(cleaned_record):
                    cleaned_prices.append(cleaned_record)

            validated_data["prices"] = cleaned_prices
            validated_data["price_count"] = len(cleaned_prices)

        # 清洗财务指标
        if "financial_metrics" in validated_data:
            validated_data["financial_metrics"] = self._data_processor.format_financial_metrics(
                validated_data["financial_metrics"]
            )

        # 清洗市场信息
        if "market_info" in validated_data:
            validated_data["market_info"] = self._data_processor.clean_numeric_data(
                validated_data["market_info"]
            )

        return validated_data

    def _validate_price_record(self, record: Dict[str, Any]) -> bool:
        """
        验证价格记录

        Args:
            record: 价格记录

        Returns:
            是否有效
        """
        required_fields = ["close", "open", "high", "low", "volume"]
        for field in required_fields:
            if field not in record or record[field] is None:
                return False

        # 验证价格逻辑
        close = record.get("close", 0)
        high = record.get("high", 0)
        low = record.get("low", 0)

        if close <= 0 or high <= 0 or low <= 0:
            return False

        if high < low:
            return False

        if not (low <= close <= high):
            return False

        return True

    def _calculate_price_statistics(self, prices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算价格统计信息

        Args:
            prices: 价格数据列表

        Returns:
            统计信息
        """
        if not prices:
            return {}

        closes = [p.get("close", 0) for p in prices if p.get("close")]
        volumes = [p.get("volume", 0) for p in prices if p.get("volume")]

        if not closes:
            return {}

        statistics = {
            "price_stats": {
                "min": min(closes),
                "max": max(closes),
                "mean": sum(closes) / len(closes),
                "latest": closes[-1] if closes else None,
                "change": ((closes[-1] - closes[0]) / closes[0] * 100) if len(closes) > 1 else 0
            }
        }

        if volumes:
            statistics["volume_stats"] = {
                "min": min(volumes),
                "max": max(volumes),
                "mean": sum(volumes) / len(volumes),
                "latest": volumes[-1] if volumes else None
            }

        return statistics

    def _generate_data_summary(self, validated_data: Dict[str, Any], ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        生成数据摘要

        Args:
            validated_data: 验证后的数据
            ticker: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            数据摘要
        """
        summary = {
            "ticker": ticker,
            "period": f"{start_date} 至 {end_date}",
            "data_collection_status": "completed",
            "data_sources": ["akshare", "market_data_api"],
            "data_quality": self._assess_data_quality(validated_data)
        }

        # 价格数据摘要
        if "prices" in validated_data:
            price_count = len(validated_data["prices"])
            summary["price_data"] = {
                "record_count": price_count,
                "date_range": f"{start_date} 至 {end_date}",
                "data_completeness": "good" if price_count > 200 else "limited"
            }

        # 财务数据摘要
        if "financial_metrics" in validated_data:
            metrics_count = len(validated_data["financial_metrics"])
            summary["financial_data"] = {
                "metrics_count": metrics_count,
                "available_metrics": list(validated_data["financial_metrics"].keys())[:5],  # 显示前5个
                "data_completeness": "good" if metrics_count > 10 else "limited"
            }

        # 市场信息摘要
        if "market_info" in validated_data:
            market_cap = validated_data["market_info"].get("market_cap", 0)
            summary["market_info"] = {
                "market_cap": market_cap,
                "market_cap_formatted": self._data_processor.format_large_number(market_cap),
                "data_status": "available" if market_cap > 0 else "unavailable"
            }

        return summary

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
            if price_count > 250:  # 一年多的交易日数据
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

        # 评估统计信息质量
        if "statistics" in data and data["statistics"]:
            quality_score += 1

        # 根据总分评估质量
        if quality_score >= 8:
            return "excellent"
        elif quality_score >= 6:
            return "good"
        elif quality_score >= 4:
            return "fair"
        else:
            return "poor"

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker", "start_date", "end_date"]

    def get_data_availability(self, ticker: str) -> Dict[str, Any]:
        """
        检查数据可用性

        Args:
            ticker: 股票代码

        Returns:
            数据可用性信息
        """
        try:
            # 尝试获取少量数据来检查可用性
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            # 检查价格数据
            prices = self._data_adapter.get_price_history_safe(ticker, start_date, end_date)
            price_available = len(prices) > 0

            # 检查财务数据
            metrics = self._data_adapter.get_financial_metrics_safe(ticker)
            metrics_available = len(metrics) > 0

            # 检查市场数据
            market_info = self._data_adapter.get_market_data_safe(ticker)
            market_available = market_info.get("market_cap", 0) > 0

            return {
                "ticker": ticker,
                "price_data_available": price_available,
                "financial_data_available": metrics_available,
                "market_data_available": market_available,
                "overall_availability": "good" if all([price_available, metrics_available, market_available]) else "limited"
            }

        except Exception as e:
            self.logger.error(f"检查数据可用性失败 {ticker}: {e}")
            return {
                "ticker": ticker,
                "price_data_available": False,
                "financial_data_available": False,
                "market_data_available": False,
                "overall_availability": "error",
                "error": str(e)
            }