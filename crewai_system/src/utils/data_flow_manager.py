"""
数据流管理器
负责管理任务间的数据传递和共享上下文的一致性
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from crewai_system.src.utils.shared_context import get_global_context
from crewai_system.src.utils.logging_config import get_logger


class DataFlowManager:
    """数据流管理器"""

    def __init__(self):
        self.logger = get_logger("data")
        self.shared_context = get_global_context()

        # 数据键映射规则
        self.key_mappings = {
            "market_data": "{ticker}_market_data",
            "news_data": "{ticker}_news_data",
            "technical_analysis": "{ticker}_technical_analysis",
            "fundamentals_analysis": "{ticker}_fundamentals_analysis",
            "sentiment_analysis": "{ticker}_sentiment_analysis",
            "valuation_analysis": "{ticker}_valuation_analysis"
        }

        # 数据依赖关系
        self.data_dependencies = {
            "technical_analysis": ["market_data"],
            "fundamentals_analysis": ["market_data"],
            "sentiment_analysis": ["news_data", "market_data"],
            "valuation_analysis": ["market_data", "fundamentals_analysis"]
        }

        # 数据验证规则
        self.validation_rules = {
            "market_data": self._validate_market_data,
            "news_data": self._validate_news_data,
            "technical_analysis": self._validate_technical_analysis,
            "fundamentals_analysis": self._validate_fundamentals_analysis,
            "sentiment_analysis": self._validate_sentiment_analysis,
            "valuation_analysis": self._validate_valuation_analysis
        }

    def store_data(self, data_type: str, ticker: str, data: Dict[str, Any],
                   source_agent: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        存储数据到共享上下文

        Args:
            data_type: 数据类型
            ticker: 股票代码
            data: 数据内容
            source_agent: 数据来源智能体
            metadata: 元数据

        Returns:
            存储是否成功
        """
        self.logger.info(f"🔍 [DEBUG] ===== 开始存储数据 =====")
        self.logger.info(f"🔍 [DEBUG] data_type: {data_type}, ticker: {ticker}, source_agent: {source_agent}")
        self.logger.info(f"🔍 [DEBUG] 原始数据大小: {len(str(data)) if data else 0}")

        try:
            key = self._get_data_key(data_type, ticker)
            self.logger.info(f"🔍 [DEBUG] 生成存储key: {key}")

            # 验证数据
            self.logger.info(f"🔍 [DEBUG] 开始验证数据...")
            if not self._validate_data(data_type, data):
                self.logger.warning(f"🔍 [DEBUG] 数据验证失败: {data_type} for {ticker}")
                return False
            self.logger.info(f"🔍 [DEBUG] 数据验证通过")

            # 添加存储元数据
            enriched_data = {
                "content": data,
                "metadata": {
                    "data_type": data_type,
                    "ticker": ticker,
                    "source_agent": source_agent,
                    "storage_time": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }

            if metadata:
                enriched_data["metadata"].update(metadata)

            # 存储数据
            self.logger.info(f"🔍 [DEBUG] 存储数据到共享上下文...")
            self.shared_context.set(
                key=key,
                value=enriched_data,
                source_agent=source_agent,
                data_type=data_type
            )

            self.logger.info(f"🔍 [DEBUG] 数据存储成功: {key}")

            # 如果是新闻数据，记录详细信息
            if data_type == "news_data":
                news_count = len(data) if isinstance(data, list) else 0
                self.logger.info(f"🔍 [DEBUG] 新闻数据存储: 数量={news_count}, ticker={ticker}")
                if isinstance(data, list) and news_count > 0:
                    self.logger.info(f"🔍 [DEBUG] 第一条新闻标题: {data[0].get('title', '无标题')[:50]}...")

            return True

        except Exception as e:
            self.logger.error(f"🔍 [DEBUG] 数据存储失败: {data_type} for {ticker}, 错误: {e}")
            import traceback
            self.logger.error(f"🔍 [DEBUG] 详细错误堆栈: {traceback.format_exc()}")
            return False

    def retrieve_data(self, data_type: str, ticker: str,
                     require_valid: bool = True) -> Optional[Dict[str, Any]]:
        """
        从共享上下文检索数据

        Args:
            data_type: 数据类型
            ticker: 股票代码
            require_valid: 是否要求数据有效

        Returns:
            检索到的数据或None
        """
        self.logger.info(f"🔍 [DEBUG] ===== 开始检索数据 =====")
        self.logger.info(f"🔍 [DEBUG] data_type: {data_type}, ticker: {ticker}, require_valid: {require_valid}")

        try:
            key = self._get_data_key(data_type, ticker)
            self.logger.info(f"🔍 [DEBUG] 生成检索key: {key}")

            data = self.shared_context.get(key, {})
            self.logger.info(f"🔍 [DEBUG] 从共享上下文获取数据: {'存在' if data else '不存在'}")

            if not data:
                self.logger.warning(f"🔍 [DEBUG] 数据不存在: {key}")
                return None

            # 验证数据完整性
            if require_valid:
                self.logger.info(f"🔍 [DEBUG] 开始验证数据完整性...")
                if not self._validate_data(data_type, data.get("content", {})):
                    self.logger.warning(f"🔍 [DEBUG] 检索到的数据无效: {key}")
                    return None
                self.logger.info(f"🔍 [DEBUG] 数据验证通过")

            # 如果是新闻数据，记录详细信息
            if data_type == "news_data":
                content = data.get("content", [])
                news_count = len(content) if isinstance(content, list) else 0
                self.logger.info(f"🔍 [DEBUG] 新闻数据检索: 数量={news_count}, ticker={ticker}")
                if isinstance(content, list) and news_count > 0:
                    self.logger.info(f"🔍 [DEBUG] 检索到的第一条新闻标题: {content[0].get('title', '无标题')[:50]}...")

            self.logger.info(f"🔍 [DEBUG] 数据检索成功: {key}")
            return data

        except Exception as e:
            self.logger.error(f"🔍 [DEBUG] 数据检索失败: {data_type} for {ticker}, 错误: {e}")
            import traceback
            self.logger.error(f"🔍 [DEBUG] 详细错误堆栈: {traceback.format_exc()}")
            return None

    def check_data_availability(self, data_type: str, ticker: str) -> Dict[str, Any]:
        """
        检查数据可用性

        Args:
            data_type: 数据类型
            ticker: 股票代码

        Returns:
            数据可用性信息
        """
        try:
            data = self.retrieve_data(data_type, ticker, require_valid=False)

            if data is None:
                return {
                    "available": False,
                    "reason": "data_not_found",
                    "quality": "none"
                }

            content = data.get("content", {})
            metadata = data.get("metadata", {})

            # 检查数据质量
            quality = self._assess_data_quality(data_type, content)

            return {
                "available": True,
                "quality": quality,
                "source_agent": metadata.get("source_agent"),
                "storage_time": metadata.get("storage_time"),
                "data_size": len(json.dumps(content, default=str))
            }

        except Exception as e:
            self.logger.error(f"检查数据可用性失败: {data_type} for {ticker}, 错误: {e}")
            return {
                "available": False,
                "reason": "check_failed",
                "error": str(e),
                "quality": "none"
            }

    def get_missing_dependencies(self, data_type: str, ticker: str) -> List[str]:
        """
        获取缺失的依赖数据

        Args:
            data_type: 数据类型
            ticker: 股票代码

        Returns:
            缺失的依赖列表
        """
        missing = []

        if data_type not in self.data_dependencies:
            return missing

        for dependency in self.data_dependencies[data_type]:
            availability = self.check_data_availability(dependency, ticker)
            if not availability["available"] or availability["quality"] == "poor":
                missing.append(dependency)

        return missing

    def ensure_data_dependencies(self, data_type: str, ticker: str) -> bool:
        """
        确保数据依赖满足

        Args:
            data_type: 数据类型
            ticker: 股票代码

        Returns:
            依赖是否满足
        """
        missing = self.get_missing_dependencies(data_type, ticker)

        if missing:
            self.logger.warning(f"数据依赖缺失: {data_type} 需要 {missing}")
            return False

        return True

    def get_data_summary(self, ticker: str) -> Dict[str, Any]:
        """
        获取指定股票的数据摘要

        Args:
            ticker: 股票代码

        Returns:
            数据摘要
        """
        summary = {
            "ticker": ticker,
            "data_types": {},
            "overall_quality": "unknown",
            "completeness": 0.0
        }

        quality_scores = []

        for data_type in self.key_mappings.keys():
            availability = self.check_data_availability(data_type, ticker)
            summary["data_types"][data_type] = availability

            if availability["available"]:
                quality_map = {"excellent": 5, "good": 4, "fair": 3, "poor": 2, "minimal": 1}
                quality = availability.get("quality", "none")
                if quality in quality_map:
                    quality_scores.append(quality_map[quality])

        # 计算整体质量
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality >= 4.5:
                summary["overall_quality"] = "excellent"
            elif avg_quality >= 3.5:
                summary["overall_quality"] = "good"
            elif avg_quality >= 2.5:
                summary["overall_quality"] = "fair"
            else:
                summary["overall_quality"] = "poor"

        # 计算完整性
        available_count = sum(1 for d in summary["data_types"].values() if d["available"])
        summary["completeness"] = available_count / len(self.key_mappings)

        return summary

    def cleanup_expired_data(self, ticker: str, ttl_hours: int = 24) -> int:
        """
        清理过期数据

        Args:
            ticker: 股票代码
            ttl_hours: 生存时间（小时）

        Returns:
            清理的数据数量
        """
        try:
            cutoff_time = datetime.now().timestamp() - (ttl_hours * 3600)
            cleaned_count = 0

            for data_type in self.key_mappings.keys():
                key = self._get_data_key(data_type, ticker)
                data = self.shared_context.get(key, {})

                if data:
                    metadata = data.get("metadata", {})
                    storage_time = metadata.get("storage_time")

                    if storage_time:
                        try:
                            storage_timestamp = datetime.fromisoformat(storage_time).timestamp()
                            if storage_timestamp < cutoff_time:
                                self.shared_context.delete(key)
                                cleaned_count += 1
                                self.logger.debug(f"清理过期数据: {key}")
                        except:
                            pass

            return cleaned_count

        except Exception as e:
            self.logger.error(f"清理过期数据失败: {ticker}, 错误: {e}")
            return 0

    def _get_data_key(self, data_type: str, ticker: str) -> str:
        """获取数据键"""
        template = self.key_mappings.get(data_type, "{ticker}_{data_type}")
        return template.format(ticker=ticker, data_type=data_type)

    def _validate_data(self, data_type: str, data: Dict[str, Any]) -> bool:
        """验证数据"""
        validator = self.validation_rules.get(data_type)
        if validator:
            return validator(data)
        return True  # 默认验证通过

    def _assess_data_quality(self, data_type: str, data: Dict[str, Any]) -> str:
        """评估数据质量"""
        # 基础质量评估
        if not data or not isinstance(data, dict):
            return "none"

        # 根据数据类型进行特定评估
        if data_type == "market_data":
            return self._assess_market_data_quality(data)
        elif data_type == "news_data":
            return self._assess_news_data_quality(data)
        elif data_type == "technical_analysis":
            return self._assess_technical_analysis_quality(data)
        elif data_type == "fundamentals_analysis":
            return self._assess_fundamentals_analysis_quality(data)
        elif data_type == "sentiment_analysis":
            return self._assess_sentiment_analysis_quality(data)
        elif data_type == "valuation_analysis":
            return self._assess_valuation_analysis_quality(data)

        return "fair"

    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """验证市场数据"""
        required_fields = ["ticker", "collection_time"]
        for field in required_fields:
            if field not in data:
                return False

        # 检查是否有价格或财务数据
        has_prices = "prices" in data and isinstance(data["prices"], list) and len(data["prices"]) > 0
        has_financial = "financial_metrics" in data and isinstance(data["financial_metrics"], dict)

        return has_prices or has_financial

    def _validate_news_data(self, data: Dict[str, Any]) -> bool:
        """验证新闻数据"""
        if not isinstance(data, list):
            return False

        # 检查新闻条目
        for news_item in data:
            if not isinstance(news_item, dict):
                return False
            if "title" not in news_item:
                return False

        return len(data) > 0

    def _validate_technical_analysis(self, data: Dict[str, Any]) -> bool:
        """验证技术分析数据"""
        required_fields = ["analysis_result"]
        for field in required_fields:
            if field not in data:
                return False

        analysis_result = data["analysis_result"]
        if not isinstance(analysis_result, dict):
            return False

        return "trends" in analysis_result or "indicators" in analysis_result

    def _validate_fundamentals_analysis(self, data: Dict[str, Any]) -> bool:
        """验证基本面分析数据"""
        required_fields = ["analysis_result"]
        for field in required_fields:
            if field not in data:
                return False

        analysis_result = data["analysis_result"]
        if not isinstance(analysis_result, dict):
            return False

        return "quality_score" in analysis_result or "financial_health" in analysis_result

    def _validate_sentiment_analysis(self, data: Dict[str, Any]) -> bool:
        """验证情绪分析数据"""
        required_fields = ["analysis_result"]
        for field in required_fields:
            if field not in data:
                return False

        analysis_result = data["analysis_result"]
        if not isinstance(analysis_result, dict):
            return False

        return "overall_sentiment" in analysis_result or "sentiment_distribution" in analysis_result

    def _validate_valuation_analysis(self, data: Dict[str, Any]) -> bool:
        """验证估值分析数据"""
        required_fields = ["analysis_result"]
        for field in required_fields:
            if field not in data:
                return False

        analysis_result = data["analysis_result"]
        if not isinstance(analysis_result, dict):
            return False

        return "valuation_methods" in analysis_result or "investment_metrics" in analysis_result

    def _assess_market_data_quality(self, data: Dict[str, Any]) -> str:
        """评估市场数据质量"""
        score = 0

        # 评估价格数据
        prices = data.get("prices", [])
        if prices and len(prices) > 100:
            score += 3
        elif prices and len(prices) > 20:
            score += 1

        # 评估财务数据
        financial_metrics = data.get("financial_metrics", {})
        if financial_metrics and len(financial_metrics) > 10:
            score += 3
        elif financial_metrics and len(financial_metrics) > 3:
            score += 1

        # 评估市场信息
        market_info = data.get("market_info", {})
        if market_info and market_info.get("market_cap", 0) > 0:
            score += 2

        # 评估统计信息
        statistics = data.get("statistics", {})
        if statistics:
            score += 1

        if score >= 7:
            return "excellent"
        elif score >= 5:
            return "good"
        elif score >= 3:
            return "fair"
        else:
            return "poor"

    def _assess_news_data_quality(self, data: Dict[str, Any]) -> str:
        """评估新闻数据质量"""
        if not isinstance(data, list):
            return "poor"

        score = min(len(data), 5)  # 最多5分

        # 检查新闻完整性
        complete_news = 0
        for news in data:
            if isinstance(news, dict) and news.get("title") and news.get("content"):
                complete_news += 1

        if complete_news == len(data) and len(data) >= 5:
            score += 2
        elif complete_news >= len(data) * 0.7:
            score += 1

        if score >= 6:
            return "excellent"
        elif score >= 4:
            return "good"
        elif score >= 2:
            return "fair"
        else:
            return "poor"

    def _assess_technical_analysis_quality(self, data: Dict[str, Any]) -> str:
        """评估技术分析质量"""
        analysis_result = data.get("analysis_result", {})
        score = 0

        # 评估趋势分析
        trends = analysis_result.get("trends", {})
        if trends:
            score += 2

        # 评估指标分析
        indicators = analysis_result.get("indicators", {})
        if indicators:
            score += 2

        # 评估信号强度
        signal_strength = analysis_result.get("signal_strength", 0)
        if signal_strength > 0.7:
            score += 2
        elif signal_strength > 0.4:
            score += 1

        if score >= 5:
            return "excellent"
        elif score >= 3:
            return "good"
        elif score >= 1:
            return "fair"
        else:
            return "poor"

    def _assess_fundamentals_analysis_quality(self, data: Dict[str, Any]) -> str:
        """评估基本面分析质量"""
        analysis_result = data.get("analysis_result", {})
        score = 0

        # 评估质量评分
        quality_score = analysis_result.get("quality_score", 0)
        if quality_score > 70:
            score += 3
        elif quality_score > 50:
            score += 2
        elif quality_score > 30:
            score += 1

        # 评估财务健康
        financial_health = analysis_result.get("financial_health", {})
        if financial_health:
            score += 2

        # 评估风险因素
        risk_factors = analysis_result.get("risk_factors", [])
        if risk_factors:
            score += 1

        if score >= 5:
            return "excellent"
        elif score >= 3:
            return "good"
        elif score >= 1:
            return "fair"
        else:
            return "poor"

    def _assess_sentiment_analysis_quality(self, data: Dict[str, Any]) -> str:
        """评估情绪分析质量"""
        analysis_result = data.get("analysis_result", {})
        score = 0

        # 评估情绪分数
        overall_sentiment = analysis_result.get("overall_sentiment", 0)
        if overall_sentiment != 0:
            score += 2

        # 评估情绪分布
        sentiment_distribution = analysis_result.get("sentiment_distribution", {})
        if sentiment_distribution:
            score += 2

        # 评估新闻覆盖
        news_coverage = analysis_result.get("news_coverage", {})
        if news_coverage:
            score += 1

        if score >= 4:
            return "excellent"
        elif score >= 2:
            return "good"
        elif score >= 1:
            return "fair"
        else:
            return "poor"

    def _assess_valuation_analysis_quality(self, data: Dict[str, Any]) -> str:
        """评估估值分析质量"""
        analysis_result = data.get("analysis_result", {})
        score = 0

        # 评估估值方法
        valuation_methods = analysis_result.get("valuation_methods", {})
        if valuation_methods:
            score += 2

        # 评估投资指标
        investment_metrics = analysis_result.get("investment_metrics", {})
        if investment_metrics:
            score += 2

        # 评估估值等级
        investment_metrics = analysis_result.get("investment_metrics", {})
        valuation_grade = investment_metrics.get("valuation_grade")
        if valuation_grade in ["undervalued", "fairly_valued", "overvalued"]:
            score += 1

        if score >= 4:
            return "excellent"
        elif score >= 2:
            return "good"
        elif score >= 1:
            return "fair"
        else:
            return "poor"


# 全局数据流管理器实例
data_flow_manager = DataFlowManager()


def get_data_flow_manager() -> DataFlowManager:
    """获取全局数据流管理器实例"""
    return data_flow_manager