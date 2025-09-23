"""
分析团队(AnalysisCrew)
负责协调各个分析师智能体进行综合投资分析
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
import os

from crewai import Crew, Process
from crewai.task import Task
from crewai_system.src.agents.base_agent import BaseAgent
from crewai_system.src.utils.shared_context import get_global_context
from crewai_system.src.utils.logging_config import get_logger, log_info, log_error, log_performance

# 导入智能体和任务
from crewai_system.src.agents.market_data_agent import MarketDataAgent
from crewai_system.src.agents.technical_analyst import TechnicalAnalyst
from crewai_system.src.agents.fundamentals_analyst import FundamentalsAnalyst
from crewai_system.src.agents.sentiment_analyst import SentimentAnalyst
from crewai_system.src.agents.valuation_analyst import ValuationAnalyst

from crewai_system.src.tasks.data_collection_task import (
    DataCollectionTask,
    NewsCollectionTask,
    TechnicalAnalysisTask,
    FundamentalsAnalysisTask,
    SentimentAnalysisTask,
    ValuationAnalysisTask
)


class AnalysisCrew:
    """分析团队"""

    def __init__(self):
        self.logger = get_logger("analysis_crew")
        self.shared_context = get_global_context()

        # 初始化智能体
        self.agents = {
            "market_data": MarketDataAgent(),
            "technical": TechnicalAnalyst(),
            "fundamentals": FundamentalsAnalyst(),
            "sentiment": SentimentAnalyst(),
            "valuation": ValuationAnalyst()
        }

        # 初始化任务
        self.tasks = {
            "data_collection": DataCollectionTask(),
            "news_collection": NewsCollectionTask(),
            "technical_analysis": TechnicalAnalysisTask(),
            "fundamentals_analysis": FundamentalsAnalysisTask(),
            "sentiment_analysis": SentimentAnalysisTask(),
            "valuation_analysis": ValuationAnalysisTask()
        }

        # CrewAI实例
        self.crew = None

    def setup_crew(self, run_context: Dict[str, Any]) -> None:
        """
        设置Crew团队

        Args:
            run_context: 运行上下文
        """
        try:
            self.logger.info("正在设置分析团队...")

            # 为当前运行创建专门的智能体和任务
            crew_agents = []
            crew_tasks = []

            # 1. 数据收集任务
            data_collection_task = Task(
                description=f"收集{run_context['ticker']}的市场数据，时间范围: {run_context['start_date']} 至 {run_context['end_date']}",
                expected_output="完整的市场数据包",
                agent=self.agents["market_data"]
            )
            crew_tasks.append(data_collection_task)
            crew_agents.append(self.agents["market_data"])

            # 2. 新闻收集任务
            news_collection_task = Task(
                description=f"收集{run_context['ticker']}的相关新闻，数量: {run_context.get('num_of_news', 10)}",
                expected_output="新闻数据包",
                agent=self.agents["market_data"]  # 暂时使用市场数据智能体
            )
            crew_tasks.append(news_collection_task)
            crew_agents.append(self.agents["market_data"])

            # 3. 技术分析任务
            technical_analysis_task = Task(
                description=f"对{run_context['ticker']}进行技术分析",
                expected_output="技术分析结果",
                agent=self.agents["technical"]
            )
            crew_tasks.append(technical_analysis_task)
            crew_agents.append(self.agents["technical"])

            # 4. 基本面分析任务
            fundamentals_analysis_task = Task(
                description=f"对{run_context['ticker']}进行基本面分析",
                expected_output="基本面分析结果",
                agent=self.agents["fundamentals"]
            )
            crew_tasks.append(fundamentals_analysis_task)
            crew_agents.append(self.agents["fundamentals"])

            # 5. 情绪分析任务
            sentiment_analysis_task = Task(
                description=f"对{run_context['ticker']}进行情绪分析",
                expected_output="情绪分析结果",
                agent=self.agents["sentiment"]
            )
            crew_tasks.append(sentiment_analysis_task)
            crew_agents.append(self.agents["sentiment"])

            # 6. 估值分析任务
            valuation_analysis_task = Task(
                description=f"对{run_context['ticker']}进行估值分析",
                expected_output="估值分析结果",
                agent=self.agents["valuation"]
            )
            crew_tasks.append(valuation_analysis_task)
            crew_agents.append(self.agents["valuation"])

            # 创建Crew实例
            self.crew = Crew(
                agents=crew_agents,
                tasks=crew_tasks,
                process=Process.sequential,  # 顺序执行
                verbose=run_context.get('show_reasoning', False)
            )

            self.logger.info("分析团队设置完成")

        except Exception as e:
            self.logger.error(f"设置分析团队失败: {e}")
            raise

    def execute_analysis(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整的投资分析

        Args:
            run_context: 运行上下文

        Returns:
            综合分析结果
        """
        try:
            ticker = run_context["ticker"]
            run_id = run_context["run_id"]

            self.logger.info(f"开始执行{ticker}的综合分析，运行ID: {run_id}")

            # 执行Crew任务
            if self.crew is None:
                self.setup_crew(run_context)

            # 记录分析开始
            start_time = datetime.now()
            self.shared_context.set(
                key=f"run_{run_id}_analysis_start",
                value=start_time.isoformat(),
                source_agent="AnalysisCrew"
            )

            # 执行分析（这里使用自定义执行逻辑，避免API密钥问题）
            result = self._execute_custom_analysis(run_context)

            # 记录分析结束
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self.shared_context.set(
                key=f"run_{run_id}_analysis_end",
                value=end_time.isoformat(),
                source_agent="AnalysisCrew"
            )

            self.shared_context.set(
                key=f"run_{run_id}_execution_time",
                value=execution_time,
                source_agent="AnalysisCrew"
            )

            self.logger.info(f"完成{ticker}的综合分析，耗时: {execution_time:.2f}秒")

            return result

        except Exception as e:
            self.logger.error(f"执行综合分析失败: {e}")
            raise

    def _execute_custom_analysis(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行自定义分析逻辑

        Args:
            run_context: 运行上下文

        Returns:
            综合分析结果
        """
        ticker = run_context["ticker"]
        run_id = run_context["run_id"]
        show_reasoning = run_context.get("show_reasoning", False)

        analysis_results = {}
        execution_summary = []

        try:
            # 1. 数据收集
            self.logger.info("步骤1: 收集市场数据")
            try:
                data_result = self.tasks["data_collection"].execute(run_context)
                analysis_results["market_data"] = data_result
                execution_summary.append("市场数据收集完成")
                self.logger.info("✓ 市场数据收集成功")
            except Exception as e:
                self.logger.error(f"✗ 市场数据收集失败: {e}")
                raise

            # 2. 新闻收集
            self.logger.info("步骤2: 收集新闻数据")
            news_result = self.tasks["news_collection"].execute(run_context)
            analysis_results["news_data"] = news_result
            execution_summary.append("新闻数据收集完成")

            # 3. 技术分析
            self.logger.info("步骤3: 执行技术分析")
            technical_result = self.tasks["technical_analysis"].execute(run_context)
            analysis_results["technical_analysis"] = technical_result
            execution_summary.append("技术分析完成")

            # 4. 基本面分析
            self.logger.info("步骤4: 执行基本面分析")
            fundamentals_result = self.tasks["fundamentals_analysis"].execute(run_context)
            analysis_results["fundamentals_analysis"] = fundamentals_result
            execution_summary.append("基本面分析完成")

            # 5. 情绪分析
            self.logger.info("步骤5: 执行情绪分析")
            sentiment_result = self.tasks["sentiment_analysis"].execute(run_context)
            analysis_results["sentiment_analysis"] = sentiment_result
            execution_summary.append("情绪分析完成")

            # 6. 估值分析
            self.logger.info("步骤6: 执行估值分析")
            valuation_result = self.tasks["valuation_analysis"].execute(run_context)
            analysis_results["valuation_analysis"] = valuation_result
            execution_summary.append("估值分析完成")

            # 7. 综合分析
            self.logger.info("步骤7: 生成综合分析结果")
            comprehensive_result = self._generate_comprehensive_analysis(
                analysis_results, run_context
            )
            execution_summary.append("综合分析完成")

            # 构建最终结果
            final_result = {
                "run_id": run_id,
                "ticker": ticker,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_results": analysis_results,
                "comprehensive_analysis": comprehensive_result,
                "execution_summary": execution_summary,
                "total_execution_time": sum(
                    r.get("metadata", {}).get("execution_time", 0)
                    for r in analysis_results.values()
                ),
                "metadata": {
                    "analysis_type": "comprehensive",
                    "agents_used": list(self.agents.keys()),
                    "tasks_completed": len(execution_summary),
                    "data_quality": self._assess_overall_data_quality(analysis_results)
                }
            }

            # 保存结果到共享上下文
            self.shared_context.set(
                key=f"run_{run_id}_final_result",
                value=final_result,
                source_agent="AnalysisCrew"
            )

            return final_result

        except Exception as e:
            self.logger.error(f"自定义分析执行失败: {e}")
            raise

    def _generate_comprehensive_analysis(self, analysis_results: Dict[str, Any],
                                       run_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合分析结果

        Args:
            analysis_results: 各个分析的结果
            run_context: 运行上下文

        Returns:
            综合分析结果
        """
        ticker = run_context["ticker"]

        # 收集各个分析的结果
        signals = []
        confidences = []
        reasonings = []

        # 技术分析信号
        if "technical_analysis" in analysis_results:
            tech_result = analysis_results["technical_analysis"]
            signals.append(tech_result.get("signal", "neutral"))
            confidences.append(tech_result.get("confidence", 50))
            reasonings.append(tech_result.get("reasoning", ""))

        # 基本面分析信号
        if "fundamentals_analysis" in analysis_results:
            fund_result = analysis_results["fundamentals_analysis"]
            signals.append(fund_result.get("signal", "neutral"))
            confidences.append(fund_result.get("confidence", 50))
            reasonings.append(fund_result.get("reasoning", ""))

        # 情绪分析信号
        if "sentiment_analysis" in analysis_results:
            sentiment_result = analysis_results["sentiment_analysis"]
            signals.append(sentiment_result.get("signal", "neutral"))
            confidences.append(sentiment_result.get("confidence", 50))
            reasonings.append(sentiment_result.get("reasoning", ""))

        # 估值分析信号
        if "valuation_analysis" in analysis_results:
            valuation_result = analysis_results["valuation_analysis"]
            signals.append(valuation_result.get("signal", "neutral"))
            confidences.append(valuation_result.get("confidence", 50))
            reasonings.append(valuation_result.get("reasoning", ""))

        # 计算综合信号
        overall_signal = self._calculate_overall_signal(signals, confidences)
        overall_confidence = np.mean(confidences) if confidences else 50

        # 生成综合推理
        comprehensive_reasoning = self._generate_comprehensive_reasoning(
            analysis_results, overall_signal
        )

        # 风险评估
        risk_assessment = self._assess_overall_risk(analysis_results)

        return {
            "ticker": ticker,
            "overall_signal": overall_signal,
            "overall_confidence": overall_confidence,
            "comprehensive_reasoning": comprehensive_reasoning,
            "individual_signals": {
                "technical": analysis_results.get("technical_analysis", {}).get("signal"),
                "fundamentals": analysis_results.get("fundamentals_analysis", {}).get("signal"),
                "sentiment": analysis_results.get("sentiment_analysis", {}).get("signal"),
                "valuation": analysis_results.get("valuation_analysis", {}).get("signal")
            },
            "risk_assessment": risk_assessment,
            "investment_recommendation": self._generate_investment_recommendation(
                overall_signal, overall_confidence, risk_assessment
            ),
            "analysis_completeness": self._assess_analysis_completeness(analysis_results)
        }

    def _calculate_overall_signal(self, signals: List[str], confidences: List[float]) -> str:
        """计算综合信号"""
        if not signals:
            return "neutral"

        # 计算加权信号
        bullish_weight = 0
        bearish_weight = 0

        for signal, confidence in zip(signals, confidences):
            if signal in ["bullish", "strong_buy", "moderately_bullish"]:
                bullish_weight += confidence
            elif signal in ["bearish", "strong_sell", "moderately_bearish"]:
                bearish_weight += confidence

        # 确定综合信号
        if bullish_weight > bearish_weight * 1.5:
            return "bullish"
        elif bearish_weight > bullish_weight * 1.5:
            return "bearish"
        else:
            return "neutral"

    def _generate_comprehensive_reasoning(self, analysis_results: Dict[str, Any],
                                        overall_signal: str) -> str:
        """生成综合推理"""
        reasoning_parts = []

        # 技术面分析
        if "technical_analysis" in analysis_results:
            tech_content = analysis_results["technical_analysis"].get("content", {})
            tech_analysis = tech_content.get("analysis_result", {})
            trends = tech_analysis.get("trends", {})
            reasoning_parts.append(f"技术面: {trends.get('overall_trend', '未知')}趋势")

        # 基本面分析
        if "fundamentals_analysis" in analysis_results:
            fund_content = analysis_results["fundamentals_analysis"].get("content", {})
            fund_analysis = fund_content.get("analysis_result", {})
            quality_score = fund_analysis.get("quality_score", 0)
            reasoning_parts.append(f"基本面: 质量评分{quality_score:.1f}/100")

        # 情绪分析
        if "sentiment_analysis" in analysis_results:
            sentiment_content = analysis_results["sentiment_analysis"].get("content", {})
            sentiment_analysis = sentiment_content.get("analysis_result", {})
            overall_sentiment = sentiment_analysis.get("overall_sentiment", 0)
            reasoning_parts.append(f"情绪面: 综合情绪{overall_sentiment:.2f}")

        # 估值分析
        if "valuation_analysis" in analysis_results:
            valuation_content = analysis_results["valuation_analysis"].get("content", {})
            valuation_analysis = valuation_content.get("analysis_result", {})
            investment_metrics = valuation_analysis.get("investment_metrics", {})
            valuation_grade = investment_metrics.get("valuation_grade", "unknown")
            reasoning_parts.append(f"估值面: {valuation_grade}")

        # 综合判断
        reasoning_parts.append(f"综合判断: {overall_signal}")

        return "; ".join(reasoning_parts)

    def _assess_overall_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估整体风险"""
        risk_factors = []

        # 技术面风险
        if "technical_analysis" in analysis_results:
            tech_content = analysis_results["technical_analysis"].get("content", {})
            tech_analysis = tech_content.get("analysis_result", {})
            volatility = tech_analysis.get("volatility", {})
            if volatility.get("volatility_level") == "very_high":
                risk_factors.append("高波动性风险")

        # 基本面风险
        if "fundamentals_analysis" in analysis_results:
            fund_content = analysis_results["fundamentals_analysis"].get("content", {})
            fund_analysis = fund_content.get("analysis_result", {})
            risk_factors.extend(fund_analysis.get("risk_factors", []))

        # 情绪面风险
        if "sentiment_analysis" in analysis_results:
            sentiment_content = analysis_results["sentiment_analysis"].get("content", {})
            sentiment_analysis = sentiment_content.get("analysis_result", {})
            extreme_sentiment = sentiment_analysis.get("extreme_sentiment", {})
            if extreme_sentiment.get("is_extreme"):
                risk_factors.append("极端情绪风险")

        # 估值面风险
        if "valuation_analysis" in analysis_results:
            valuation_content = analysis_results["valuation_analysis"].get("content", {})
            valuation_analysis = valuation_content.get("analysis_result", {})
            investment_metrics = valuation_analysis.get("investment_metrics", {})
            if investment_metrics.get("valuation_grade") in ["overvalued", "significantly_overvalued"]:
                risk_factors.append("估值过高风险")

        # 风险等级
        if len(risk_factors) >= 3:
            risk_level = "high"
        elif len(risk_factors) >= 2:
            risk_level = "medium"
        elif len(risk_factors) >= 1:
            risk_level = "low"
        else:
            risk_level = "minimal"

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "risk_count": len(risk_factors)
        }

    def _generate_investment_recommendation(self, overall_signal: str,
                                          overall_confidence: float,
                                          risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """生成投资建议"""
        risk_level = risk_assessment.get("risk_level", "minimal")

        # 基于信号和风险等级生成建议
        if overall_signal == "bullish" and risk_level in ["minimal", "low"]:
            recommendation = "strong_buy"
            reasoning = "技术面、基本面、情绪面、估值面综合看好，风险较低"
        elif overall_signal == "bullish" and risk_level == "medium":
            recommendation = "buy"
            reasoning = "综合看好，但需注意中等风险"
        elif overall_signal == "neutral":
            recommendation = "hold"
            reasoning = "多空因素平衡，建议观望"
        elif overall_signal == "bearish" and risk_level in ["minimal", "low"]:
            recommendation = "sell"
            reasoning = "综合偏空，风险相对可控"
        else:
            recommendation = "strong_sell"
            reasoning = "综合偏空且风险较高，建议谨慎"

        return {
            "recommendation": recommendation,
            "confidence": overall_confidence,
            "reasoning": reasoning,
            "risk_level": risk_level,
            "time_horizon": "medium_term"  # 中期投资建议
        }

    def _assess_analysis_completeness(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估分析完整性"""
        expected_analyses = [
            "market_data", "news_data", "technical_analysis",
            "fundamentals_analysis", "sentiment_analysis", "valuation_analysis"
        ]

        completed_analyses = [key for key in expected_analyses if key in analysis_results]
        completeness = len(completed_analyses) / len(expected_analyses)

        return {
            "completeness_ratio": completeness,
            "completed_analyses": completed_analyses,
            "missing_analyses": [key for key in expected_analyses if key not in analysis_results],
            "overall_completeness": "complete" if completeness == 1.0 else "partial"
        }

    def _assess_overall_data_quality(self, analysis_results: Dict[str, Any]) -> str:
        """评估整体数据质量"""
        quality_scores = []

        for key, result in analysis_results.items():
            metadata = result.get("metadata", {})
            data_quality = metadata.get("data_quality", "unknown")
            if data_quality in ["excellent", "good", "fair", "poor"]:
                quality_map = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
                quality_scores.append(quality_map.get(data_quality, 1))

        if not quality_scores:
            return "unknown"

        avg_quality = np.mean(quality_scores)

        if avg_quality >= 3.5:
            return "excellent"
        elif avg_quality >= 2.5:
            return "good"
        elif avg_quality >= 1.5:
            return "fair"
        else:
            return "poor"

    def get_crew_status(self) -> Dict[str, Any]:
        """获取团队状态"""
        return {
            "crew_name": "AnalysisCrew",
            "agents_count": len(self.agents),
            "tasks_count": len(self.tasks),
            "agents": list(self.agents.keys()),
            "tasks": list(self.tasks.keys()),
            "is_initialized": self.crew is not None
        }

    def reset_crew(self):
        """重置团队"""
        self.crew = None
        self.logger.info("分析团队已重置")