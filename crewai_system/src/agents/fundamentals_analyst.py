"""
基本面分析师智能体
负责分析公司财务指标、基本面数据和经营状况
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent
from utils.data_processing import get_data_processor
from utils.shared_context import get_global_context


class FundamentalsAnalyst(BaseAgent):
    """基本面分析师智能体"""

    def __init__(self):
        super().__init__(
            role="基本面分析专家",
            goal="分析公司财务指标和基本面数据，评估公司内在价值和投资价值",
            backstory="""你是一位资深的财务分析师，精通公司基本面分析和财务估值。
            你擅长解读财务报表，分析盈利能力、偿债能力、运营效率等关键指标，
            并能够评估公司的行业地位、竞争优势和成长潜力。
            你的分析为投资决策提供重要的基本面依据。""",
            agent_name="FundamentalsAnalyst"
        )

        self._data_processor = get_data_processor()

    @property
    def data_processor(self):
        """获取数据处理器"""
        return getattr(self, '_data_processor', None)

    def process_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理基本面分析任务

        Args:
            task_context: 任务上下文，包含财务数据等信息

        Returns:
            基本面分析结果
        """
        self.log_execution_start("执行基本面分析")

        try:
            # 验证输入
            required_fields = ["ticker", "financial_metrics"]
            if not self.validate_input(task_context, required_fields):
                raise ValueError(f"缺少必需字段: {required_fields}")

            ticker = task_context["ticker"]
            financial_data = task_context["financial_metrics"]
            market_info = task_context.get("market_info", {})
            show_reasoning = task_context.get("show_reasoning", False)

            # 执行基本面分析
            analysis_result = self._perform_fundamentals_analysis(
                financial_data, market_info, ticker
            )

            # 生成投资建议
            investment_recommendation = self._generate_investment_recommendation(
                analysis_result
            )

            # 记录推理过程
            if show_reasoning:
                reasoning = self._generate_reasoning_report(analysis_result, investment_recommendation)
                self.log_reasoning(reasoning, "基本面分析推理过程")

            result = self.format_agent_output(
                content={
                    "analysis_result": analysis_result,
                    "investment_recommendation": investment_recommendation
                },
                signal=investment_recommendation["direction"],
                confidence=investment_recommendation["confidence"],
                reasoning=investment_recommendation["reasoning"],
                metadata={
                    "ticker": ticker,
                    "analysis_date": datetime.now().isoformat(),
                    "data_quality": self._assess_data_quality(financial_data)
                }
            )

            self.log_execution_complete(f"完成{ticker}的基本面分析")
            return result

        except Exception as e:
            self.log_execution_error(e, "基本面分析执行失败")
            raise

    def _perform_fundamentals_analysis(
        self,
        financial_data: Dict[str, Any],
        market_info: Dict[str, Any],
        ticker: str
    ) -> Dict[str, Any]:
        """
        执行基本面分析

        Args:
            financial_data: 财务数据
            market_info: 市场信息
            ticker: 股票代码

        Returns:
            基本面分析结果
        """
        self.logger.info(f"开始对{ticker}进行基本面分析")

        analysis_result = {
            "ticker": ticker,
            "analysis_timestamp": datetime.now().isoformat(),
            "profitability": {},
            "financial_health": {},
            "efficiency": {},
            "growth": {},
            "valuation": {},
            "quality_score": 0,
            "risk_factors": []
        }

        # 盈利能力分析
        analysis_result["profitability"] = self._analyze_profitability(financial_data)

        # 财务健康分析
        analysis_result["financial_health"] = self._analyze_financial_health(financial_data)

        # 运营效率分析
        analysis_result["efficiency"] = self._analyze_efficiency(financial_data)

        # 成长性分析
        analysis_result["growth"] = self._analyze_growth(financial_data)

        # 估值分析
        analysis_result["valuation"] = self._analyze_valuation(financial_data, market_info)

        # 计算综合质量评分
        analysis_result["quality_score"] = self._calculate_quality_score(analysis_result)

        # 识别风险因素
        analysis_result["risk_factors"] = self._identify_risk_factors(analysis_result)

        return analysis_result

    def _analyze_profitability(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析盈利能力"""
        profitability = {
            "metrics": {},
            "trend_analysis": {},
            "quality_assessment": {},
            "assessment": "unknown",
            "score": 0,
            "stability_score": 0
        }

        try:
            # 扩展盈利指标
            metrics = {
                # 基础盈利指标
                "roe": financial_data.get("roe"),  # 净资产收益率
                "roa": financial_data.get("roa"),  # 总资产收益率
                "gross_margin": financial_data.get("gross_margin"),  # 毛利率
                "net_margin": financial_data.get("net_margin"),  # 净利率
                "ebitda_margin": financial_data.get("ebitda_margin"),  # EBITDA利润率
                "operating_margin": financial_data.get("operating_margin"),  # 营业利润率
                # 高级盈利指标
                "return_on_capital": financial_data.get("return_on_capital"),  # 资本回报率
                "return_on_invested_capital": financial_data.get("return_on_invested_capital"),  # 投入资本回报率
                "free_cash_flow_margin": financial_data.get("free_cash_flow_margin"),  # 自由现金流利润率
                "asset_turnover": financial_data.get("asset_turnover"),  # 资产周转率
                "equity_multiplier": financial_data.get("equity_multiplier")  # 权益乘数
            }

            # 清理和验证数据
            for key, value in metrics.items():
                metrics[key] = self._clean_financial_value(value)

            profitability["metrics"] = metrics

            # 盈利能力评估
            score = 0
            stability_score = 0
            assessments = []

            # ROE评估（更细致的分级）
            roe = metrics.get("roe")
            if roe is not None:
                if roe > 25:
                    score += 30
                    stability_score += 20
                    assessments.append("卓越净资产收益率")
                elif roe > 18:
                    score += 25
                    stability_score += 15
                    assessments.append("优秀净资产收益率")
                elif roe > 12:
                    score += 18
                    stability_score += 10
                    assessments.append("良好净资产收益率")
                elif roe > 8:
                    score += 10
                    stability_score += 5
                    assessments.append("一般净资产收益率")
                elif roe > 5:
                    score += 5
                    assessments.append("较低净资产收益率")
                else:
                    assessments.append("较差净资产收益率")

            # ROIC评估
            roic = metrics.get("return_on_invested_capital")
            if roic is not None:
                if roic > 15:
                    score += 20
                    assessments.append("卓越投入资本回报率")
                elif roic > 10:
                    score += 15
                    assessments.append("良好投入资本回报率")
                elif roic > 5:
                    score += 8
                    assessments.append("一般投入资本回报率")

            # 毛利率评估（考虑行业特性）
            gross_margin = metrics.get("gross_margin")
            if gross_margin is not None:
                if gross_margin > 0.5:
                    score += 20
                    stability_score += 10
                    assessments.append("卓越毛利率")
                elif gross_margin > 0.35:
                    score += 15
                    stability_score += 8
                    assessments.append("优秀毛利率")
                elif gross_margin > 0.25:
                    score += 10
                    stability_score += 5
                    assessments.append("良好毛利率")
                elif gross_margin > 0.15:
                    score += 5
                    assessments.append("一般毛利率")
                else:
                    assessments.append("较低毛利率")

            # 净利率评估
            net_margin = metrics.get("net_margin")
            if net_margin is not None:
                if net_margin > 0.2:
                    score += 20
                    stability_score += 10
                    assessments.append("卓越净利率")
                elif net_margin > 0.12:
                    score += 15
                    stability_score += 8
                    assessments.append("优秀净利率")
                elif net_margin > 0.08:
                    score += 10
                    stability_score += 5
                    assessments.append("良好净利率")
                elif net_margin > 0.03:
                    score += 5
                    assessments.append("一般净利率")
                else:
                    assessments.append("较低净利率")

            # FCF利润率评估
            fcf_margin = metrics.get("free_cash_flow_margin")
            if fcf_margin is not None:
                if fcf_margin > 0.15:
                    score += 15
                    stability_score += 15
                    assessments.append("卓越现金流生成能力")
                elif fcf_margin > 0.08:
                    score += 10
                    stability_score += 10
                    assessments.append("良好现金流生成能力")
                elif fcf_margin > 0.03:
                    score += 5
                    stability_score += 5
                    assessments.append("一般现金流生成能力")

            # 杜邦分析（三要素分析）
            if all(metrics.get(key) is not None for key in ["roe", "net_margin", "asset_turnover", "equity_multiplier"]):
                dupont_analysis = {
                    "roe": roe,
                    "net_margin": metrics["net_margin"],
                    "asset_turnover": metrics["asset_turnover"],
                    "equity_multiplier": metrics["equity_multiplier"],
                    "analysis": "盈利能力来源于" +
                               ("高利润率" if metrics["net_margin"] > 0.1 else "一般利润率") + "、" +
                               ("高资产周转" if metrics["asset_turnover"] > 1.5 else "一般资产周转") + "和" +
                               ("高财务杠杆" if metrics["equity_multiplier"] > 2.0 else "适中财务杠杆")
                }
                profitability["dupont_analysis"] = dupont_analysis

            # 盈利质量评估
            profitability_quality = self._assess_profitability_quality(metrics)
            profitability["quality_assessment"] = profitability_quality

            # 盈利趋势分析（如果有历史数据）
            historical_data = financial_data.get("historical_data", {})
            if historical_data:
                trend_analysis = self._analyze_profitability_trend(historical_data, metrics)
                profitability["trend_analysis"] = trend_analysis
                # 根据趋势调整分数
                if trend_analysis.get("trend") == "improving":
                    score += 10
                    stability_score += 5
                elif trend_analysis.get("trend") == "declining":
                    score -= 10
                    stability_score -= 5

            # 综合评估
            total_score = min(score, 100)
            if total_score >= 75:
                assessment = "excellent"
            elif total_score >= 60:
                assessment = "good"
            elif total_score >= 40:
                assessment = "average"
            else:
                assessment = "poor"

            profitability["assessment"] = assessment
            profitability["score"] = total_score
            profitability["stability_score"] = min(stability_score, 50)
            profitability["details"] = assessments

        except Exception as e:
            self.logger.error(f"盈利能力分析失败: {e}")
            profitability["error"] = str(e)

        return profitability

    def _assess_profitability_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估盈利质量"""
        quality_score = 0
        quality_factors = []

        # 现金流一致性评估
        net_margin = metrics.get("net_margin", 0)
        fcf_margin = metrics.get("free_cash_flow_margin", 0)

        if fcf_margin is not None and net_margin is not None:
            if abs(fcf_margin - net_margin) < 0.02:  # 现金流与会计利润一致
                quality_score += 30
                quality_factors.append("现金流与会计利润高度一致")
            elif fcf_margin > net_margin * 0.8:  # 现金流较好
                quality_score += 20
                quality_factors.append("现金流转化良好")
            else:
                quality_factors.append("现金流转化需要关注")

        # ROE稳定性评估
        roe = metrics.get("roe")
        if roe is not None:
            if roe > 15 and roe < 25:  # 适中且稳定的ROE
                quality_score += 25
                quality_factors.append("ROE稳定且可持续")
            elif roe > 25:
                quality_score += 15
                quality_factors.append("高ROE但需关注可持续性")
            else:
                quality_factors.append("ROE水平有待提升")

        # 利润率稳定性评估
        gross_margin = metrics.get("gross_margin")
        if gross_margin is not None:
            if gross_margin > 0.3:  # 较高的毛利率通常意味着竞争优势
                quality_score += 20
                quality_factors.append("具备一定竞争优势")
            elif gross_margin > 0.2:
                quality_score += 10
                quality_factors.append("毛利率处于合理水平")
            else:
                quality_factors.append("毛利率偏低，竞争压力较大")

        # 运营效率评估
        asset_turnover = metrics.get("asset_turnover")
        if asset_turnover is not None:
            if asset_turnover > 1.0:  # 良好的资产使用效率
                quality_score += 15
                quality_factors.append("资产使用效率良好")
            elif asset_turnover > 0.5:
                quality_score += 8
                quality_factors.append("资产使用效率一般")
            else:
                quality_factors.append("资产使用效率有待提升")

        # 综合质量评估
        if quality_score >= 70:
            quality_grade = "excellent"
        elif quality_score >= 50:
            quality_grade = "good"
        elif quality_score >= 30:
            quality_grade = "average"
        else:
            quality_grade = "poor"

        return {
            "quality_score": quality_score,
            "quality_grade": quality_grade,
            "quality_factors": quality_factors
        }

    def _analyze_profitability_trend(self, historical_data: Dict[str, Any], current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析盈利趋势"""
        trend_analysis = {
            "trend": "stable",
            "consistency": "unknown",
            "volatility": "unknown",
            "growth_rate": 0
        }

        try:
            # 模拟历史数据分析（实际应用中应该使用真实历史数据）
            # 这里假设historical_data包含过去几年的财务数据
            if "roe_history" in historical_data:
                roe_history = historical_data["roe_history"]
                if len(roe_history) >= 3:
                    # 计算趋势
                    recent_roe = roe_history[-1]
                    avg_roe = np.mean(roe_history)

                    if recent_roe > avg_roe * 1.1:
                        trend_analysis["trend"] = "improving"
                    elif recent_roe < avg_roe * 0.9:
                        trend_analysis["trend"] = "declining"
                    else:
                        trend_analysis["trend"] = "stable"

                    # 计算波动性
                    roe_std = np.std(roe_history)
                    if roe_std < 3:
                        trend_analysis["volatility"] = "low"
                    elif roe_std < 8:
                        trend_analysis["volatility"] = "medium"
                    else:
                        trend_analysis["volatility"] = "high"

                    # 计算增长率
                    if len(roe_history) >= 2:
                        growth_rate = (roe_history[-1] - roe_history[0]) / roe_history[0] * 100
                        trend_analysis["growth_rate"] = growth_rate

        except Exception as e:
            self.logger.warning(f"盈利趋势分析失败: {e}")

        return trend_analysis

    def _analyze_financial_health(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析财务健康状况"""
        financial_health = {
            "metrics": {},
            "assessment": "unknown",
            "score": 0
        }

        try:
            # 关键财务健康指标
            metrics = {
                "debt_ratio": financial_data.get("debt_ratio"),  # 资产负债率
                "current_ratio": financial_data.get("current_ratio"),  # 流动比率
                "quick_ratio": financial_data.get("quick_ratio"),  # 速动比率
                "interest_coverage": financial_data.get("interest_coverage"),  # 利息保障倍数
                "cash_ratio": financial_data.get("cash_ratio")  # 现金比率
            }

            # 清理和验证数据
            for key, value in metrics.items():
                metrics[key] = self._clean_financial_value(value)

            financial_health["metrics"] = metrics

            # 财务健康评估
            score = 0
            assessments = []

            # 资产负债率评估
            debt_ratio = metrics.get("debt_ratio")
            if debt_ratio is not None:
                if debt_ratio < 0.3:
                    score += 25
                    assessments.append("优秀资产负债率")
                elif debt_ratio < 0.5:
                    score += 15
                    assessments.append("良好资产负债率")
                elif debt_ratio < 0.7:
                    score += 5
                    assessments.append("一般资产负债率")
                else:
                    assessments.append("较高资产负债率")

            # 流动比率评估
            current_ratio = metrics.get("current_ratio")
            if current_ratio is not None:
                if current_ratio > 2.0:
                    score += 20
                    assessments.append("优秀流动比率")
                elif current_ratio > 1.5:
                    score += 10
                    assessments.append("良好流动比率")
                elif current_ratio > 1.0:
                    score += 5
                    assessments.append("一般流动比率")
                else:
                    assessments.append("较低流动比率")

            # 速动比率评估
            quick_ratio = metrics.get("quick_ratio")
            if quick_ratio is not None:
                if quick_ratio > 1.5:
                    score += 15
                    assessments.append("优秀速动比率")
                elif quick_ratio > 1.0:
                    score += 10
                    assessments.append("良好速动比率")
                elif quick_ratio > 0.5:
                    score += 5
                    assessments.append("一般速动比率")
                else:
                    assessments.append("较低速动比率")

            # 综合评估
            if score >= 50:
                assessment = "excellent"
            elif score >= 30:
                assessment = "good"
            elif score >= 15:
                assessment = "average"
            else:
                assessment = "poor"

            financial_health["assessment"] = assessment
            financial_health["score"] = score
            financial_health["details"] = assessments

        except Exception as e:
            self.logger.error(f"财务健康分析失败: {e}")
            financial_health["error"] = str(e)

        return financial_health

    def _analyze_efficiency(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析运营效率"""
        efficiency = {
            "metrics": {},
            "assessment": "unknown",
            "score": 0
        }

        try:
            # 关键效率指标
            metrics = {
                "asset_turnover": financial_data.get("asset_turnover"),  # 总资产周转率
                "inventory_turnover": financial_data.get("inventory_turnover"),  # 存货周转率
                "receivables_turnover": financial_data.get("receivables_turnover"),  # 应收账款周转率
                "working_capital_turnover": financial_data.get("working_capital_turnover")  # 营运资金周转率
            }

            # 清理和验证数据
            for key, value in metrics.items():
                metrics[key] = self._clean_financial_value(value)

            efficiency["metrics"] = metrics

            # 运营效率评估
            score = 0
            assessments = []

            # 总资产周转率评估
            asset_turnover = metrics.get("asset_turnover")
            if asset_turnover is not None:
                if asset_turnover > 1.5:
                    score += 30
                    assessments.append("优秀总资产周转率")
                elif asset_turnover > 1.0:
                    score += 20
                    assessments.append("良好总资产周转率")
                elif asset_turnover > 0.5:
                    score += 10
                    assessments.append("一般总资产周转率")
                else:
                    assessments.append("较低总资产周转率")

            # 存货周转率评估
            inventory_turnover = metrics.get("inventory_turnover")
            if inventory_turnover is not None:
                if inventory_turnover > 8:
                    score += 20
                    assessments.append("优秀存货周转率")
                elif inventory_turnover > 5:
                    score += 10
                    assessments.append("良好存货周转率")
                elif inventory_turnover > 3:
                    score += 5
                    assessments.append("一般存货周转率")
                else:
                    assessments.append("较低存货周转率")

            # 应收账款周转率评估
            receivables_turnover = metrics.get("receivables_turnover")
            if receivables_turnover is not None:
                if receivables_turnover > 12:
                    score += 15
                    assessments.append("优秀应收账款周转率")
                elif receivables_turnover > 8:
                    score += 10
                    assessments.append("良好应收账款周转率")
                elif receivables_turnover > 4:
                    score += 5
                    assessments.append("一般应收账款周转率")
                else:
                    assessments.append("较低应收账款周转率")

            # 综合评估
            if score >= 60:
                assessment = "excellent"
            elif score >= 40:
                assessment = "good"
            elif score >= 20:
                assessment = "average"
            else:
                assessment = "poor"

            efficiency["assessment"] = assessment
            efficiency["score"] = score
            efficiency["details"] = assessments

        except Exception as e:
            self.logger.error(f"运营效率分析失败: {e}")
            efficiency["error"] = str(e)

        return efficiency

    def _analyze_growth(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析成长性"""
        growth = {
            "metrics": {},
            "assessment": "unknown",
            "score": 0
        }

        try:
            # 关键成长指标
            metrics = {
                "revenue_growth": financial_data.get("revenue_growth"),  # 营收增长率
                "net_income_growth": financial_data.get("net_income_growth"),  # 净利润增长率
                "eps_growth": financial_data.get("eps_growth"),  # 每股收益增长率
                "asset_growth": financial_data.get("asset_growth"),  # 总资产增长率
                "equity_growth": financial_data.get("equity_growth")  # 净资产增长率
            }

            # 清理和验证数据
            for key, value in metrics.items():
                metrics[key] = self._clean_financial_value(value)

            growth["metrics"] = metrics

            # 成长性评估
            score = 0
            assessments = []

            # 营收增长率评估
            revenue_growth = metrics.get("revenue_growth")
            if revenue_growth is not None:
                if revenue_growth > 0.25:
                    score += 30
                    assessments.append("优秀营收增长率")
                elif revenue_growth > 0.15:
                    score += 20
                    assessments.append("良好营收增长率")
                elif revenue_growth > 0.08:
                    score += 10
                    assessments.append("一般营收增长率")
                elif revenue_growth > 0:
                    score += 5
                    assessments.append("缓慢营收增长率")
                else:
                    assessments.append("营收负增长")

            # 净利润增长率评估
            net_income_growth = metrics.get("net_income_growth")
            if net_income_growth is not None:
                if net_income_growth > 0.30:
                    score += 30
                    assessments.append("优秀净利润增长率")
                elif net_income_growth > 0.20:
                    score += 20
                    assessments.append("良好净利润增长率")
                elif net_income_growth > 0.10:
                    score += 10
                    assessments.append("一般净利润增长率")
                elif net_income_growth > 0:
                    score += 5
                    assessments.append("缓慢净利润增长率")
                else:
                    assessments.append("净利润负增长")

            # 每股收益增长率评估
            eps_growth = metrics.get("eps_growth")
            if eps_growth is not None:
                if eps_growth > 0.20:
                    score += 20
                    assessments.append("优秀每股收益增长率")
                elif eps_growth > 0.15:
                    score += 10
                    assessments.append("良好每股收益增长率")
                elif eps_growth > 0.08:
                    score += 5
                    assessments.append("一般每股收益增长率")
                else:
                    assessments.append("较低每股收益增长率")

            # 综合评估
            if score >= 70:
                assessment = "excellent"
            elif score >= 50:
                assessment = "good"
            elif score >= 25:
                assessment = "average"
            else:
                assessment = "poor"

            growth["assessment"] = assessment
            growth["score"] = score
            growth["details"] = assessments

        except Exception as e:
            self.logger.error(f"成长性分析失败: {e}")
            growth["error"] = str(e)

        return growth

    def _analyze_valuation(self, financial_data: Dict[str, Any], market_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析估值水平"""
        valuation = {
            "metrics": {},
            "assessment": "unknown",
            "score": 0
        }

        try:
            # 关键估值指标
            metrics = {
                "pe_ratio": financial_data.get("pe_ratio"),  # 市盈率
                "pb_ratio": financial_data.get("pb_ratio"),  # 市净率
                "ps_ratio": financial_data.get("ps_ratio"),  # 市销率
                "ev_ebitda": financial_data.get("ev_ebitda"),  # EV/EBITDA
                "peg_ratio": financial_data.get("peg_ratio")  # PEG比率
            }

            # 清理和验证数据
            for key, value in metrics.items():
                metrics[key] = self._clean_financial_value(value)

            valuation["metrics"] = metrics

            # 估值水平评估
            score = 0
            assessments = []

            # 市盈率评估
            pe_ratio = metrics.get("pe_ratio")
            if pe_ratio is not None:
                if pe_ratio < 10:
                    score += 25
                    assessments.append("低市盈率，估值便宜")
                elif pe_ratio < 20:
                    score += 15
                    assessments.append("合理市盈率")
                elif pe_ratio < 30:
                    score += 5
                    assessments.append("偏高市盈率")
                else:
                    assessments.append("高市盈率，估值昂贵")

            # 市净率评估
            pb_ratio = metrics.get("pb_ratio")
            if pb_ratio is not None:
                if pb_ratio < 1.0:
                    score += 20
                    assessments.append("低市净率，估值便宜")
                elif pb_ratio < 2.0:
                    score += 10
                    assessments.append("合理市净率")
                elif pb_ratio < 4.0:
                    score += 5
                    assessments.append("偏高市净率")
                else:
                    assessments.append("高市净率，估值昂贵")

            # PEG比率评估
            peg_ratio = metrics.get("peg_ratio")
            if peg_ratio is not None:
                if peg_ratio < 0.5:
                    score += 20
                    assessments.append("优秀PEG比率")
                elif peg_ratio < 1.0:
                    score += 10
                    assessments.append("良好PEG比率")
                elif peg_ratio < 1.5:
                    score += 5
                    assessments.append("一般PEG比率")
                else:
                    assessments.append("较高PEG比率")

            # 综合评估
            if score >= 50:
                assessment = "undervalued"
            elif score >= 25:
                assessment = "fair_value"
            elif score >= 10:
                assessment = "overvalued"
            else:
                assessment = "expensive"

            valuation["assessment"] = assessment
            valuation["score"] = score
            valuation["details"] = assessments

        except Exception as e:
            self.logger.error(f"估值分析失败: {e}")
            valuation["error"] = str(e)

        return valuation

    def _clean_financial_value(self, value: Any) -> Optional[float]:
        """清理财务数据值"""
        if value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                return float(value)

            # 处理字符串格式的数值
            if isinstance(value, str):
                # 移除百分号
                cleaned = str(value).replace('%', '').replace(',', '').strip()
                return float(cleaned) if cleaned else None

            return None
        except (ValueError, TypeError):
            return None

    def _calculate_quality_score(self, analysis_result: Dict[str, Any]) -> float:
        """计算综合质量评分"""
        scores = []

        # 各维度评分
        dimensions = ["profitability", "financial_health", "efficiency", "growth", "valuation"]

        for dimension in dimensions:
            dim_data = analysis_result.get(dimension, {})
            score = dim_data.get("score", 0)
            scores.append(score)

        # 加权平均
        weights = [0.25, 0.2, 0.15, 0.2, 0.2]  # 各维度权重

        if scores:
            weighted_score = sum(score * weight for score, weight in zip(scores, weights))
            return min(weighted_score, 100)  # 最大100分
        else:
            return 0

    def _identify_risk_factors(self, analysis_result: Dict[str, Any]) -> List[str]:
        """识别风险因素"""
        risk_factors = []

        # 盈利能力风险
        profitability = analysis_result.get("profitability", {})
        if profitability.get("assessment") == "poor":
            risk_factors.append("盈利能力较弱")

        # 财务健康风险
        financial_health = analysis_result.get("financial_health", {})
        if financial_health.get("assessment") == "poor":
            risk_factors.append("财务状况较差")

        # 高负债风险
        debt_ratio = financial_health.get("metrics", {}).get("debt_ratio")
        if debt_ratio and debt_ratio > 0.7:
            risk_factors.append("负债率较高")

        # 流动性风险
        current_ratio = financial_health.get("metrics", {}).get("current_ratio")
        if current_ratio and current_ratio < 1.0:
            risk_factors.append("流动性风险")

        # 成长性风险
        growth = analysis_result.get("growth", {})
        revenue_growth = growth.get("metrics", {}).get("revenue_growth")
        if revenue_growth is not None and revenue_growth < 0:
            risk_factors.append("营收负增长")

        # 估值风险
        valuation = analysis_result.get("valuation", {})
        if valuation.get("assessment") == "expensive":
            risk_factors.append("估值过高")

        return risk_factors

    def _generate_investment_recommendation(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成投资建议"""
        quality_score = analysis_result.get("quality_score", 0)
        risk_factors = analysis_result.get("risk_factors", [])
        risk_count = len(risk_factors)

        # 基于质量评分和风险因素生成建议
        if quality_score >= 70 and risk_count <= 1:
            direction = "bullish"
            confidence = min(quality_score * 0.9, 95)
            reasoning = "基本面优秀，投资价值较高"
        elif quality_score >= 50 and risk_count <= 2:
            direction = "moderately_bullish"
            confidence = min(quality_score * 0.8, 85)
            reasoning = "基本面良好，具有投资价值"
        elif quality_score >= 30 and risk_count <= 3:
            direction = "neutral"
            confidence = min(quality_score * 0.7, 75)
            reasoning = "基本面一般，需谨慎考虑"
        elif quality_score >= 20:
            direction = "moderately_bearish"
            confidence = min((100 - quality_score) * 0.8, 80)
            reasoning = "基本面较差，存在投资风险"
        else:
            direction = "bearish"
            confidence = min((100 - quality_score) * 0.9, 90)
            reasoning = "基本面很差，投资风险较高"

        return {
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "quality_score": quality_score,
            "risk_factors": risk_factors,
            "risk_count": risk_count
        }

    def _generate_reasoning_report(self, analysis_result: Dict[str, Any],
                                 investment_recommendation: Dict[str, Any]) -> str:
        """生成推理报告"""
        report = []

        # 质量评分
        quality_score = analysis_result.get("quality_score", 0)
        report.append(f"综合质量评分: {quality_score:.1f}/100")

        # 各维度分析
        dimensions = {
            "profitability": "盈利能力",
            "financial_health": "财务健康",
            "efficiency": "运营效率",
            "growth": "成长性",
            "valuation": "估值水平"
        }

        for dim_key, dim_name in dimensions.items():
            dim_data = analysis_result.get(dim_key, {})
            assessment = dim_data.get("assessment", "unknown")
            score = dim_data.get("score", 0)
            report.append(f"{dim_name}: {assessment} (评分: {score})")

        # 风险因素
        risk_factors = analysis_result.get("risk_factors", [])
        if risk_factors:
            report.append(f"风险因素: {', '.join(risk_factors)}")
        else:
            report.append("风险因素: 未发现明显风险")

        # 投资建议
        direction = investment_recommendation.get("direction", "neutral")
        confidence = investment_recommendation.get("confidence", 50)
        reasoning = investment_recommendation.get("reasoning", "")

        report.append(f"投资建议: {direction}")
        report.append(f"建议置信度: {confidence:.1f}%")
        report.append(f"主要理由: {reasoning}")

        return "\n".join(report)

    def _assess_data_quality(self, financial_data: Dict[str, Any]) -> str:
        """评估数据质量"""
        if not financial_data:
            return "no_data"

        # 计算有效数据点
        valid_metrics = 0
        total_metrics = 0

        important_metrics = [
            "roe", "roa", "debt_ratio", "current_ratio", "revenue_growth",
            "net_income_growth", "pe_ratio", "pb_ratio"
        ]

        for metric in important_metrics:
            total_metrics += 1
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None:
                    valid_metrics += 1

        # 数据完整性评估
        if total_metrics == 0:
            return "no_data"

        completeness = valid_metrics / total_metrics

        if completeness >= 0.8:
            return "excellent"
        elif completeness >= 0.6:
            return "good"
        elif completeness >= 0.4:
            return "fair"
        else:
            return "poor"

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker", "financial_metrics"]