"""
估值分析师智能体
负责进行公司估值分析，计算内在价值和评估投资价值
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent
from utils.data_processing import get_data_processor
from utils.shared_context import get_global_context


class ValuationAnalyst(BaseAgent):
    """估值分析师智能体"""

    def __init__(self):
        super().__init__(
            role="估值分析专家",
            goal="进行公司估值分析，计算内在价值和评估投资价值",
            backstory="""你是一位资深的企业估值专家，精通各种估值方法和模型。
            你能够运用DCF估值、相对估值、资产估值等多种方法对公司进行全面估值，
            并结合市场环境、行业特点和公司发展前景，提供专业的投资建议。
            你的分析为投资决策提供重要的估值参考依据。""",
            agent_name="ValuationAnalyst"
        )

        self._data_processor = get_data_processor()

    @property
    def data_processor(self):
        """获取数据处理器"""
        return getattr(self, '_data_processor', None)

    def process_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理估值分析任务

        Args:
            task_context: 任务上下文，包含财务数据、市场数据等信息

        Returns:
            估值分析结果
        """
        self.log_execution_start("执行估值分析")

        try:
            # 验证输入
            required_fields = ["ticker", "financial_metrics", "market_info"]
            if not self.validate_input(task_context, required_fields):
                raise ValueError(f"缺少必需字段: {required_fields}")

            ticker = task_context["ticker"]
            financial_data = task_context["financial_metrics"]
            market_info = task_context["market_info"]
            show_reasoning = task_context.get("show_reasoning", False)

            # 执行估值分析
            analysis_result = self._perform_valuation_analysis(
                financial_data, market_info, ticker
            )

            # 生成投资建议
            investment_recommendation = self._generate_investment_recommendation(
                analysis_result, market_info
            )

            # 记录推理过程
            if show_reasoning:
                reasoning = self._generate_reasoning_report(analysis_result, investment_recommendation)
                self.log_reasoning(reasoning, "估值分析推理过程")

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
                    "valuation_methods": ["dcf", "relative", "asset"],
                    "data_quality": self._assess_valuation_data_quality(financial_data, market_info)
                }
            )

            self.log_execution_complete(f"完成{ticker}的估值分析")
            return result

        except Exception as e:
            self.log_execution_error(e, "估值分析执行失败")
            raise

    def _perform_valuation_analysis(
        self,
        financial_data: Dict[str, Any],
        market_info: Dict[str, Any],
        ticker: str
    ) -> Dict[str, Any]:
        """
        执行估值分析

        Args:
            financial_data: 财务数据
            market_info: 市场信息
            ticker: 股票代码

        Returns:
            估值分析结果
        """
        self.logger.info(f"开始对{ticker}进行估值分析")

        analysis_result = {
            "ticker": ticker,
            "analysis_timestamp": datetime.now().isoformat(),
            "dcf_valuation": {},
            "relative_valuation": {},
            "asset_valuation": {},
            "dividend_valuation": {},
            "residual_income_valuation": {},
            "real_options_valuation": {},
            "sotp_valuation": {},
            "eva_valuation": {},
            "comprehensive_valuation": {},
            "valuation_range": {},
            "investment_metrics": {}
        }

        # DCF估值
        analysis_result["dcf_valuation"] = self._perform_advanced_dcf_valuation(financial_data)

        # 相对估值
        analysis_result["relative_valuation"] = self._perform_enhanced_relative_valuation(
            financial_data, market_info
        )

        # 资产估值
        analysis_result["asset_valuation"] = self._perform_enhanced_asset_valuation(financial_data)

        # 股利折现模型
        analysis_result["dividend_valuation"] = self._perform_dividend_valuation(financial_data)

        # 剩余收益模型
        analysis_result["residual_income_valuation"] = self._perform_residual_income_valuation(financial_data)

        # 实物期权估值
        analysis_result["real_options_valuation"] = self._perform_real_options_valuation(financial_data)

        # 分部估值法
        analysis_result["sotp_valuation"] = self._perform_sotp_valuation(financial_data)

        # 经济增加值模型
        analysis_result["eva_valuation"] = self._perform_eva_valuation(financial_data)

        # 综合估值
        analysis_result["comprehensive_valuation"] = self._calculate_enhanced_comprehensive_valuation(
            analysis_result
        )

        # 估值区间
        analysis_result["valuation_range"] = self._calculate_enhanced_valuation_range(
            analysis_result, market_info
        )

        # 投资指标
        analysis_result["investment_metrics"] = self._calculate_enhanced_investment_metrics(
            analysis_result, market_info
        )

        return analysis_result

    def _perform_advanced_dcf_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行高级DCF估值（多阶段模型）"""
        dcf_result = {
            "intrinsic_value": 0,
            "per_share_value": 0,
            "two_stage_dcf": {},
            "three_stage_dcf": {},
            "gordon_growth_model": {},
            "assumptions": {},
            "calculation_details": {},
            "sensitivity_analysis": {},
            "scenario_analysis": {},
            "confidence": 0
        }

        try:
            # 获取关键财务指标
            fcf = self._clean_financial_value(financial_data.get("free_cash_flow"))
            revenue = self._clean_financial_value(financial_data.get("revenue"))
            growth_rate = self._clean_financial_value(financial_data.get("revenue_growth"))
            roic = self._clean_financial_value(financial_data.get("roic"))  # 投入资本回报率
            wacc = self._clean_financial_value(financial_data.get("wacc")) or 0.10  # 加权平均资本成本

            # 增长阶段分析
            current_growth = growth_rate if growth_rate else self._estimate_growth_rate(financial_data)
            high_growth_years = min(5, max(2, int(10 / (current_growth + 0.01))))  # 高增长年限
            transition_years = 3  # 过渡年限

            # DCF参数假设
            assumptions = {
                "high_growth_rate": current_growth,
                "high_growth_years": high_growth_years,
                "transition_growth_rate": (current_growth + 0.03) / 2,  # 过渡增长率
                "transition_years": transition_years,
                "terminal_growth_rate": 0.03,  # 终端增长率
                "discount_rate": wacc,
                "margin_of_safety": 0.20,
                "tax_rate": 0.25  # 企业所得税率
            }

            # 如果没有足够数据，使用行业平均值
            if fcf is None or fcf <= 0:
                fcf = revenue * 0.15 if revenue else 1000000  # 假设FCF占收入15%

            # 两阶段DCF模型
            two_stage_result = self._two_stage_dcf(fcf, assumptions)
            dcf_result["two_stage_dcf"] = two_stage_result

            # 三阶段DCF模型
            three_stage_result = self._three_stage_dcf(fcf, assumptions)
            dcf_result["three_stage_dcf"] = three_stage_result

            # Gordon增长模型
            gordon_result = self._gordon_growth_model(fcf, assumptions, financial_data)
            dcf_result["gordon_growth_model"] = gordon_result

            # 综合DCF价值
            dcf_values = [
                two_stage_result.get("per_share_value", 0),
                three_stage_result.get("per_share_value", 0),
                gordon_result.get("per_share_value", 0)
            ]
            valid_values = [v for v in dcf_values if v > 0]

            if valid_values:
                # 加权平均（三阶段模型权重最高）
                weights = [0.25, 0.5, 0.25]  # 两阶段、三阶段、Gordon
                weighted_value = sum(v * w for v, w in zip(valid_values, weights[:len(valid_values)]))
                dcf_result["intrinsic_value"] = weighted_value * 100000000  # 假设1亿股
                dcf_result["per_share_value"] = weighted_value

            # 置信度评估
            data_quality = self._assess_enhanced_dcf_data_quality(financial_data)
            confidence = data_quality * 0.85

            dcf_result.update({
                "assumptions": assumptions,
                "calculation_details": {
                    "base_fcf": fcf,
                    "fcf_yield": fcf / revenue if revenue else 0,
                    "growth_profile": {
                        "high_growth": f"{current_growth:.1%} for {high_growth_years} years",
                        "transition": f"{assumptions['transition_growth_rate']:.1%} for {transition_years} years",
                        "terminal": f"{assumptions['terminal_growth_rate']:.1%} perpetual"
                    }
                },
                "confidence": confidence
            })

            # 敏感性分析
            dcf_result["sensitivity_analysis"] = self._perform_enhanced_dcf_sensitivity_analysis(
                fcf, assumptions
            )

            # 情景分析
            dcf_result["scenario_analysis"] = self._perform_dcf_scenario_analysis(
                fcf, assumptions
            )

        except Exception as e:
            self.logger.error(f"高级DCF估值失败: {e}")
            dcf_result["error"] = str(e)

        return dcf_result

    def _two_stage_dcf(self, fcf: float, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """两阶段DCF模型"""
        result = {
            "intrinsic_value": 0,
            "per_share_value": 0,
            "stage1_value": 0,
            "terminal_value": 0,
            "fcf_projections": []
        }

        try:
            # 第一阶段：高增长期
            stage1_pv = 0
            fcf_projections = []
            current_fcf = fcf

            for year in range(assumptions["high_growth_years"]):
                current_fcf = current_fcf * (1 + assumptions["high_growth_rate"])
                fcf_projections.append(current_fcf)
                pv = current_fcf / ((1 + assumptions["discount_rate"]) ** (year + 1))
                stage1_pv += pv

            # 终端价值
            terminal_growth_rate = assumptions["terminal_growth_rate"]
            discount_rate = assumptions["discount_rate"]
            terminal_fcf = fcf_projections[-1] * (1 + terminal_growth_rate)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
            terminal_pv = terminal_value / ((1 + discount_rate) ** assumptions["high_growth_years"])

            # 总价值
            total_value = stage1_pv + terminal_pv
            shares_outstanding = 100000000
            per_share_value = total_value / shares_outstanding

            result.update({
                "intrinsic_value": total_value,
                "per_share_value": per_share_value,
                "stage1_value": stage1_pv,
                "terminal_value": terminal_pv,
                "fcf_projections": fcf_projections
            })

        except Exception as e:
            self.logger.error(f"两阶段DCF计算失败: {e}")
            result["error"] = str(e)

        return result

    def _three_stage_dcf(self, fcf: float, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """三阶段DCF模型"""
        result = {
            "intrinsic_value": 0,
            "per_share_value": 0,
            "stage1_value": 0,
            "stage2_value": 0,
            "terminal_value": 0,
            "fcf_projections": []
        }

        try:
            total_pv = 0
            fcf_projections = []
            current_fcf = fcf

            # 第一阶段：高增长期
            for year in range(assumptions["high_growth_years"]):
                current_fcf = current_fcf * (1 + assumptions["high_growth_rate"])
                fcf_projections.append(current_fcf)
                pv = current_fcf / ((1 + assumptions["discount_rate"]) ** (year + 1))
                total_pv += pv

            stage1_pv = total_pv

            # 第二阶段：过渡期（增长率线性下降）
            for year in range(assumptions["transition_years"]):
                year_offset = assumptions["high_growth_years"] + year
                # 线性递减增长率
                growth_rate = assumptions["transition_growth_rate"] - \
                             (assumptions["transition_growth_rate"] - assumptions["terminal_growth_rate"]) * \
                             (year / assumptions["transition_years"])

                current_fcf = current_fcf * (1 + growth_rate)
                fcf_projections.append(current_fcf)
                pv = current_fcf / ((1 + assumptions["discount_rate"]) ** (year_offset + 1))
                total_pv += pv

            stage2_pv = total_pv - stage1_pv

            # 终端价值
            terminal_growth_rate = assumptions["terminal_growth_rate"]
            discount_rate = assumptions["discount_rate"]
            terminal_fcf = fcf_projections[-1] * (1 + terminal_growth_rate)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
            terminal_pv = terminal_value / ((1 + discount_rate) ** (assumptions["high_growth_years"] + assumptions["transition_years"]))

            # 总价值
            total_value = total_pv + terminal_pv
            shares_outstanding = 100000000
            per_share_value = total_value / shares_outstanding

            result.update({
                "intrinsic_value": total_value,
                "per_share_value": per_share_value,
                "stage1_value": stage1_pv,
                "stage2_value": stage2_pv,
                "terminal_value": terminal_pv,
                "fcf_projections": fcf_projections
            })

        except Exception as e:
            self.logger.error(f"三阶段DCF计算失败: {e}")
            result["error"] = str(e)

        return result

    def _gordon_growth_model(self, fcf: float, assumptions: Dict[str, Any], financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Gordon增长模型（单阶段）"""
        result = {
            "intrinsic_value": 0,
            "per_share_value": 0,
            "sustainable_growth_rate": 0
        }

        try:
            # 计算可持续增长率
            if financial_data:
                roe = self._clean_financial_value(financial_data.get("roe")) or 0.15
            else:
                roe = 0.15  # 默认净资产收益率

            retention_rate = 0.6  # 留存收益率假设
            sustainable_growth_rate = roe * retention_rate

            # 使用可持续增长率和终端增长率的较小值
            growth_rate = min(sustainable_growth_rate, assumptions["terminal_growth_rate"])

            # Gordon增长模型公式
            next_fcf = fcf * (1 + growth_rate)
            discount_rate = assumptions["discount_rate"]

            if discount_rate > growth_rate:
                intrinsic_value = next_fcf / (discount_rate - growth_rate)
                shares_outstanding = 100000000
                per_share_value = intrinsic_value / shares_outstanding

                result.update({
                    "intrinsic_value": intrinsic_value,
                    "per_share_value": per_share_value,
                    "sustainable_growth_rate": sustainable_growth_rate,
                    "used_growth_rate": growth_rate
                })
            else:
                result["error"] = "折现率必须大于增长率"

        except Exception as e:
            self.logger.error(f"Gordon增长模型计算失败: {e}")
            result["error"] = str(e)

        return result

    def _estimate_growth_rate(self, financial_data: Dict[str, Any]) -> float:
        """估算增长率"""
        try:
            # 尝试从历史数据估算增长率
            revenue_growth = self._clean_financial_value(financial_data.get("revenue_growth"))
            eps_growth = self._clean_financial_value(financial_data.get("eps_growth"))

            if revenue_growth and eps_growth:
                # 使用收入增长和EPS增长的几何平均
                return (revenue_growth * eps_growth) ** 0.5
            elif revenue_growth:
                return revenue_growth
            elif eps_growth:
                return eps_growth
            else:
                # 使用行业平均增长率
                return 0.08  # 8%的默认增长率

        except Exception:
            return 0.08  # 默认增长率

    def _assess_enhanced_dcf_data_quality(self, financial_data: Dict[str, Any]) -> float:
        """评估增强DCF数据质量"""
        quality_score = 0
        max_score = 0

        # 基础指标
        basic_metrics = ["free_cash_flow", "revenue", "revenue_growth"]
        for metric in basic_metrics:
            max_score += 1
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None and cleaned > 0:
                    quality_score += 1

        # 高级指标
        advanced_metrics = ["roic", "wacc", "roe", "eps_growth"]
        for metric in advanced_metrics:
            max_score += 0.5
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None:
                    quality_score += 0.5

        # 盈利能力指标
        profitability_metrics = ["net_margin", "operating_margin", "return_on_assets"]
        for metric in profitability_metrics:
            max_score += 0.3
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None and cleaned > 0:
                    quality_score += 0.3

        return min(1.0, quality_score / max_score) if max_score > 0 else 0

    def _perform_relative_valuation(self, financial_data: Dict[str, Any],
                                 market_info: Dict[str, Any]) -> Dict[str, Any]:
        """执行相对估值"""
        relative_result = {
            "peer_comparison": {},
            "multiples_valuation": {},
            "industry_average": {},
            "relative_value": 0
        }

        try:
            # 获取估值倍数
            pe_ratio = self._clean_financial_value(financial_data.get("pe_ratio"))
            pb_ratio = self._clean_financial_value(financial_data.get("pb_ratio"))
            ps_ratio = self._clean_financial_value(financial_data.get("ps_ratio"))
            ev_ebitda = self._clean_financial_value(financial_data.get("ev_ebitda"))

            # 行业平均倍数（假设值）
            industry_averages = {
                "pe_ratio": 20.0,
                "pb_ratio": 2.5,
                "ps_ratio": 3.0,
                "ev_ebitda": 12.0
            }

            # 当前股价
            current_price = market_info.get("current_price", 0)

            # 计算相对价值
            relative_values = []

            # PE相对估值
            if pe_ratio and industry_averages["pe_ratio"]:
                pe_relative_value = current_price * (industry_averages["pe_ratio"] / pe_ratio)
                relative_values.append(("PE相对估值", pe_relative_value))

            # PB相对估值
            if pb_ratio and industry_averages["pb_ratio"]:
                pb_relative_value = current_price * (industry_averages["pb_ratio"] / pb_ratio)
                relative_values.append(("PB相对估值", pb_relative_value))

            # PS相对估值
            if ps_ratio and industry_averages["ps_ratio"]:
                ps_relative_value = current_price * (industry_averages["ps_ratio"] / ps_ratio)
                relative_values.append(("PS相对估值", ps_relative_value))

            # 计算平均相对价值
            if relative_values:
                values = [value for _, value in relative_values]
                avg_relative_value = np.mean(values) if values else current_price
            else:
                avg_relative_value = current_price

            relative_result.update({
                "peer_comparison": {
                    "current_pe": pe_ratio,
                    "industry_pe": industry_averages["pe_ratio"],
                    "pe_discount": (industry_averages["pe_ratio"] - pe_ratio) / industry_averages["pe_ratio"] if pe_ratio else 0,
                    "current_pb": pb_ratio,
                    "industry_pb": industry_averages["pb_ratio"],
                    "pb_discount": (industry_averages["pb_ratio"] - pb_ratio) / industry_averages["pb_ratio"] if pb_ratio else 0
                },
                "multiples_valuation": relative_values,
                "industry_average": industry_averages,
                "relative_value": avg_relative_value
            })

        except Exception as e:
            self.logger.error(f"相对估值失败: {e}")
            relative_result["error"] = str(e)

        return relative_result

    def _perform_asset_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行资产估值"""
        asset_result = {
            "net_asset_value": 0,
            "liquidation_value": 0,
            "replacement_value": 0,
            "asset_based_value": 0
        }

        try:
            # 获取资产相关数据
            total_assets = self._clean_financial_value(financial_data.get("total_assets"))
            total_liabilities = self._clean_financial_value(financial_data.get("total_liabilities"))
            net_assets = total_assets - total_liabilities if total_assets and total_liabilities else 0

            # 净资产价值
            asset_result["net_asset_value"] = net_assets

            # 清算价值（假设为净资产的70%）
            asset_result["liquidation_value"] = net_assets * 0.7

            # 重置价值（假设为净资产的120%）
            asset_result["replacement_value"] = net_assets * 1.2

            # 资产基础估值（平均三种价值）
            asset_values = [v for v in [net_assets, asset_result["liquidation_value"], asset_result["replacement_value"]] if v > 0]
            asset_result["asset_based_value"] = np.mean(asset_values) if asset_values else net_assets

        except Exception as e:
            self.logger.error(f"资产估值失败: {e}")
            asset_result["error"] = str(e)

        return asset_result

    def _calculate_comprehensive_valuation(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合估值"""
        comprehensive_result = {
            "weighted_value": 0,
            "valuation_method_weights": {},
            "method_contributions": {},
            "confidence_score": 0
        }

        try:
            # 获取各种估值结果
            dcf_val = analysis_result.get("dcf_valuation", {})
            relative_val = analysis_result.get("relative_valuation", {})
            asset_val = analysis_result.get("asset_valuation", {})

            # 各方法的估值
            dcf_value = dcf_val.get("per_share_value", 0)
            relative_value = relative_val.get("relative_value", 0)
            asset_value = asset_val.get("asset_based_value", 0)

            # 方法权重（根据置信度动态调整）
            dcf_confidence = dcf_val.get("confidence", 0)
            relative_confidence = 0.6  # 相对估值中等置信度
            asset_confidence = 0.4  # 资产估值较低置信度

            # 计算权重
            total_confidence = dcf_confidence + relative_confidence + asset_confidence
            if total_confidence > 0:
                dcf_weight = dcf_confidence / total_confidence
                relative_weight = relative_confidence / total_confidence
                asset_weight = asset_confidence / total_confidence
            else:
                dcf_weight = relative_weight = asset_weight = 1/3

            # 计算加权价值
            valid_values = []
            weights = []

            if dcf_value > 0:
                valid_values.append(dcf_value)
                weights.append(dcf_weight)

            if relative_value > 0:
                valid_values.append(relative_value)
                weights.append(relative_weight)

            if asset_value > 0:
                valid_values.append(asset_value)
                weights.append(asset_weight)

            if valid_values:
                # 归一化权重
                normalized_weights = [w / sum(weights) for w in weights]
                weighted_value = sum(v * w for v, w in zip(valid_values, normalized_weights))
            else:
                weighted_value = 0

            comprehensive_result.update({
                "weighted_value": weighted_value,
                "valuation_method_weights": {
                    "dcf_weight": dcf_weight,
                    "relative_weight": relative_weight,
                    "asset_weight": asset_weight
                },
                "method_contributions": {
                    "dcf_contribution": dcf_value * dcf_weight if dcf_value > 0 else 0,
                    "relative_contribution": relative_value * relative_weight if relative_value > 0 else 0,
                    "asset_contribution": asset_value * asset_weight if asset_value > 0 else 0
                },
                "confidence_score": max(dcf_confidence, relative_confidence, asset_confidence)
            })

        except Exception as e:
            self.logger.error(f"综合估值计算失败: {e}")
            comprehensive_result["error"] = str(e)

        return comprehensive_result

    def _calculate_valuation_range(self, analysis_result: Dict[str, Any],
                                 market_info: Dict[str, Any]) -> Dict[str, Any]:
        """计算估值区间"""
        valuation_range = {
            "lower_bound": 0,
            "upper_bound": 0,
            "fair_value": 0,
            "safety_margin": 0
        }

        try:
            comprehensive_val = analysis_result.get("comprehensive_valuation", {})
            weighted_value = comprehensive_val.get("weighted_value", 0)

            if weighted_value > 0:
                # 设置估值区间（±20%）
                lower_bound = weighted_value * 0.8
                upper_bound = weighted_value * 1.2
                fair_value = weighted_value

                # 当前股价
                current_price = market_info.get("current_price", 0)

                # 计算安全边际
                if current_price > 0:
                    safety_margin = (fair_value - current_price) / current_price * 100
                else:
                    safety_margin = 0

                valuation_range.update({
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "fair_value": fair_value,
                    "safety_margin": safety_margin,
                    "current_price": current_price
                })

        except Exception as e:
            self.logger.error(f"估值区间计算失败: {e}")
            valuation_range["error"] = str(e)

        return valuation_range

    def _calculate_investment_metrics(self, analysis_result: Dict[str, Any],
                                     market_info: Dict[str, Any]) -> Dict[str, Any]:
        """计算投资指标"""
        investment_metrics = {
            "upside_potential": 0,
            "downside_risk": 0,
            "risk_reward_ratio": 0,
            "valuation_grade": "unknown"
        }

        try:
            valuation_range = analysis_result.get("valuation_range", {})
            fair_value = valuation_range.get("fair_value", 0)
            lower_bound = valuation_range.get("lower_bound", 0)
            upper_bound = valuation_range.get("upper_bound", 0)

            current_price = market_info.get("current_price", 0)

            if current_price > 0 and fair_value > 0:
                # 上涨潜力
                upside_potential = (fair_value - current_price) / current_price * 100

                # 下跌风险
                downside_risk = (current_price - lower_bound) / current_price * 100

                # 风险收益比
                if downside_risk > 0:
                    risk_reward_ratio = upside_potential / downside_risk
                else:
                    risk_reward_ratio = 0

                # 估值评级
                if upside_potential > 50:
                    valuation_grade = "significantly_undervalued"
                elif upside_potential > 20:
                    valuation_grade = "undervalued"
                elif upside_potential > -20:
                    valuation_grade = "fairly_valued"
                elif upside_potential > -40:
                    valuation_grade = "overvalued"
                else:
                    valuation_grade = "significantly_overvalued"

                investment_metrics.update({
                    "upside_potential": upside_potential,
                    "downside_risk": downside_risk,
                    "risk_reward_ratio": risk_reward_ratio,
                    "valuation_grade": valuation_grade
                })

        except Exception as e:
            self.logger.error(f"投资指标计算失败: {e}")
            investment_metrics["error"] = str(e)

        return investment_metrics

    def _clean_financial_value(self, value: Any) -> Optional[float]:
        """清理财务数据值"""
        if value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                return float(value)

            # 处理字符串格式的数值
            if isinstance(value, str):
                cleaned = str(value).replace('%', '').replace(',', '').strip()
                return float(cleaned) if cleaned else None

            return None
        except (ValueError, TypeError):
            return None

    def _assess_dcf_data_quality(self, financial_data: Dict[str, Any]) -> float:
        """评估DCF数据质量"""
        quality_score = 0

        # 检查关键数据
        key_metrics = ["free_cash_flow", "revenue", "revenue_growth"]
        available_metrics = 0

        for metric in key_metrics:
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None and cleaned > 0:
                    available_metrics += 1

        # 计算质量分数
        quality_score = available_metrics / len(key_metrics)

        return quality_score

    def _perform_dcf_sensitivity_analysis(self, fcf: float,
                                        assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """执行DCF敏感性分析"""
        sensitivity = {}

        try:
            base_growth = assumptions["growth_rate_5yr"]
            base_discount = assumptions["discount_rate"]

            # 增长率敏感性
            growth_scenarios = [-0.02, 0, 0.02]  # ±2%的变化
            growth_effects = []

            for growth_change in growth_scenarios:
                new_growth = base_growth + growth_change
                growth_effects.append({
                    "growth_rate": new_growth,
                    "value_change": growth_change / base_growth * 100
                })

            # 折现率敏感性
            discount_scenarios = [-0.01, 0, 0.01]  # ±1%的变化
            discount_effects = []

            for discount_change in discount_scenarios:
                new_discount = base_discount + discount_change
                discount_effects.append({
                    "discount_rate": new_discount,
                    "value_change": -discount_change / base_discount * 100
                })

            sensitivity.update({
                "growth_sensitivity": growth_effects,
                "discount_sensitivity": discount_effects
            })

        except Exception as e:
            self.logger.error(f"D敏感性分析失败: {e}")
            sensitivity["error"] = str(e)

        return sensitivity

    def _perform_dividend_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """股利折现模型估值"""
        ddm_result = {
            "intrinsic_value": 0,
            "per_share_value": 0,
            "gordon_ddm": {},
            "multi_stage_ddm": {},
            "assumptions": {},
            "confidence": 0
        }

        try:
            # 获取股利相关数据
            dividend_per_share = self._clean_financial_value(financial_data.get("dividend_per_share"))
            payout_ratio = self._clean_financial_value(financial_data.get("payout_ratio")) or 0.4
            eps = self._clean_financial_value(financial_data.get("eps"))
            roe = self._clean_financial_value(financial_data.get("roe")) or 0.12
            discount_rate = self._clean_financial_value(financial_data.get("wacc")) or 0.10

            # 估算股利增长率
            retention_rate = 1 - payout_ratio
            sustainable_growth_rate = roe * retention_rate

            # 如果没有股利数据，估算
            if dividend_per_share is None or dividend_per_share <= 0:
                if eps:
                    dividend_per_share = eps * payout_ratio
                else:
                    dividend_per_share = 1.0  # 默认股利

            # DDM参数
            assumptions = {
                "current_dividend": dividend_per_share,
                "payout_ratio": payout_ratio,
                "retention_rate": retention_rate,
                "roe": roe,
                "sustainable_growth_rate": sustainable_growth_rate,
                "discount_rate": discount_rate,
                "high_growth_years": 5,
                "terminal_growth_rate": 0.03
            }

            # Gordon增长模型DDM
            gordon_ddm = self._gordon_ddm(dividend_per_share, sustainable_growth_rate, discount_rate)
            ddm_result["gordon_ddm"] = gordon_ddm

            # 多阶段DDM
            multi_stage_ddm = self._multi_stage_ddm(dividend_per_share, assumptions)
            ddm_result["multi_stage_ddm"] = multi_stage_ddm

            # 综合DDM价值
            ddm_values = [
                gordon_ddm.get("per_share_value", 0),
                multi_stage_ddm.get("per_share_value", 0)
            ]
            valid_values = [v for v in ddm_values if v > 0]

            if valid_values:
                # 加权平均（多阶段权重更高）
                weighted_value = valid_values[0] * 0.4 + valid_values[-1] * 0.6 if len(valid_values) > 1 else valid_values[0]
                ddm_result["intrinsic_value"] = weighted_value
                ddm_result["per_share_value"] = weighted_value

            # 置信度评估
            data_quality = self._assess_ddm_data_quality(financial_data)
            confidence = data_quality * 0.75  # DDM模型置信度较低

            ddm_result.update({
                "assumptions": assumptions,
                "confidence": confidence
            })

        except Exception as e:
            self.logger.error(f"股利折现模型估值失败: {e}")
            ddm_result["error"] = str(e)

        return ddm_result

    def _gordon_ddm(self, dividend: float, growth_rate: float, discount_rate: float) -> Dict[str, Any]:
        """Gordon增长DDM模型"""
        result = {"intrinsic_value": 0, "per_share_value": 0}

        try:
            if discount_rate > growth_rate:
                next_dividend = dividend * (1 + growth_rate)
                intrinsic_value = next_dividend / (discount_rate - growth_rate)

                result.update({
                    "intrinsic_value": intrinsic_value,
                    "per_share_value": intrinsic_value
                })
            else:
                result["error"] = "折现率必须大于增长率"

        except Exception as e:
            self.logger.error(f"Gordon DDM计算失败: {e}")
            result["error"] = str(e)

        return result

    def _multi_stage_ddm(self, dividend: float, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """多阶段DDM模型"""
        result = {"intrinsic_value": 0, "per_share_value": 0, "dividend_projections": []}

        try:
            total_pv = 0
            dividend_projections = []
            current_dividend = dividend

            # 第一阶段：高增长期
            high_growth_rate = assumptions["sustainable_growth_rate"]
            discount_rate = assumptions["discount_rate"]

            for year in range(assumptions["high_growth_years"]):
                current_dividend = current_dividend * (1 + high_growth_rate)
                dividend_projections.append(current_dividend)
                pv = current_dividend / ((1 + discount_rate) ** (year + 1))
                total_pv += pv

            # 终端价值
            terminal_growth_rate = assumptions["terminal_growth_rate"]
            terminal_dividend = dividend_projections[-1] * (1 + terminal_growth_rate)
            terminal_value = terminal_dividend / (discount_rate - terminal_growth_rate)
            terminal_pv = terminal_value / ((1 + discount_rate) ** assumptions["high_growth_years"])

            # 总价值
            total_value = total_pv + terminal_pv

            result.update({
                "intrinsic_value": total_value,
                "per_share_value": total_value,
                "dividend_projections": dividend_projections
            })

        except Exception as e:
            self.logger.error(f"多阶段DDM计算失败: {e}")
            result["error"] = str(e)

        return result

    def _perform_residual_income_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """剩余收益模型估值"""
        rim_result = {
            "intrinsic_value": 0,
            "per_share_value": 0,
            "book_value_contribution": 0,
            "residual_income_contribution": 0,
            "assumptions": {},
            "confidence": 0
        }

        try:
            # 获取剩余收益相关数据
            book_value_per_share = self._clean_financial_value(financial_data.get("book_value_per_share"))
            roe = self._clean_financial_value(financial_data.get("roe")) or 0.12
            cost_of_equity = self._clean_financial_value(financial_data.get("cost_of_equity")) or 0.10
            eps = self._clean_financial_value(financial_data.get("eps"))

            # 如果没有账面价值，估算
            if book_value_per_share is None or book_value_per_share <= 0:
                if eps and roe:
                    book_value_per_share = eps / roe
                else:
                    book_value_per_share = 10.0  # 默认账面价值

            # RIM参数
            assumptions = {
                "book_value_per_share": book_value_per_share,
                "roe": roe,
                "cost_of_equity": cost_of_equity,
                "forecast_years": 5,
                "terminal_roe": 0.10,
                "terminal_growth_rate": 0.03
            }

            # 计算剩余收益
            residual_income_pv = 0
            current_eps = eps or (book_value_per_share * roe)

            for year in range(assumptions["forecast_years"]):
                # 剩余收益 = (ROE - 权益成本) × 期初账面价值
                residual_income = (roe - cost_of_equity) * book_value_per_share
                residual_income_pv += residual_income / ((1 + cost_of_equity) ** (year + 1))

                # 更新账面价值
                book_value_per_share += current_eps * (1 - 0.4)  # 假设40%分红率
                current_eps = book_value_per_share * roe

            # 终端剩余收益价值
            terminal_residual_income = (assumptions["terminal_roe"] - cost_of_equity) * book_value_per_share
            terminal_growth_rate = assumptions["terminal_growth_rate"]
            terminal_value = terminal_residual_income / (cost_of_equity - terminal_growth_rate)
            terminal_pv = terminal_value / ((1 + cost_of_equity) ** assumptions["forecast_years"])

            # 总内在价值
            initial_book_value = assumptions["book_value_per_share"]
            intrinsic_value = initial_book_value + residual_income_pv + terminal_pv

            rim_result.update({
                "intrinsic_value": intrinsic_value,
                "per_share_value": intrinsic_value,
                "book_value_contribution": initial_book_value,
                "residual_income_contribution": residual_income_pv + terminal_pv,
                "assumptions": assumptions
            })

            # 置信度评估
            data_quality = self._assess_rim_data_quality(financial_data)
            confidence = data_quality * 0.80

            rim_result["confidence"] = confidence

        except Exception as e:
            self.logger.error(f"剩余收益模型估值失败: {e}")
            rim_result["error"] = str(e)

        return rim_result

    def _perform_real_options_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """实物期权估值"""
        rov_result = {
            "option_value": 0,
            "per_share_option_value": 0,
            "growth_options": {},
            "expansion_options": {},
            "abandonment_options": {},
            "confidence": 0
        }

        try:
            # 获取期权相关数据
            current_value = self._clean_financial_value(financial_data.get("market_cap")) or 1000000000
            volatility = self._clean_financial_value(financial_data.get("volatility")) or 0.3
            risk_free_rate = self._clean_financial_value(financial_data.get("risk_free_rate")) or 0.03

            # 期权参数
            option_params = {
                "current_value": current_value,
                "volatility": volatility,
                "risk_free_rate": risk_free_rate,
                "time_to_maturity": 5,  # 5年期期权
                "investment_cost": current_value * 0.3,  # 30%的投资成本
                "expansion_factor": 2.0  # 扩张倍数
            }

            # 增长期权价值
            growth_option = self._calculate_growth_option(option_params)
            rov_result["growth_options"] = growth_option

            # 扩张期权价值
            expansion_option = self._calculate_expansion_option(option_params)
            rov_result["expansion_options"] = expansion_option

            # 放弃期权价值
            abandonment_option = self._calculate_abandonment_option(option_params)
            rov_result["abandonment_options"] = abandonment_option

            # 总期权价值
            total_option_value = (
                growth_option.get("option_value", 0) +
                expansion_option.get("option_value", 0) +
                abandonment_option.get("option_value", 0)
            )

            shares_outstanding = 100000000
            per_share_option_value = total_option_value / shares_outstanding

            rov_result.update({
                "option_value": total_option_value,
                "per_share_option_value": per_share_option_value
            })

            # 置信度评估（实物期权估值不确定性较高）
            confidence = 0.60  # 固定置信度
            rov_result["confidence"] = confidence

        except Exception as e:
            self.logger.error(f"实物期权估值失败: {e}")
            rov_result["error"] = str(e)

        return rov_result

    def _calculate_growth_option(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """计算增长期权价值（使用Black-Scholes模型简化版）"""
        result = {"option_value": 0}

        try:
            S = params["current_value"] * 0.2  # 增长期权价值占企业价值的20%
            K = params["investment_cost"]
            T = params["time_to_maturity"]
            r = params["risk_free_rate"]
            sigma = params["volatility"]

            # 简化的Black-Scholes计算
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # 使用近似计算避免复杂的正态分布函数
            option_value = S * max(0, d1 / 10) - K * np.exp(-r * T) * max(0, d2 / 10)
            option_value = max(0, option_value)

            result["option_value"] = option_value

        except Exception as e:
            self.logger.error(f"增长期权计算失败: {e}")
            result["error"] = str(e)

        return result

    def _calculate_expansion_option(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """计算扩张期权价值"""
        result = {"option_value": 0}

        try:
            # 扩张期权通常比增长期权更有价值
            S = params["current_value"] * 0.5  # 扩张期权价值占企业价值的50%
            K = params["investment_cost"] * params["expansion_factor"]
            T = params["time_to_maturity"] * 0.8  # 扩张期权的执行期通常较短
            r = params["risk_free_rate"]
            sigma = params["volatility"] * 1.2  # 扩张期权波动性更大

            # 简化计算
            expansion_value = S * 0.3 * np.exp(-r * T)  # 扩张价值为当前价值的30%现值
            result["option_value"] = max(0, expansion_value)

        except Exception as e:
            self.logger.error(f"扩张期权计算失败: {e}")
            result["error"] = str(e)

        return result

    def _calculate_abandonment_option(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """计算放弃期权价值"""
        result = {"option_value": 0}

        try:
            # 放弃期权是一种看跌期权
            salvage_value = params["current_value"] * 0.6  # 清算价值为企业价值的60%
            abandonment_cost = params["investment_cost"] * 0.1  # 放弃成本
            T = params["time_to_maturity"]
            r = params["risk_free_rate"]

            # 简化计算：放弃期权价值 = 清算价值现值 - 放弃成本
            abandonment_value = salvage_value * np.exp(-r * T) - abandonment_cost
            result["option_value"] = max(0, abandonment_value)

        except Exception as e:
            self.logger.error(f"放弃期权计算失败: {e}")
            result["error"] = str(e)

        return result

    def _perform_sotp_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """分部估值法"""
        sotp_result = {
            "total_value": 0,
            "per_share_value": 0,
            "business_segments": {},
            "consolidation_adjustment": 0,
            "confidence": 0
        }

        try:
            # 模拟业务分部（在实际应用中，这些数据应该从财报中获取）
            business_segments = {
                "core_business": {
                    "revenue_contribution": 0.6,
                    "ebitda_margin": 0.15,
                    "growth_rate": 0.08,
                    "multiple": 12.0
                },
                "emerging_business": {
                    "revenue_contribution": 0.25,
                    "ebitda_margin": 0.10,
                    "growth_rate": 0.15,
                    "multiple": 15.0
                },
                "other_business": {
                    "revenue_contribution": 0.15,
                    "ebitda_margin": 0.05,
                    "growth_rate": 0.03,
                    "multiple": 8.0
                }
            }

            total_revenue = self._clean_financial_value(financial_data.get("revenue")) or 1000000000

            segment_values = {}

            for segment_name, segment_data in business_segments.items():
                segment_revenue = total_revenue * segment_data["revenue_contribution"]
                segment_ebitda = segment_revenue * segment_data["ebitda_margin"]
                segment_value = segment_ebitda * segment_data["multiple"]

                segment_values[segment_name] = {
                    "revenue": segment_revenue,
                    "ebitda": segment_ebitda,
                    "value": segment_value,
                    "valuation_multiple": segment_data["multiple"]
                }

            # 总分部价值
            total_segment_value = sum(data["value"] for data in segment_values.values())

            # 协同效应调整（±10%）
            synergy_adjustment = total_segment_value * 0.05  # 假设5%的协同效应
            consolidated_value = total_segment_value + synergy_adjustment

            # 公司层面的调整（负债、现金等）
            net_debt = self._clean_financial_value(financial_data.get("net_debt")) or 0
            cash_and_equivalents = self._clean_financial_value(financial_data.get("cash_and_equivalents")) or 0
            enterprise_value_adjustment = net_debt - cash_and_equivalents

            final_value = consolidated_value + enterprise_value_adjustment

            shares_outstanding = 100000000
            per_share_value = final_value / shares_outstanding

            sotp_result.update({
                "total_value": final_value,
                "per_share_value": per_share_value,
                "business_segments": segment_values,
                "consolidation_adjustment": synergy_adjustment,
                "enterprise_value_adjustment": enterprise_value_adjustment
            })

            # 置信度评估
            confidence = 0.70  # 分部估值法的置信度
            sotp_result["confidence"] = confidence

        except Exception as e:
            self.logger.error(f"分部估值法失败: {e}")
            sotp_result["error"] = str(e)

        return sotp_result

    def _perform_eva_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """经济增加值模型估值"""
        eva_result = {
            "intrinsic_value": 0,
            "per_share_value": 0,
            "current_eva": 0,
            "eva_projections": [],
            "eva_pv": 0,
            "assumptions": {},
            "confidence": 0
        }

        try:
            # 获取EVA相关数据
            nopat = self._clean_financial_value(financial_data.get("nopat"))  # 税后净营业利润
            total_capital = self._clean_financial_value(financial_data.get("total_capital"))
            wacc = self._clean_financial_value(financial_data.get("wacc")) or 0.10
            roic = self._clean_financial_value(financial_data.get("roic")) or 0.12

            # 如果没有NOPAT，估算
            if nopat is None:
                ebit = self._clean_financial_value(financial_data.get("ebit")) or 0
                tax_rate = 0.25
                nopat = ebit * (1 - tax_rate)

            # 如果没有总资本，估算
            if total_capital is None:
                total_assets = self._clean_financial_value(financial_data.get("total_assets")) or 0
                current_liabilities = self._clean_financial_value(financial_data.get("current_liabilities")) or 0
                total_capital = total_assets - current_liabilities

            # EVA参数
            assumptions = {
                "nopat": nopat,
                "total_capital": total_capital,
                "wacc": wacc,
                "roic": roic,
                "forecast_years": 5,
                "terminal_roic": 0.10,
                "terminal_growth_rate": 0.03
            }

            # 计算当前EVA
            current_eva = nopat - (total_capital * wacc)
            eva_result["current_eva"] = current_eva

            # 预测未来EVA
            eva_projections = []
            eva_pv = 0
            current_capital = total_capital

            for year in range(assumptions["forecast_years"]):
                # 预测NOPAT增长
                projected_nopat = nopat * (1 + roic) ** year
                # 预测资本增长
                projected_capital = current_capital * (1 + roic) ** year
                # 计算EVA
                projected_eva = projected_nopat - (projected_capital * wacc)
                # 计算EVA现值
                eva_pv_component = projected_eva / ((1 + wacc) ** (year + 1))
                eva_pv += eva_pv_component

                eva_projections.append({
                    "year": year + 1,
                    "nopat": projected_nopat,
                    "capital": projected_capital,
                    "eva": projected_eva,
                    "pv": eva_pv_component
                })

            # 终端EVA价值
            terminal_nopat = nopat * (1 + assumptions["terminal_roic"]) ** assumptions["forecast_years"]
            terminal_capital = current_capital * (1 + assumptions["terminal_roic"]) ** assumptions["forecast_years"]
            terminal_eva = terminal_nopat - (terminal_capital * wacc)

            terminal_eva_value = terminal_eva / (wacc - assumptions["terminal_growth_rate"])
            terminal_pv = terminal_eva_value / ((1 + wacc) ** assumptions["forecast_years"])

            # 总EVA价值
            total_eva_value = eva_pv + terminal_pv

            # 加上初始资本价值
            intrinsic_value = total_capital + total_eva_value

            shares_outstanding = 100000000
            per_share_value = intrinsic_value / shares_outstanding

            eva_result.update({
                "intrinsic_value": intrinsic_value,
                "per_share_value": per_share_value,
                "eva_projections": eva_projections,
                "eva_pv": eva_pv,
                "terminal_eva_pv": terminal_pv,
                "assumptions": assumptions
            })

            # 置信度评估
            data_quality = self._assess_eva_data_quality(financial_data)
            confidence = data_quality * 0.85

            eva_result["confidence"] = confidence

        except Exception as e:
            self.logger.error(f"EVA模型估值失败: {e}")
            eva_result["error"] = str(e)

        return eva_result

    def _perform_enhanced_relative_valuation(self, financial_data: Dict[str, Any],
                                          market_info: Dict[str, Any]) -> Dict[str, Any]:
        """增强相对估值"""
        # 首先保持原有相对估值逻辑
        relative_result = self._perform_relative_valuation(financial_data, market_info)

        # 添加增强功能
        try:
            # 动态行业倍数
            industry_multiples = self._get_dynamic_industry_multiples(financial_data)

            # 可比公司分析
            peer_analysis = self._perform_peer_analysis(financial_data, market_info)

            # 修正的相对估值
            adjusted_relative_value = self._calculate_adjusted_relative_value(
                financial_data, market_info, industry_multiples
            )

            relative_result.update({
                "industry_multiples": industry_multiples,
                "peer_analysis": peer_analysis,
                "adjusted_relative_value": adjusted_relative_value
            })

        except Exception as e:
            self.logger.error(f"增强相对估值失败: {e}")
            relative_result["enhanced_error"] = str(e)

        return relative_result

    def _perform_enhanced_asset_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """增强资产估值"""
        # 首先保持原有资产估值逻辑
        asset_result = self._perform_asset_valuation(financial_data)

        # 添加增强功能
        try:
            # 无形资产估值
            intangible_assets = self._valuate_intangible_assets(financial_data)

            # 品牌价值估值
            brand_value = self._valuate_brand(financial_data)

            # 客户关系价值
            customer_value = self._valuate_customer_relationships(financial_data)

            # 技术价值
            technology_value = self._valuate_technology(financial_data)

            enhanced_asset_value = asset_result.get("asset_based_value", 0) + \
                                 intangible_assets + brand_value + customer_value + technology_value

            asset_result.update({
                "intangible_assets_value": intangible_assets,
                "brand_value": brand_value,
                "customer_value": customer_value,
                "technology_value": technology_value,
                "enhanced_asset_value": enhanced_asset_value
            })

        except Exception as e:
            self.logger.error(f"增强资产估值失败: {e}")
            asset_result["enhanced_error"] = str(e)

        return asset_result

    def _valuate_intangible_assets(self, financial_data: Dict[str, Any]) -> float:
        """估值无形资产"""
        try:
            # 使用收入倍数法估算无形资产价值
            revenue = self._clean_financial_value(financial_data.get("revenue")) or 0
            intangible_ratio = 0.15  # 无形资产占收入的比例
            return revenue * intangible_ratio
        except Exception:
            return 0

    def _valuate_brand(self, financial_data: Dict[str, Any]) -> float:
        """估值品牌价值"""
        try:
            # 使用收入倍数法估算品牌价值
            revenue = self._clean_financial_value(financial_data.get("revenue")) or 0
            brand_ratio = 0.10  # 品牌价值占收入的比例
            return revenue * brand_ratio
        except Exception:
            return 0

    def _valuate_customer_relationships(self, financial_data: Dict[str, Any]) -> float:
        """估值客户关系"""
        try:
            # 使用收入倍数法估算客户关系价值
            revenue = self._clean_financial_value(financial_data.get("revenue")) or 0
            customer_ratio = 0.05  # 客户关系价值占收入的比例
            return revenue * customer_ratio
        except Exception:
            return 0

    def _valuate_technology(self, financial_data: Dict[str, Any]) -> float:
        """估值技术价值"""
        try:
            # 使用研发支出倍数法估算技术价值
            rd_expense = self._clean_financial_value(financial_data.get("rd_expense")) or 0
            tech_multiple = 3.0  # 研发支出的倍数
            return rd_expense * tech_multiple
        except Exception:
            return 0

    def _get_dynamic_industry_multiples(self, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """获取动态行业倍数"""
        # 基于公司特征调整行业倍数
        growth_rate = self._clean_financial_value(financial_data.get("revenue_growth")) or 0.08
        profitability = self._clean_financial_value(financial_data.get("net_margin")) or 0.10

        # 根据成长性和盈利能力调整倍数
        growth_adjustment = 1.0 + (growth_rate - 0.08) * 2  # 增长率每高于基准1%，倍数调整2%
        profitability_adjustment = 1.0 + (profitability - 0.10) * 3  # 利润率每高于基准1%，倍数调整3%

        base_multiples = {
            "pe_ratio": 20.0,
            "pb_ratio": 2.5,
            "ps_ratio": 3.0,
            "ev_ebitda": 12.0
        }

        return {
            multiple: base_value * growth_adjustment * profitability_adjustment
            for multiple, base_value in base_multiples.items()
        }

    def _perform_peer_analysis(self, financial_data: Dict[str, Any],
                             market_info: Dict[str, Any]) -> Dict[str, Any]:
        """进行可比公司分析"""
        # 模拟可比公司数据
        peer_comparison = {
            "current_pe": self._clean_financial_value(financial_data.get("pe_ratio")) or 0,
            "current_pb": self._clean_financial_value(financial_data.get("pb_ratio")) or 0,
            "current_ps": self._clean_financial_value(financial_data.get("ps_ratio")) or 0,
            "peer_avg_pe": 22.0,
            "peer_avg_pb": 2.8,
            "peer_avg_ps": 3.2,
            "peer_median_pe": 20.5,
            "peer_median_pb": 2.6,
            "peer_median_ps": 3.0,
            "valuation_discount": {}
        }

        # 计算估值折价/溢价
        for metric in ["pe", "pb", "ps"]:
            current = peer_comparison[f"current_{metric}"]
            peer_avg = peer_comparison[f"peer_avg_{metric}"]
            if current > 0 and peer_avg > 0:
                discount = (current - peer_avg) / peer_avg * 100
                peer_comparison["valuation_discount"][f"{metric}_discount"] = discount

        return peer_comparison

    def _calculate_adjusted_relative_value(self, financial_data: Dict[str, Any],
                                         market_info: Dict[str, Any],
                                         industry_multiples: Dict[str, float]) -> float:
        """计算调整后的相对估值"""
        try:
            current_price = market_info.get("current_price", 0)
            if current_price == 0:
                return 0

            # 获取当前倍数
            current_pe = self._clean_financial_value(financial_data.get("pe_ratio")) or 0
            current_pb = self._clean_financial_value(financial_data.get("pb_ratio")) or 0
            current_ps = self._clean_financial_value(financial_data.get("ps_ratio")) or 0

            # 计算各方法的调整后价值
            adjusted_values = []

            if current_pe > 0:
                pe_adjusted_value = current_price * (industry_multiples["pe_ratio"] / current_pe)
                adjusted_values.append(pe_adjusted_value)

            if current_pb > 0:
                pb_adjusted_value = current_price * (industry_multiples["pb_ratio"] / current_pb)
                adjusted_values.append(pb_adjusted_value)

            if current_ps > 0:
                ps_adjusted_value = current_price * (industry_multiples["ps_ratio"] / current_ps)
                adjusted_values.append(ps_adjusted_value)

            # 计算平均调整价值
            return np.mean(adjusted_values) if adjusted_values else current_price

        except Exception:
            return market_info.get("current_price", 0)

    def _assess_ddm_data_quality(self, financial_data: Dict[str, Any]) -> float:
        """评估DDM数据质量"""
        quality_score = 0
        max_score = 0

        ddm_metrics = ["dividend_per_share", "payout_ratio", "eps", "roe"]
        for metric in ddm_metrics:
            max_score += 1
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None and cleaned > 0:
                    quality_score += 1

        return quality_score / max_score if max_score > 0 else 0

    def _assess_rim_data_quality(self, financial_data: Dict[str, Any]) -> float:
        """评估RIM数据质量"""
        quality_score = 0
        max_score = 0

        rim_metrics = ["book_value_per_share", "roe", "eps", "cost_of_equity"]
        for metric in rim_metrics:
            max_score += 1
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None and cleaned > 0:
                    quality_score += 1

        return quality_score / max_score if max_score > 0 else 0

    def _assess_eva_data_quality(self, financial_data: Dict[str, Any]) -> float:
        """评估EVA数据质量"""
        quality_score = 0
        max_score = 0

        eva_metrics = ["nopat", "total_capital", "wacc", "roic"]
        for metric in eva_metrics:
            max_score += 1
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None and cleaned > 0:
                    quality_score += 1

        return quality_score / max_score if max_score > 0 else 0

    def _perform_enhanced_dcf_sensitivity_analysis(self, fcf: float,
                                                 assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """执行增强DCF敏感性分析"""
        sensitivity = {}

        try:
            # 增长率敏感性
            growth_scenarios = [-0.03, -0.015, 0, 0.015, 0.03]  # ±3%的变化
            growth_effects = []

            for growth_change in growth_scenarios:
                new_growth = max(0.01, assumptions["high_growth_rate"] + growth_change)
                growth_effects.append({
                    "growth_rate": new_growth,
                    "value_change": growth_change / assumptions["high_growth_rate"] * 100 if assumptions["high_growth_rate"] > 0 else 0
                })

            # 折现率敏感性
            discount_scenarios = [-0.02, -0.01, 0, 0.01, 0.02]  # ±2%的变化
            discount_effects = []

            for discount_change in discount_scenarios:
                new_discount = max(0.02, assumptions["discount_rate"] + discount_change)
                discount_effects.append({
                    "discount_rate": new_discount,
                    "value_change": -discount_change / assumptions["discount_rate"] * 100
                })

            # 终端增长率敏感性
            terminal_scenarios = [-0.01, -0.005, 0, 0.005, 0.01]  # ±1%的变化
            terminal_effects = []

            for terminal_change in terminal_scenarios:
                new_terminal = max(0.01, assumptions["terminal_growth_rate"] + terminal_change)
                terminal_effects.append({
                    "terminal_growth_rate": new_terminal,
                    "value_change": terminal_change / assumptions["terminal_growth_rate"] * 100 if assumptions["terminal_growth_rate"] > 0 else 0
                })

            sensitivity.update({
                "growth_sensitivity": growth_effects,
                "discount_sensitivity": discount_effects,
                "terminal_sensitivity": terminal_effects
            })

        except Exception as e:
            self.logger.error(f"增强DCF敏感性分析失败: {e}")
            sensitivity["error"] = str(e)

        return sensitivity

    def _perform_dcf_scenario_analysis(self, fcf: float,
                                      assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """执行DCF情景分析"""
        scenarios = {}

        try:
            # 乐观情景
            optimistic_assumptions = assumptions.copy()
            optimistic_assumptions.update({
                "high_growth_rate": min(0.25, assumptions["high_growth_rate"] * 1.3),
                "transition_growth_rate": min(0.15, assumptions["transition_growth_rate"] * 1.2),
                "discount_rate": max(0.06, assumptions["discount_rate"] * 0.9)
            })
            optimistic_value = self._calculate_scenario_value(fcf, optimistic_assumptions)

            # 基准情景
            base_value = self._calculate_scenario_value(fcf, assumptions)

            # 悲观情景
            pessimistic_assumptions = assumptions.copy()
            pessimistic_assumptions.update({
                "high_growth_rate": max(0.02, assumptions["high_growth_rate"] * 0.7),
                "transition_growth_rate": max(0.02, assumptions["transition_growth_rate"] * 0.8),
                "discount_rate": min(0.15, assumptions["discount_rate"] * 1.1)
            })
            pessimistic_value = self._calculate_scenario_value(fcf, pessimistic_assumptions)

            scenarios.update({
                "optimistic": {
                    "value": optimistic_value,
                    "upside": (optimistic_value - base_value) / base_value * 100 if base_value > 0 else 0
                },
                "base": {
                    "value": base_value
                },
                "pessimistic": {
                    "value": pessimistic_value,
                    "downside": (base_value - pessimistic_value) / base_value * 100 if base_value > 0 else 0
                }
            })

        except Exception as e:
            self.logger.error(f"DCF情景分析失败: {e}")
            scenarios["error"] = str(e)

        return scenarios

    def _calculate_scenario_value(self, fcf: float, assumptions: Dict[str, Any]) -> float:
        """计算情景价值"""
        try:
            # 使用三阶段模型计算情景价值
            three_stage_result = self._three_stage_dcf(fcf, assumptions)
            return three_stage_result.get("intrinsic_value", 0)
        except Exception:
            return 0

    def _calculate_enhanced_comprehensive_valuation(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """计算增强综合估值"""
        comprehensive_result = {
            "weighted_value": 0,
            "valuation_method_weights": {},
            "method_contributions": {},
            "confidence_score": 0,
            "valuation_diversity": {},
            "model_consensus": {}
        }

        try:
            # 获取各种估值结果
            valuation_methods = [
                ("dcf", analysis_result.get("dcf_valuation", {}), 0.25),
                ("relative", analysis_result.get("relative_valuation", {}), 0.20),
                ("asset", analysis_result.get("asset_valuation", {}), 0.15),
                ("dividend", analysis_result.get("dividend_valuation", {}), 0.10),
                ("residual_income", analysis_result.get("residual_income_valuation", {}), 0.10),
                ("real_options", analysis_result.get("real_options_valuation", {}), 0.05),
                ("sotp", analysis_result.get("sotp_valuation", {}), 0.10),
                ("eva", analysis_result.get("eva_valuation", {}), 0.05)
            ]

            # 收集有效估值和置信度
            method_values = []
            confidences = []

            for method_name, method_data, base_weight in valuation_methods:
                value = 0
                confidence = 0

                if method_name == "dcf":
                    value = method_data.get("per_share_value", 0)
                    confidence = method_data.get("confidence", 0)
                elif method_name == "relative":
                    value = method_data.get("relative_value", 0)
                    confidence = 0.6
                elif method_name == "asset":
                    enhanced_asset_value = method_data.get("enhanced_asset_value", 0)
                    if enhanced_asset_value > 0:
                        value = enhanced_asset_value / 100000000  # 转换为每股价值
                    else:
                        value = method_data.get("asset_based_value", 0) / 100000000
                    confidence = 0.4
                elif method_name in ["dividend", "residual_income", "sotp", "eva"]:
                    value = method_data.get("per_share_value", 0)
                    confidence = method_data.get("confidence", 0.5)
                elif method_name == "real_options":
                    value = method_data.get("per_share_option_value", 0)
                    confidence = method_data.get("confidence", 0.3)

                if value > 0 and confidence > 0:
                    method_values.append((method_name, value, confidence, base_weight))
                    confidences.append(confidence)

            # 计算动态权重
            total_confidence = sum(confidences)
            if total_confidence > 0:
                dynamic_weights = {
                    method_name: (confidence * base_weight) / total_confidence
                    for method_name, _, confidence, base_weight in method_values
                }
            else:
                # 使用等权重
                count = len(method_values)
                dynamic_weights = {
                    method_name: 1 / count for method_name, _, _, _ in method_values
                }

            # 计算加权价值
            weighted_sum = sum(value * dynamic_weights[method_name]
                             for method_name, value, _, _ in method_values)

            comprehensive_result.update({
                "weighted_value": weighted_sum,
                "valuation_method_weights": dynamic_weights,
                "method_contributions": {
                    method_name: value * dynamic_weights[method_name]
                    for method_name, value, _, _ in method_values
                },
                "confidence_score": np.mean(confidences) if confidences else 0
            })

            # 计算估值多样性指标
            if len(method_values) > 1:
                values = [value for _, value, _, _ in method_values]
                if values:
                    value_std = np.std(values)
                    value_mean = np.mean(values)
                    if value_mean > 0:
                        cv = value_std / value_mean  # 变异系数
                    comprehensive_result["valuation_diversity"] = {
                        "coefficient_of_variation": cv,
                        "diversity_level": "high" if cv > 0.3 else "medium" if cv > 0.15 else "low"
                    }

            # 计算模型一致性
            if len(method_values) >= 3:
                median_value = np.median([value for _, value, _, _ in method_values])
                above_median = sum(1 for _, value, _, _ in method_values if value > median_value)
                below_median = len(method_values) - above_median

                comprehensive_result["model_consensus"] = {
                    "median_value": median_value,
                    "above_median_count": above_median,
                    "below_median_count": below_median,
                    "consensus_level": "strong" if abs(above_median - below_median) <= 1 else "weak"
                }

        except Exception as e:
            self.logger.error(f"增强综合估值计算失败: {e}")
            comprehensive_result["error"] = str(e)

        return comprehensive_result

    def _calculate_enhanced_valuation_range(self, analysis_result: Dict[str, Any],
                                         market_info: Dict[str, Any]) -> Dict[str, Any]:
        """计算增强估值区间"""
        valuation_range = {
            "lower_bound": 0,
            "upper_bound": 0,
            "fair_value": 0,
            "safety_margin": 0,
            "confidence_interval": {},
            "valuation_scenarios": {}
        }

        try:
            comprehensive_val = analysis_result.get("comprehensive_valuation", {})
            weighted_value = comprehensive_val.get("weighted_value", 0)

            if weighted_value > 0:
                # 基于估值多样性调整区间宽度
                diversity = comprehensive_val.get("valuation_diversity", {})
                cv = diversity.get("coefficient_of_variation", 0.2)

                # 根据变异系数调整区间宽度
                base_width = 0.25  # 基础宽度±25%
                adjustment_factor = 1 + cv * 2  # 多样性调整因子
                width = base_width * adjustment_factor

                # 计算区间
                lower_bound = weighted_value * (1 - width)
                upper_bound = weighted_value * (1 + width)
                fair_value = weighted_value

                # 当前股价
                current_price = market_info.get("current_price", 0)

                # 计算安全边际
                if current_price > 0:
                    safety_margin = (fair_value - current_price) / current_price * 100
                else:
                    safety_margin = 0

                # 置信区间
                confidence_score = comprehensive_val.get("confidence_score", 0.5)
                confidence_width = width * (2 - confidence_score)  # 置信度越低，区间越宽

                valuation_range.update({
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "fair_value": fair_value,
                    "safety_margin": safety_margin,
                    "current_price": current_price,
                    "confidence_interval": {
                        "lower_90": weighted_value * (1 - confidence_width * 0.9),
                        "upper_90": weighted_value * (1 + confidence_width * 0.9),
                        "lower_50": weighted_value * (1 - confidence_width * 0.5),
                        "upper_50": weighted_value * (1 + confidence_width * 0.5)
                    }
                })

                # 估值情景
                scenarios = analysis_result.get("dcf_valuation", {}).get("scenario_analysis", {})
                if scenarios:
                    valuation_range["valuation_scenarios"] = {
                        "optimistic_scenarios": {
                            "upper_bound": scenarios.get("optimistic", {}).get("value", 0) / 100000000,
                            "probability": 0.2
                        },
                        "pessimistic_scenarios": {
                            "lower_bound": scenarios.get("pessimistic", {}).get("value", 0) / 100000000,
                            "probability": 0.2
                        },
                        "base_case": {
                            "value": scenarios.get("base", {}).get("value", 0) / 100000000,
                            "probability": 0.6
                        }
                    }

        except Exception as e:
            self.logger.error(f"增强估值区间计算失败: {e}")
            valuation_range["error"] = str(e)

        return valuation_range

    def _calculate_enhanced_investment_metrics(self, analysis_result: Dict[str, Any],
                                             market_info: Dict[str, Any]) -> Dict[str, Any]:
        """计算增强投资指标"""
        investment_metrics = {
            "upside_potential": 0,
            "downside_risk": 0,
            "risk_reward_ratio": 0,
            "valuation_grade": "unknown",
            "enhanced_metrics": {},
            "risk_assessment": {}
        }

        try:
            valuation_range = analysis_result.get("valuation_range", {})
            fair_value = valuation_range.get("fair_value", 0)
            lower_bound = valuation_range.get("lower_bound", 0)
            upper_bound = valuation_range.get("upper_bound", 0)

            current_price = market_info.get("current_price", 0)

            if current_price > 0 and fair_value > 0:
                # 基础指标
                upside_potential = (fair_value - current_price) / current_price * 100
                downside_risk = (current_price - lower_bound) / current_price * 100 if current_price > lower_bound else 0
                risk_reward_ratio = upside_potential / downside_risk if downside_risk > 0 else 0

                # 增强指标
                comprehensive_val = analysis_result.get("comprehensive_valuation", {})
                confidence_score = comprehensive_val.get("confidence_score", 0.5)

                # 调整后的上涨潜力（考虑置信度）
                adjusted_upside = upside_potential * confidence_score

                # 估值确定性指标
                valuation_diversity = comprehensive_val.get("valuation_diversity", {})
                cv = valuation_diversity.get("coefficient_of_variation", 0.2)
                valuation_certainty = 1 - min(1, cv * 2)  # 将变异系数转换为确定性

                # 风险调整回报
                risk_adjusted_return = adjusted_upside * valuation_certainty

                # 增强的估值评级
                if risk_adjusted_return > 30 and valuation_certainty > 0.7:
                    valuation_grade = "strong_buy"
                elif risk_adjusted_return > 15 and valuation_certainty > 0.5:
                    valuation_grade = "buy"
                elif risk_adjusted_return > -5:
                    valuation_grade = "hold"
                elif risk_adjusted_return > -20:
                    valuation_grade = "sell"
                else:
                    valuation_grade = "strong_sell"

                investment_metrics.update({
                    "upside_potential": upside_potential,
                    "downside_risk": downside_risk,
                    "risk_reward_ratio": risk_reward_ratio,
                    "valuation_grade": valuation_grade,
                    "enhanced_metrics": {
                        "adjusted_upside": adjusted_upside,
                        "valuation_certainty": valuation_certainty,
                        "risk_adjusted_return": risk_adjusted_return,
                        "confidence_score": confidence_score
                    },
                    "risk_assessment": {
                        "high_potential_return": bool(adjusted_upside > 25),
                        "low_valuation_risk": bool(downside_risk < 20),
                        "high_confidence": bool(confidence_score > 0.7),
                        "good_risk_reward": bool(risk_reward_ratio > 2)
                    }
                })

        except Exception as e:
            self.logger.error(f"增强投资指标计算失败: {e}")
            investment_metrics["error"] = str(e)

        return investment_metrics

    def _assess_valuation_data_quality(self, financial_data: Dict[str, Any],
                                     market_info: Dict[str, Any]) -> str:
        """评估估值数据质量"""
        if not financial_data:
            return "no_data"

        # 计算有效数据点
        valid_metrics = 0
        total_metrics = 0

        important_metrics = [
            "revenue", "net_income", "free_cash_flow", "pe_ratio", "pb_ratio",
            "total_assets", "total_liabilities", "revenue_growth", "roe", "eps",
            "dividend_per_share", "book_value_per_share", "wacc", "market_cap"
        ]

        for metric in important_metrics:
            total_metrics += 1
            value = financial_data.get(metric)
            if value is not None:
                cleaned = self._clean_financial_value(value)
                if cleaned is not None:
                    valid_metrics += 1

        # 检查市场数据
        if market_info.get("current_price"):
            valid_metrics += 1
        total_metrics += 1

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

    def _generate_investment_recommendation(self, analysis_result: Dict[str, Any],
                                          market_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成投资建议"""
        investment_metrics = analysis_result.get("investment_metrics", {})
        valuation_range = analysis_result.get("valuation_range", {})

        upside_potential = investment_metrics.get("upside_potential", 0)
        risk_reward_ratio = investment_metrics.get("risk_reward_ratio", 0)
        valuation_grade = investment_metrics.get("valuation_grade", "fairly_valued")

        # 基于估值结果生成建议
        if valuation_grade == "significantly_undervalued" and risk_reward_ratio > 3:
            direction = "strong_buy"
            confidence = min(90, upside_potential * 0.9)
            reasoning = "显著低估，风险收益比优秀"
        elif valuation_grade == "undervalued" and risk_reward_ratio > 2:
            direction = "buy"
            confidence = min(80, upside_potential * 0.8)
            reasoning = "低估，具有投资价值"
        elif valuation_grade == "fairly_valued":
            direction = "hold"
            confidence = 60
            reasoning = "估值合理，建议持有"
        elif valuation_grade == "overvalued":
            direction = "sell"
            confidence = min(80, abs(upside_potential) * 0.8)
            reasoning = "估值偏高，建议减仓"
        else:
            direction = "strong_sell"
            confidence = min(90, abs(upside_potential) * 0.9)
            reasoning = "显著高估，建议卖出"

        return {
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "valuation_grade": valuation_grade,
            "upside_potential": upside_potential,
            "risk_reward_ratio": risk_reward_ratio,
            "fair_value": valuation_range.get("fair_value", 0),
            "safety_margin": valuation_range.get("safety_margin", 0)
        }

    def _generate_reasoning_report(self, analysis_result: Dict[str, Any],
                                 investment_recommendation: Dict[str, Any]) -> str:
        """生成推理报告"""
        report = []

        # 综合估值结果
        comprehensive_val = analysis_result.get("comprehensive_valuation", {})
        weighted_value = comprehensive_val.get("weighted_value", 0)

        report.append(f"综合估值结果: {weighted_value:.2f}元")

        # 估值区间
        valuation_range = analysis_result.get("valuation_range", {})
        fair_value = valuation_range.get("fair_value", 0)
        lower_bound = valuation_range.get("lower_bound", 0)
        upper_bound = valuation_range.get("upper_bound", 0)

        report.append(f"估值区间: {lower_bound:.2f} - {upper_bound:.2f}元")
        report.append(f"公允价值: {fair_value:.2f}元")

        # 投资指标
        investment_metrics = analysis_result.get("investment_metrics", {})
        upside_potential = investment_metrics.get("upside_potential", 0)
        downside_risk = investment_metrics.get("downside_risk", 0)
        risk_reward_ratio = investment_metrics.get("risk_reward_ratio", 0)

        report.append(f"上涨潜力: {upside_potential:.1f}%")
        report.append(f"下跌风险: {downside_risk:.1f}%")
        report.append(f"风险收益比: {risk_reward_ratio:.2f}")

        # 估值评级
        valuation_grade = investment_metrics.get("valuation_grade", "unknown")
        report.append(f"估值评级: {valuation_grade}")

        # 各估值方法权重
        method_weights = comprehensive_val.get("valuation_method_weights", {})
        dcf_weight = method_weights.get("dcf_weight", 0) * 100
        relative_weight = method_weights.get("relative_weight", 0) * 100
        asset_weight = method_weights.get("asset_weight", 0) * 100

        report.append(f"估值方法权重: DCF({dcf_weight:.0f}%) 相对估值({relative_weight:.0f}%) 资产估值({asset_weight:.0f}%)")

        # 投资建议
        direction = investment_recommendation.get("direction", "hold")
        confidence = investment_recommendation.get("confidence", 50)
        reasoning = investment_recommendation.get("reasoning", "")
        safety_margin = investment_recommendation.get("safety_margin", 0)

        report.append(f"投资建议: {direction}")
        report.append(f"建议置信度: {confidence:.1f}%")
        report.append(f"安全边际: {safety_margin:.1f}%")
        report.append(f"主要理由: {reasoning}")

        return "\n".join(report)

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker", "financial_metrics", "market_info"]