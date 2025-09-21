from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
import numpy as np
from langchain_core.messages import HumanMessage

# 初始化 logger
logger = setup_logger('valuation_agent')


@agent_endpoint("valuation", "估值分析师，使用DCF和所有者收益法评估公司内在价值")
def valuation_agent(state: AgentState):
    """Responsible for valuation analysis"""
    logger.info("估值分析师Agent开始执行")
    show_workflow_status("Valuation Agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    logger.debug(f"推理显示设置: {show_reasoning}")
    
    data = state["data"]
    logger.debug(f"接收到的数据: 股票代码={data.get('ticker')}, 市场数据存在={str(bool(data.get('market_data')))}")
    
    # Safely access financial metrics with default values
    metrics = data["financial_metrics"][0] if data.get("financial_metrics") and len(data["financial_metrics"]) > 0 else {}
    logger.debug(f"获取到的财务指标数量: {len(metrics)}")
    
    # Safely access financial line items with default values
    current_financial_line_item = data["financial_line_items"][0] if data.get("financial_line_items") and len(data["financial_line_items"]) > 0 else {}
    previous_financial_line_item = data["financial_line_items"][1] if data.get("financial_line_items") and len(data["financial_line_items"]) > 1 else current_financial_line_item
    logger.debug(f"当前财务报表数据项数: {len(current_financial_line_item)}, 上年数据项数: {len(previous_financial_line_item)}")
    
    # Safely access market_cap with a default value
    market_cap = data.get("market_cap", 0)
    logger.info(f"市值信息: {market_cap:,.2f}")

    reasoning = {}

    # Calculate working capital change
    working_capital_change = (current_financial_line_item.get(
        'working_capital') or 0) - (previous_financial_line_item.get('working_capital') or 0)

    # Get earnings growth with a default value if missing
    earnings_growth = metrics.get("earnings_growth", 0.05)  # Default to 5% growth rate

    # Owner Earnings Valuation (Buffett Method)
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get('net_income', 0),
        depreciation=current_financial_line_item.get(
            'depreciation_and_amortization', 0),
        capex=current_financial_line_item.get('capital_expenditure', 0),
        working_capital_change=working_capital_change,
        growth_rate=earnings_growth,
        required_return=0.15,
        margin_of_safety=0.25
    )

    # DCF Valuation
    dcf_value = calculate_intrinsic_value(
        free_cash_flow=current_financial_line_item.get('free_cash_flow', 0),
        growth_rate=earnings_growth,
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        num_years=5,
    )

    # Calculate combined valuation gap (average of both methods)
    # Handle division by zero by setting reasonable default values
    # 计算估值差距并处理极端情况
    if market_cap == 0:
        # 如果市值为0，避免除以0错误
        dcf_gap = 0.0
        owner_earnings_gap = 0.0
        valuation_gap = 0.0
        logger.warning("市值为0，无法计算估值差距，使用默认值")
    else:
        # 计算DCF估值差距并处理可能的NaN值
        if np.isnan(dcf_value):
            dcf_gap = 0.0
            logger.warning("DCF价值为NaN，使用默认估值差距")
        else:
            dcf_gap = (dcf_value - market_cap) / market_cap
            # 限制估值差距范围，避免极端值
            dcf_gap = max(min(dcf_gap, 2.0), -1.0)  # 限制在-100%到200%之间
        
        # 计算所有者收益估值差距并处理可能的NaN值
        if np.isnan(owner_earnings_value):
            owner_earnings_gap = 0.0
            logger.warning("所有者收益价值为NaN，使用默认估值差距")
        else:
            owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
            # 限制估值差距范围，避免极端值
            owner_earnings_gap = max(min(owner_earnings_gap, 2.0), -1.0)  # 限制在-100%到200%之间
        
        # 计算综合估值差距
        valuation_gap = (dcf_gap + owner_earnings_gap) / 2
        # 再次检查是否为NaN
        if np.isnan(valuation_gap):
            valuation_gap = 0.0
            logger.warning("综合估值差距为NaN，使用默认值")

    if valuation_gap > 0.10:  # Changed from 0.15 to 0.10 (10% undervalued)
        signal = 'bullish'
    elif valuation_gap < -0.20:  # Changed from -0.15 to -0.20 (20% overvalued)
        signal = 'bearish'
    else:
        signal = 'neutral'

    reasoning["dcf_analysis"] = {
        "signal": "bullish" if dcf_gap > 0.10 else "bearish" if dcf_gap < -0.20 else "neutral",
        "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}"
    }

    reasoning["owner_earnings_analysis"] = {
        "signal": "bullish" if owner_earnings_gap > 0.10 else "bearish" if owner_earnings_gap < -0.20 else "neutral",
        "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}"
    }

    # 计算置信度并确保不会出现NaN
    confidence_value = abs(valuation_gap)
    if np.isnan(confidence_value):
        confidence_value = 0.0
        logger.warning("置信度计算为NaN，使用默认值")
    
    # 限制置信度范围在0%到100%之间
    confidence_value = min(confidence_value, 1.0)

    message_content = {
        "signal": signal,
        "confidence": f"{confidence_value:.0%}",
        "reasoning": reasoning
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Valuation Analysis Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Valuation Agent", "completed")
    logger.info(f"估值分析完成: 信号={signal}, 置信度={confidence_value:.0%}, 估值差距={valuation_gap:.1%}")
    logger.debug(f"DCF价值=${dcf_value:,.2f}, 所有者收益价值=${owner_earnings_value:,.2f}")
    return {
        "messages": [message],
        "data": {
            **data,
            "valuation_analysis": message_content
        },
        "metadata": state["metadata"],
    }


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5


) -> float:
    logger.debug(f"开始计算所有者收益价值: 净利润={net_income:,.2f}, 折旧={depreciation:,.2f}, 资本支出={capex:,.2f}")
    """
    使用改进的所有者收益法计算公司价值。

    Args:
        net_income: 净利润
        depreciation: 折旧和摊销
        capex: 资本支出
        working_capital_change: 营运资金变化
        growth_rate: 预期增长率
        required_return: 要求回报率
        margin_of_safety: 安全边际
        num_years: 预测年数

    Returns:
        float: 计算得到的公司价值
    """
    try:
        # 数据有效性检查
        if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
            return 0

        # 计算初始所有者收益
        owner_earnings = (
            net_income +
            depreciation -
            capex -
            working_capital_change
        )

        # 处理异常情况
        if owner_earnings <= 0 or required_return <= 0:
            logger.warning(f"所有者收益或要求回报率为负值或零: 所有者收益={owner_earnings:,.2f}, 要求回报率={required_return:,.2f}")
            return 0

        # 调整增长率，确保合理性
        growth_rate = min(max(growth_rate, -0.2), 0.25)  # 限制在-20%到25%之间

        # 计算预测期收益现值
        future_values = []
        for year in range(1, num_years + 1):
            # 使用递减增长率模型
            year_growth = growth_rate * (1 - year / (2 * num_years))  # 增长率逐年递减
            future_value = owner_earnings * (1 + year_growth) ** year
            discounted_value = future_value / (1 + required_return) ** year
            future_values.append(discounted_value)

        # 计算永续价值，确保分母不为零
        terminal_growth = min(max(growth_rate * 0.4, 0.01), 0.03)  # 永续增长率控制在1%-3%之间
        if required_return <= terminal_growth:
            logger.warning(f"要求回报率小于或等于永续增长率，无法计算永续价值: 要求回报率={required_return:,.2f}, 永续增长率={terminal_growth:,.2f}")
            return sum(future_values)  # 只返回预测期现值

        terminal_value = (
            future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
        terminal_value_discounted = terminal_value / \
            (1 + required_return) ** num_years

        # 计算总价值并应用安全边际
        intrinsic_value = sum(future_values) + terminal_value_discounted
        value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

        return max(value_with_safety_margin, 0)  # 确保不返回负值

    except Exception as e:
        logger.error(f"所有者收益计算错误: {e}")
        return 0


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    使用改进的DCF方法计算内在价值，考虑增长率和风险因素。

    Args:
        free_cash_flow: 自由现金流
        growth_rate: 预期增长率
        discount_rate: 基础折现率
        terminal_growth_rate: 永续增长率
        num_years: 预测年数

    Returns:
        float: 计算得到的内在价值
    """
    logger.debug(f"开始计算DCF价值: 自由现金流={free_cash_flow:,.2f}, 增长率={growth_rate:.2%}")
    try:
        # 数据类型和有效性检查
        if not isinstance(free_cash_flow, (int, float)):
            logger.warning(f"自由现金流不是有效数值: {free_cash_flow}")
            return 0
        
        # 如果自由现金流为0或负数，尝试使用其他替代方法
        if free_cash_flow <= 0:
            logger.warning(f"自由现金流为负值或零: {free_cash_flow:,.2f}")
            # 可以考虑使用净利润的一部分作为替代
            return 0

        # 调整增长率，确保合理性
        growth_rate = min(max(growth_rate, -0.2), 0.25)  # 限制在-20%到25%之间

        # 调整永续增长率，不能超过经济平均增长
        terminal_growth_rate = min(max(growth_rate * 0.4, 0.01), 0.03)  # 取增长率的40%或3%的较小值
        
        # 确保折现率有效
        if discount_rate <= 0 or discount_rate <= terminal_growth_rate:
            logger.warning(f"折现率无效或小于等于永续增长率: 折现率={discount_rate:,.2f}, 永续增长率={terminal_growth_rate:,.2f}")
            discount_rate = max(0.05, terminal_growth_rate + 0.02)  # 设置最低折现率

        # 计算预测期现金流现值
        present_values = []
        for year in range(1, num_years + 1):
            future_cf = free_cash_flow * (1 + growth_rate) ** year
            present_value = future_cf / (1 + discount_rate) ** year
            present_values.append(present_value)

        # 计算永续价值
        terminal_year_cf = free_cash_flow * (1 + growth_rate) ** num_years
        terminal_value = terminal_year_cf * \
            (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        terminal_present_value = terminal_value / \
            (1 + discount_rate) ** num_years

        # 总价值
        total_value = sum(present_values) + terminal_present_value
        logger.debug(f"DCF价值计算结果: 预测期现值总和={sum(present_values):,.2f}, 永续价值现值={terminal_present_value:,.2f}, 总价值={total_value:,.2f}")

        return max(total_value, 0)  # 确保不返回负值

    except Exception as e:
        logger.error(f"DCF计算错误: {e}")
        return 0


def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change means more capital is tied up in working capital (cash outflow).
    A negative change means less capital is tied up (cash inflow).

    Args:
        current_working_capital: Current period's working capital
        previous_working_capital: Previous period's working capital

    Returns:
        float: Change in working capital (current - previous)
    """
    logger.debug(f"计算营运资金变化: 当前期={current_working_capital:,.2f}, 上期={previous_working_capital:,.2f}")
    change = current_working_capital - previous_working_capital
    logger.debug(f"营运资金变化结果: {change:,.2f}")
    return change
