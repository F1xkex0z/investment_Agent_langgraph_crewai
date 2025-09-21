from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction

import json

# 初始化 logger
logger = setup_logger('fundamentals_agent')

##### Fundamental Agent #####


@agent_endpoint("fundamentals", "基本面分析师，分析公司财务指标、盈利能力和增长潜力")
def fundamentals_agent(state: AgentState):
    """Responsible for fundamental analysis"""
    show_workflow_status("Fundamentals Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]
    symbol = data.get("ticker", "unknown")
    
    logger.info(f"开始基本面分析: {symbol}")

    # Initialize signals list for different fundamental aspects
    signals = []
    reasoning = {}

    # 1. Profitability Analysis
    logger.debug("开始盈利能力分析")
    return_on_equity = metrics.get("return_on_equity", 0)
    net_margin = metrics.get("net_margin", 0)
    operating_margin = metrics.get("operating_margin", 0)
    logger.debug(f"盈利能力指标原始数据: ROE={return_on_equity}, 净利率={net_margin}, 营业利润率={operating_margin}")

    thresholds = [
        (return_on_equity, 0.15),  # Strong ROE above 15%
        (net_margin, 0.20),  # Healthy profit margins
        (operating_margin, 0.15)  # Strong operating efficiency
    ]
    profitability_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds
    )
    logger.debug(f"盈利能力得分: {profitability_score}/3")

    signals.append('bullish' if profitability_score >=
                   2 else 'bearish' if profitability_score == 0 else 'neutral')
    reasoning["profitability_signal"] = {
        "signal": signals[0],
        "details": (
            f"ROE: {metrics.get('return_on_equity', 0):.2%}" if metrics.get(
                "return_on_equity") is not None else "ROE: N/A"
        ) + ", " + (
            f"Net Margin: {metrics.get('net_margin', 0):.2%}" if metrics.get(
                "net_margin") is not None else "Net Margin: N/A"
        ) + ", " + (
            f"Op Margin: {metrics.get('operating_margin', 0):.2%}" if metrics.get(
                "operating_margin") is not None else "Op Margin: N/A"
        )
    }

    # 2. Growth Analysis
    logger.debug("开始增长潜力分析")
    revenue_growth = metrics.get("revenue_growth", 0)
    earnings_growth = metrics.get("earnings_growth", 0)
    book_value_growth = metrics.get("book_value_growth", 0)
    logger.debug(f"增长指标原始数据: 收入增长={revenue_growth}, 盈利增长={earnings_growth}, 账面价值增长={book_value_growth}")

    thresholds = [
        (revenue_growth, 0.10),  # 10% revenue growth
        (earnings_growth, 0.10),  # 10% earnings growth
        (book_value_growth, 0.10)  # 10% book value growth
    ]
    growth_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds
    )
    logger.debug(f"增长潜力得分: {growth_score}/3")

    signals.append('bullish' if growth_score >=
                   2 else 'bearish' if growth_score == 0 else 'neutral')
    reasoning["growth_signal"] = {
        "signal": signals[1],
        "details": (
            f"Revenue Growth: {metrics.get('revenue_growth', 0):.2%}" if metrics.get(
                "revenue_growth") is not None else "Revenue Growth: N/A"
        ) + ", " + (
            f"Earnings Growth: {metrics.get('earnings_growth', 0):.2%}" if metrics.get(
                "earnings_growth") is not None else "Earnings Growth: N/A"
        )
    }

    # 3. Financial Health
    logger.debug("开始财务健康分析")
    current_ratio = metrics.get("current_ratio", 0)
    debt_to_equity = metrics.get("debt_to_equity", 0)
    free_cash_flow_per_share = metrics.get("free_cash_flow_per_share", 0)
    earnings_per_share = metrics.get("earnings_per_share", 0)
    logger.debug(f"财务健康指标原始数据: 流动比率={current_ratio}, 债务权益比={debt_to_equity}, FCF每股={free_cash_flow_per_share}, EPS={earnings_per_share}")

    health_score = 0
    if current_ratio and current_ratio > 1.5:  # Strong liquidity
        health_score += 1
        logger.debug("财务健康得分+1: 流动比率充足")
    if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
        health_score += 1
        logger.debug("财务健康得分+1: 债务水平保守")
    if (free_cash_flow_per_share and earnings_per_share and
            free_cash_flow_per_share > earnings_per_share * 0.8):  # Strong FCF conversion
        health_score += 1
        logger.debug("财务健康得分+1: FCF转换能力强")
    logger.debug(f"财务健康得分: {health_score}/3")

    signals.append('bullish' if health_score >=
                   2 else 'bearish' if health_score == 0 else 'neutral')
    reasoning["financial_health_signal"] = {
        "signal": signals[2],
        "details": (
            f"Current Ratio: {metrics.get('current_ratio', 0):.2f}" if metrics.get(
                "current_ratio") is not None else "Current Ratio: N/A"
        ) + ", " + (
            f"D/E: {metrics.get('debt_to_equity', 0):.2f}" if metrics.get(
                "debt_to_equity") is not None else "D/E: N/A"
        )
    }

    # 4. Price to X ratios
    logger.debug("开始估值比率分析")
    pe_ratio = metrics.get("pe_ratio", 0)
    price_to_book = metrics.get("price_to_book", 0)
    price_to_sales = metrics.get("price_to_sales", 0)
    logger.debug(f"估值比率原始数据: P/E={pe_ratio}, P/B={price_to_book}, P/S={price_to_sales}")

    thresholds = [
        (pe_ratio, 25),  # Reasonable P/E ratio
        (price_to_book, 3),  # Reasonable P/B ratio
        (price_to_sales, 5)  # Reasonable P/S ratio
    ]
    price_ratio_score = sum(
        metric is not None and metric < threshold
        for metric, threshold in thresholds
    )
    logger.debug(f"估值比率得分: {price_ratio_score}/3")

    signals.append('bullish' if price_ratio_score >=
                   2 else 'bearish' if price_ratio_score == 0 else 'neutral')
    reasoning["price_ratios_signal"] = {
        "signal": signals[3],
        "details": (
            f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A"
        ) + ", " + (
            f"P/B: {price_to_book:.2f}" if price_to_book else "P/B: N/A"
        ) + ", " + (
            f"P/S: {price_to_sales:.2f}" if price_to_sales else "P/S: N/A"
        )
    }

    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    logger.debug(f"信号统计: 看涨={bullish_signals}, 看跌={bearish_signals}, 中性={signals.count('neutral')}")

    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'

    # Calculate confidence level
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    logger.debug(f"置信度计算: {max(bullish_signals, bearish_signals)}/{total_signals} = {confidence:.2f}")

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning
    }

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="fundamentals_agent",
    )

    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Fundamental Analysis Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    # 记录分析结果
    logger.info(f"基本面分析完成: {symbol}, 信号: {overall_signal}, 置信度: {round(confidence * 100)}%")
    logger.info(f"各项指标分析结果: 盈利能力={signals[0]}, 增长潜力={signals[1]}, 财务健康={signals[2]}, 估值={signals[3]}")
    
    show_workflow_status("Fundamentals Analyst", "completed")
    logger.debug(f"--- DEBUG: fundamentals_agent RETURN messages: {[msg.name for msg in [message]]} ---")
    return {
        "messages": [message],
        "data": {
            **data,
            "fundamental_analysis": message_content
        },
        "metadata": state["metadata"],
    }
