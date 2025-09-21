from langchain_core.messages import HumanMessage
from src.tools.openrouter_config import get_chat_completion
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import get_financial_metrics, get_financial_statements, get_market_data, get_price_history
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction

from datetime import datetime, timedelta
import pandas as pd

# 设置日志记录
logger = setup_logger('market_data_agent')


@agent_endpoint("market_data", "市场数据收集，负责获取股价历史、财务指标和市场信息")
def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    logger.info("市场数据收集Agent开始执行")
    show_workflow_status("Market Data Agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    logger.debug(f"推理显示设置: {show_reasoning}")

    messages = state["messages"]
    data = state["data"]
    logger.debug(f"接收到的状态数据: 消息数量={len(messages)}, 股票代码={data.get('ticker')}, 开始日期={data.get('start_date')}, 结束日期={data.get('end_date')}")

    # Set default dates
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = data["end_date"] or yesterday.strftime('%Y-%m-%d')
    logger.debug(f"当前日期: {current_date.strftime('%Y-%m-%d')}, 设置默认结束日期: {end_date}")

    # Ensure end_date is not in the future
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    if end_date_obj > yesterday:
        logger.warning(f"结束日期{end_date}在未来，已调整为{yesterday.strftime('%Y-%m-%d')}")
        end_date = yesterday.strftime('%Y-%m-%d')
        end_date_obj = yesterday

    if not data["start_date"]:
        # Calculate 1 year before end_date
        start_date = end_date_obj - timedelta(days=365)  # 默认获取一年的数据
        start_date = start_date.strftime('%Y-%m-%d')
        logger.debug(f"未提供开始日期，自动设置为结束日期前一年: {start_date}")
    else:
        start_date = data["start_date"]
        logger.debug(f"使用提供的开始日期: {start_date}")

    # Get all required data
    ticker = data["ticker"]
    logger.info(f"开始收集{ticker}的市场数据，时间范围: {start_date} 至 {end_date}")

    # 获取价格数据并验证
    logger.debug(f"正在获取{ticker}的价格历史数据")
    prices_df = get_price_history(ticker, start_date, end_date)
    if prices_df is None or prices_df.empty:
        logger.warning(f"警告：无法获取{ticker}的价格数据，将使用空数据继续")
        prices_df = pd.DataFrame(
            columns=['close', 'open', 'high', 'low', 'volume'])
    else:
        logger.debug(f"成功获取{ticker}的价格历史数据，包含{len(prices_df)}条记录")

    # 获取财务指标
    logger.debug(f"正在获取{ticker}的财务指标")
    try:
        financial_metrics = get_financial_metrics(ticker)
        logger.debug(f"成功获取{ticker}的财务指标，包含{len(financial_metrics)}个指标")
    except Exception as e:
        logger.error(f"获取财务指标失败: {str(e)}")
        financial_metrics = {}

    # 获取财务报表
    logger.debug(f"正在获取{ticker}的财务报表")
    try:
        financial_line_items = get_financial_statements(ticker)
        logger.debug(f"成功获取{ticker}的财务报表，包含{len(financial_line_items)}个项目")
    except Exception as e:
        logger.error(f"获取财务报表失败: {str(e)}")
        financial_line_items = {}

    # 获取市场数据
    logger.debug(f"正在获取{ticker}的市场数据")
    try:
        market_data = get_market_data(ticker)
        logger.debug(f"成功获取{ticker}的市场数据，包含市值: {market_data.get('market_cap', '未知')}")
    except Exception as e:
        logger.error(f"获取市场数据失败: {str(e)}")
        market_data = {"market_cap": 0}

    # 确保数据格式正确
    if not isinstance(prices_df, pd.DataFrame):
        logger.warning(f"价格数据格式不正确，转换为空DataFrame")
        prices_df = pd.DataFrame(
            columns=['close', 'open', 'high', 'low', 'volume'])

    # 转换价格数据为字典格式
    prices_dict = prices_df.to_dict('records')
    logger.debug(f"价格数据转换为字典格式完成，记录数: {len(prices_dict)}")

    # 保存推理信息到metadata供API使用
    market_data_summary = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "data_collected": {
            "price_history": len(prices_dict) > 0,
            "financial_metrics": len(financial_metrics) > 0,
            "financial_statements": len(financial_line_items) > 0,
            "market_data": len(market_data) > 0
        },
        "summary": f"为{ticker}收集了从{start_date}到{end_date}的市场数据，包括价格历史、财务指标和市场信息"
    }
    logger.debug(f"创建市场数据摘要完成，数据收集状态: {market_data_summary['data_collected']}")

    if show_reasoning:
        logger.debug(f"显示Agent推理信息")
        show_agent_reasoning(market_data_summary, "Market Data Agent")
        state["metadata"]["agent_reasoning"] = market_data_summary

    logger.info(f"市场数据收集Agent执行完成，成功为{ticker}收集了所需市场数据")
    return {
        "messages": messages,
        "data": {
            **data,
            "prices": prices_dict,
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "financial_line_items": financial_line_items,
            "market_cap": market_data.get("market_cap", 0),
            "market_data": market_data,
        },
        "metadata": state["metadata"],
    }
