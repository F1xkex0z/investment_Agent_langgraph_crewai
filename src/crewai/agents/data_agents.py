"""
Data Collection Agents for CrewAI Investment System

This module contains data collection agents responsible for gathering market data,
financial information, and news data.
"""

from typing import List, Any, Dict
from datetime import datetime, timedelta
import pandas as pd

from crewai.tools import BaseTool
from src.tools.api import get_financial_metrics, get_financial_statements, get_market_data, get_price_history
from src.utils.logging_config import setup_logger

from .base import DataAgent
from ..config.state import InvestmentState


logger = setup_logger('crewai_data_agents')


class MarketDataTool(BaseTool):
    """Tool for collecting market data"""
    name: str = "collect_market_data"
    description: str = "Collect market data including price history, financial metrics, and market information"

    def _run(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Collect market data for given ticker and date range"""
        try:
            # Get price history
            prices_df = get_price_history(ticker, start_date, end_date)
            if prices_df is None or prices_df.empty:
                logger.warning(f"Could not get price data for {ticker}")
                prices_dict = []
            else:
                prices_dict = prices_df.to_dict('records')

            # Get financial metrics
            try:
                financial_metrics = get_financial_metrics(ticker)
            except Exception as e:
                logger.error(f"Failed to get financial metrics: {str(e)}")
                financial_metrics = {}

            # Get financial statements
            try:
                financial_line_items = get_financial_statements(ticker)
            except Exception as e:
                logger.error(f"Failed to get financial statements: {str(e)}")
                financial_line_items = {}

            # Get market data
            try:
                market_data = get_market_data(ticker)
            except Exception as e:
                logger.error(f"Failed to get market data: {str(e)}")
                market_data = {"market_cap": 0}

            return {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "prices": prices_dict,
                "financial_metrics": financial_metrics,
                "financial_line_items": financial_line_items,
                "market_cap": market_data.get("market_cap", 0),
                "market_data": market_data,
                "data_quality": {
                    "price_history": len(prices_dict) > 0,
                    "financial_metrics": len(financial_metrics) > 0,
                    "financial_statements": len(financial_line_items) > 0,
                    "market_data": len(market_data) > 0
                }
            }

        except Exception as e:
            logger.error(f"Market data collection failed: {str(e)}")
            return {
                "ticker": ticker,
                "error": str(e),
                "data_quality": {
                    "price_history": False,
                    "financial_metrics": False,
                    "financial_statements": False,
                    "market_data": False
                }
            }


class NewsDataTool(BaseTool):
    """Tool for collecting news data"""
    name: str = "collect_news_data"
    description: str = "Collect news articles for sentiment analysis"

    def _run(self, ticker: str, num_of_news: int = 5) -> Dict[str, Any]:
        """Collect news data for given ticker"""
        try:
            from src.tools.news_crawler import get_stock_news

            news_articles = get_stock_news(ticker, num_of_news)

            return {
                "ticker": ticker,
                "news_count": len(news_articles),
                "news_articles": news_articles,
                "collection_time": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"News data collection failed: {str(e)}")
            return {
                "ticker": ticker,
                "error": str(e),
                "news_count": 0,
                "news_articles": []
            }


class DataCollectionAgent(DataAgent):
    """
    Data Collection Agent responsible for gathering all necessary market data,
    financial information, and news data for investment analysis.
    """

    def __init__(self, tools: List[BaseTool] = None):
        super().__init__(
            role="Senior Market Data Analyst",
            goal="Collect comprehensive and accurate market data, financial metrics, and news information for investment analysis",
            backstory="""You are an expert market data analyst with 10+ years of experience in financial data collection.
            You specialize in gathering high-quality data from various sources including stock exchanges, financial
            databases, and news outlets. You understand the importance of data accuracy and completeness in
            investment decision making. You are meticulous about data validation and error handling.""",
            tools=tools or [MarketDataTool(), NewsDataTool()]
        )

    def preprocess_state(self, state: InvestmentState) -> Dict[str, Any]:
        """Preprocess state for data collection"""
        context = super().preprocess_state(state)

        # Set default dates if not provided
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)
        end_date = state.end_date or yesterday.strftime('%Y-%m-%d')

        # Ensure end_date is not in the future
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        if end_date_obj > yesterday:
            end_date = yesterday.strftime('%Y-%m-%d')
            end_date_obj = yesterday

        if not state.start_date:
            # Calculate 1 year before end_date
            start_date = end_date_obj - timedelta(days=365)
            start_date = start_date.strftime('%Y-%m-%d')
        else:
            start_date = state.start_date

        context.update({
            "data_collection_params": {
                "ticker": state.ticker,
                "start_date": start_date,
                "end_date": end_date,
                "num_of_news": state.config.num_of_news
            }
        })

        return context

    def create_task_description(self, state: InvestmentState) -> str:
        """Create task description for data collection"""
        context = self.preprocess_state(state)
        params = context["data_collection_params"]

        return f"""
        Collect comprehensive market data for {params['ticker']} from {params['start_date']} to {params['end_date']}.

        Your tasks:
        1. Collect historical price data (OHLCV) for the specified date range
        2. Gather key financial metrics and ratios
        3. Collect financial statement data
        4. Get current market information (market cap, etc.)
        5. Collect recent news articles (up to {params['num_of_news']} articles) for sentiment analysis

        Requirements:
        - Ensure data quality and handle missing data appropriately
        - Validate data completeness and accuracy
        - Provide data quality summary
        - Handle API errors gracefully
        - Cache data when possible to avoid redundant API calls

        Expected Output:
        A comprehensive data package containing all necessary information for investment analysis,
        including price history, financial metrics, financial statements, market data, and news articles.
        """

    def postprocess_output(self, output: Any, state: InvestmentState) -> Any:
        """Postprocess data collection output and update state"""
        try:
            # Parse the output if it's a string
            if isinstance(output, str):
                import json
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    # If not JSON, create structured output from text
                    output = {
                        "summary": output,
                        "ticker": state.ticker,
                        "collection_time": datetime.now().isoformat()
                    }

            # Update state data cache
            if isinstance(output, dict):
                state.data_cache['market_data'] = output

                # Extract and store specific data components
                if 'prices' in output:
                    state.data_cache['price_history'] = output['prices']
                if 'financial_metrics' in output:
                    state.data_cache['financial_metrics'] = output['financial_metrics']
                if 'financial_line_items' in output:
                    state.data_cache['financial_statements'] = output['financial_line_items']
                if 'market_data' in output:
                    state.data_cache['market_info'] = output['market_data']
                if 'news_articles' in output:
                    state.data_cache['news_articles'] = output['news_articles']

                # Update portfolio with current price if available
                if 'prices' in output and output['prices']:
                    latest_price = output['prices'][-1].get('close')
                    if latest_price:
                        state.portfolio.update_value(latest_price)

            # Update state with analysis result
            state.update_analysis_result('data_collection', output)

            return output

        except Exception as e:
            logger.error(f"Failed to postprocess data collection output: {str(e)}")
            error_output = {
                "error": str(e),
                "ticker": state.ticker,
                "collection_time": datetime.now().isoformat()
            }
            state.update_analysis_result('data_collection', error_output)
            return error_output


class NewsAnalysisAgent(DataAgent):
    """
    News Analysis Agent responsible for collecting and analyzing news data
    for sentiment analysis.
    """

    def __init__(self, tools: List[BaseTool] = None):
        super().__init__(
            role="Financial News Analyst",
            goal="Collect and analyze relevant news articles for sentiment analysis and market impact assessment",
            backstory="""You are a specialized financial news analyst with expertise in identifying and collecting
            relevant news articles that impact stock prices. You understand the importance of timely, relevant
            news in investment decision making and can distinguish between noise and significant market-moving
            information. You are skilled at searching multiple news sources and filtering for quality content.""",
            tools=tools or [NewsDataTool()]
        )

    def create_task_description(self, state: InvestmentState) -> str:
        """Create task description for news analysis"""
        return f"""
        Collect and analyze recent news articles for {state.ticker} to support sentiment analysis.

        Your tasks:
        1. Search for recent news articles related to {state.ticker}
        2. Filter for relevant financial and business news
        3. Collect up to {state.config.num_of_news} articles
        4. Ensure articles are from credible sources
        5. Categorize articles by relevance and potential market impact

        Requirements:
        - Focus on recent news (last 30 days)
        - Prioritize news from reputable financial sources
        - Include both positive and negative news
        - Filter out irrelevant or duplicate content
        - Provide brief summaries of key articles

        Expected Output:
        A curated collection of news articles with metadata, ready for sentiment analysis.
        """

    def postprocess_output(self, output: Any, state: InvestmentState) -> Any:
        """Postprocess news analysis output and update state"""
        try:
            if isinstance(output, str):
                import json
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    output = {
                        "summary": output,
                        "ticker": state.ticker,
                        "analysis_time": datetime.now().isoformat()
                    }

            # Store news data in state
            if isinstance(output, dict):
                state.data_cache['news_analysis'] = output
                if 'news_articles' in output:
                    state.data_cache['news_articles'] = output['news_articles']

            state.update_analysis_result('news_analysis', output)
            return output

        except Exception as e:
            logger.error(f"Failed to postprocess news analysis output: {str(e)}")
            error_output = {
                "error": str(e),
                "ticker": state.ticker,
                "analysis_time": datetime.now().isoformat()
            }
            state.update_analysis_result('news_analysis', error_output)
            return error_output