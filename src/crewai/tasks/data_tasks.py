"""
Data Collection Tasks for CrewAI Investment System

This module defines tasks for data collection and preparation.
"""

from typing import Dict, Any, Optional
from crewai import Task
from datetime import datetime
import time

from ..config.state import InvestmentState
from ..agents.data_agents import DataCollectionAgent, NewsAnalysisAgent

# Import logging
from src.utils.logging_config import setup_logger, setup_detailed_logger

logger = setup_logger('crewai_data_tasks')
detailed_logger = setup_detailed_logger('crewai_data_tasks')

class DataCollectionTasks:
    """Factory class for creating data collection tasks"""

    def __init__(self, state: InvestmentState):
        self.state = state
        detailed_logger.info(f"Initializing DataCollectionTasks for ticker: {state.ticker}")
        detailed_logger.debug(f"State run_id: {state.run_id}")

    def create_market_data_collection_task(self, agent: DataCollectionAgent) -> Task:
        """
        Create task for comprehensive market data collection

        Args:
            agent: DataCollectionAgent instance

        Returns:
            CrewAI Task instance
        """
        detailed_logger.info("Creating market data collection task")
        detailed_logger.debug(f"Creating task for ticker: {self.state.ticker}")
        
        task = Task(
            description=agent.create_task_description(self.state),
            expected_output=f"""
            A comprehensive data package for {self.state.ticker} including:
            - Historical price data (OHLCV) from {self.state.start_date} to {self.state.end_date}
            - Key financial metrics and ratios
            - Financial statement data
            - Current market information (market cap, etc.)
            - Recent news articles for sentiment analysis
            - Data quality assessment and metadata
            """,
            agent=agent,
            context=[],  # No context needed for data collection
            async_execution=False,
            human_input=False,
            output_json=False
        )
        
        detailed_logger.debug("Market data collection task created successfully")
        return task

    def create_news_analysis_task(self, agent: NewsAnalysisAgent) -> Task:
        """
        Create task for news collection and analysis

        Args:
            agent: NewsAnalysisAgent instance

        Returns:
            CrewAI Task instance
        """
        detailed_logger.info("Creating news analysis task")
        detailed_logger.debug(f"Creating task for ticker: {self.state.ticker}")
        
        task = Task(
            description=agent.create_task_description(self.state),
            expected_output=f"""
            A curated collection of news articles for {self.state.ticker} including:
            - Up to {self.state.config.num_of_news} relevant articles
            - Article metadata (source, date, title, summary)
            - Relevance scoring and categorization
            - Brief summaries of key market-moving news
            - Sentiment analysis-ready content
            """,
            agent=agent,
            context=[],  # Can be executed in parallel with market data collection
            async_execution=True,  # Enable async execution
            human_input=False,
            output_json=False
        )
        
        detailed_logger.debug("News analysis task created successfully")
        return task

    def create_parallel_data_collection_task(self, market_agent: DataCollectionAgent, news_agent: NewsAnalysisAgent) -> Task:
        """
        Create a composite task for parallel data collection

        Args:
            market_agent: DataCollectionAgent for market data
            news_agent: NewsAnalysisAgent for news data

        Returns:
            CrewAI Task instance that coordinates parallel execution
        """
        return Task(
            description=f"""
            Coordinate parallel data collection for {self.state.ticker} investment analysis.

            This task manages the simultaneous execution of:
            1. Market data collection (price history, financial metrics, statements)
            2. News data collection (recent articles, sentiment analysis preparation)

            Ensure both data streams are collected efficiently and accurately,
            with proper error handling and data validation.
            """,
            expected_output=f"""
            A unified data package containing:
            - Market data (price history, financial metrics, statements)
            - News data (articles, metadata, relevance scores)
            - Data quality assessment and completeness report
            - Timestamps and collection metadata
            - Error logs and data gap analysis
            """,
            agent=market_agent,  # Use market agent as coordinator
            context=[],
            async_execution=True,
            human_input=False,
            output_json=False
        )


def create_market_data_collection_task(state: InvestmentState, agent: DataCollectionAgent) -> Task:
    """
    Convenience function to create market data collection task

    Args:
        state: Current investment state
        agent: DataCollectionAgent instance

    Returns:
        CrewAI Task instance
    """
    task_factory = DataCollectionTasks(state)
    return task_factory.create_market_data_collection_task(agent)


def create_news_analysis_task(state: InvestmentState, agent: NewsAnalysisAgent) -> Task:
    """
    Convenience function to create news analysis task

    Args:
        state: Current investment state
        agent: NewsAnalysisAgent instance

    Returns:
        CrewAI Task instance
    """
    task_factory = DataCollectionTasks(state)
    return task_factory.create_news_analysis_task(agent)


def create_parallel_data_tasks(state: InvestmentState, market_agent: DataCollectionAgent, news_agent: NewsAnalysisAgent) -> list:
    """
    Create parallel data collection tasks

    Args:
        state: Current investment state
        market_agent: DataCollectionAgent for market data
        news_agent: NewsAnalysisAgent for news data

    Returns:
        List of CrewAI Task instances
    """
    task_factory = DataCollectionTasks(state)

    # Create individual tasks
    market_task = task_factory.create_market_data_collection_task(market_agent)
    news_task = task_factory.create_news_analysis_task(news_agent)

    # Configure for parallel execution
    market_task.async_execution = True
    news_task.async_execution = True

    return [market_task, news_task]


# Task result processing functions
def process_market_data_result(task_result: Any, state: InvestmentState) -> Dict[str, Any]:
    """
    Process market data collection task result

    Args:
        task_result: Raw task result
        state: Current investment state

    Returns:
        Processed result dictionary
    """
    detailed_logger.info("Processing market data result")
    detailed_logger.debug(f"Processing result for ticker: {state.ticker}, run_id: {state.run_id}")
    detailed_logger.debug(f"Task result type: {type(task_result)}")
    
    try:
        if isinstance(task_result, str):
            import json
            detailed_logger.debug("Parsing task result as JSON")
            result = json.loads(task_result)
        else:
            result = task_result

        # Update state with processed data
        if isinstance(result, dict):
            state.data_cache['market_data_collection'] = result
            state.update_analysis_result('market_data_collection', result)
            detailed_logger.debug("Updated state with market data collection result")

        detailed_logger.info("Market data result processed successfully")
        return {
            "success": True,
            "data": result,
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker
        }

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker
        }
        state.update_analysis_result('market_data_collection', error_result)
        logger.error(f"Failed to process market data result: {str(e)}")
        detailed_logger.error(f"Failed to process market data result", exc_info=True)
        return error_result


def process_news_analysis_result(task_result: Any, state: InvestmentState) -> Dict[str, Any]:
    """
    Process news analysis task result

    Args:
        task_result: Raw task result
        state: Current investment state

    Returns:
        Processed result dictionary
    """
    try:
        if isinstance(task_result, str):
            import json
            result = json.loads(task_result)
        else:
            result = task_result

        # Update state with processed data
        if isinstance(result, dict):
            state.data_cache['news_analysis_result'] = result
            state.update_analysis_result('news_analysis', result)

        return {
            "success": True,
            "data": result,
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker
        }

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker
        }
        state.update_analysis_result('news_analysis', error_result)
        return error_result