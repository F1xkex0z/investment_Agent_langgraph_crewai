"""
CrewAI Tasks for Investment Analysis System

This module defines all tasks that will be executed by the CrewAI agents.
"""

from .data_tasks import *
from .analysis_tasks import *
from .research_tasks import *

__all__ = [
    # Data Tasks
    "create_market_data_collection_task",
    "create_news_analysis_task",
    "create_parallel_data_tasks",

    # Analysis Tasks
    "create_technical_analysis_task",
    "create_fundamental_analysis_task",
    "create_parallel_analysis_tasks",

    # Research Tasks
    "create_bullish_research_task",
    "create_bearish_research_task",
    "create_debate_moderation_task",
    "create_parallel_research_tasks",
    "create_sequential_research_tasks"
]