"""
CrewAI Agents for Investment Analysis System

This module contains all CrewAI agents specialized for different aspects of investment analysis.
"""

from .data_agents import *
from .analysis_agents import *
from .research_agents import *

__all__ = [
    # Data Agents
    "DataCollectionAgent",
    "NewsAnalysisAgent",

    # Analysis Agents
    "TechnicalAnalysisAgent",
    "FundamentalAnalysisAgent",

    # Research Agents
    "BullishResearchAgent",
    "BearishResearchAgent",
    "DebateModeratorAgent"
]