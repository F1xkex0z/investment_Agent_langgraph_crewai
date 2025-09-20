"""
CrewAI Crews for Investment Analysis System

This module defines different crews that coordinate groups of agents for specific goals.
"""

from .analysis_crew import *
from .research_crew import *
from .decision_crew import *
from .main_crew import *

__all__ = [
    "AnalysisCrew",
    "ResearchCrew",
    "DecisionCrew",
    "MainInvestmentCrew"
]