"""
Research Agents for CrewAI Investment System

This module defines research agents that analyze data from different perspectives
and engage in debates to reach balanced investment conclusions.
"""

from typing import List, Dict, Any, Optional
from crewai import Agent
from datetime import datetime

from .base import ResearchAgent
from ..config.state import InvestmentState
from ..tools.research_tools import BullishResearchTool, BearishResearchTool, DebateModerationTool


class BullishResearchAgent(ResearchAgent):
    """
    Bullish Research Agent that analyzes data from optimistic perspective
    and generates bullish investment thesis
    """

    def __init__(self, tools: Optional[List] = None):
        super().__init__(
            role="Senior Bullish Research Analyst",
            goal="Analyze investment opportunities from an optimistic perspective, identify growth potential, and construct compelling bullish investment theses",
            backstory="""You are an experienced bullish research analyst with a talent for identifying
            undervalued opportunities and growth potential. You excel at finding positive signals in
            market data and constructing optimistic investment narratives. While you naturally lean
            bullish, you maintain analytical rigor and avoid irrational exuberance.""",
            research_type="bullish_research",
            tools=tools or [BullishResearchTool()],
            verbose=True
        )

    def create_task_description(self, state: InvestmentState) -> str:
        """Create task description for bullish research analysis"""
        return f"""
        Conduct comprehensive bullish research analysis for {state.ticker} investment opportunity.

        Your task involves:
        1. Analyzing technical indicators from bullish perspective - identifying momentum, breakouts, and upward trends
        2. Evaluating fundamental strengths - growth potential, competitive advantages, and financial health
        3. Assessing positive sentiment factors - market optimism, favorable news, and investor sentiment
        4. Identifying valuation opportunities - potential undervaluation and growth catalysts

        Available data includes:
        - Technical analysis results with signals and indicators
        - Fundamental analysis with financial metrics and ratios
        - Market sentiment and news analysis
        - Valuation metrics and comparisons

        Based on this analysis, develop a compelling bullish investment thesis with:
        - Clear bullish arguments and supporting evidence
        - Confidence assessment for each bullish point
        - Identification of key growth catalysts and drivers
        - Risk factors that could undermine the bullish thesis

        Analysis period: {state.start_date} to {state.end_date}
        Current portfolio: Cash ${state.portfolio.cash:,.0f}, {state.portfolio.stock_position} shares
        """


class BearishResearchAgent(ResearchAgent):
    """
    Bearish Research Agent that analyzes data from pessimistic perspective
    and identifies risks and cautionary factors
    """

    def __init__(self, tools: Optional[List] = None):
        super().__init__(
            role="Senior Bearish Research Analyst",
            goal="Analyze investment risks from a cautious perspective, identify potential threats, and provide balanced risk assessment",
            backstory="""You are a seasoned bearish research analyst specializing in risk identification
            and conservative analysis. You have a keen eye for spotting red flags, overvaluations,
            and potential pitfalls. While you naturally lean toward caution, you maintain objectivity
            and avoid undue pessimism. Your analysis helps prevent costly investment mistakes.""",
            research_type="bearish_research",
            tools=tools or [BearishResearchTool()],
            verbose=True
        )

    def create_task_description(self, state: InvestmentState) -> str:
        """Create task description for bearish research analysis"""
        return f"""
        Conduct comprehensive bearish research analysis for {state.ticker} investment opportunity.

        Your task involves:
        1. Analyzing technical indicators from bearish perspective - identifying weaknesses, breakdowns, and downward trends
        2. Evaluating fundamental concerns - financial risks, competitive threats, and operational challenges
        3. Assessing negative sentiment factors - market pessimism, unfavorable news, and investor concerns
        4. Identifying valuation risks - potential overvaluation and value traps

        Available data includes:
        - Technical analysis results with signals and indicators
        - Fundamental analysis with financial metrics and ratios
        - Market sentiment and news analysis
        - Valuation metrics and comparisons

        Based on this analysis, develop a comprehensive bearish risk assessment with:
        - Clear bearish arguments and supporting evidence
        - Confidence assessment for each bearish point
        - Identification of key risk factors and potential catalysts
        - Positive factors that could mitigate the bearish thesis

        Analysis period: {state.start_date} to {state.end_date}
        Current portfolio: Cash ${state.portfolio.cash:,.0f}, {state.portfolio.stock_position} shares
        """


class DebateModeratorAgent(ResearchAgent):
    """
    Debate Moderator Agent that facilitates balanced discussions between
    bullish and bearish perspectives to reach informed investment conclusions
    """

    def __init__(self, tools: Optional[List] = None):
        super().__init__(
            role="Investment Debate Moderator",
            goal="Facilitate balanced investment debates, synthesize diverse perspectives, and reach well-reasoned investment conclusions",
            backstory="""You are an expert investment debate moderator with exceptional analytical
            and reasoning skills. You excel at evaluating competing investment theses, identifying
            the strongest arguments, and synthesizing diverse viewpoints into balanced conclusions.
            You maintain objectivity while applying critical thinking to assess the merits of different
            perspectives.""",
            research_type="debate_moderation",
            tools=tools or [DebateModerationTool()],
            verbose=True
        )

    def create_task_description(self, state: InvestmentState) -> str:
        """Create task description for debate moderation"""
        return f"""
        Moderate a comprehensive investment debate for {state.ticker} between bullish and bearish perspectives.

        Your task involves:
        1. Reviewing bullish research arguments and confidence levels
        2. Evaluating bearish risk assessments and cautionary factors
        3. Facilitating a balanced debate that considers all perspectives
        4. Synthesizing conflicting viewpoints into a coherent conclusion
        5. Providing an objective assessment with reasoned final recommendation

        Available inputs include:
        - Bullish research thesis with arguments and confidence scores
        - Bearish risk assessment with identified threats and concerns
        - Technical, fundamental, sentiment, and valuation analysis data
        - Market context and timing considerations

        Your analysis should result in:
        - Balanced debate summary highlighting key arguments from both sides
        - Objective assessment of argument strength and credibility
        - Final investment signal (bullish/bearish/neutral) with confidence level
        - Clear reasoning for the conclusion and investment recommendation
        - Identification of critical factors that could change the thesis

        Analysis period: {state.start_date} to {state.end_date}
        Current portfolio: Cash ${state.portfolio.cash:,.0f}, {state.portfolio.stock_position} shares
        """


# Factory functions for creating research agents
def create_bullish_research_agent(tools: Optional[List] = None) -> BullishResearchAgent:
    """Create a bullish research agent"""
    return BullishResearchAgent(tools)


def create_bearish_research_agent(tools: Optional[List] = None) -> BearishResearchAgent:
    """Create a bearish research agent"""
    return BearishResearchAgent(tools)


def create_debate_moderator_agent(tools: Optional[List] = None) -> DebateModeratorAgent:
    """Create a debate moderator agent"""
    return DebateModeratorAgent(tools)


def create_all_research_agents(tools: Optional[Dict[str, List]] = None) -> Dict[str, Agent]:
    """
    Create all research agents for the investment system

    Args:
        tools: Optional dictionary mapping agent types to their tools

    Returns:
        Dictionary of all research agents
    """
    if tools is None:
        tools = {}

    agents = {
        'bullish_research': BullishResearchAgent(tools.get('bullish')),
        'bearish_research': BearishResearchAgent(tools.get('bearish')),
        'debate_moderator': DebateModeratorAgent(tools.get('debate'))
    }

    return agents