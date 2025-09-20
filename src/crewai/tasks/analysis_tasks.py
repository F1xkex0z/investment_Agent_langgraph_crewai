"""
Analysis Tasks for CrewAI Investment System

This module defines tasks for technical, fundamental, sentiment, and valuation analysis.
"""

from typing import List, Dict, Any
from crewai import Task
from datetime import datetime

from ..config.state import InvestmentState
from ..agents.analysis_agents import TechnicalAnalysisAgent, FundamentalAnalysisAgent


class AnalysisTasks:
    """Factory class for creating analysis tasks"""

    def __init__(self, state: InvestmentState):
        self.state = state

    def create_technical_analysis_task(self, agent: TechnicalAnalysisAgent) -> Task:
        """
        Create task for technical analysis

        Args:
            agent: TechnicalAnalysisAgent instance

        Returns:
            CrewAI Task instance
        """
        return Task(
            description=agent.create_task_description(self.state),
            expected_output=f"""
            A comprehensive technical analysis report for {self.state.ticker} including:
            - Technical indicator calculations (MACD, RSI, Bollinger Bands, etc.)
            - Trading signals (bullish/bearish/neutral) with confidence levels
            - Support and resistance levels
            - Trend analysis and market structure assessment
            - Volume analysis and pattern recognition
            - Overall trading recommendation with detailed reasoning
            """,
            agent=agent,
            context=[],  # Will be populated with data collection results
            async_execution=True,  # Enable parallel execution
            human_input=False,
            output_json=False
        )

    def create_fundamental_analysis_task(self, agent: FundamentalAnalysisAgent) -> Task:
        """
        Create task for fundamental analysis

        Args:
            agent: FundamentalAnalysisAgent instance

        Returns:
            CrewAI Task instance
        """
        return Task(
            description=agent.create_task_description(self.state),
            expected_output=f"""
            A comprehensive fundamental analysis report for {self.state.ticker} including:
            - Financial health assessment (liquidity, solvency, efficiency)
            - Profitability analysis (margins, returns, operational efficiency)
            - Growth trends and sustainability assessment
            - Competitive position and business model analysis
            - Management quality evaluation
            - Valuation assessment and intrinsic value calculation
            - Investment recommendation (buy/hold/sell) with detailed reasoning
            """,
            agent=agent,
            context=[],  # Will be populated with data collection results
            async_execution=True,  # Enable parallel execution
            human_input=False,
            output_json=False
        )

    def create_parallel_analysis_tasks(self, technical_agent: TechnicalAnalysisAgent, fundamental_agent: FundamentalAnalysisAgent) -> List[Task]:
        """
        Create parallel analysis tasks for simultaneous execution

        Args:
            technical_agent: TechnicalAnalysisAgent instance
            fundamental_agent: FundamentalAnalysisAgent instance

        Returns:
            List of CrewAI Task instances configured for parallel execution
        """
        # Create individual tasks
        technical_task = self.create_technical_analysis_task(technical_agent)
        fundamental_task = self.create_fundamental_analysis_task(fundamental_agent)

        # Both tasks can execute in parallel as they don't depend on each other
        technical_task.async_execution = True
        fundamental_task.async_execution = True

        return [technical_task, fundamental_task]

    def create_sequential_analysis_tasks(self, technical_agent: TechnicalAnalysisAgent, fundamental_agent: FundamentalAnalysisAgent) -> List[Task]:
        """
        Create sequential analysis tasks where fundamental depends on technical

        Args:
            technical_agent: TechnicalAnalysisAgent instance
            fundamental_agent: FundamentalAnalysisAgent instance

        Returns:
            List of CrewAI Task instances configured for sequential execution
        """
        # Create individual tasks
        technical_task = self.create_technical_analysis_task(technical_agent)
        fundamental_task = self.create_fundamental_analysis_task(fundamental_agent)

        # Configure sequential execution - fundamental depends on technical
        technical_task.async_execution = False
        fundamental_task.async_execution = False
        fundamental_task.context = [technical_task]  # Fundamental analysis can reference technical results

        return [technical_task, fundamental_task]


def create_technical_analysis_task(state: InvestmentState, agent: TechnicalAnalysisAgent) -> Task:
    """
    Convenience function to create technical analysis task

    Args:
        state: Current investment state
        agent: TechnicalAnalysisAgent instance

    Returns:
        CrewAI Task instance
    """
    task_factory = AnalysisTasks(state)
    return task_factory.create_technical_analysis_task(agent)


def create_fundamental_analysis_task(state: InvestmentState, agent: FundamentalAnalysisAgent) -> Task:
    """
    Convenience function to create fundamental analysis task

    Args:
        state: Current investment state
        agent: FundamentalAnalysisAgent instance

    Returns:
        CrewAI Task instance
    """
    task_factory = AnalysisTasks(state)
    return task_factory.create_fundamental_analysis_task(agent)


def create_parallel_analysis_tasks(state: InvestmentState, technical_agent: TechnicalAnalysisAgent, fundamental_agent: FundamentalAnalysisAgent) -> List[Task]:
    """
    Convenience function to create parallel analysis tasks

    Args:
        state: Current investment state
        technical_agent: TechnicalAnalysisAgent instance
        fundamental_agent: FundamentalAnalysisAgent instance

    Returns:
        List of CrewAI Task instances
    """
    task_factory = AnalysisTasks(state)
    return task_factory.create_parallel_analysis_tasks(technical_agent, fundamental_agent)


def create_sequential_analysis_tasks(state: InvestmentState, technical_agent: TechnicalAnalysisAgent, fundamental_agent: FundamentalAnalysisAgent) -> List[Task]:
    """
    Convenience function to create sequential analysis tasks

    Args:
        state: Current investment state
        technical_agent: TechnicalAnalysisAgent instance
        fundamental_agent: FundamentalAnalysisAgent instance

    Returns:
        List of CrewAI Task instances
    """
    task_factory = AnalysisTasks(state)
    return task_factory.create_sequential_analysis_tasks(technical_agent, fundamental_agent)


# Task result processing functions
def process_technical_analysis_result(task_result: Any, state: InvestmentState) -> Dict[str, Any]:
    """
    Process technical analysis task result

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
            state.data_cache['technical_analysis_result'] = result
            state.update_analysis_result('technical_analysis', result)

            # Extract trading signal for easy access
            if 'signal' in result:
                state.data_cache['technical_signal'] = {
                    'signal': result['signal'],
                    'confidence': result.get('confidence', 0.5),
                    'timestamp': datetime.now().isoformat()
                }

        return {
            "success": True,
            "data": result,
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "technical"
        }

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "technical"
        }
        state.update_analysis_result('technical_analysis', error_result)
        return error_result


def process_fundamental_analysis_result(task_result: Any, state: InvestmentState) -> Dict[str, Any]:
    """
    Process fundamental analysis task result

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
            state.data_cache['fundamental_analysis_result'] = result
            state.update_analysis_result('fundamental_analysis', result)

            # Extract investment recommendation for easy access
            if 'signal' in result:
                state.data_cache['fundamental_signal'] = {
                    'signal': result['signal'],
                    'confidence': result.get('confidence', 0.5),
                    'timestamp': datetime.now().isoformat()
                }

        return {
            "success": True,
            "data": result,
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "fundamental"
        }

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "fundamental"
        }
        state.update_analysis_result('fundamental_analysis', error_result)
        return error_result


def combine_analysis_results(technical_result: Dict[str, Any], fundamental_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine technical and fundamental analysis results

    Args:
        technical_result: Processed technical analysis result
        fundamental_result: Processed fundamental analysis result

    Returns:
        Combined analysis result
    """
    combined = {
        "ticker": technical_result.get("ticker") or fundamental_result.get("ticker"),
        "analysis_timestamp": datetime.now().isoformat(),
        "technical_analysis": technical_result.get("data", {}),
        "fundamental_analysis": fundamental_result.get("data", {}),
        "combined_signal": "neutral",
        "combined_confidence": 0.0,
        "analysis_summary": ""
    }

    # Extract signals
    tech_signal = technical_result.get("data", {}).get("signal", "neutral")
    fund_signal = fundamental_result.get("data", {}).get("signal", "neutral")
    tech_confidence = technical_result.get("data", {}).get("confidence", 0.5)
    fund_confidence = fundamental_result.get("data", {}).get("confidence", 0.5)

    # Calculate combined signal
    if tech_signal == fund_signal:
        combined["combined_signal"] = tech_signal
        combined["combined_confidence"] = (tech_confidence + fund_confidence) / 2
    else:
        # Signals differ - use confidence weighted approach
        if tech_confidence > fund_confidence:
            combined["combined_signal"] = tech_signal
            combined["combined_confidence"] = tech_confidence * 0.7
        else:
            combined["combined_signal"] = fund_signal
            combined["combined_confidence"] = fund_confidence * 0.7

    # Generate summary
    combined["analysis_summary"] = f"""
    Technical Analysis: {tech_signal.upper()} (confidence: {tech_confidence:.2f})
    Fundamental Analysis: {fund_signal.upper()} (confidence: {fund_confidence:.2f})
    Combined Signal: {combined['combined_signal'].upper()} (confidence: {combined['combined_confidence']:.2f})
    """

    return combined