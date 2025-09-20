"""
Research Tasks for CrewAI Investment System

This module defines tasks for investment research, perspective analysis,
and debate moderation.
"""

from typing import List, Dict, Any, Optional
from crewai import Task
from datetime import datetime

from ..config.state import InvestmentState
from ..agents.research_agents import BullishResearchAgent, BearishResearchAgent, DebateModeratorAgent


class ResearchTasks:
    """Factory class for creating research tasks"""

    def __init__(self, state: InvestmentState):
        self.state = state

    def create_bullish_research_task(self, agent: BullishResearchAgent) -> Task:
        """
        Create task for bullish research analysis

        Args:
            agent: BullishResearchAgent instance

        Returns:
            CrewAI Task instance
        """
        return Task(
            description=agent.create_task_description(self.state),
            expected_output=f"""
            A comprehensive bullish research analysis for {self.state.ticker} including:
            - Bullish arguments and supporting evidence
            - Growth catalysts and positive factors
            - Confidence assessment for each bullish point
            - Risk factors that could undermine the bullish thesis
            - Overall bullish confidence level and investment rationale
            """,
            agent=agent,
            context=[],  # Will be populated with analysis results
            async_execution=True,  # Can execute in parallel with bearish research
            human_input=False,
            output_json=False
        )

    def create_bearish_research_task(self, agent: BearishResearchAgent) -> Task:
        """
        Create task for bearish research analysis

        Args:
            agent: BearishResearchAgent instance

        Returns:
            CrewAI Task instance
        """
        return Task(
            description=agent.create_task_description(self.state),
            expected_output=f"""
            A comprehensive bearish risk assessment for {self.state.ticker} including:
            - Bearish arguments and risk factors
            - Downside threats and cautionary factors
            - Confidence assessment for each bearish point
            - Mitigating factors that could reduce concerns
            - Overall bearish confidence level and risk rationale
            """,
            agent=agent,
            context=[],  # Will be populated with analysis results
            async_execution=True,  # Can execute in parallel with bullish research
            human_input=False,
            output_json=False
        )

    def create_debate_moderation_task(self, agent: DebateModeratorAgent,
                                   bullish_task: Optional[Task] = None,
                                   bearish_task: Optional[Task] = None) -> Task:
        """
        Create task for debate moderation and synthesis

        Args:
            agent: DebateModeratorAgent instance
            bullish_task: Optional bullish research task for context
            bearish_task: Optional bearish research task for context

        Returns:
            CrewAI Task instance
        """
        context = []
        if bullish_task:
            context.append(bullish_task)
        if bearish_task:
            context.append(bearish_task)

        return Task(
            description=agent.create_task_description(self.state),
            expected_output=f"""
            A balanced investment debate analysis for {self.state.ticker} including:
            - Summary of bullish and bearish arguments
            - Assessment of argument strength and credibility
            - Final investment signal (bullish/bearish/neutral) with confidence
            - Detailed reasoning for the conclusion
            - Investment recommendation and position guidance
            - Key factors that could change the investment thesis
            """,
            agent=agent,
            context=context,  # Depends on research results
            async_execution=False,  # Must wait for research to complete
            human_input=False,
            output_json=False
        )

    def create_parallel_research_tasks(self,
                                     bullish_agent: BullishResearchAgent,
                                     bearish_agent: BearishResearchAgent) -> List[Task]:
        """
        Create parallel research tasks for simultaneous execution

        Args:
            bullish_agent: BullishResearchAgent instance
            bearish_agent: BearishResearchAgent instance

        Returns:
            List of CrewAI Task instances configured for parallel execution
        """
        # Create individual research tasks
        bullish_task = self.create_bullish_research_task(bullish_agent)
        bearish_task = self.create_bearish_research_task(bearish_agent)

        # Configure for parallel execution
        bullish_task.async_execution = True
        bearish_task.async_execution = True

        return [bullish_task, bearish_task]

    def create_sequential_research_tasks(self,
                                        bullish_agent: BullishResearchAgent,
                                        bearish_agent: BearishResearchAgent,
                                        debate_agent: DebateModeratorAgent) -> List[Task]:
        """
        Create sequential research tasks with debate moderation

        Args:
            bullish_agent: BullishResearchAgent instance
            bearish_agent: BearishResearchAgent instance
            debate_agent: DebateModeratorAgent instance

        Returns:
            List of CrewAI Task instances configured for sequential execution
        """
        # Create research tasks (parallel)
        research_tasks = self.create_parallel_research_tasks(bullish_agent, bearish_agent)

        # Create debate task that depends on research
        debate_task = self.create_debate_moderation_task(
            debate_agent,
            bullish_task=research_tasks[0],
            bearish_task=research_tasks[1]
        )

        # Configure sequential execution
        for task in research_tasks:
            task.async_execution = True  # Research can be parallel
        debate_task.async_execution = False  # Debate must wait

        return research_tasks + [debate_task]


# Convenience functions for creating research tasks
def create_bullish_research_task(state: InvestmentState, agent: BullishResearchAgent) -> Task:
    """
    Convenience function to create bullish research task

    Args:
        state: Current investment state
        agent: BullishResearchAgent instance

    Returns:
        CrewAI Task instance
    """
    task_factory = ResearchTasks(state)
    return task_factory.create_bullish_research_task(agent)


def create_bearish_research_task(state: InvestmentState, agent: BearishResearchAgent) -> Task:
    """
    Convenience function to create bearish research task

    Args:
        state: Current investment state
        agent: BearishResearchAgent instance

    Returns:
        CrewAI Task instance
    """
    task_factory = ResearchTasks(state)
    return task_factory.create_bearish_research_task(agent)


def create_debate_moderation_task(state: InvestmentState, agent: DebateModeratorAgent,
                                bullish_task: Optional[Task] = None,
                                bearish_task: Optional[Task] = None) -> Task:
    """
    Convenience function to create debate moderation task

    Args:
        state: Current investment state
        agent: DebateModeratorAgent instance
        bullish_task: Optional bullish research task for context
        bearish_task: Optional bearish research task for context

    Returns:
        CrewAI Task instance
    """
    task_factory = ResearchTasks(state)
    return task_factory.create_debate_moderation_task(agent, bullish_task, bearish_task)


def create_parallel_research_tasks(state: InvestmentState,
                                   bullish_agent: BullishResearchAgent,
                                   bearish_agent: BearishResearchAgent) -> List[Task]:
    """
    Convenience function to create parallel research tasks

    Args:
        state: Current investment state
        bullish_agent: BullishResearchAgent instance
        bearish_agent: BearishResearchAgent instance

    Returns:
        List of CrewAI Task instances
    """
    task_factory = ResearchTasks(state)
    return task_factory.create_parallel_research_tasks(bullish_agent, bearish_agent)


def create_sequential_research_tasks(state: InvestmentState,
                                    bullish_agent: BullishResearchAgent,
                                    bearish_agent: BearishResearchAgent,
                                    debate_agent: DebateModeratorAgent) -> List[Task]:
    """
    Convenience function to create sequential research tasks with debate

    Args:
        state: Current investment state
        bullish_agent: BullishResearchAgent instance
        bearish_agent: BearishResearchAgent instance
        debate_agent: DebateModeratorAgent instance

    Returns:
        List of CrewAI Task instances
    """
    task_factory = ResearchTasks(state)
    return task_factory.create_sequential_research_tasks(bullish_agent, bearish_agent, debate_agent)


# Task result processing functions
def process_bullish_research_result(task_result: Any, state: InvestmentState) -> Dict[str, Any]:
    """
    Process bullish research task result

    Args:
        task_result: Raw task result
        state: Current investment state

    Returns:
        Processed result dictionary
    """
    try:
        if isinstance(task_result, str):
            result = json.loads(task_result)
        else:
            result = task_result

        # Update state with processed data
        if isinstance(result, dict):
            state.data_cache['bullish_research_result'] = result
            state.update_analysis_result('bullish_research', result)

            # Extract bullish signal for easy access
            if 'confidence' in result:
                state.data_cache['bullish_signal'] = {
                    'perspective': 'bullish',
                    'confidence': result.get('confidence', 0.5),
                    'timestamp': datetime.now().isoformat()
                }

        return {
            "success": True,
            "data": result,
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "bullish_research"
        }

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "bullish_research"
        }
        state.update_analysis_result('bullish_research', error_result)
        return error_result


def process_bearish_research_result(task_result: Any, state: InvestmentState) -> Dict[str, Any]:
    """
    Process bearish research task result

    Args:
        task_result: Raw task result
        state: Current investment state

    Returns:
        Processed result dictionary
    """
    try:
        if isinstance(task_result, str):
            result = json.loads(task_result)
        else:
            result = task_result

        # Update state with processed data
        if isinstance(result, dict):
            state.data_cache['bearish_research_result'] = result
            state.update_analysis_result('bearish_research', result)

            # Extract bearish signal for easy access
            if 'confidence' in result:
                state.data_cache['bearish_signal'] = {
                    'perspective': 'bearish',
                    'confidence': result.get('confidence', 0.5),
                    'timestamp': datetime.now().isoformat()
                }

        return {
            "success": True,
            "data": result,
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "bearish_research"
        }

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "bearish_research"
        }
        state.update_analysis_result('bearish_research', error_result)
        return error_result


def process_debate_result(task_result: Any, state: InvestmentState) -> Dict[str, Any]:
    """
    Process debate moderation task result

    Args:
        task_result: Raw task result
        state: Current investment state

    Returns:
        Processed result dictionary
    """
    try:
        if isinstance(task_result, str):
            result = json.loads(task_result)
        else:
            result = task_result

        # Update state with processed data
        if isinstance(result, dict):
            state.data_cache['debate_result'] = result
            state.update_analysis_result('debate_analysis', result)

            # Extract final signal for easy access
            if 'final_signal' in result:
                state.data_cache['final_signal'] = {
                    'signal': result.get('final_signal', 'neutral'),
                    'confidence': result.get('confidence', 0.5),
                    'reasoning': result.get('reasoning', ''),
                    'timestamp': datetime.now().isoformat()
                }

            # Update final decision in state
            if 'recommendation' in result:
                state.set_final_decision(result['recommendation'])

        return {
            "success": True,
            "data": result,
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "debate_analysis"
        }

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
            "ticker": state.ticker,
            "analysis_type": "debate_analysis"
        }
        state.update_analysis_result('debate_analysis', error_result)
        return error_result


def combine_research_results(bullish_result: Dict[str, Any],
                           bearish_result: Dict[str, Any],
                           debate_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine all research results into comprehensive analysis

    Args:
        bullish_result: Processed bullish research result
        bearish_result: Processed bearish research result
        debate_result: Processed debate result

    Returns:
        Combined research analysis result
    """
    combined = {
        "ticker": bullish_result.get("ticker") or bearish_result.get("ticker") or debate_result.get("ticker"),
        "research_timestamp": datetime.now().isoformat(),
        "bullish_research": bullish_result.get("data", {}),
        "bearish_research": bearish_result.get("data", {}),
        "debate_analysis": debate_result.get("data", {}),
        "final_assessment": {}
    }

    # Extract final assessment from debate result
    debate_data = debate_result.get("data", {})
    if debate_data:
        combined["final_assessment"] = {
            "signal": debate_data.get("final_signal", "neutral"),
            "confidence": debate_data.get("confidence", 0.5),
            "reasoning": debate_data.get("reasoning", ""),
            "recommendation": debate_data.get("recommendation", {}),
            "bull_confidence": debate_data.get("bull_confidence", 0.0),
            "bear_confidence": debate_data.get("bear_confidence", 0.0),
            "confidence_differential": debate_data.get("confidence_differential", 0.0)
        }

    # Generate research summary
    bull_data = bullish_result.get("data", {})
    bear_data = bearish_result.get("data", {})

    combined["research_summary"] = f"""
    Bullish Research: {bull_data.get('confidence', 0.0):.2f} confidence
    Bearish Research: {bear_data.get('confidence', 0.0):.2f} confidence

    Final Signal: {combined['final_assessment'].get('signal', 'neutral').upper()}
    Final Confidence: {combined['final_assessment'].get('confidence', 0.0):.2f}

    Key Reasoning: {combined['final_assessment'].get('reasoning', 'No reasoning provided')}
    """

    return combined