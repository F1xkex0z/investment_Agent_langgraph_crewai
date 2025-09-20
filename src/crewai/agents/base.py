"""
Base Agent Classes for CrewAI Investment System

This module defines the base classes for all CrewAI investment agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from crewai import Agent
from crewai.tools import BaseTool
from src.utils.logging_config import setup_logger

from ..config.state import InvestmentState
from ..config.llm_config import get_default_llm_config
from ..config.settings import CrewAISettings


logger = setup_logger('crewai_base_agents')


class BaseInvestmentAgent(Agent, ABC):
    """
    Base class for all investment agents in the CrewAI system.
    Extends CrewAI Agent with investment-specific functionality.
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[BaseTool]] = None,
        llm_config=None,
        agent_type: str = "general",
        **kwargs
    ):
        """
        Initialize base investment agent

        Args:
            role: Agent's role in the investment process
            goal: Agent's specific goal
            backstory: Agent's backstory and expertise
            tools: List of tools available to the agent
            llm_config: LLM configuration (uses default if None)
            agent_type: Type of agent for configuration lookup
            **kwargs: Additional CrewAI Agent parameters
        """
        # Get LLM configuration
        if llm_config is None:
            llm_config = get_default_llm_config()

        # Get agent-specific configuration
        settings = CrewAISettings.from_environment()
        agent_config = settings.get_analysis_config(agent_type)

        # Initialize CrewAI Agent
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            llm=llm_config.get_llm(),
            verbose=kwargs.get('verbose', True),
            allow_delegation=kwargs.get('allow_delegation', False),
            max_iter=kwargs.get('max_iter', 20),
            max_rpm=kwargs.get('max_rpm', 60)
        )

        self.agent_type = agent_type
        self.logger = setup_logger(f'crewai_agent_{agent_type}')

    def preprocess_state(self, state: InvestmentState) -> Dict[str, Any]:
        """
        Preprocess investment state for agent consumption
        Override in subclasses for agent-specific preprocessing

        Args:
            state: Current investment state

        Returns:
            Processed context dictionary for the agent
        """
        context = {
            "ticker": state.ticker,
            "portfolio": state.portfolio.to_dict(),
            "analysis_results": state.analysis_results,
            "research_findings": state.research_findings,
            "debate_results": state.debate_results,
            "config": state.config.to_dict()
        }

        # Add agent-specific data
        if self.agent_type in state.data_cache:
            context["agent_data"] = state.data_cache[self.agent_type]

        return context

    def postprocess_output(self, output: Any, state: InvestmentState) -> Any:
        """
        Postprocess agent output and update state
        Override in subclasses for agent-specific postprocessing

        Args:
            output: Raw output from agent
            state: Current investment state to update

        Returns:
            Processed output
        """
        # Update state with analysis result
        state.update_analysis_result(self.agent_type, output)
        return output

    def create_task_description(self, state: InvestmentState) -> str:
        """
        Create task description based on current state
        Override in subclasses for agent-specific task descriptions

        Args:
            state: Current investment state

        Returns:
            Task description string
        """
        context = self.preprocess_state(state)
        return f"""
        Analyze investment opportunity for {state.ticker} based on the following context:

        Context: {context}

        Portfolio Status: {state.portfolio.to_dict()}

        Provide your analysis as {self.role}.
        """


class DataAgent(BaseInvestmentAgent):
    """Base class for data collection agents"""

    def __init__(self, role: str, goal: str, backstory: str, **kwargs):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            agent_type="data_collection",
            allow_delegation=False,  # Data agents typically don't delegate
            **kwargs
        )


class AnalysisAgent(BaseInvestmentAgent):
    """Base class for analysis agents"""

    def __init__(self, role: str, goal: str, backstory: str, analysis_type: str, **kwargs):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            agent_type=analysis_type,
            allow_delegation=False,  # Analysis agents typically don't delegate
            **kwargs
        )
        self.analysis_type = analysis_type


class ResearchAgent(BaseInvestmentAgent):
    """Base class for research agents"""

    def __init__(self, role: str, goal: str, backstory: str, research_type: str, **kwargs):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            agent_type=research_type,
            allow_delegation=True,  # Research agents can delegate
            **kwargs
        )
        self.research_type = research_type


class DecisionAgent(BaseInvestmentAgent):
    """Base class for decision-making agents"""

    def __init__(self, role: str, goal: str, backstory: str, decision_type: str, **kwargs):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            agent_type=decision_type,
            allow_delegation=False,  # Decision agents typically don't delegate
            max_iter=25,  # Decision agents may need more iterations
            **kwargs
        )
        self.decision_type = decision_type


# Agent factory function
def create_agent(
    agent_type: str,
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[List[BaseTool]] = None,
    **kwargs
) -> BaseInvestmentAgent:
    """
    Factory function to create appropriate agent type

    Args:
        agent_type: Type of agent ("data", "analysis", "research", "decision")
        role: Agent's role
        goal: Agent's goal
        backstory: Agent's backstory
        tools: List of tools
        **kwargs: Additional parameters

    Returns:
        Appropriate agent instance
    """
    if agent_type == "data":
        return DataAgent(role, goal, backstory, tools=tools, **kwargs)
    elif agent_type == "analysis":
        return AnalysisAgent(role, goal, backstory, kwargs.get('analysis_type', 'general'), tools=tools, **kwargs)
    elif agent_type == "research":
        return ResearchAgent(role, goal, backstory, kwargs.get('research_type', 'general'), tools=tools, **kwargs)
    elif agent_type == "decision":
        return DecisionAgent(role, goal, backstory, kwargs.get('decision_type', 'general'), tools=tools, **kwargs)
    else:
        return BaseInvestmentAgent(role, goal, backstory, tools=tools, agent_type=agent_type, **kwargs)


# Agent validation function
def validate_agent_config(agent: BaseInvestmentAgent) -> bool:
    """
    Validate agent configuration

    Args:
        agent: Agent to validate

    Returns:
        True if configuration is valid
    """
    try:
        # Check required attributes
        if not agent.role:
            logger.error("Agent role is required")
            return False

        if not agent.goal:
            logger.error("Agent goal is required")
            return False

        if not agent.backstory:
            logger.error("Agent backstory is required")
            return False

        # Check LLM configuration
        if not agent.llm:
            logger.error("Agent LLM configuration is required")
            return False

        # Agent-specific validation
        if isinstance(agent, DataAgent) and agent.allow_delegation:
            logger.warning("Data agents typically should not allow delegation")

        if isinstance(agent, DecisionAgent) and agent.max_iter < 20:
            logger.warning("Decision agents may need more iterations")

        return True

    except Exception as e:
        logger.error(f"Agent validation failed: {str(e)}")
        return False