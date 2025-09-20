"""
CrewAI Configuration

This module contains Crew-specific configuration settings.
"""

from typing import Dict, Any, List, Optional
from crewai import Crew, Process
from dataclasses import dataclass
import json


@dataclass
class CrewProcessConfig:
    """Configuration for Crew execution process"""
    process_type: str = "hierarchical"  # "sequential", "hierarchical", "parallel"
    verbose: bool = True
    max_rpm: Optional[int] = None  # Rate limiting
    max_execution_time: Optional[int] = None  # In seconds
    memory: bool = True  # Enable crew memory


@dataclass
class CrewAgentConfig:
    """Configuration for individual agent behavior"""
    allow_delegation: bool = True
    cache: bool = True
    max_iter: int = 20
    max_rpm: Optional[int] = None
    verbose: bool = True


@dataclass
class CrewTaskConfig:
    """Configuration for task execution"""
    async_execution: bool = False
    human_input: bool = False
    context: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    output_json: bool = False
    output_pydantic: bool = False
    output_file: Optional[str] = None


class CrewAIConfig:
    """Main configuration class for CrewAI system"""

    def __init__(self):
        self.process_config = CrewProcessConfig()
        self.agent_config = CrewAgentConfig()
        self.task_config = CrewTaskConfig()

        # Agent-specific configurations
        self.agent_configs = {
            "data_collection": CrewAgentConfig(
                allow_delegation=False,
                cache=True,
                max_iter=10
            ),
            "technical_analysis": CrewAgentConfig(
                allow_delegation=False,
                cache=True,
                max_iter=15
            ),
            "fundamental_analysis": CrewAgentConfig(
                allow_delegation=False,
                cache=True,
                max_iter=15
            ),
            "sentiment_analysis": CrewAgentConfig(
                allow_delegation=False,
                cache=True,
                max_iter=15
            ),
            "valuation_analysis": CrewAgentConfig(
                allow_delegation=False,
                cache=True,
                max_iter=15
            ),
            "bull_research": CrewAgentConfig(
                allow_delegation=True,
                cache=True,
                max_iter=20
            ),
            "bear_research": CrewAgentConfig(
                allow_delegation=True,
                cache=True,
                max_iter=20
            ),
            "debate_moderator": CrewAgentConfig(
                allow_delegation=True,
                cache=False,  # Debate should be fresh each time
                max_iter=10
            ),
            "risk_assessment": CrewAgentConfig(
                allow_delegation=False,
                cache=True,
                max_iter=15
            ),
            "macro_analysis": CrewAgentConfig(
                allow_delegation=False,
                cache=True,
                max_iter=15
            ),
            "portfolio_management": CrewAgentConfig(
                allow_delegation=False,
                cache=False,  # Fresh decision each time
                max_iter=25
            )
        }

        # Task-specific configurations
        self.task_configs = {
            "data_collection": CrewTaskConfig(
                async_execution=False,
                human_input=False,
                cache=True
            ),
            "parallel_analysis": CrewTaskConfig(
                async_execution=True,  # Enable parallel execution
                human_input=False,
                cache=True
            ),
            "research_debate": CrewTaskConfig(
                async_execution=False,
                human_input=False,
                cache=False
            ),
            "risk_assessment": CrewTaskConfig(
                async_execution=False,
                human_input=False,
                cache=True
            ),
            "final_decision": CrewTaskConfig(
                async_execution=False,
                human_input=False,
                cache=False
            )
        }

    def get_agent_config(self, agent_type: str) -> CrewAgentConfig:
        """Get configuration for specific agent type"""
        return self.agent_configs.get(agent_type, self.agent_config)

    def get_task_config(self, task_type: str) -> CrewTaskConfig:
        """Get configuration for specific task type"""
        return self.task_configs.get(task_type, self.task_config)

    def create_crew_config(self, process_type: str = "hierarchical") -> Dict[str, Any]:
        """Create Crew configuration dictionary"""
        return {
            "process": Process.hierarchical if process_type == "hierarchical" else Process.sequential,
            "verbose": self.process_config.verbose,
            "max_rpm": self.process_config.max_rpm,
            "memory": self.process_config.memory
        }

    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        config_dict = {
            "process_config": self.process_config.__dict__,
            "agent_config": self.agent_config.__dict__,
            "task_config": self.task_config.__dict__,
            "agent_configs": {k: v.__dict__ for k, v in self.agent_configs.items()},
            "task_configs": {k: v.__dict__ for k, v in self.task_configs.items()}
        }
        return json.dumps(config_dict, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CrewAIConfig':
        """Create configuration from JSON string"""
        config_dict = json.loads(json_str)
        config = cls()

        # Load process config
        if "process_config" in config_dict:
            for key, value in config_dict["process_config"].items():
                setattr(config.process_config, key, value)

        # Load agent config
        if "agent_config" in config_dict:
            for key, value in config_dict["agent_config"].items():
                setattr(config.agent_config, key, value)

        # Load task config
        if "task_config" in config_dict:
            for key, value in config_dict["task_config"].items():
                setattr(config.task_config, key, value)

        return config


def get_crew_config() -> CrewAIConfig:
    """Get default CrewAI configuration"""
    return CrewAIConfig()


def get_analysis_crew_config() -> Dict[str, Any]:
    """Get specific configuration for analysis crew"""
    return {
        "process": Process.parallel,  # Parallel analysis
        "verbose": True,
        "memory": True
    }


def get_research_crew_config() -> Dict[str, Any]:
    """Get specific configuration for research crew"""
    return {
        "process": Process.sequential,  # Sequential research and debate
        "verbose": True,
        "memory": True
    }


def get_main_crew_config() -> Dict[str, Any]:
    """Get specific configuration for main investment crew"""
    return {
        "process": Process.hierarchical,  # Hierarchical decision making
        "verbose": True,
        "memory": True
    }