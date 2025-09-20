"""
CrewAI Investment State Management

This module defines the state management classes for the CrewAI investment system.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import json


@dataclass
class InvestmentConfig:
    """Configuration for investment analysis"""
    show_reasoning: bool = False
    num_of_news: int = 5
    initial_capital: float = 100000.0
    initial_position: int = 0
    show_summary: bool = False

    # LLM配置
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.1

    # 分析配置
    enable_risk_management: bool = True
    enable_macro_analysis: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "show_reasoning": self.show_reasoning,
            "num_of_news": self.num_of_news,
            "initial_capital": self.initial_capital,
            "initial_position": self.initial_position,
            "show_summary": self.show_summary,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "enable_risk_management": self.enable_risk_management,
            "enable_macro_analysis": self.enable_macro_analysis
        }


@dataclass
class PortfolioState:
    """Portfolio state management"""
    cash: float = 100000.0
    stock_position: int = 0
    stock_value: float = 0.0
    total_value: float = 100000.0

    def update_value(self, current_price: float):
        """Update portfolio value based on current stock price"""
        self.stock_value = self.stock_position * current_price
        self.total_value = self.cash + self.stock_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "cash": self.cash,
            "stock_position": self.stock_position,
            "stock_value": self.stock_value,
            "total_value": self.total_value
        }


@dataclass
class InvestmentState:
    """
    Central state management for CrewAI investment system.
    Replaces the original AgentState with a more structured approach.
    """

    # 基本信息
    ticker: str
    run_id: str
    start_date: str
    end_date: str

    # 配置和投资组合
    config: InvestmentConfig
    portfolio: PortfolioState

    # 数据缓存
    data_cache: Dict[str, Any] = field(default_factory=dict)

    # 分析结果
    analysis_results: Dict[str, Any] = field(default_factory=dict)

    # 研究和辩论结果
    research_findings: Dict[str, Any] = field(default_factory=dict)
    debate_results: Dict[str, Any] = field(default_factory=dict)

    # 决策上下文
    decision_context: Dict[str, Any] = field(default_factory=dict)

    # 执行元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 任务执行历史
    task_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default values after creation"""
        if not hasattr(self, 'portfolio') or self.portfolio is None:
            self.portfolio = PortfolioState(
                cash=self.config.initial_capital,
                stock_position=self.config.initial_position
            )

    def update_analysis_result(self, agent_name: str, result: Any):
        """Update analysis result from a specific agent"""
        self.analysis_results[agent_name] = result
        self.updated_at = datetime.now()

        # Add to task history
        self.task_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": "analysis",
            "result_summary": str(result)[:200] if result else None
        })

    def update_research_finding(self, researcher_type: str, finding: Any):
        """Update research finding from bull/bear researchers"""
        self.research_findings[researcher_type] = finding
        self.updated_at = datetime.now()

        self.task_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": researcher_type,
            "type": "research",
            "result_summary": str(finding)[:200] if finding else None
        })

    def update_debate_result(self, debate_result: Any):
        """Update debate room result"""
        self.debate_results = debate_result
        self.updated_at = datetime.now()

        self.task_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "debate_room",
            "type": "debate",
            "result_summary": str(debate_result)[:200] if debate_result else None
        })

    def set_final_decision(self, decision: Any):
        """Set final portfolio management decision"""
        self.decision_context['final_decision'] = decision
        self.updated_at = datetime.now()

        self.task_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "portfolio_manager",
            "type": "decision",
            "result_summary": str(decision)[:200] if decision else None
        })

    def get_agent_messages(self) -> List[Dict[str, Any]]:
        """Get all agent messages for compatibility with original system"""
        messages = []

        # Add analysis results as messages
        for agent_name, result in self.analysis_results.items():
            messages.append({
                "role": "assistant",
                "content": json.dumps(result) if isinstance(result, dict) else str(result),
                "name": agent_name
            })

        # Add research findings as messages
        for researcher_type, finding in self.research_findings.items():
            messages.append({
                "role": "assistant",
                "content": json.dumps(finding) if isinstance(finding, dict) else str(finding),
                "name": researcher_type
            })

        # Add debate result as message
        if self.debate_results:
            messages.append({
                "role": "assistant",
                "content": json.dumps(self.debate_results) if isinstance(self.debate_results, dict) else str(self.debate_results),
                "name": "debate_room"
            })

        return messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "ticker": self.ticker,
            "run_id": self.run_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "config": self.config.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "data_cache": self.data_cache,
            "analysis_results": self.analysis_results,
            "research_findings": self.research_findings,
            "debate_results": self.debate_results,
            "decision_context": self.decision_context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "task_history": self.task_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InvestmentState':
        """Create instance from dictionary"""
        config_data = data.get('config', {})
        portfolio_data = data.get('portfolio', {})

        config = InvestmentConfig(**config_data)
        portfolio = PortfolioState(**portfolio_data)

        # Parse datetime strings
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now()

        return cls(
            ticker=data['ticker'],
            run_id=data['run_id'],
            start_date=data['start_date'],
            end_date=data['end_date'],
            config=config,
            portfolio=portfolio,
            data_cache=data.get('data_cache', {}),
            analysis_results=data.get('analysis_results', {}),
            research_findings=data.get('research_findings', {}),
            debate_results=data.get('debate_results', {}),
            decision_context=data.get('decision_context', {}),
            created_at=created_at,
            updated_at=updated_at,
            task_history=data.get('task_history', [])
        )

    def get_task_output(self, task_name: str) -> Optional[Any]:
        """Get output of a specific task by name"""
        for task_record in reversed(self.task_history):
            if task_record.get('agent') == task_name:
                return task_record.get('result_summary')
        return None

    def set_task_output(self, task_name: str, output: Any):
        """Set output for a specific task (for CrewAI compatibility)"""
        # Determine task type based on name
        if 'research' in task_name.lower():
            self.update_research_finding(task_name, output)
        elif 'debate' in task_name.lower():
            self.update_debate_result(output)
        elif 'portfolio' in task_name.lower() or 'decision' in task_name.lower():
            self.set_final_decision(output)
        else:
            self.update_analysis_result(task_name, output)