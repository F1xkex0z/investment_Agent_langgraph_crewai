"""
CrewAI Investment System - Main Entry Point

This module provides the main entry point for the CrewAI-based investment system.
"""

import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from crewai import Crew, Process

from .config.state import InvestmentState, InvestmentConfig, PortfolioState
from .config.llm_config import get_default_llm_config
from .config.crew_config import get_crew_config
from .agents.data_agents import DataCollectionAgent, NewsAnalysisAgent
from .agents.analysis_agents import TechnicalAnalysisAgent, FundamentalAnalysisAgent
from .agents.research_agents import BullishResearchAgent, BearishResearchAgent, DebateModeratorAgent
from .tasks.data_tasks import create_parallel_data_tasks
from .tasks.analysis_tasks import create_parallel_analysis_tasks
from .tasks.research_tasks import create_sequential_research_tasks

from src.utils.logging_config import setup_logger, setup_detailed_logger

logger = setup_logger('crewai_main')
detailed_logger = setup_detailed_logger('crewai_main')


class CrewAIInvestmentSystem:
    """
    Main CrewAI Investment System that orchestrates all agents and tasks
    """

    def __init__(self, config: Optional[InvestmentConfig] = None):
        """Initialize the CrewAI investment system"""
        self.config = config or InvestmentConfig()
        self.llm_config = get_default_llm_config()
        self.crew_config = get_crew_config()
        self.logger = setup_logger('crewai_system')
        self.detailed_logger = setup_detailed_logger('crewai_system')
        
        self.logger.info("Initializing CrewAI Investment System")
        self.detailed_logger.debug(f"System config: {self.config.to_dict()}")
        self.detailed_logger.debug(f"LLM config: {self.llm_config}")
        self.detailed_logger.debug(f"Crew config: {self.crew_config}")

    def run_analysis(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        portfolio: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run complete investment analysis using CrewAI

        Args:
            ticker: Stock ticker symbol
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            portfolio: Portfolio state dictionary

        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        self.logger.info(f"Starting CrewAI analysis for {ticker}")
        self.detailed_logger.debug(f"Analysis parameters - Ticker: {ticker}, Start: {start_date}, End: {end_date}")
        self.detailed_logger.debug(f"Portfolio: {portfolio}")
        
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        self.detailed_logger.info(f"Generated run ID: {run_id}")

        # Set default dates
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)
        end_date = end_date or yesterday.strftime('%Y-%m-%d')

        if not start_date:
            start_date_obj = yesterday - timedelta(days=365)
            start_date = start_date_obj.strftime('%Y-%m-%d')

        self.detailed_logger.debug(f"Final dates - Start: {start_date}, End: {end_date}")

        # Initialize portfolio
        if portfolio is None:
            portfolio = PortfolioState(
                cash=self.config.initial_capital,
                stock_position=self.config.initial_position
            )
        else:
            portfolio = PortfolioState(**portfolio)

        # Create investment state
        investment_state = InvestmentState(
            ticker=ticker,
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            config=self.config,
            portfolio=portfolio
        )

        self.logger.info(f"Starting CrewAI analysis for {ticker} (Run ID: {run_id})")
        self.detailed_logger.debug(f"Investment state created: {investment_state.to_dict()}")

        try:
            # Create agents
            self.detailed_logger.info("Creating agents...")
            agents_start_time = time.time()
            agents = self._create_agents()
            agents_end_time = time.time()
            self.detailed_logger.info(f"Agents created successfully in {agents_end_time - agents_start_time:.2f} seconds")
            self.detailed_logger.debug(f"Created agents: {list(agents.keys())}")

            # Create tasks
            self.detailed_logger.info("Creating tasks...")
            tasks_start_time = time.time()
            tasks = self._create_tasks(investment_state, agents)
            tasks_end_time = time.time()
            self.detailed_logger.info(f"Tasks created successfully in {tasks_end_time - tasks_start_time:.2f} seconds")
            self.detailed_logger.debug(f"Created {len(tasks)} tasks")

            # Create and run crew
            self.detailed_logger.info("Creating and running crew...")
            crew_start_time = time.time()
            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                verbose=True,
                process=Process.sequential  # Start with sequential for simplicity
            )
            self.detailed_logger.debug(f"Crew configuration: {crew.__dict__}")

            # Execute analysis
            self.detailed_logger.info("Executing crew analysis...")
            execution_start_time = time.time()
            result = crew.kickoff()
            execution_end_time = time.time()
            self.detailed_logger.info(f"Crew execution completed in {execution_end_time - execution_start_time:.2f} seconds")
            self.detailed_logger.debug(f"Crew result type: {type(result)}")

            # Process results
            self.detailed_logger.info("Processing results...")
            processing_start_time = time.time()
            final_result = self._process_results(result, investment_state)
            processing_end_time = time.time()
            self.detailed_logger.info(f"Results processed in {processing_end_time - processing_start_time:.2f} seconds")

            end_time = time.time()
            self.logger.info(f"Analysis completed successfully for {ticker} in {end_time - start_time:.2f} seconds")
            self.detailed_logger.info(f"Complete analysis execution time: {end_time - start_time:.2f} seconds")
            
            return final_result

        except Exception as e:
            self.logger.error(f"Analysis failed for {ticker}: {str(e)}")
            self.detailed_logger.error(f"Analysis failed for {ticker}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat()
            }

    def _create_agents(self) -> Dict[str, Any]:
        """Create all required agents"""
        self.detailed_logger.info("Starting agent creation process")
        agents = {}

        # Data collection agents
        self.detailed_logger.debug("Creating data collection agents")
        agents['data_collection'] = DataCollectionAgent()
        agents['news_analysis'] = NewsAnalysisAgent()

        # Analysis agents
        self.detailed_logger.debug("Creating analysis agents")
        agents['technical_analysis'] = TechnicalAnalysisAgent()
        agents['fundamental_analysis'] = FundamentalAnalysisAgent()

        # Research agents
        self.detailed_logger.debug("Creating research agents")
        agents['bullish_research'] = BullishResearchAgent()
        agents['bearish_research'] = BearishResearchAgent()
        agents['debate_moderator'] = DebateModeratorAgent()

        self.detailed_logger.info(f"Agent creation completed. Total agents created: {len(agents)}")
        self.detailed_logger.debug(f"Agent types: {list(agents.keys())}")
        
        return agents

    def _create_tasks(self, state: InvestmentState, agents: Dict[str, Any]) -> list:
        """Create all required tasks"""
        self.detailed_logger.info("Starting task creation process")
        self.detailed_logger.debug(f"Creating tasks for state: {state.ticker} with run_id: {state.run_id}")
        tasks = []

        # Data collection tasks (parallel)
        self.detailed_logger.debug("Creating data collection tasks")
        data_tasks = create_parallel_data_tasks(
            state,
            agents['data_collection'],
            agents['news_analysis']
        )
        tasks.extend(data_tasks)
        self.detailed_logger.debug(f"Created {len(data_tasks)} data collection tasks")

        # Analysis tasks (parallel)
        self.detailed_logger.debug("Creating analysis tasks")
        analysis_tasks = create_parallel_analysis_tasks(
            state,
            agents['technical_analysis'],
            agents['fundamental_analysis']
        )
        tasks.extend(analysis_tasks)
        self.detailed_logger.debug(f"Created {len(analysis_tasks)} analysis tasks")

        # Research tasks (sequential with debate)
        self.detailed_logger.debug("Creating research tasks")
        research_tasks = create_sequential_research_tasks(
            state,
            agents['bullish_research'],
            agents['bearish_research'],
            agents['debate_moderator']
        )
        tasks.extend(research_tasks)
        self.detailed_logger.debug(f"Created {len(research_tasks)} research tasks")

        self.detailed_logger.info(f"Task creation completed. Total tasks created: {len(tasks)}")
        return tasks

    def _process_results(self, crew_result: Any, state: InvestmentState) -> Dict[str, Any]:
        """Process crew execution results"""
        self.detailed_logger.info("Starting results processing")
        self.detailed_logger.debug(f"Processing results for ticker: {state.ticker}, run_id: {state.run_id}")
        self.detailed_logger.debug(f"Crew result type: {type(crew_result)}")
        
        try:
            # Create comprehensive result
            self.detailed_logger.debug("Building comprehensive result structure")
            result = {
                "success": True,
                "ticker": state.ticker,
                "run_id": state.run_id,
                "timestamp": datetime.now().isoformat(),
                "analysis_period": {
                    "start_date": state.start_date,
                    "end_date": state.end_date
                },
                "portfolio": state.portfolio.to_dict(),
                "data_collection": state.data_cache.get('market_data_collection', {}),
                "news_analysis": state.data_cache.get('news_analysis_result', {}),
                "technical_analysis": state.data_cache.get('technical_analysis_result', {}),
                "fundamental_analysis": state.data_cache.get('fundamental_analysis_result', {}),
                "bullish_research": state.data_cache.get('bullish_research_result', {}),
                "bearish_research": state.data_cache.get('bearish_research_result', {}),
                "debate_analysis": state.data_cache.get('debate_result', {}),
                "final_recommendation": self._generate_final_recommendation(state),
                "crew_output": str(crew_result)  # Raw crew output for debugging
            }

            self.detailed_logger.debug("Result structure built successfully")
            self.detailed_logger.debug(f"Data cache keys: {list(state.data_cache.keys())}")
            
            # Log key components of the result
            self.detailed_logger.debug(f"Portfolio status: {result['portfolio']}")
            self.detailed_logger.debug(f"Final recommendation action: {result['final_recommendation'].get('action', 'N/A')}")
            self.detailed_logger.debug(f"Final recommendation confidence: {result['final_recommendation'].get('confidence', 'N/A')}")
            
            self.detailed_logger.info("Results processing completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Failed to process results: {str(e)}")
            self.detailed_logger.error(f"Failed to process results", exc_info=True)
            return {
                "success": False,
                "error": f"Result processing failed: {str(e)}",
                "ticker": state.ticker,
                "run_id": state.run_id,
                "timestamp": datetime.now().isoformat()
            }

    def _generate_final_recommendation(self, state: InvestmentState) -> Dict[str, Any]:
        """Generate final investment recommendation based on all analyses including debate results"""
        self.detailed_logger.info("Generating final investment recommendation")
        self.detailed_logger.debug(f"Generating recommendation for ticker: {state.ticker}")
        
        try:
            # Prioritize debate results if available
            self.detailed_logger.debug("Checking for debate results in state data cache")
            debate_result = state.data_cache.get('debate_result', {})
            final_signal = state.data_cache.get('final_signal', {})
            
            self.detailed_logger.debug(f"Debate result present: {bool(debate_result)}, Final signal present: {bool(final_signal)}")

            if final_signal and debate_result:
                self.detailed_logger.info("Using debate-based recommendation")
                # Use debate analysis as primary source
                signal = final_signal.get('signal', 'neutral')
                confidence = final_signal.get('confidence', 0.5)
                reasoning = final_signal.get('reasoning', 'Based on comprehensive debate analysis')
                recommendation = debate_result.get('recommendation', {})

                # Extract action from recommendation
                action = recommendation.get('action', 'hold')

                # Calculate position size based on confidence and portfolio
                max_investment = state.portfolio.cash * 0.2  # Max 20% in single position
                suggested_investment = max_investment * confidence
                
                self.detailed_logger.debug(f"Debate-based recommendation - Action: {action}, Confidence: {confidence}")

                return {
                    "action": action,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "max_investment": max_investment,
                    "suggested_investment": suggested_investment,
                    "position_size_guidance": recommendation.get('position_size_guidance', 'Maintain current position'),
                    "risk_level": recommendation.get('risk_level', 'Medium'),
                    "time_horizon": recommendation.get('time_horizon', 'Medium to Long-term'),
                    "key_considerations": recommendation.get('key_considerations', []),
                    "debate_based": True,
                    "bull_confidence": debate_result.get('bull_confidence', 0.0),
                    "bear_confidence": debate_result.get('bear_confidence', 0.0),
                    "confidence_differential": debate_result.get('confidence_differential', 0.0)
                }

            # Fallback to original logic if debate results not available
            self.detailed_logger.info("Using fallback recommendation logic")
            technical_signal = state.data_cache.get('technical_signal', {})
            fundamental_signal = state.data_cache.get('fundamental_signal', {})

            # Simple signal combination logic
            signals = []
            confidences = []

            if technical_signal:
                signals.append(technical_signal.get('signal', 'neutral'))
                confidences.append(technical_signal.get('confidence', 0.5))

            if fundamental_signal:
                signals.append(fundamental_signal.get('signal', 'neutral'))
                confidences.append(fundamental_signal.get('confidence', 0.5))

            if not signals:
                self.detailed_logger.warning("No signals available for recommendation")
                return {
                    "action": "hold",
                    "confidence": 0.5,
                    "reasoning": "Insufficient analysis data for recommendation",
                    "debate_based": False
                }

            # Calculate weighted recommendation
            bullish_count = signals.count('bullish')
            bearish_count = signals.count('bearish')
            total_signals = len(signals)
            
            self.detailed_logger.debug(f"Signal counts - Bullish: {bullish_count}, Bearish: {bearish_count}, Total: {total_signals}")

            if bullish_count > bearish_count:
                action = "buy"
                confidence = sum(confidences) / len(confidences) if confidences else 0.5
            elif bearish_count > bullish_count:
                action = "sell"
                confidence = sum(confidences) / len(confidences) if confidences else 0.5
            else:
                action = "hold"
                confidence = 0.5

            # Calculate position size based on confidence and portfolio
            max_investment = state.portfolio.cash * 0.2  # Max 20% in single position
            suggested_investment = max_investment * confidence
            
            self.detailed_logger.debug(f"Fallback recommendation - Action: {action}, Confidence: {confidence}")

            return {
                "action": action,
                "confidence": confidence,
                "reasoning": f"Based on {bullish_count} bullish and {bearish_count} bearish signals out of {total_signals} analyses",
                "max_investment": max_investment,
                "suggested_investment": suggested_investment,
                "signal_count": {
                    "bullish": bullish_count,
                    "bearish": bearish_count,
                    "neutral": total_signals - bullish_count - bearish_count
                },
                "debate_based": False
            }

        except Exception as e:
            self.logger.error(f"Failed to generate recommendation: {str(e)}")
            self.detailed_logger.error(f"Failed to generate recommendation", exc_info=True)
            return {
                "action": "hold",
                "confidence": 0.5,
                "reasoning": f"Error generating recommendation: {str(e)}",
                "debate_based": False
            }


def run_investment_crew(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    portfolio: Optional[Dict[str, Any]] = None,
    show_reasoning: bool = False,
    num_of_news: int = 5,
    initial_capital: float = 100000.0,
    initial_position: int = 0
) -> Dict[str, Any]:
    """
    Convenience function to run CrewAI investment analysis

    Args:
        ticker: Stock ticker symbol
        start_date: Analysis start date (YYYY-MM-DD)
        end_date: Analysis end date (YYYY-MM-DD)
        portfolio: Portfolio state dictionary
        show_reasoning: Whether to show detailed reasoning
        num_of_news: Number of news articles to analyze
        initial_capital: Initial cash amount
        initial_position: Initial stock position

    Returns:
        Analysis results dictionary
    """
    # Create configuration
    config = InvestmentConfig(
        show_reasoning=show_reasoning,
        num_of_news=num_of_news,
        initial_capital=initial_capital,
        initial_position=initial_position
    )

    # Create and run system
    system = CrewAIInvestmentSystem(config)
    return system.run_analysis(ticker, start_date, end_date, portfolio)


# Backward compatibility with original system
def run_hedge_fund(
    run_id: str,
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: Dict[str, Any],
    show_reasoning: bool = False,
    num_of_news: int = 5,
    show_summary: bool = False
) -> str:
    """
    Backward compatibility function that matches original run_hedge_fund signature

    Args:
        run_id: Unique run identifier
        ticker: Stock ticker symbol
        start_date: Analysis start date
        end_date: Analysis end date
        portfolio: Portfolio state
        show_reasoning: Whether to show reasoning
        num_of_news: Number of news articles
        show_summary: Whether to show summary

    Returns:
        String result (for compatibility with original system)
    """
    try:
        # Convert to CrewAI format
        config = InvestmentConfig(
            show_reasoning=show_reasoning,
            num_of_news=num_of_news,
            show_summary=show_summary,
            initial_capital=portfolio.get('cash', 100000.0),
            initial_position=portfolio.get('stock', 0)
        )

        system = CrewAIInvestmentSystem(config)
        result = system.run_analysis(ticker, start_date, end_date, portfolio)

        # Convert result to string format for compatibility
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Backward compatibility function failed: {str(e)}")
        return json.dumps({
            "error": str(e),
            "run_id": run_id,
            "ticker": ticker
        }, indent=2)


if __name__ == "__main__":
    # Example usage
    import json

    # Run a sample analysis
    result = run_investment_crew(
        ticker="000001",  # Example stock
        show_reasoning=True,
        num_of_news=5
    )

    print("CrewAI Investment Analysis Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))