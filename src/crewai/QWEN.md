# CrewAI Investment Analysis System - Development Context

## Project Overview

This is a CrewAI-based multi-agent system for A-Share (Chinese stock market) investment analysis and decision making. The system has been migrated from an original LangGraph framework to leverage CrewAI's collaborative multi-agent capabilities while maintaining all original functionality.

### Core Features

1. **Multi-layered Agent Architecture**: Data collection, analysis, research, and decision-making layers
2. **Parallel Processing**: Supports concurrent execution of multiple tasks for improved efficiency
3. **Debate-based Analysis**: Unique bullish/bearish debate mechanism for balanced investment decisions
4. **State Management**: Comprehensive investment state tracking and management
5. **Backward Compatibility**: Maintains API compatibility with the original system

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CrewAI Investment System                 │
├─────────────────────────────────────────────────────────────┤
│  Data Collection Layer       │  Analysis Layer              │
│  ┌─────────────┐            │  ┌─────────────┐             │
│  │DataCollector│            │  │Tech Analyzer│             │
│  └─────────────┘            │  └─────────────┘             │
│  ┌─────────────┐            │  ┌─────────────┐             │
│  │News Analyzer│            │  │Fund Analyzer│             │
│  └─────────────┘            │  └─────────────┘             │
├─────────────────────────────────────────────────────────────┤
│  Research Debate Layer       │  Decision Layer              │
│  ┌─────────────┐            │  ┌─────────────┐             │
│  │BullishAgent │            │  │DebateCoord. │             │
│  └─────────────┘            │  └─────────────┘             │
│  ┌─────────────┤            │                             │
│  │BearishAgent │            │  Final Investment Decision  │
│  └─────────────┘            │                             │
└─────────────────────────────────────────────────────────────┘
```

### Agent Types

1. **Data Collection Layer**
   - `DataCollectionAgent`: Collects market data, financial metrics, and historical prices
   - `NewsAnalysisAgent`: Analyzes news sentiment and market events

2. **Analysis Layer**
   - `TechnicalAnalysisAgent`: Performs technical indicator analysis and trend identification
   - `FundamentalAnalysisAgent`: Conducts fundamental analysis and financial health assessment

3. **Research Debate Layer**
   - `BullishResearchAgent`: Analyzes opportunities from an optimistic perspective
   - `BearishResearchAgent`: Identifies risks from a cautious perspective
   - `DebateModeratorAgent`: Coordinates debate and synthesizes balanced conclusions

## Key Components

### State Management
The system uses `InvestmentState` as the central state management class, which includes:
- Configuration settings (`InvestmentConfig`)
- Portfolio state (`PortfolioState`)
- Data cache for intermediate results
- Analysis results from different agents
- Research findings from bullish/bearish perspectives
- Debate results and final decisions

### Configuration
- `InvestmentConfig`: System configuration including LLM settings, analysis parameters
- LLM configuration supporting both Gemini and OpenAI-compatible APIs
- Crew configuration for agent collaboration settings

### Tools
- Specialized research tools for bullish/bearish analysis and debate moderation
- Market data collection tools
- News analysis tools

## Usage Examples

### Basic Usage
```python
from src.crewai.main import run_investment_crew

# Execute investment analysis
result = run_investment_crew(
    ticker="000001",  # Stock code
    show_reasoning=True,  # Show reasoning process
    num_of_news=5,  # News analysis count
    initial_capital=100000.0  # Initial funds
)

print(f"Investment recommendation: {result['final_recommendation']['action']}")
print(f"Confidence: {result['final_recommendation']['confidence']:.1%}")
```

### Advanced Configuration
```python
from src.crewai.config.state import InvestmentConfig
from src.crewai.main import CrewAIInvestmentSystem

# Create custom configuration
config = InvestmentConfig(
    show_reasoning=True,
    num_of_news=10,
    initial_capital=500000.0,
    initial_position=100,
    max_analysis_workers=4
)

# Create system instance
system = CrewAIInvestmentSystem(config)

# Execute analysis
result = system.run_analysis(
    ticker="000001",
    start_date="2024-01-01",
    end_date="2024-12-01"
)
```

### Backward Compatibility
```python
from src.crewai.main import run_hedge_fund

# Use original API format
result = run_hedge_fund(
    run_id="analysis-001",
    ticker="000001",
    start_date="2024-01-01",
    end_date="2024-12-01",
    portfolio={"cash": 100000.0, "stock": 0},
    show_reasoning=True,
    num_of_news=5
)
```

## Development Guidelines

### Code Structure
- Follow the existing modular architecture with clear separation of concerns
- Maintain backward compatibility when modifying core APIs
- Use the state management system for data sharing between agents
- Implement proper error handling and logging

### Best Practices
1. **Multi-perspective Analysis**: Always consider both bullish and bearish viewpoints
2. **Risk Management**: Implement strict position sizing and risk controls
3. **Data Quality**: Validate all data inputs and handle missing data gracefully
4. **Performance**: Optimize for parallel execution where possible
5. **Testing**: Ensure all changes are thoroughly tested

### Testing
Run the test suite to verify functionality:
```bash
python test_crewai.py
```

Expected output:
```
🧪 CrewAI Investment System Test Suite
==================================================
✅ PASS Basic Functionality
✅ PASS State Management
✅ PASS Analysis Function
✅ PASS Research Agents
✅ PASS Backward Compatibility
📊 Overall: 5/5 tests passed
🎉 All tests passed! CrewAI migration is working correctly.
```

## Environment Setup

### Dependencies
The system requires the following key dependencies:
- CrewAI framework
- Python 3.8+
- API keys for LLM services (Gemini or OpenAI-compatible)

### Configuration
Set environment variables for API access:
```bash
export GEMINI_API_KEY="your_gemini_api_key"
# OR
export OPENAI_COMPATIBLE_API_KEY="your_openai_api_key"
```

## Performance Optimization

1. **Parallel Processing**: Configure `max_analysis_workers` in `InvestmentConfig`
2. **Result Caching**: Enable `cache_results` for repeated analyses
3. **Resource Monitoring**: Monitor CPU and memory usage during execution
4. **Batch Processing**: Process multiple stocks in batches for efficiency

## Troubleshooting

### Common Issues
1. **API Key Configuration**: Verify environment variables are set correctly
2. **Module Import Errors**: Ensure working directory is set to project root
3. **Dependency Conflicts**: Check CrewAI version compatibility
4. **Data Access Issues**: Verify network connectivity and API quotas

### Debugging
Enable detailed logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extension Development

### Adding New Agents
```python
from src.crewai.agents.base import AnalysisAgent
from src.crewai.tools.base_tool import BaseTool

class CustomAnalysisAgent(AnalysisAgent):
    def __init__(self, tools=None):
        super().__init__(
            role="Custom Analysis Expert",
            goal="Perform custom analysis",
            backstory="Expert in custom analysis techniques",
            agent_type="custom_analysis",
            tools=tools or []
        )
```

### Custom Tools
```python
from crewai.tools import BaseTool

class CustomAnalysisTool(BaseTool):
    name: str = "custom_analysis_tool"
    description: str = "Perform custom analysis operations"

    def _run(self, input_data: str) -> str:
        # Implement custom analysis logic
        return json.dumps({"result": "custom_analysis_result"})
```