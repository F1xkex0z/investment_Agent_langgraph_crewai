# CrewAI A-Share Investment Analysis System - Qwen Context

## Project Overview

This is a CrewAI-based multi-agent system for A-Share (Chinese stock market) investment analysis. The system uses 12 specialized agents working together to provide comprehensive investment recommendations through technical analysis, fundamental analysis, sentiment analysis, and valuation analysis.

### Key Features
- **Multi-Agent Architecture**: 12 specialized agents collaborating through CrewAI framework
- **Comprehensive Analysis**: Technical, fundamental, sentiment, and valuation analysis
- **LLM-Enhanced Decision Making**: Multi-agent debate mechanism with LLM as third-party evaluator
- **Full Risk Management**: End-to-end risk control throughout the investment process
- **Efficient Execution**: Optimized with CrewAI framework for performance

### System Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                    CrewAI Investment System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────────────────────────────┐    │
│  │   Entry     │───▶│         Main Controller             │    │
│  │ (run.py)    │    │  (CrewAIInvestmentSystem)           │    │
│  └─────────────┘    └─────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│                   ┌──────────────────────┐                      │
│                   │  Market Data Agent   │                      │
│                   │   (Data Collection)  │                      │
│                   └──────────────────────┘                      │
│                              │                                  │
│                    ┌─────────┴─────────┐                       │
│                    ▼                   ▼                       │
│        ┌──────────────────┐  ┌──────────────────┐             │
│        │ Technical Agent  │  │ Fundamentals Agent│             │
│        │(Price Analysis)  │  │ (Financial Metrics)│             │
│        └──────────────────┘  └──────────────────┘             │
│                    │                   │                       │
│                    ▼                   ▼                       │
│        ┌──────────────────┐  ┌──────────────────┐             │
│        │ Sentiment Agent  │  │ Valuation Agent  │             │
│        │ (News Analysis)  │  │ (Value Analysis) │             │
│        └──────────────────┘  └──────────────────┘             │
│                              │                   │             │
│                    ┌─────────┴─────────┐         │             │
│                    ▼                   ▼         │             │
│        ┌──────────────────┐  ┌──────────────────┐│             │
│        │ Bull Researcher  │  │ Bear Researcher  ││             │
│        │  (Positive View) │  │ (Negative View)  ││             │
│        └──────────────────┘  └──────────────────┘│             │
│                    │                   │         │             │
│                    └─────────┬─────────┘         │             │
│                              ▼                   │             │
│                   ┌──────────────────────┐       │             │
│                   │    Debate Room       │       │             │
│                   │ (LLM as Moderator)   │       │             │
│                   └──────────────────────┘       │             │
│                              │                   │             │
│                    ┌─────────┴─────────┐         │             │
│                    ▼                   ▼         ▼             │
│        ┌──────────────────┐  ┌──────────────────┐│             │
│        │ Risk Manager     │  │ Macro Analyst    ││             │
│        │ (Risk Control)   │  │ (Economic Data)  ││             │
│        └──────────────────┘  └──────────────────┘│             │
│                    │                   │         │             │
│                    └─────────┬─────────┘         │             │
│                              ▼                   │             │
│                   ┌──────────────────────┐       │             │
│                   │ Portfolio Manager    │       │             │
│                   │ (Final Decision)     │◀──────┘             │
│                   └──────────────────────┘                     │
│                              │                                  │
│                              ▼                                  │
│                   ┌──────────────────────┐                      │
│                   │     Results          │                      │
│                   │ (Investment Advice)  │                      │
│                   └──────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack
- **Framework**: CrewAI (v0.22.0+)
- **Language**: Python 3.x
- **Core Dependencies**: 
  - crewai, crewai-tools
  - langchain, langchain-openai
  - pandas, numpy
  - akshare (for Chinese market data)
  - openai, google-generativeai
- **Development Tools**: pytest, black, isort, flake8

## Project Structure

```
crewai_system/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── base_agent.py      # Base agent class with common functionality
│   │   └── specialized agents (market_data_agent.py, technical_analyst.py, etc.)
│   ├── tasks/            # Task definitions
│   │   ├── base_task.py       # Base task class with common functionality
│   │   └── specific tasks (data_collection_task.py, etc.)
│   ├── tools/            # Tools and data sources
│   │   ├── data_sources.py    # Data source adapters
│   │   └── real_data_sources.py
│   ├── utils/            # Utility functions
│   │   ├── logging_config.py  # Logging configuration
│   │   ├── shared_context.py  # Shared context management
│   │   ├── data_processing.py # Data processing utilities
│   │   └── data_flow_manager.py # Data flow management
│   ├── crews/           # Crew definitions
│   │   └── analysis_crew.py   # Main analysis crew
│   └── main.py          # Main entry point
├── config.py            # System configuration
├── requirements.txt     # Python dependencies
├── run.py              # Execution script
└── README.md          # Documentation
```

## System Architecture

### Agent Architecture
1. **Market Data Agent**: Collects stock prices, financial metrics, and market information
2. **Technical Analyst**: Analyzes price trends and technical indicators
3. **Fundamentals Analyst**: Evaluates company financial metrics and performance
4. **Sentiment Analyst**: Analyzes market news and sentiment
5. **Valuation Analyst**: Performs company valuation and intrinsic value analysis
6. **Bull/Bear Researchers**: Provide opposing market viewpoints
7. **Debate Room**: Coordinates bull/bear debates with LLM as third-party evaluator
8. **Risk Manager**: Assesses risk and sets trading limits
9. **Macro Analyst**: Analyzes macroeconomic environment
10. **Portfolio Manager**: Makes final trading decisions

### Data Flow
1. **Data Collection**: Market data and news collection agents gather information
2. **Analysis**: Specialized analysts process data in parallel
3. **Debate**: Bull/bear researchers present opposing views in a debate
4. **Risk Assessment**: Risk manager evaluates potential risks
5. **Decision**: Portfolio manager makes final investment recommendation

### Shared Context System
The system uses a shared context mechanism to:
- Share data between agents
- Maintain consistency across the analysis
- Track data provenance and quality
- Enable caching and reuse of results

## Core Components

### BaseAgent (src/agents/base_agent.py)
- Abstract base class for all agents
- Provides common functionality like logging, execution tracking, and error handling
- Implements standardized output formatting
- Includes safety mechanisms and retry logic

### BaseTask (src/tasks/base_task.py)
- Abstract base class for all tasks
- Implements retry mechanisms, caching, and validation
- Provides standardized task execution flow
- Supports both sequential and parallel task execution

### SharedContext (src/utils/shared_context.py)
- Thread-safe data sharing mechanism between agents
- Implements TTL-based data expiration
- Provides data change history tracking
- Supports subscription-based notifications

### DataFlowManager (src/utils/data_flow_manager.py)
- Manages data dependencies between tasks
- Ensures data quality and validation
- Handles data storage and retrieval
- Provides data availability checking

## Running the System

### Basic Usage
```bash
# Basic analysis
python run.py --ticker 000001

# Show detailed reasoning
python run.py --ticker 000001 --show-reasoning

# Custom date range
python run.py --ticker 000001 --start-date 2024-01-01 --end-date 2024-12-31

# Custom portfolio
python run.py --ticker 000001 --initial-capital 200000 --initial-position 1000
```

### System Management
```bash
# Show system status
python run.py --status

# Clean up system resources
python run.py --cleanup
```

### Configuration

### Environment Variables (.env)
```
# LLM Configuration
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_COMPATIBLE_API_KEY=your_openai_api_key_here
OPENAI_COMPATIBLE_BASE_URL=your_api_base_url_here
OPENAI_COMPATIBLE_MODEL=your_model_name_here

# System Configuration
LOG_LEVEL=INFO
MAX_WORKERS=4
CACHE_ENABLED=true
CACHE_TTL=3600
```

### Key Configuration Options
- **LLM Settings**: API keys and model configuration
- **Performance**: Worker count and caching settings
- **Logging**: Log level and output configuration
- **Data Sources**: Timeout and retry settings for data retrieval

## Development Guidelines

### Adding New Agents
1. Inherit from `BaseAgent` class:
```python
from crewai_system.src.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Custom Role",
            goal="Agent Goal",
            backstory="Background Story",
            tools=[your_tools]
        )

    def process_task(self, task_context):
        # Implementation
        return result
```

2. Register with Crew team:
```python
from crewai import Crew

crew = Crew(
    agents=[custom_agent, ...],
    tasks=[custom_task, ...],
    process=Process.sequential
)
```

### Adding New Tasks
1. Inherit from `BaseTask` class:
```python
from crewai_system.src.tasks.base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, agent):
        super().__init__(
            description="Task Description",
            expected_output="Expected Output",
            agent=agent
        )

    def get_required_fields(self):
        return ["ticker", "start_date"]

    def execute_task_logic(self, context):
        # Implementation
        return result
```

### Shared Context Usage
```python
from crewai_system.src.utils.shared_context import get_global_context, ContextManager

context = get_global_context()

# Set data
context.set("key", value, source_agent="agent_name")

# Get data
value = context.get("key")

# Use context manager
with ContextManager(context, "agent_name") as ctx:
    ctx.set("key", value)
    result = ctx.get("key")
```

## Performance Optimization

### Caching Mechanism
The system implements multi-level caching:
1. **Data Source Cache**: Reduces duplicate API calls
2. **Task Result Cache**: Avoids recomputation
3. **Context Cache**: Enables data sharing between agents

### Concurrent Execution
- Supports parallel agent execution
- Thread-safe shared context
- Configurable maximum worker threads

## Monitoring and Logging

### Log Levels
```python
import logging
logging.getLogger('crewai_system').setLevel(logging.INFO)
```

### System Monitoring
```bash
# View system status
python run.py --status

# View cache statistics
python -c "from crewai_system.src.tools.data_sources import get_data_adapter; print(get_data_adapter().get_cache_stats())"
```

## Troubleshooting

### Common Issues
1. **API Key Errors**: Check .env file configuration
2. **Data Source Connection Failures**: Check network connection and akshare installation
3. **Memory Issues**: Adjust MAX_WORKERS configuration
4. **Cache Problems**: Use --cleanup to clear cache

### Debug Mode
```bash
# Enable verbose logging
LOG_LEVEL=DEBUG python run.py --ticker 000001

# Show reasoning process
python run.py --ticker 000001 --show-reasoning
```

## Testing

The system includes unit tests in the `tests/` directory. Run tests with:
```bash
pytest
```

## Contributing

1. Fork the project
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License.

## Important Notice

This system is for educational and research purposes only. It does not constitute actual investment advice. Investing carries risks, please make decisions carefully.