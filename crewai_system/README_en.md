# CrewAI A-Share Investment Analysis System

This is a CrewAI-based multi-agent system for A-Share (Chinese stock market) investment analysis. The system uses specialized agents working together to provide comprehensive investment recommendations through technical analysis, fundamental analysis, sentiment analysis, and valuation analysis.

## Features

- 🤖 **Multi-Agent Architecture**: Specialized agents collaborating through CrewAI framework
- 📊 **Comprehensive Analysis**: Technical, fundamental, sentiment, and valuation analysis
- 🎯 **LLM-Enhanced Decision Making**: Multi-agent debate mechanism with LLM as third-party evaluator
- 🔒 **Full Risk Management**: End-to-end risk control throughout the investment process
- ⚡ **Efficient Execution**: Optimized with CrewAI framework for performance

## System Architecture Flowchart

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

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd crewai-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys and configuration
   ```

## Usage

### Basic Analysis
```bash
python run.py --ticker 000001
```

### Show Detailed Reasoning
```bash
python run.py --ticker 000001 --show-reasoning
```

### Custom Date Range
```bash
python run.py --ticker 000001 --start-date 2024-01-01 --end-date 2024-12-31
```

### Custom Portfolio
```bash
python run.py --ticker 000001 --initial-capital 200000 --initial-position 1000
```

## Configuration

Configure the system by editing the `.env` file:

```env
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

## Project Structure

```
crewai_system/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── base_agent.py      # Base agent class with common functionality
│   │   └── specialized agents
│   ├── tasks/            # Task definitions
│   │   ├── base_task.py       # Base task class with common functionality
│   │   └── specific tasks
│   ├── tools/            # Tools and data sources
│   │   ├── data_sources.py    # Data source adapters
│   │   └── real_data_sources.py
│   ├── utils/            # Utility functions
│   │   ├── logging_config.py  # Logging configuration
│   │   ├── shared_context.py  # Shared context management
│   │   └── data_processing.py # Data processing utilities
│   └── crews/           # Crew definitions
├── config.py            # System configuration
├── requirements.txt     # Python dependencies
├── run.py              # Execution script
└── README.md          # Documentation
```

## Development

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

## Testing

Run tests with:
```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for educational and research purposes only. It does not constitute actual investment advice. Investing carries risks, please make decisions carefully.