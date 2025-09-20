# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered A-Share (Chinese stock market) investment system that uses multiple agents to make trading decisions. The system is based on a multi-agent architecture with LLM-enhanced decision making through a debate room mechanism.

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry
poetry lock --no-update
poetry install

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys
```

### Running the System
```bash
# Command-line analysis mode (basic output)
poetry run python src/main.py --ticker 000000

# Command-line analysis with detailed reasoning
poetry run python src/main.py --ticker 000000 --show-reasoning

# Backtesting
poetry run python src/backtester.py --ticker 301157 --start-date 2024-12-11 --end-date 2025-01-07 --num-of-news 20

# Backend API service mode
poetry run python run_with_backend.py

# Backend with immediate analysis
poetry run python run_with_backend.py --ticker 002848 --show-reasoning
```

### Testing and Quality
```bash
# Run tests
poetry run pytest

# Code formatting
poetry run black .

# Import sorting
poetry run isort .

# Linting
poetry run flake8 .
```

## System Architecture

### Core Components

**Multi-Agent Workflow**:
- Market Data Agent: Collects and preprocesses market data via akshare API
- Technical Analyst: Analyzes price trends and technical indicators
- Fundamentals Analyst: Evaluates company financial metrics and performance
- Sentiment Analyst: Analyzes market news and sentiment using LLMs
- Valuation Analyst: Performs company valuation and intrinsic value analysis
- Researcher Bull/Bear: Provides opposing market perspectives
- Debate Room: Facilitates structured debates with LLM as third-party evaluator
- Risk Manager: Assesses risks and sets trading limits
- Portfolio Manager: Makes final trading decisions

**Backend API** (FastAPI):
- `/api/*` endpoints: Real-time agent status and workflow information
- `/runs/*` endpoints: Historical execution logs and analysis results
- `/logs/*` endpoints: LLM interaction logs and detailed execution traces
- Swagger UI available at `http://localhost:8000/docs`

### Key Technologies

- **Framework**: LangGraph for agent orchestration, FastAPI for backend services
- **LLM Integration**: Supports Google Gemini API and OpenAI-compatible APIs
- **Data Sources**: akshare for Chinese market data, multiple financial news sources
- **State Management**: AgentState for inter-agent communication, api_state for real-time status
- **Logging**: Comprehensive execution logging with structured output

### Data Flow

1. **Data Collection**: Market data, financial statements, and news articles are gathered
2. **Parallel Analysis**: Multiple agents analyze different aspects simultaneously
3. **Debate Enhancement**: Bull/Bear researchers debate with LLM evaluation
4. **Risk Assessment**: All signals are evaluated against risk parameters
5. **Decision Making**: Portfolio manager makes final trading decision with confidence scoring
6. **API Exposure**: All results and execution traces available via REST API

## Agent Communication

Agents communicate through a shared `AgentState` object containing:
- `messages`: LangChain message sequence for reasoning traces
- `data`: Market data, analysis results, and trading signals
- `metadata`: Execution timestamps, agent status, and workflow information

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google AI Studio API key (primary)
- `OPENAI_COMPATIBLE_*`: Alternative LLM provider configuration
- System prioritizes OpenAI-compatible APIs if configured, otherwise uses Gemini

### Agent Configuration
- Each agent has configurable LLM models and parameters
- Debate room uses hybrid confidence scoring combining traditional signals with LLM evaluation
- Risk manager sets position limits based on portfolio state and market conditions

## Development Notes

- The system is designed for educational/research purposes only, not actual trading
- All agents use the `@agent_endpoint` decorator for execution logging
- LLM interactions should use the `@log_llm_interaction` decorator for complete traceability
- Backend uses dependency injection for log storage, supporting both memory and persistent storage implementations
- Chinese market data is primarily sourced through akshare, requiring proper network configuration