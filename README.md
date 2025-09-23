# CrewAI A-Share Investment Analysis System

基于CrewAI框架的A股智能投资分析系统，使用多智能体协作进行投资决策。

![CrewAI Architecture](docs/architecture.png)

## 🌟 系统概述

本系统是一个先进的AI驱动的投资分析平台，通过12个专业智能体的协作，为A股投资提供全方位的分析支持。系统结合了最新的LLM技术、多智能体协作机制和深度学习算法，为投资者提供专业、客观、及时的投资建议。

介绍参考文章： https://linux.do/t/topic/978220

### 🎯 核心特性

- **🤖 多智能体协作**: 12个专业AI智能体协同工作，各司其职
- **📊 多维分析**: 覆盖技术分析、基本面分析、情绪分析、估值分析等多个维度
- **🧠 LLM增强**: 集成先进的语言模型，提供智能辩论和推理能力
- **🔒 风险管理**: 全流程风险控制和投资组合管理
- **⚡ 高效执行**: 优化的CrewAI框架，支持并行处理和智能调度
- **📈 实时数据**: 接入真实的A股市场数据，包括股价、财务、新闻等
- **🛡️ 安全可靠**: 完善的错误处理、日志记录和监控机制

## 🏗️ 系统架构

### 智能体工作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CrewAI Investment System                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📥 Entry Point                                                          │
│  ┌─────────────┐                                                          │
│  │  Main App   │                                                          │
│  │  (run.py)   │                                                          │
│  └─────────────┘                                                          │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────┐                                            │
│  │   CrewAI System        │                                            │
│  │   (Orchestrator)       │                                            │
│  └─────────────────────────┘                                            │
│           │                                                                 │
│    ┌──────┴───────┐                                                        │
│    │              │                                                        │
│    ▼              ▼                                                        │
│ ┌─────────┐   ┌─────────────┐                                              │
│ │Data     │   │Task         │                                              │
│ │Manager  │   │Manager      │                                              │
│ └─────────┘   └─────────────┘                                              │
│    │              │                                                        │
│    ▼              ▼                                                        │
│ ┌─────────────────────────────────────────────────────────┐              │
│ │                 AGENTS LAYER                           │              │
│ │                                                         │              │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │              │
│ │ │Market Data │ │Technical    │ │Fundamentals │         │              │
│ │ │Agent       │ │Analyst      │ │Analyst      │         │              │
│ │ │(数据收集)   │ │(技术分析)    │ │(基本面分析)   │         │              │
│ │ └─────────────┘ └─────────────┘ └─────────────┘         │              │
│ │                                                         │              │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │              │
│ │ │Sentiment   │ │Valuation    │ │Macro        │         │              │
│ │ │Analyst     │ │Analyst      │ │Analyst      │         │              │
│ │ │(情绪分析)   │ │(估值分析)    │ │(宏观分析)    │         │              │
│ │ └─────────────┘ └─────────────┘ └─────────────┘         │              │
│ │                                                         │              │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │              │
│ │ │Bull        │ │Bear         │ │Debate       │         │              │
│ │ │Researcher   │ │Researcher   │ │Room         │         │              │
│ │ │(看多研究)   │ │(看空研究)   │ │(辩论室)      │         │              │
│ │ └─────────────┘ └─────────────┘ └─────────────┘         │              │
│ │                                                         │              │
│ │ ┌─────────────┐ ┌─────────────┐                         │              │
│ │ │Risk        │ │Portfolio    │                         │              │
│ │ │Manager     │ │Manager      │                         │              │
│ │ │(风险管理)   │ │(投资组合)    │                         │              │
│ │ └─────────────┘ └─────────────┘                         │              │
│ └─────────────────────────────────────────────────────────┘              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────┐                                            │
│  │   Results & Reports    │                                            │
│  │   (结果输出与报告)       │                                            │
│  └─────────────────────────┘                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 智能体详细介绍

#### 1. 市场数据智能体 (Market Data Agent)
- **职责**: 收集和处理股价历史、财务指标、市场新闻等数据
- **数据源**: akshare、东方财富、财经新闻API
- **输出**: 标准化的市场数据包

#### 2. 技术分析师 (Technical Analyst)
- **职责**: 分析价格趋势、技术指标、图表形态
- **分析方法**: 移动平均线、RSI、MACD、布林带等技术指标
- **输出**: 技术面分析报告和交易信号

#### 3. 基本面分析师 (Fundamentals Analyst)
- **职责**: 评估公司财务健康状况和经营业绩
- **分析指标**: ROE、净利润率、营收增长、负债率等
- **输出**: 基本面评级和财务分析报告

#### 4. 情绪分析师 (Sentiment Analyst)
- **职责**: 分析市场情绪、新闻情感、投资者情绪
- **技术**: 使用LLM进行深度文本情感分析
- **输出**: 市场情绪评分和情感趋势分析

#### 5. 估值分析师 (Valuation Analyst)
- **职责**: 进行公司估值和内在价值分析
- **方法**: DCF、可比公司分析、多重估值法
- **输出**: 估值结果和投资建议
 

## 🚀 快速开始

### 运行方法：
langgraph版本：
 poetry run python src/main.py --ticker 301155 --show-reasoning

 langgraph版本是基于： https://github.com/24mlight/A_Share_investment_Agent ， 进行少量改造，这里只是作为对比参考，主要是crewAI版本

crewAI版本：
E:\investment_Agent_langgraph_crewai-main>  python -m crewai_system.src.main --ticker 002594  --show-reasoning

### 1. 环境要求

- Python 3.9+
- Poetry 或 pip
- 充足的内存 (建议8GB+)
- 稳定的网络连接 (访问A股数据API)

### 2. 安装步骤

#### 使用Poetry (推荐)

```bash
# 克隆项目
git clone <repository-url>
cd crewai_system

# 安装依赖
poetry install

# 激活虚拟环境
poetry shell

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

#### 使用pip

```bash
# 克隆项目
git clone <repository-url>
cd crewai_system

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 3. 环境配置

编辑 `.env` 文件：

```env
# ==================== LLM配置 ====================
 
# OpenAI兼容API配置 
OPENAI_COMPATIBLE_API_KEY=your_openai_api_key_here
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_MODEL=gpt-4o

# ==================== 系统配置 ====================
LOG_LEVEL=INFO
MAX_WORKERS=4
CACHE_ENABLED=true
CACHE_TTL=3600

# ==================== 数据源配置 ====================
AKSHARE_TIMEOUT=30
AKSHARE_RETRY_COUNT=3

# ==================== API服务配置 ====================
API_HOST=0.0.0.0
API_PORT=8001
API_RELOAD=true

# ==================== 安全配置 ====================
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 4. 基本使用

#### 命令行分析模式

```bash
# 基础分析
python -m crewai_system.src.main --ticker 000002

# 显示详细推理过程
python -m crewai_system.src.main --ticker 000002 --show-reasoning

# 自定义分析参数
python -m crewai_system.src.main --ticker 000002 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --num-of-news 20

# 自定义投资组合
python -m crewai_system.src.main --ticker 000002 \
  --initial-capital 1000000 \
  --initial-position 1000
```
 
## 🛠️ 开发指南

### 项目结构

```
crewai_system/
├── src/
│   ├── agents/                    # 智能体定义
│   │   ├── __init__.py
│   │   ├── base_agent.py          # 基础智能体类
│   │   ├── market_data_agent.py   # 市场数据智能体
│   │   ├── technical_analyst.py   # 技术分析师
│   │   ├── fundamentals_analyst.py # 基本面分析师
│   │   ├── sentiment_analyst.py   # 情绪分析师
│   │   ├── valuation_analyst.py    # 估值分析师
│   │   ├── researcher_bull.py     # 看多研究员
│   │   ├── researcher_bear.py     # 看空研究员
│   │   ├── debate_room.py         # 辩论室
│   │   ├── risk_manager.py        # 风险管理师
│   │   ├── macro_analyst.py       # 宏观分析师
│   │   └── portfolio_manager.py   # 投资组合经理
│   │
│   ├── tools/                     # 工具和接口
│   │   ├── __init__.py
│   │   ├── data_sources.py        # 数据源接口
│   │   ├── market_data_tools.py   # 市场数据工具
│   │   ├── financial_tools.py     # 财务分析工具
│   │   └── news_tools.py          # 新闻分析工具
│   │
│   ├── utils/                     # 工具类
│   │   ├── __init__.py
│   │   ├── logging_config.py      # 日志配置
│   │   ├── shared_context.py      # 共享上下文
│   │   ├── data_processing.py     # 数据处理
│   │   ├── llm_clients.py         # LLM客户端
│   │   ├── llm_config.py          # LLM配置
│   │   └── api_utils.py           # API工具
│   │
│   ├── backend/                   # 后端API服务
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI主程序
│   │   ├── routers/               # API路由
│   │   ├── schemas/               # 数据模型
│   │   └── storage/               # 数据存储
│   │
│   └── logs/                      # 日志文件
│
├── config.py                      # 系统配置
├── requirements.txt               # 依赖列表
├── pyproject.toml                 # Poetry配置
├── .env.example                   # 环境变量模板
├── run.py                         # 启动脚本
└── README.md                      # 项目文档
```

### 添加新智能体

#### 1. 创建智能体类

```python
# src/agents/custom_agent.py
from typing import Any, Dict, List
from .base_agent import BaseAgent
from crewai import Tool

class CustomAgent(BaseAgent):
    """自定义智能体示例"""

    def __init__(self):
        super().__init__(
            role="自定义分析师",
            goal="执行特定的分析任务",
            backstory="""你是一位专业的分析师，专注于特定领域的分析。
            你具有深厚的专业知识和丰富的实战经验。""",
            agent_name="CustomAnalyst"
        )

        # 添加工具
        self.tools = [
            Tool(
                name="custom_analysis_tool",
                func=self.custom_analysis,
                description="执行自定义分析任务"
            )
        ]

    def process_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """处理任务"""
        self.log_execution_start("执行自定义分析")

        try:
            # 验证输入
            required_fields = ["ticker", "data"]
            if not self.validate_input(task_context, required_fields):
                raise ValueError(f"缺少必需字段: {required_fields}")

            # 执行分析
            result = self.custom_analysis(task_context)

            # 格式化输出
            return self.format_agent_output(
                content=result,
                signal=result.get("signal", "neutral"),
                confidence=result.get("confidence", 50),
                reasoning=result.get("reasoning", ""),
                metadata={
                    "ticker": task_context["ticker"],
                    "analysis_type": "custom",
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.log_execution_error(e, "自定义分析失败")
            raise

    def custom_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """自定义分析逻辑"""
        # 实现具体的分析逻辑
        return {
            "signal": "bullish",
            "confidence": 75,
            "reasoning": "基于自定义分析得出的投资建议",
            "details": {}
        }
```

#### 2. 注册智能体到系统

```python
# 在系统初始化代码中
from src.agents.custom_agent import CustomAgent

# 创建智能体实例
custom_agent = CustomAgent()

# 添加到智能体列表
agents = [
    market_data_agent,
    technical_analyst,
    fundamentals_analyst,
    # ... 其他智能体
    custom_agent,  # 添加自定义智能体
]
```

### 添加新工具

```python
# src/tools/custom_tools.py
from crewai import Tool
from typing import Any, Dict

def custom_analysis_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """自定义分析工具函数"""
    # 实现工具逻辑
    return {
        "result": "分析结果",
        "confidence": 0.85
    }

# 创建工具
custom_tool = Tool(
    name="custom_analysis",
    func=custom_analysis_function,
    description="执行自定义分析任务"
)
```

### 共享上下文使用

```python
from src.utils.shared_context import get_global_context, ContextManager

# 获取全局上下文
context = get_global_context()

# 设置数据
context.set("market_data", data, source_agent="market_data_agent")

# 获取数据
market_data = context.get("market_data")

# 使用上下文管理器
with ContextManager(context, "sentiment_analyst") as ctx:
    # 设置分析结果
    ctx.set("sentiment_score", 0.75)

    # 获取其他智能体的数据
    market_data = ctx.get("market_data")

    # 执行分析
    result = analyze_sentiment(market_data)

    # 设置最终结果
    ctx.set("sentiment_result", result)
```

## 🔧 配置详解

### LLM配置

系统支持多种LLM提供商：

#### 1. Google Gemini (推荐)
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash  # 或 gemini-1.5-pro
```

#### 2. OpenAI兼容API
```env
OPENAI_COMPATIBLE_API_KEY=your_api_key
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_MODEL=gpt-4-turbo-preview
```

#### 3. 本地模型
```env
OPENAI_COMPATIBLE_BASE_URL=http://localhost:8000/v1
OPENAI_COMPATIBLE_MODEL=llama2-7b-chat
```

### 数据源配置

#### akshare配置
```env
AKSHARE_TIMEOUT=30          # 超时时间(秒)
AKSHARE_RETRY_COUNT=3       # 重试次数
```

#### 缓存配置
```env
CACHE_ENABLED=true         # 启用缓存
CACHE_TTL=3600            # 缓存有效期(秒)
```

### 系统性能配置

```env
MAX_WORKERS=4              # 最大工作线程数
LOG_LEVEL=INFO            # 日志级别
API_PORT=8001             # API服务端口
```

## 📈 性能优化

### 缓存策略

系统采用多级缓存机制：

```python
# 查看缓存统计
from src.tools.data_sources import get_data_adapter
adapter = get_data_adapter()
stats = adapter.get_cache_stats()
print(f"缓存命中率: {stats['hit_rate']:.2%}")
print(f"缓存大小: {stats['cache_size']}")
```

### 并发处理

```python
# 配置并发处理
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 创建线程池
executor = ThreadPoolExecutor(max_workers=4)

# 并发执行多个任务
async def concurrent_analysis(tickers):
    tasks = [analyze_ticker(ticker) for ticker in tickers]
    return await asyncio.gather(*tasks)
```

### 内存管理

```python
# 清理内存
import gc
gc.collect()

# 清理缓存
adapter.clear_cache()
```

## 🚨 监控和日志

### 日志配置

系统提供详细的日志记录：

```python
import logging
from src.utils.logging_config import get_logger

# 获取日志器
logger = get_logger('sentiment_analyst')

# 记录日志
logger.info("开始情绪分析")
logger.debug(f"分析参数: {params}")
logger.error("分析失败", exc_info=True)
```

### 日志文件结构

```
logs/
├── investment_system.log     # 主系统日志
├── debug.log                  # 调试日志
├── data.log                   # 数据访问日志
└── api.log                    # API调用日志
```

### 性能监控

```python
# 查看系统性能
from src.utils.shared_context import get_global_context
context = get_global_context()

# 获取执行统计
execution_stats = context.get_execution_stats()
print(f"总执行时间: {execution_stats['total_time']:.2f}秒")
print(f"智能体执行时间: {execution_stats['agent_times']}")
```

## 🐛 故障排除

### 常见问题

#### 1. API密钥错误
```
错误: 未找到 GEMINI_API_KEY 环境变量
解决: 检查 .env 文件中的 API 密钥配置
```

#### 2. 数据源连接失败
```
错误: akshare连接超时
解决:
1. 检查网络连接
2. 增加AKSHARE_TIMEOUT值
3. 使用VPN访问
```

#### 3. 内存不足
```
错误: MemoryError
解决:
1. 减少MAX_WORKERS数量
2. 清理缓存: python run.py --cleanup
3. 增加系统内存
```

#### 4. LLM调用失败
```
错误: Gemini API调用失败
解决:
1. 检查API密钥是否有效
2. 确认网络连接正常
3. 检查API配额是否用完
```

### 调试模式

```bash
# 启用详细日志
LOG_LEVEL=DEBUG python -m crewai_system.src.main --ticker 000002

# 显示推理过程
python -m crewai_system.src.main --ticker 000002 --show-reasoning

# 测试单个智能体
python -c "
from src.agents.sentiment_analyst import SentimentAnalyst
agent = SentimentAnalyst()
result = agent.process_task({'ticker': '000002', 'news_data': []})
print(result)
"
```

### 性能诊断

```bash
# 查看系统资源使用
python -c "
import psutil
print(f'CPU使用率: {psutil.cpu_percent()}%')
print(f'内存使用: {psutil.virtual_memory().percent}%')
print(f'磁盘使用: {psutil.disk_usage('/').percent}%')
"

# 查看网络连接
python -c "
import socket
socket.gethostbyname('api.akshare.xyz')
"
```

## 📚 API参考

### 核心类

#### CrewAIInvestmentSystem
主系统类，负责协调所有智能体的执行。

```python
from src.main import CrewAIInvestmentSystem

system = CrewAIInvestmentSystem()
result = system.analyze_ticker("000002")
```

#### BaseAgent
所有智能体的基类。

```python
from src.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def process_task(self, task_context):
        # 实现逻辑
        return result
```

#### DataSourceAdapter
数据源适配器，提供统一的数据访问接口。

```python
from src.tools.data_sources import get_data_adapter

adapter = get_data_adapter()
market_data = adapter.get_price_history("000002")
financial_data = adapter.get_financial_metrics("000002")
```

### 工具函数

```python
# 获取全局上下文
from src.utils.shared_context import get_global_context
context = get_global_context()

# 获取数据处理器
from src.utils.data_processing import get_data_processor
processor = get_data_processor()

# 获取LLM客户端
from src.utils.llm_clients import LLMClientFactory
client = LLMClientFactory.create_client()
```

## 🔄 版本历史

### v0.2.0 (当前版本)
- ✨ 重构情绪分析师，使用LLM进行情绪分析
- 🗑️ 移除所有模拟数据，仅使用真实数据
- 🚀 优化LLM客户端，支持多种模型提供商
- 🐛 修复新闻数据获取bug
- 📝 完善文档和错误处理

### v0.1.0
- 🎉 初始版本发布
- 🤖 实现基础多智能体架构
- 📊 集成akshare数据源
- 🧠 添加LLM支持
- 📈 实现基础分析功能

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. **Fork项目** - 从主仓库创建分支
2. **创建功能分支** - `git checkout -b feature/amazing-feature`
3. **提交更改** - `git commit -m 'Add amazing feature'`
4. **推送分支** - `git push origin feature/amazing-feature`
5. **创建Pull Request** - 详细描述你的更改

### 开发规范

- 遵循PEP 8代码风格
- 添加适当的注释和文档字符串
- 编写单元测试
- 确保所有测试通过
- 更新相关文档

### 测试

```bash
# 运行所有测试
poetry run pytest

# 运行特定测试
poetry run pytest tests/test_agents.py

# 代码覆盖率
poetry run pytest --cov=src
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## ⚠️ 免责声明

**重要提示**: 本系统仅供教育和研究目的，不构成实际投资建议。

- 投资有风险，决策需谨慎
- 过往表现不代表未来收益
- 系统分析结果仅供参考
- 用户应自行承担投资风险
- 建议结合专业投资顾问意见

## 📞 联系我们

- 📧 Email: [your-email@example.com](mailto:your-email@example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 文档: [完整文档](https://your-docs-url.com)
- 💬 讨论: [GitHub Discussions](https://github.com/your-repo/discussions)

---

<div align="center">

**CrewAI A-Share Investment Analysis System**

*让AI为您的投资决策提供智能支持*

[⭐ Star](https://github.com/your-repo) | [🐛 报告问题](https://github.com/your-repo/issues) | [💡 功能建议](https://github.com/your-repo/discussions)

</div>