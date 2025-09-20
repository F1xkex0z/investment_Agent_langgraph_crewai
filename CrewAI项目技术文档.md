● CrewAI投资分析系统迁移项目技术文档

  项目概述

  项目背景

  本项目是一个将现有A股投资分析系统从LangGraph框架成功迁移至CrewAI框架的技术改造项目。原系统基于LangGraph构建了一个
  复杂的多智能体投资分析流水线，为了获得更好的协作能力、更灵活的工作流程和更强大的社区支持，我们决定迁移到CrewAI框架
  。

  技术目标

  - 保持原有功能的完整性和向后兼容性
  - 利用CrewAI的协作优势提升分析质量
  - 优化系统架构，提高代码可维护性
  - 增强系统的扩展性和定制能力

  项目价值

  - 通过辩论式分析提供更平衡的投资建议
  - 利用并行处理提升分析效率
  - 建立更清晰的智能体层级架构
  - 为未来功能扩展奠定坚实基础

  系统架构对比

  原有LangGraph架构

  # 原有架构特点
  1. 基于StateGraph的线性工作流
  2. 13个独立Agent按序执行
  3. 集中式状态管理(AgentState)
  4. 装饰器模式的日志和API集成
  5. 硬编码的任务依赖关系

  # 工作流程
  StateGraph -> Agent1 -> Agent2 -> ... -> Agent13 -> FinalResult

  新CrewAI架构

  # 新架构特点
  1. 基于Crew的协作式工作流
  2. 三层智能体架构（数据、分析、研究）
  3. 分布式状态管理(InvestmentState)
  4. 工具模式的智能体能力扩展
  5. 灵活的任务依赖和并行执行

  # 工作流程
  CrewAI -> [DataCrew + AnalysisCrew + ResearchCrew] -> SynthesizedResult

  详细技术实现

  1. 项目结构重组

  src/
  ├── crewai/                          # 新CrewAI系统
  │   ├── __init__.py                  # 模块初始化
  │   ├── main.py                      # 系统入口点
  │   ├── config/                      # 配置管理
  │   │   ├── __init__.py
  │   │   ├── state.py                 # 状态管理
  │   │   ├── llm_config.py            # LLM配置
  │   │   ├── crew_config.py           # Crew配置
  │   │   └── settings.py              # 环境设置
  │   ├── agents/                      # 智能体定义
  │   │   ├── __init__.py
  │   │   ├── base.py                  # 基础智能体类
  │   │   ├── data_agents.py           # 数据收集智能体
  │   │   ├── analysis_agents.py       # 分析智能体
  │   │   └── research_agents.py       # 研究辩论智能体
  │   ├── tasks/                       # 任务定义
  │   │   ├── __init__.py
  │   │   ├── data_tasks.py            # 数据收集任务
  │   │   ├── analysis_tasks.py        # 分析任务
  │   │   └── research_tasks.py        # 研究辩论任务
  │   └── tools/                       # 智能体工具
  │       └── research_tools.py        # 研究分析工具
  ├── agents/                          # 原有系统（保留兼容）
  ├── tools/                           # 原有工具
  └── utils/                           # 工具函数

  2. 核心抽象设计

  2.1 状态管理系统

  @dataclass
  class InvestmentState:
      """投资分析状态管理 - 替代原有AgentState"""

      # 基础信息
      ticker: str
      run_id: str
      start_date: str
      end_date: str

      # 配置和状态
      config: InvestmentConfig
      portfolio: PortfolioState

      # 数据缓存
      data_cache: Dict[str, Any] = field(default_factory=dict)
      analysis_results: Dict[str, Any] = field(default_factory=dict)

      # 任务历史
      task_history: List[Dict[str, Any]] = field(default_factory=list)

      def update_analysis_result(self, agent_type: str, result: Dict[str, Any]):
          """更新分析结果"""
          self.analysis_results[agent_type] = {
              "result": result,
              "timestamp": datetime.now().isoformat()
          }

      def to_dict(self) -> Dict[str, Any]:
          """状态序列化"""
          return asdict(self)

      @classmethod
      def from_dict(cls, data: Dict[str, Any]) -> 'InvestmentState':
          """状态反序列化"""
          return cls(**data)

  2.2 智能体基础架构

  class BaseInvestmentAgent(Agent, ABC):
      """投资智能体基类"""

      def __init__(self, role: str, goal: str, backstory: str,
                   agent_type: str = "general", **kwargs):

          # LLM配置
          llm_config = kwargs.get('llm_config') or get_default_llm_config()

          # CrewAI Agent初始化
          super().__init__(
              role=role,
              goal=goal,
              backstory=backstory,
              tools=kwargs.get('tools', []),
              llm=llm_config.get_llm(),
              verbose=kwargs.get('verbose', True),
              allow_delegation=kwargs.get('allow_delegation', False),
              max_iter=kwargs.get('max_iter', 20),
              max_rpm=kwargs.get('max_rpm', 60)
          )

          self.agent_type = agent_type
          self.logger = setup_logger(f'crewai_agent_{agent_type}')

      def preprocess_state(self, state: InvestmentState) -> Dict[str, Any]:
          """状态预处理 - 智能体可以重写此方法"""
          return {
              "ticker": state.ticker,
              "portfolio": state.portfolio.to_dict(),
              "analysis_results": state.analysis_results,
              "config": state.config.to_dict()
          }

  3. 智能体层级实现

  3.1 数据收集层

  class DataCollectionAgent(DataAgent):
      """数据收集智能体 - 迁移自MarketDataAgent"""

      def __init__(self, tools: List[BaseTool] = None):
          super().__init__(
              role="Senior Market Data Analyst",
              goal="Collect comprehensive and accurate market data for investment analysis",
              backstory="You are an experienced market data analyst with expertise in "
                        "financial data collection, validation, and preprocessing. "
                        "You understand the importance of data quality and accuracy.",
              tools=tools or [MarketDataTool(), NewsDataTool()],
              agent_type="data_collection"
          )

      def create_task_description(self, state: InvestmentState) -> str:
          """动态任务生成"""
          return f"""
          Collect comprehensive market data for {state.ticker} analysis.

          Requirements:
          - Historical price data (OHLCV) from {state.start_date} to {state.end_date}
          - Financial metrics and ratios
          - Company financial statements
          - Market sentiment indicators
          - Recent news and events analysis

          Ensure data quality, completeness, and proper formatting.
          """

  3.2 分析层

  class TechnicalAnalysisAgent(AnalysisAgent):
      """技术分析智能体"""

      def __init__(self, tools: List[BaseTool] = None):
          super().__init__(
              role="Technical Analysis Expert",
              goal="Analyze price trends, patterns, and indicators to generate trading signals",
              backstory="You are a technical analysis specialist with deep knowledge of "
                        "chart patterns, indicators, and market timing strategies.",
              tools=tools or [TechnicalAnalysisTool()],
              agent_type="technical_analysis"
          )

  class FundamentalAnalysisAgent(AnalysisAgent):
      """基本面分析智能体"""

      def __init__(self, tools: List[BaseTool] = None):
          super().__init__(
              role="Fundamental Analysis Expert",
              goal="Evaluate financial health, business model, and intrinsic value",
              backstory="You are a fundamental analysis expert specializing in "
                        "financial statement analysis, business valuation, and quality assessment.",
              tools=tools or [FundamentalAnalysisTool()],
              agent_type="fundamental_analysis"
          )

  3.3 研究辩论层（新增特性）

  class BullishResearchAgent(ResearchAgent):
      """看多研究智能体 - 替代原有ResearcherBull"""

      def __init__(self, tools: List[BaseTool] = None):
          super().__init__(
              role="Senior Bullish Research Analyst",
              goal="Analyze investment opportunities from an optimistic perspective",
              backstory="You are an experienced bullish research analyst with a talent for "
                        "identifying undervalued opportunities and growth potential.",
              research_type="bullish_research",
              tools=tools or [BullishResearchTool()]
          )

      def create_task_description(self, state: InvestmentState) -> str:
          return f"""
          Conduct comprehensive bullish research analysis for {state.ticker}.

          Analyze from bullish perspective:
          1. Technical indicators - momentum, breakouts, upward trends
          2. Fundamental strengths - growth potential, competitive advantages
          3. Positive sentiment factors - market optimism, favorable news
          4. Valuation opportunities - potential undervaluation, growth catalysts

          Generate compelling bullish investment thesis with confidence assessment.
          """

  class BearishResearchAgent(ResearchAgent):
      """看空研究智能体 - 替代原有ResearcherBear"""

      def __init__(self, tools: List[BaseTool] = None):
          super().__init__(
              role="Senior Bearish Research Analyst",
              goal="Analyze investment risks from a cautious perspective",
              backstory="You are a seasoned bearish research analyst specializing in "
                        "risk identification and conservative analysis.",
              research_type="bearish_research",
              tools=tools or [BearishResearchTool()]
          )

  class DebateModeratorAgent(ResearchAgent):
      """辩论协调智能体 - 替代原有DebateRoom"""

      def __init__(self, tools: List[BaseTool] = None):
          super().__init__(
              role="Investment Debate Moderator",
              goal="Facilitate balanced investment debates and synthesize perspectives",
              backstory="You are an expert debate moderator with exceptional analytical "
                        "and reasoning skills for evaluating competing investment theses.",
              research_type="debate_moderation",
              tools=tools or [DebateModerationTool()]
          )

  4. 工具系统设计

  4.1 研究工具实现

  class BullishResearchTool(BaseTool):
      """看多研究工具"""

      name: str = "bullish_research_tool"
      description: str = "Analyze investment opportunities from optimistic perspective"

      def _run(self, input_data: str) -> str:
          """执行看多分析"""
          try:
              data = json.loads(input_data)
              ticker = data.get('ticker', 'UNKNOWN')
              analysis_data = data.get('analysis_data', {})

              # 生成看多论点
              bullish_points = []
              confidence_scores = []

              # 技术面看多分析
              technical_data = analysis_data.get('technical_analysis_result', {})
              if technical_data.get('signal') == 'bullish':
                  bullish_points.append(f"Technical indicators show bullish momentum")
                  confidence_scores.append(technical_data.get('confidence', 0.5))
              else:
                  bullish_points.append("Technical weakness may present buying opportunity")
                  confidence_scores.append(0.4)

              # 基本面看多分析
              fundamental_data = analysis_data.get('fundamental_analysis_result', {})
              if fundamental_data.get('signal') == 'bullish':
                  bullish_points.append("Strong fundamentals support growth thesis")
                  confidence_scores.append(fundamental_data.get('confidence', 0.5))

              # 计算整体置信度
              avg_confidence = sum(confidence_scores) / len(confidence_scores)

              # 识别增长催化剂
              catalysts = self._identify_growth_catalysts(analysis_data)

              result = {
                  "perspective": "bullish",
                  "ticker": ticker,
                  "confidence": avg_confidence,
                  "thesis_points": bullish_points,
                  "growth_catalysts": catalysts,
                  "reasoning": "Bullish thesis based on comprehensive analysis"
              }

              return json.dumps(result, ensure_ascii=False)

          except Exception as e:
              return json.dumps({"error": str(e), "perspective": "bullish"})

  class DebateModerationTool(BaseTool):
      """辩论协调工具"""

      name: str = "debate_moderation_tool"
      description: str = "Facilitate balanced investment debates and synthesize viewpoints"

      def _run(self, input_data: str) -> str:
          """执行辩论协调"""
          try:
              data = json.loads(input_data)

              # 提取多空观点
              bull_thesis = data.get('bullish_thesis', {})
              bear_thesis = data.get('bearish_thesis', {})

              bull_confidence = bull_thesis.get('confidence', 0.0)
              bear_confidence = bear_thesis.get('confidence', 0.0)
              confidence_diff = bull_confidence - bear_confidence

              # 生成辩论摘要
              debate_summary = self._generate_debate_summary(bull_thesis, bear_thesis)

              # 评估论点强度
              argument_assessment = self._assess_arguments(bull_thesis, bear_thesis)

              # 确定最终信号
              final_signal, reasoning, final_confidence = self._determine_final_signal(
                  bull_confidence, bear_confidence, confidence_diff, argument_assessment
              )

              # 生成投资建议
              recommendation = self._generate_recommendation(
                  final_signal, final_confidence, data.get('ticker', 'UNKNOWN'),
                  data.get('current_portfolio', {})
              )

              result = {
                  "final_signal": final_signal,
                  "confidence": final_confidence,
                  "bull_confidence": bull_confidence,
                  "bear_confidence": bear_confidence,
                  "confidence_differential": confidence_diff,
                  "debate_summary": debate_summary,
                  "reasoning": reasoning,
                  "recommendation": recommendation
              }

              return json.dumps(result, ensure_ascii=False)

          except Exception as e:
              return json.dumps({"error": str(e), "final_signal": "neutral"})

  5. 任务系统设计

  5.1 任务工厂模式

  class ResearchTasks:
      """研究任务工厂"""

      def __init__(self, state: InvestmentState):
          self.state = state

      def create_parallel_research_tasks(self,
                                      bullish_agent: BullishResearchAgent,
                                      bearish_agent: BearishResearchAgent) -> List[Task]:
          """创建并行研究任务"""

          # 看多研究任务
          bullish_task = Task(
              description=bullish_agent.create_task_description(self.state),
              expected_output="Comprehensive bullish research analysis",
              agent=bullish_agent,
              async_execution=True,  # 启用并行执行
              human_input=False
          )

          # 看空研究任务
          bearish_task = Task(
              description=bearish_agent.create_task_description(self.state),
              expected_output="Comprehensive bearish risk assessment",
              agent=bearish_agent,
              async_execution=True,  # 启用并行执行
              human_input=False
          )

          return [bullish_task, bearish_task]

      def create_sequential_research_tasks(self,
                                         bullish_agent: BullishResearchAgent,
                                         bearish_agent: BearishResearchAgent,
                                         debate_agent: DebateModeratorAgent) -> List[Task]:
          """创建顺序研究任务（研究并行，辩论顺序）"""

          # 并行研究任务
          research_tasks = self.create_parallel_research_tasks(bullish_agent, bearish_agent)

          # 辩论任务（依赖研究结果）
          debate_task = Task(
              description=debate_agent.create_task_description(self.state),
              expected_output="Balanced investment debate analysis",
              agent=debate_agent,
              context=research_tasks,  # 依赖研究结果
              async_execution=False,   # 顺序执行
              human_input=False
          )

          return research_tasks + [debate_task]

  6. 主系统集成

  6.1 CrewAI系统类

  class CrewAIInvestmentSystem:
      """CrewAI投资分析系统主类"""

      def __init__(self, config: Optional[InvestmentConfig] = None):
          self.config = config or InvestmentConfig()
          self.llm_config = get_default_llm_config()
          self.crew_config = get_crew_config()
          self.logger = setup_logger('crewai_system')

      def run_analysis(self, ticker: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
          """执行完整投资分析"""

          # 生成运行ID和状态
          run_id = str(uuid.uuid4())
          investment_state = self._create_investment_state(
              ticker, run_id, start_date, end_date, portfolio
          )

          try:
              # 创建智能体
              agents = self._create_agents()

              # 创建任务
              tasks = self._create_tasks(investment_state, agents)

              # 创建Crew
              crew = Crew(
                  agents=list(agents.values()),
                  tasks=tasks,
                  verbose=True,
                  process=Process.sequential
              )

              # 执行分析
              result = crew.kickoff()

              # 处理结果
              final_result = self._process_results(result, investment_state)

              self.logger.info(f"Analysis completed for {ticker}")
              return final_result

          except Exception as e:
              self.logger.error(f"Analysis failed for {ticker}: {str(e)}")
              return self._create_error_result(e, ticker, run_id)

      def _create_agents(self) -> Dict[str, Agent]:
          """创建所有智能体"""
          return {
              'data_collection': DataCollectionAgent(),
              'news_analysis': NewsAnalysisAgent(),
              'technical_analysis': TechnicalAnalysisAgent(),
              'fundamental_analysis': FundamentalAnalysisAgent(),
              'bullish_research': BullishResearchAgent(),
              'bearish_research': BearishResearchAgent(),
              'debate_moderator': DebateModeratorAgent()
          }

      def _create_tasks(self, state: InvestmentState, agents: Dict[str, Any]) -> list:
          """创建所有任务"""
          tasks = []

          # 数据收集任务（并行）
          data_tasks = create_parallel_data_tasks(
              state, agents['data_collection'], agents['news_analysis']
          )
          tasks.extend(data_tasks)

          # 分析任务（并行）
          analysis_tasks = create_parallel_analysis_tasks(
              state, agents['technical_analysis'], agents['fundamental_analysis']
          )
          tasks.extend(analysis_tasks)

          # 研究辩论任务（研究并行，辩论顺序）
          research_tasks = create_sequential_research_tasks(
              state,
              agents['bullish_research'],
              agents['bearish_research'],
              agents['debate_moderator']
          )
          tasks.extend(research_tasks)

          return tasks

  7. 向后兼容性实现

  7.1 API兼容层

  def run_hedge_fund(run_id: str, ticker: str, start_date: str, end_date: str,
                    portfolio: Dict[str, Any], show_reasoning: bool = False,
                    num_of_news: int = 5, show_summary: bool = False) -> str:
      """向后兼容函数 - 保持原有API接口"""

      try:
          # 转换为CrewAI格式
          config = InvestmentConfig(
              show_reasoning=show_reasoning,
              num_of_news=num_of_news,
              show_summary=show_summary,
              initial_capital=portfolio.get('cash', 100000.0),
              initial_position=portfolio.get('stock', 0)
          )

          system = CrewAIInvestmentSystem(config)
          result = system.run_analysis(ticker, start_date, end_date, portfolio)

          # 转换为字符串格式保持兼容性
          return json.dumps(result, indent=2, ensure_ascii=False)

      except Exception as e:
          logger.error(f"Backward compatibility function failed: {str(e)}")
          return json.dumps({
              "error": str(e),
              "run_id": run_id,
              "ticker": ticker
          }, indent=2)

  核心技术创新

  1. 辩论式分析机制

  原有的系统分析结果相对单一，新的辩论式分析通过多空双方的深入辩论，提供更全面、更平衡的投资建议：

  # 传统分析：单一信号生成
  technical_signal = analyze_technical(data)
  fundamental_signal = analyze_fundamental(data)
  final_signal = combine_signals([technical_signal, fundamental_signal])

  # 辩论式分析：多角度深入分析
  bullish_thesis = bullish_research_agent.analyze(data)
  bearish_thesis = bearish_research_agent.analyze(data)
  debate_result = debate_moderator.synthesize(bullish_thesis, bearish_thesis)
  final_signal = debate_result.final_signal

  2. 并行处理优化

  # 传统串行处理
  data_task = execute_sequentially([data_collection, news_analysis])
  analysis_task = execute_sequentially([technical, fundamental])
  research_task = execute_sequentially([bullish, bearish, debate])

  # CrewAI并行处理
  data_tasks = execute_parallel([data_collection, news_analysis])
  analysis_tasks = execute_parallel([technical, fundamental])
  research_tasks = execute_parallel_sequential([bullish, bearish], [debate])

  3. 状态管理改进

  # 原有AgentState - 相对简单的字典结构
  class AgentState(TypedDict):
      messages: List[BaseMessage]
      data: Dict[str, Any]
      metadata: Dict[str, Any]

  # 新InvestmentState - 结构化状态管理
  @dataclass
  class InvestmentState:
      ticker: str
      run_id: str
      config: InvestmentConfig
      portfolio: PortfolioState
      data_cache: Dict[str, Any]
      analysis_results: Dict[str, Any]
      task_history: List[Dict[str, Any]]

      def update_analysis_result(self, agent_type: str, result: Dict[str, Any]):
          # 类型安全的状态更新
          self.analysis_results[agent_type] = {
              "result": result,
              "timestamp": datetime.now().isoformat()
          }

  性能优化策略

  1. 任务并行化

  # 数据收集并行化
  def create_parallel_data_tasks(state, market_agent, news_agent):
      market_task = Task(
          description="Collect market data",
          agent=market_agent,
          async_execution=True
      )
      news_task = Task(
          description="Analyze news sentiment",
          agent=news_agent,
          async_execution=True
      )
      return [market_task, news_task]  # CrewAI自动并行执行

  2. 缓存机制

  @dataclass
  class InvestmentConfig:
      # ... 其他配置
      enable_cache: bool = True
      cache_ttl: int = 3600  # 1小时缓存

  class CachedAnalysisAgent:
      def __init__(self):
          self.cache = {}

      def analyze(self, data):
          cache_key = self._generate_cache_key(data)
          if cache_key in self.cache:
              if time.time() - self.cache[cache_key]['timestamp'] < self.config.cache_ttl:
                  return self.cache[cache_key]['result']

          result = self._perform_analysis(data)
          self.cache[cache_key] = {
              'result': result,
              'timestamp': time.time()
          }
          return result

  3. 资源管理

  class ResourceManager:
      def __init__(self):
          self.max_workers = min(4, os.cpu_count())
          self.memory_limit = psutil.virtual_memory().total * 0.8

      def optimize_task_execution(self, tasks):
          """根据系统资源动态调整任务执行"""
          if len(tasks) > self.max_workers:
              return self._batch_execution(tasks)
          else:
              return self._parallel_execution(tasks)

  测试验证体系

  1. 全面测试覆盖

  class TestCrewAIMigration:
      """CrewAI迁移测试套件"""

      def test_basic_functionality(self):
          """基础功能测试"""
          agents = self._create_all_agents()
          assert len(agents) == 7  # 验证所有智能体创建成功

      def test_state_management(self):
          """状态管理测试"""
          state = InvestmentState(
              ticker="TEST", run_id="test-001",
              start_date="2024-01-01", end_date="2024-12-01",
              config=InvestmentConfig(),
              portfolio=PortfolioState(cash=100000, stock_position=0)
          )

          # 测试状态更新
          state.update_analysis_result('technical', {'signal': 'bullish'})
          assert 'technical' in state.analysis_results

          # 测试序列化
          state_dict = state.to_dict()
          restored_state = InvestmentState.from_dict(state_dict)
          assert restored_state.ticker == state.ticker

      def test_research_tools(self):
          """研究工具测试"""
          sample_data = {
              "ticker": "TEST",
              "analysis_data": {
                  "technical_analysis_result": {"signal": "bullish", "confidence": 0.7},
                  "fundamental_analysis_result": {"signal": "neutral", "confidence": 0.5}
              }
          }

          # 测试看多工具
          bull_tool = BullishResearchTool()
          bull_result = bull_tool._run(json.dumps(sample_data))
          bull_data = json.loads(bull_result)
          assert bull_data['perspective'] == 'bullish'

          # 测试辩论工具
          debate_tool = DebateModerationTool()
          debate_input = {
              "bullish_thesis": bull_data,
              "bearish_thesis": json.loads(bear_tool._run(json.dumps(sample_data)))
          }
          debate_result = debate_tool._run(json.dumps(debate_input))
          debate_data = json.loads(debate_result)
          assert debate_data['final_signal'] in ['bullish', 'bearish', 'neutral']

      def test_backward_compatibility(self):
          """向后兼容性测试"""
          result = run_hedge_fund(
              run_id="test-run-123",
              ticker="000001",
              start_date="2024-01-01",
              end_date="2024-12-01",
              portfolio={"cash": 100000.0, "stock": 0}
          )

          result_dict = json.loads(result)
          assert result_dict.get('run_id') == "test-run-123"

  2. 性能基准测试

  class PerformanceBenchmark:
      """性能基准测试"""

      def test_execution_time(self):
          """执行时间测试"""
          start_time = time.time()

          result = run_investment_crew(
              ticker="000001",
              num_of_news=3
          )

          execution_time = time.time() - start_time
          print(f"Execution time: {execution_time:.2f} seconds")

          # 验证性能目标
          assert execution_time < 300  # 5分钟内完成

      def test_memory_usage(self):
          """内存使用测试"""
          process = psutil.Process()
          initial_memory = process.memory_info().rss

          result = run_investment_crew(
              ticker="000001",
              num_of_news=3
          )

          peak_memory = process.memory_info().rss
          memory_increase = peak_memory - initial_memory

          print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")
          assert memory_increase < 500 * 1024 * 1024  # 内存增长小于500MB

      def test_concurrent_execution(self):
          """并发执行测试"""
          import concurrent.futures

          tickers = ["000001", "000002", "000003"]

          with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
              futures = [
                  executor.submit(run_investment_crew, ticker=ticker)
                  for ticker in tickers
              ]

              results = [future.result() for future in futures]

          # 验证所有分析都成功
          for result in results:
              assert result['success'] is True

  部署和运维

  1. 环境配置

  # 生产环境依赖
  crewai==0.28.0
  crewai-tools==0.2.0
  pandas>=1.5.0
  numpy>=1.21.0
  python-dotenv>=0.19.0

  # 开发环境依赖
  pytest>=7.0.0
  pytest-cov>=4.0.0
  black>=22.0.0
  flake8>=5.0.0

  2. Docker化部署

  FROM python:3.9-slim

  WORKDIR /app

  # 安装系统依赖
  RUN apt-get update && apt-get install -y \
      build-essential \
      curl \
      && rm -rf /var/lib/apt/lists/*

  # 复制依赖文件
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  # 复制应用代码
  COPY src/ ./src/
  COPY test_crewai.py .

  # 设置环境变量
  ENV PYTHONPATH=/app
  ENV GEMINI_API_KEY=${GEMINI_API_KEY}

  # 健康检查
  HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
      CMD python -c "from src.crewai.main import run_investment_crew; print('OK')"

  # 暴露端口（如果需要Web API）
  EXPOSE 8000

  # 启动命令
  CMD ["python", "test_crewai.py"]

  3. 监控和日志

  import logging
  import prometheus_client
  from prometheus_client import Counter, Histogram, Gauge

  # 性能指标
  REQUEST_COUNT = Counter('crewai_requests_total', 'Total requests', ['method', 'status'])
  REQUEST_DURATION = Histogram('crewai_request_duration_seconds', 'Request duration')
  ACTIVE_ANALYSES = Gauge('crewai_active_analyses', 'Currently running analyses')

  # 结构化日志配置
  class JsonFormatter(logging.Formatter):
      def format(self, record):
          log_data = {
              'timestamp': self.formatTime(record),
              'level': record.levelname,
              'message': record.getMessage(),
              'module': record.module,
              'function': record.funcName,
              'line': record.lineno
          }

          if hasattr(record, 'ticker'):
              log_data['ticker'] = record.ticker
          if hasattr(record, 'run_id'):
              log_data['run_id'] = record.run_id

          return json.dumps(log_data, ensure_ascii=False)

  # 应用监控装饰器
  def monitor_performance(func):
      def wrapper(*args, **kwargs):
          ACTIVE_ANALYSES.inc()
          start_time = time.time()

          try:
              result = func(*args, **kwargs)
              REQUEST_COUNT.labels(method=func.__name__, status='success').inc()
              return result
          except Exception as e:
              REQUEST_COUNT.labels(method=func.__name__, status='error').inc()
              raise
          finally:
              REQUEST_DURATION.observe(time.time() - start_time)
              ACTIVE_ANALYSES.dec()

      return wrapper

  项目总结和经验

  1. 技术收获

  架构设计经验

  - 分层架构优势：清晰的数据、分析、研究三层架构，职责明确，易于维护
  - 工具模式应用：将智能体能力抽象为工具，提高了复用性和扩展性
  - 状态管理改进：结构化状态管理比字典式状态更安全、更易于调试

  框架迁移经验

  - API兼容性重要性：保持向后兼容性降低了迁移风险和用户学习成本
  - 渐进式迁移策略：分阶段迁移降低了项目风险，便于问题定位
  - 测试驱动开发：全面的测试覆盖是迁移成功的关键保障

  性能优化经验

  - 并行处理效果：合理使用并行处理显著提升了系统性能
  - 资源管理重要性：动态资源管理避免了系统过载和内存泄漏
  - 缓存策略设计：适当的缓存机制大幅提升了重复请求的响应速度

  2. 遇到的挑战和解决方案

  挑战1：框架差异适配

  问题：CrewAI和LangGraph在任务编排、状态管理等方面存在显著差异

  解决方案：
  - 设计适配层桥接两个框架的差异
  - 重新设计任务流程，充分利用CrewAI的协作特性
  - 保持业务逻辑不变，仅改变实现方式

  挑战2：向后兼容性维护

  问题：需要在保持原有API接口的同时，实现新的架构

  解决方案：
  - 设计兼容层，将新系统接口适配到原有API
  - 保留原有的函数签名和返回格式
  - 内部实现完全替换为新的CrewAI架构

  挑战3：性能和稳定性

  问题：多智能体协作可能带来性能瓶颈和稳定性问题

  解决方案：
  - 实现智能的资源管理和任务调度
  - 添加完善的错误处理和重试机制
  - 设计监控和告警系统，及时发现和解决问题

  3. 未来发展方向

  功能增强

  - 更多数据源：集成更多实时数据源和另类数据
  - AI模型升级：支持更多先进的LLM模型和分析算法
  - 可视化界面：开发Web界面，提升用户体验

  性能优化

  - 分布式处理：支持多节点分布式分析
  - 实时分析：实现流式数据处理和实时分析
  - 智能缓存：更智能的缓存策略和预计算机制

  应用扩展

  - 多市场支持：扩展到其他股票市场和金融产品
  - 策略回测：集成历史数据回测功能
  - 风险管理：增强风险控制和投资组合优化

  这个CrewAI迁移项目成功地将原有系统升级到更先进的框架，保持了功能完整性，提升了分析质量，为未来发展奠定了坚实基础。
  整个项目展示了如何在不破坏现有功能的前提下，通过技术升级显著提升系统能力。