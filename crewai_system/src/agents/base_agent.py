"""
基础智能体类
提供所有智能体的通用功能和接口
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from crewai import Agent
from crewai.tools import BaseTool

from crewai_system.src.utils.logging_config import get_logger, log_agent_info, log_error, log_success, log_failure
from crewai_system.src.utils.llm_config import get_llm_config


class BaseAgent(Agent, ABC):
    """
    基础智能体类，集成原系统的功能特性
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[Any] = None,
        agent_name: Optional[str] = None,
        **kwargs
    ):
        """
        初始化基础智能体

        Args:
            role: 智能体角色
            goal: 智能体目标
            backstory: 智能体背景故事
            tools: 可用工具列表
            llm: 语言模型实例
            agent_name: 智能体名称（用于日志记录）
            **kwargs: 其他参数
        """
        import os

        # 设置智能体名称
        self._agent_name = agent_name or self.__class__.__name__

        # 获取LLM配置并设置环境变量，让CrewAI使用我们的配置
        llm_config_instance = get_llm_config()

        # 如果配置了OpenAI兼容API，设置标准环境变量
        if llm_config_instance.config.openai_compatible_api_key:
            os.environ['OPENAI_API_KEY'] = llm_config_instance.config.openai_compatible_api_key
            os.environ['OPENAI_BASE_URL'] = str(llm_config_instance.config.openai_compatible_base_url)
            # 使用模型名称字符串
            if llm is None:
                llm = llm_config_instance.config.openai_compatible_model
        elif llm_config_instance.config.gemini_api_key:
            # Gemini配置
            os.environ['GOOGLE_API_KEY'] = llm_config_instance.config.gemini_api_key
            if llm is None:
                llm = llm_config_instance.config.gemini_model

        # 初始化父类（不传递额外属性，避免被CrewAI Agent类过滤掉）
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            llm=llm,
            **kwargs
        )

    @property
    def agent_name(self) -> str:
        """获取智能体名称"""
        return getattr(self, '_agent_name', self.__class__.__name__)

    @agent_name.setter
    def agent_name(self, value: str):
        """设置智能体名称"""
        self._agent_name = value

    @property
    def logger(self):
        """获取日志记录器"""
        logger = getattr(self, '_logger', None)
        if logger is None:
            logger = get_logger(f"agent.{self.agent_name}")
            self._logger = logger
        return logger

    @logger.setter
    def logger(self, value):
        """设置日志记录器"""
        self._logger = value

    @property
    def execution_start_time(self) -> Optional[float]:
        """获取执行开始时间"""
        return getattr(self, '_execution_start_time', None)

    @execution_start_time.setter
    def execution_start_time(self, value: Optional[float]):
        """设置执行开始时间"""
        self._execution_start_time = value

    @property
    def execution_end_time(self) -> Optional[float]:
        """获取执行结束时间"""
        return getattr(self, '_execution_end_time', None)

    @execution_end_time.setter
    def execution_end_time(self, value: Optional[float]):
        """设置执行结束时间"""
        self._execution_end_time = value

    @property
    def execution_status(self) -> str:
        """获取执行状态"""
        return getattr(self, '_execution_status', 'idle')

    @execution_status.setter
    def execution_status(self, value: str):
        """设置执行状态"""
        self._execution_status = value

    def log_execution_start(self, task_description: str = ""):
        """记录执行开始"""
        self.execution_start_time = time.time()
        self.execution_status = "running"
        log_agent_info(self.agent_name, f"开始执行任务: {task_description}")

    def log_execution_complete(self, result_summary: str = ""):
        """记录执行完成"""
        self.execution_end_time = time.time()
        execution_time = self.execution_end_time - self.execution_start_time

        self.execution_status = "completed"
        log_success(f"{self.agent_name} 执行完成，耗时: {execution_time:.2f}秒 - {result_summary}")

    def log_execution_error(self, error: Exception, context: str = ""):
        """记录执行错误"""
        self.execution_end_time = time.time()
        execution_time = self.execution_end_time - self.execution_start_time

        self.execution_status = "error"
        log_failure(f"{self.agent_name} 执行失败，耗时: {execution_time:.2f}秒 - {context}")
        log_error(f"错误详情: {str(error)}", self.agent_name)

    def log_reasoning(self, reasoning_data: Any, title: str = "推理过程"):
        """记录推理过程"""
        try:
            if isinstance(reasoning_data, (dict, list)):
                reasoning_str = json.dumps(reasoning_data, ensure_ascii=False, indent=2)
            else:
                reasoning_str = str(reasoning_data)

            print(f"{'='*20} {self.agent_name} {title} {'='*20}")
            print(reasoning_str)
        except Exception as e:
            print(f"记录推理过程失败: {e}")

    def format_agent_output(
        self,
        content: Any,
        signal: Optional[str] = None,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        格式化智能体输出，与原系统兼容

        Args:
            content: 主要内容
            signal: 信号（bullish/bearish/neutral等）
            confidence: 置信度 (0-1)
            reasoning: 推理说明
            metadata: 元数据

        Returns:
            格式化的输出字典
        """
        output = {
            "agent_name": self.agent_name,
            "timestamp": time.time(),
            "content": content,
        }

        if signal is not None:
            output["signal"] = signal

        if confidence is not None:
            output["confidence"] = max(0.0, min(1.0, confidence))

        if reasoning is not None:
            output["reasoning"] = reasoning

        if metadata is not None:
            output["metadata"] = metadata

        return output

    def safe_execute(self, func, *args, **kwargs):
        """
        安全执行函数，包含错误处理和重试机制

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            执行结果或错误信息
        """
        agent_max_retries = 3
        retry_delay = 1.0

        for attempt in range(agent_max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.logger.warning(f"执行失败（尝试 {attempt + 1}/{agent_max_retries}）: {e}")

                if attempt < agent_max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    self.log_execution_error(e, f"函数执行失败: {func.__name__}")
                    return {
                        "error": str(e),
                        "agent_name": self.agent_name,
                        "timestamp": time.time(),
                        "status": "error"
                    }

    @abstractmethod
    def process_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理任务的抽象方法，子类必须实现

        Args:
            task_context: 任务上下文

        Returns:
            处理结果
        """
        pass

    def execute_with_context(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        在上下文中执行任务

        Args:
            task_context: 任务上下文

        Returns:
            处理结果
        """
        task_description = task_context.get("description", "未指定任务")
        self.log_execution_start(task_description)

        try:
            result = self.process_task(task_context)
            self.log_execution_complete(f"成功处理: {task_description}")
            return result
        except Exception as e:
            self.log_execution_error(e, task_description)
            raise

    def validate_input(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据
            required_fields: 必需字段列表

        Returns:
            验证是否通过
        """
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)

        if missing_fields:
            self.logger.warning(f"缺少必需字段: {missing_fields}")
            return False

        return True

    def __repr__(self):
        """智能体字符串表示"""
        return f"<{self.__class__.__name__}: {self.agent_name}>"