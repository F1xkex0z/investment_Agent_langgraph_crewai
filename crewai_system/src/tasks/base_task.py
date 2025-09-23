"""
基础任务类
提供所有任务的通用功能和接口
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod

from crewai import Task
from crewai_system.src.utils.logging_config import get_logger, log_info, log_error, log_performance


class BaseTask(Task, ABC):
    """
    基础任务类，集成原系统的功能特性
    """

    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: Any,
        context: Optional[List[Task]] = None,
        tools: Optional[List[Any]] = None,
        task_id: Optional[str] = None,
        **kwargs
    ):
        """
        初始化基础任务

        Args:
            description: 任务描述
            expected_output: 预期输出
            agent: 执行任务的智能体
            context: 依赖的任务列表
            tools: 可用工具列表
            task_id: 任务ID（用于日志记录）
            **kwargs: 其他参数
        """
        # 初始化父类
        super().__init__(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=context or [],
            tools=tools or [],
            **kwargs
        )

        # 设置任务ID（在父类初始化后设置，避免被过滤）
        self._task_id = task_id or f"{self.__class__.__name__}_{int(time.time())}"
        # 保存agent引用
        self._agent = agent

        # 任务状态跟踪
        self._execution_start_time: Optional[float] = None
        self._execution_end_time: Optional[float] = None
        self._execution_status: str = "pending"
        self._retry_count: int = 0
        self._task_max_retries: int = 3

        # 任务结果缓存
        self._cached_result: Optional[Dict[str, Any]] = None
        self._cache_key: Optional[str] = None

    @property
    def agent(self):
        """获取执行任务的智能体"""
        return self._agent

    @agent.setter
    def agent(self, value):
        """设置执行任务的智能体"""
        self._agent = value

    @property
    def task_id(self) -> str:
        """获取任务ID"""
        return self._task_id

    @property
    def task_logger(self):
        """获取日志记录器"""
        return self._logger

    @property
    def execution_start_time(self) -> Optional[float]:
        """获取执行开始时间"""
        return self._execution_start_time

    @execution_start_time.setter
    def execution_start_time(self, value: Optional[float]):
        """设置执行开始时间"""
        self._execution_start_time = value

    @property
    def execution_end_time(self) -> Optional[float]:
        """获取执行结束时间"""
        return self._execution_end_time

    @execution_end_time.setter
    def execution_end_time(self, value: Optional[float]):
        """设置执行结束时间"""
        self._execution_end_time = value

    @property
    def execution_status(self) -> str:
        """获取执行状态"""
        return self._execution_status

    @execution_status.setter
    def execution_status(self, value: str):
        """设置执行状态"""
        self._execution_status = value

    @property
    def retry_count(self) -> int:
        """获取重试次数"""
        return self._retry_count

    @retry_count.setter
    def retry_count(self, value: int):
        """设置重试次数"""
        self._retry_count = value

    @property
    def max_retries(self) -> int:
        """获取最大重试次数"""
        return getattr(self, '_task_max_retries', 3)

    @max_retries.setter
    def max_retries(self, value: int):
        """设置最大重试次数"""
        self._task_max_retries = value

    @property
    def cached_result(self) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        return self._cached_result

    @cached_result.setter
    def cached_result(self, value: Optional[Dict[str, Any]]):
        """设置缓存结果"""
        self._cached_result = value

    @property
    def cache_key(self) -> Optional[str]:
        """获取缓存键"""
        return self._cache_key

    @cache_key.setter
    def cache_key(self, value: Optional[str]):
        """设置缓存键"""
        self._cache_key = value

    def log_task_start(self, context: Optional[Dict[str, Any]] = None):
        """记录任务开始"""
        self.execution_start_time = time.time()
        self.execution_status = "running"
        context_info = f" - 上下文: {context}" if context else ""
        self.task_logger.info(f"🚀 任务 {self.task_id} 开始执行{context_info}")
        self._debug_logger.debug(f"任务开始详情: task_id={self.task_id}, task_type={self.__class__.__name__}, context={context or {}}, agent_name={self.agent.agent_name if hasattr(self.agent, 'agent_name') else str(self.agent)}")

    def log_task_complete(self, result_summary: str = ""):
        """记录任务完成"""
        self.execution_end_time = time.time()
        execution_time = self.execution_end_time - self.execution_start_time

        self.execution_status = "completed"
        self.task_logger.info(
            f"✅ 任务 {self.task_id} 执行完成，耗时: {execution_time:.2f}秒 - {result_summary}"
        )

        # 记录智能体性能日志
        if hasattr(self.agent, 'agent_name'):
            log_agent_performance(self.task_logger, self.agent.agent_name, self.__class__.__name__,
                                datetime.fromtimestamp(self.execution_start_time),
                                datetime.fromtimestamp(self.execution_end_time),
                                True, result_summary)

        self._debug_logger.debug(f"任务完成详情: task_id={self.task_id}, execution_time={execution_time}, result_summary={result_summary}, success=True")

    def log_task_error(self, error: Exception, context: str = ""):
        """记录任务错误"""
        self.execution_end_time = time.time()
        execution_time = self.execution_end_time - self.execution_start_time

        self.execution_status = "error"
        self.task_logger.error(
            f"❌ 任务 {self.task_id} 执行失败，耗时: {execution_time:.2f}秒 - {context}"
        )
        self.task_logger.error(f"错误详情: {str(error)}")

        # 记录智能体性能日志
        if hasattr(self.agent, 'agent_name'):
            log_agent_performance(self.task_logger, self.agent.agent_name, self.__class__.__name__,
                                datetime.fromtimestamp(self.execution_start_time),
                                datetime.fromtimestamp(self.execution_end_time),
                                False, str(error))

        self._debug_logger.debug(f"任务错误详情: task_id={self.task_id}, execution_time={execution_time}, error_type={type(error).__name__}, error_message={str(error)}, context={context}, success=False")

    def log_task_retry(self, error: Exception, retry_count: int):
        """记录任务重试"""
        max_retries_value = self.max_retries  # 使用property getter获取实际值
        self.task_logger.warning(
            f"🔄 任务 {self.task_id} 重试 ({retry_count}/{max_retries_value}): {str(error)}"
        )
        self._debug_logger.debug(f"任务重试详情: task_id={self.task_id}, retry_count={retry_count}, max_retries={max_retries_value}, error_type={type(error).__name__}, error_message={str(error)}")

    def generate_cache_key(self, context: Dict[str, Any]) -> str:
        """
        生成缓存键

        Args:
            context: 任务上下文

        Returns:
            缓存键字符串
        """
        # 基于任务描述和关键上下文生成缓存键
        cache_data = {
            "description": self.description,
            "ticker": context.get("ticker", ""),
            "start_date": context.get("start_date", ""),
            "end_date": context.get("end_date", ""),
        }
        return json.dumps(cache_data, sort_keys=True)

    def is_cache_valid(self, cache_key: str, ttl: int = 3600) -> bool:
        """
        检查缓存是否有效

        Args:
            cache_key: 缓存键
            ttl: 缓存生存时间（秒）

        Returns:
            缓存是否有效
        """
        if self.cached_result is None or self.cache_key != cache_key:
            return False

        cached_time = self.cached_result.get("timestamp", 0)
        return (time.time() - cached_time) < ttl

    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存结果

        Args:
            cache_key: 缓存键

        Returns:
            缓存的结果或None
        """
        if self.is_cache_valid(cache_key):
            self.logger.info(f"📦 任务 {self.task_id} 使用缓存结果")
            return self.cached_result
        return None

    def cache_result(self, result: Dict[str, Any], cache_key: str):
        """
        缓存任务结果

        Args:
            result: 任务结果
            cache_key: 缓存键
        """
        self.cached_result = {
            **result,
            "timestamp": time.time(),
            "cache_key": cache_key
        }
        self.cache_key = cache_key

    def validate_context(self, context: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        验证任务上下文

        Args:
            context: 任务上下文
            required_fields: 必需字段列表

        Returns:
            验证是否通过
        """
        missing_fields = []
        for field in required_fields:
            if field not in context or context[field] is None:
                missing_fields.append(field)

        if missing_fields:
            self.task_logger.warning(f"任务 {self.task_id} 缺少必需字段: {missing_fields}")
            return False

        return True

    def format_task_output(
        self,
        content: Any,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        格式化任务输出

        Args:
            content: 主要内容
            status: 任务状态
            metadata: 元数据
            execution_time: 执行时间

        Returns:
            格式化的输出字典
        """
        if execution_time is None and self.execution_start_time and self.execution_end_time:
            execution_time = self.execution_end_time - self.execution_start_time

        output = {
            "task_id": self.task_id,
            "agent_name": self.agent.agent_name if hasattr(self.agent, 'agent_name') else str(self.agent),
            "status": status,
            "content": content,
            "timestamp": time.time(),
        }

        if execution_time is not None:
            output["execution_time"] = execution_time

        if metadata is not None:
            output["metadata"] = metadata

        return output

    def execute_with_retry(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        带重试机制的任务执行

        Args:
            context: 任务上下文

        Returns:
            任务执行结果
        """
        # 检查缓存
        cache_key = self.generate_cache_key(context)
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            return cached_result

        # 执行任务
        max_retries_value = getattr(self, '_task_max_retries', 3)

        for attempt in range(max_retries_value):
                try:
                    self.log_task_start(context)

                    # 验证上下文
                    if not self.validate_context(context, self.get_required_fields()):
                        raise ValueError(f"任务上下文验证失败")

                    # 执行具体任务逻辑
                    result = self.execute_task_logic(context)

                    # 缓存结果
                    self.cache_result(result, cache_key)

                    self.log_task_complete("任务执行成功")
                    return result

                except Exception as e:
                    self.retry_count = attempt + 1
                    if attempt < max_retries_value - 1:
                        self.log_task_retry(e, self.retry_count)
                        time.sleep(2 ** attempt)  # 指数退避
                    else:
                        self.log_task_error(e, "达到最大重试次数")
                        return self.format_task_output(
                            content={"error": str(e)},
                            status="error",
                            metadata={"retry_count": self.retry_count}
                        )

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """
        获取任务必需字段

        Returns:
            必需字段列表
        """
        pass

    @abstractmethod
    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务逻辑，子类必须实现

        Args:
            context: 任务上下文

        Returns:
            任务执行结果
        """
        pass

    def __repr__(self):
        """任务字符串表示"""
        return f"<{self.__class__.__name__}: {self.task_id}>"


class SequentialTask(BaseTask):
    """顺序执行的任务"""

    def __init__(self, subtasks: List[BaseTask], **kwargs):
        super().__init__(**kwargs)
        self.subtasks = subtasks

    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """顺序执行子任务"""
        results = []
        current_context = context.copy()

        for subtask in self.subtasks:
            subtask_result = subtask.execute_with_retry(current_context)
            results.append(subtask_result)
            # 将子任务结果合并到上下文中
            current_context.update(subtask_result.get("content", {}))

        return self.format_task_output(
            content={
                "subtask_results": results,
                "final_context": current_context
            },
            status="success"
        )


class ParallelTask(BaseTask):
    """并行执行的任务"""

    def __init__(self, subtasks: List[BaseTask], **kwargs):
        super().__init__(**kwargs)
        self.subtasks = subtasks

    def execute_task_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """并行执行子任务"""
        import concurrent.futures
        import threading

        results = []
        lock = threading.Lock()

        def execute_subtask(subtask):
            try:
                result = subtask.execute_with_retry(context)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    results.append({
                        "task_id": subtask.task_id,
                        "error": str(e),
                        "status": "error"
                    })

        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.subtasks)) as executor:
            futures = [executor.submit(execute_subtask, subtask) for subtask in self.subtasks]
            concurrent.futures.wait(futures)

        return self.format_task_output(
            content={
                "subtask_results": results,
                "completed_count": len([r for r in results if r.get("status") == "success"]),
                "failed_count": len([r for r in results if r.get("status") == "error"])
            },
            status="success"
        )