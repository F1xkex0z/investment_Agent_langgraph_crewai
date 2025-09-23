"""
共享上下文机制
实现智能体间的数据共享和状态管理
"""

import json
import threading
import time
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict

from .logging_config import get_logger


@dataclass
class ContextEntry:
    """上下文条目"""
    key: str
    value: Any
    timestamp: float
    source_agent: str
    data_type: str = "raw"
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class SharedContext:
    """
    共享上下文管理器
    提供线程安全的智能体间数据共享机制
    """

    def __init__(self):
        self.logger = get_logger("debug")
        self._data: Dict[str, ContextEntry] = {}
        self._lock = threading.RLock()
        self._subscribers: Dict[str, Set[str]] = defaultdict(set)  # key -> set of agent names
        self._change_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000

    def set(
        self,
        key: str,
        value: Any,
        source_agent: str,
        data_type: str = "raw",
        ttl: Optional[float] = None,
        notify: bool = True
    ):
        """
        设置共享上下文数据

        Args:
            key: 数据键
            value: 数据值
            source_agent: 来源智能体
            data_type: 数据类型
            ttl: 生存时间（秒）
            notify: 是否通知订阅者
        """
        with self._lock:
            # 创建新的上下文条目
            entry = ContextEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                source_agent=source_agent,
                data_type=data_type,
                ttl=ttl
            )

            # 记录变更历史
            old_entry = self._data.get(key)
            change_record = {
                "timestamp": time.time(),
                "key": key,
                "old_value": old_entry.value if old_entry else None,
                "new_value": value,
                "source_agent": source_agent,
                "change_type": "update" if old_entry else "create"
            }
            self._add_to_history(change_record)

            # 设置数据
            self._data[key] = entry

            self.logger.debug(
                f"智能体 {source_agent} 设置共享数据 {key} (类型: {data_type})"
            )

            # 通知订阅者
            if notify:
                self._notify_subscribers(key, entry)

    def get(self, key: str, default: Any = None, check_expiry: bool = True) -> Any:
        """
        获取共享上下文数据

        Args:
            key: 数据键
            default: 默认值
            check_expiry: 是否检查过期

        Returns:
            数据值或默认值
        """
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return default

            # 检查是否过期
            if check_expiry and entry.is_expired():
                self.logger.debug(f"共享数据 {key} 已过期，将被删除")
                del self._data[key]
                return default

            return entry.value

    def exists(self, key: str, check_expiry: bool = True) -> bool:
        """
        检查键是否存在

        Args:
            key: 数据键
            check_expiry: 是否检查过期

        Returns:
            是否存在
        """
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False

            if check_expiry and entry.is_expired():
                del self._data[key]
                return False

            return True

    def delete(self, key: str, notify: bool = True):
        """
        删除共享上下文数据

        Args:
            key: 数据键
            notify: 是否通知订阅者
        """
        with self._lock:
            if key in self._data:
                entry = self._data[key]
                del self._data[key]

                # 记录变更历史
                change_record = {
                    "timestamp": time.time(),
                    "key": key,
                    "old_value": entry.value,
                    "new_value": None,
                    "source_agent": "system",
                    "change_type": "delete"
                }
                self._add_to_history(change_record)

                self.logger.debug(f"删除共享数据 {key}")

                # 通知订阅者
                if notify:
                    self._notify_subscribers(key, None, is_delete=True)

    def get_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """
        根据前缀获取所有匹配的数据

        Args:
            prefix: 键前缀

        Returns:
            匹配的数据字典
        """
        with self._lock:
            result = {}
            for key, entry in self._data.items():
                if key.startswith(prefix) and not entry.is_expired():
                    result[key] = entry.value
            return result

    def get_by_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        获取指定智能体设置的所有数据

        Args:
            agent_name: 智能体名称

        Returns:
            该智能体设置的数据字典
        """
        with self._lock:
            result = {}
            for key, entry in self._data.items():
                if entry.source_agent == agent_name and not entry.is_expired():
                    result[key] = entry.value
            return result

    def subscribe(self, key: str, agent_name: str):
        """
        订阅键变更通知

        Args:
            key: 数据键
            agent_name: 智能体名称
        """
        with self._lock:
            self._subscribers[key].add(agent_name)
            self.logger.debug(f"智能体 {agent_name} 订阅了键 {key} 的变更通知")

    def unsubscribe(self, key: str, agent_name: str):
        """
        取消订阅键变更通知

        Args:
            key: 数据键
            agent_name: 智能体名称
        """
        with self._lock:
            if key in self._subscribers:
                self._subscribers[key].discard(agent_name)
                if not self._subscribers[key]:
                    del self._subscribers[key]
                self.logger.debug(f"智能体 {agent_name} 取消订阅键 {key}")

    def _notify_subscribers(self, key: str, entry: Optional[ContextEntry], is_delete: bool = False):
        """通知订阅者"""
        if key not in self._subscribers:
            return

        for agent_name in self._subscribers[key]:
            try:
                # 这里可以实现智能体通知机制
                # 例如通过事件总线或消息队列
                self.logger.debug(
                    f"通知智能体 {agent_name}: 键 {key} {'已删除' if is_delete else '已更新'}"
                )
            except Exception as e:
                self.logger.error(f"通知智能体 {agent_name} 失败: {e}")

    def _add_to_history(self, change_record: Dict[str, Any]):
        """添加变更历史记录"""
        self._change_history.append(change_record)
        if len(self._change_history) > self._max_history_size:
            self._change_history.pop(0)

    def get_history(self, key: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取变更历史

        Args:
            key: 特定键（可选）
            limit: 返回记录数量限制

        Returns:
            变更历史记录列表
        """
        with self._lock:
            if key:
                history = [record for record in self._change_history if record["key"] == key]
            else:
                history = self._change_history.copy()

            return history[-limit:] if limit else history

    def cleanup_expired(self):
        """清理过期数据"""
        with self._lock:
            expired_keys = []
            for key, entry in self._data.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._data[key]
                self.logger.debug(f"清理过期数据: {key}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取上下文统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            total_entries = len(self._data)
            expired_entries = sum(1 for entry in self._data.values() if entry.is_expired())
            total_subscriptions = sum(len(subscribers) for subscribers in self._subscribers.values())

            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries,
                "total_subscriptions": total_subscriptions,
                "subscribed_keys": len(self._subscribers),
                "history_size": len(self._change_history)
            }

    def export_context(self, include_expired: bool = False) -> Dict[str, Any]:
        """
        导出整个上下文

        Args:
            include_expired: 是否包含过期数据

        Returns:
            导出的上下文字典
        """
        with self._lock:
            result = {}
            for key, entry in self._data.items():
                if include_expired or not entry.is_expired():
                    result[key] = {
                        "value": entry.value,
                        "timestamp": entry.timestamp,
                        "source_agent": entry.source_agent,
                        "data_type": entry.data_type,
                        "expired": entry.is_expired()
                    }
            return result

    def import_context(self, context_data: Dict[str, Any], merge_strategy: str = "overwrite"):
        """
        导入上下文数据

        Args:
            context_data: 上下文数据字典
            merge_strategy: 合并策略 ("overwrite", "skip", "merge")
        """
        with self._lock:
            for key, data in context_data.items():
                if merge_strategy == "overwrite" or not self.exists(key, check_expiry=False):
                    self.set(
                        key=key,
                        value=data["value"],
                        source_agent=data.get("source_agent", "import"),
                        data_type=data.get("data_type", "raw"),
                        notify=False
                    )
                elif merge_strategy == "merge":
                    # 实现更复杂的合并逻辑
                    existing_value = self.get(key, check_expiry=False)
                    if isinstance(existing_value, dict) and isinstance(data["value"], dict):
                        merged_value = {**existing_value, **data["value"]}
                        self.set(
                            key=key,
                            value=merged_value,
                            source_agent=data.get("source_agent", "import"),
                            data_type=data.get("data_type", "raw"),
                            notify=False
                        )

    def clear(self):
        """清空所有上下文数据"""
        with self._lock:
            self._data.clear()
            self._subscribers.clear()
            self._change_history.clear()
            self.logger.info("共享上下文已清空")

    def __str__(self):
        """字符串表示"""
        stats = self.get_stats()
        return f"SharedContext(entries={stats['active_entries']}, subscriptions={stats['total_subscriptions']})"


# 全局共享上下文实例
global_context = SharedContext()


def get_global_context() -> SharedContext:
    """获取全局共享上下文实例"""
    return global_context


def reset_global_context():
    """重置全局共享上下文"""
    global_context.clear()


# 上下文管理器
class ContextManager:
    """上下文管理器，提供with语句支持"""

    def __init__(self, context: SharedContext, agent_name: str):
        self.context = context
        self.agent_name = agent_name
        self._changes = []

    def set(self, key: str, value: Any, **kwargs):
        """设置上下文数据"""
        self.context.set(key, value, source_agent=self.agent_name, **kwargs)
        self._changes.append(("set", key, value))

    def get(self, key: str, default: Any = None):
        """获取上下文数据"""
        return self.context.get(key, default)

    def __enter__(self):
        """进入上下文"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if exc_type is not None:
            self.logger.error(f"上下文执行异常: {exc_val}")
        return False

    @property
    def logger(self):
        """获取日志记录器"""
        return get_logger("agents")