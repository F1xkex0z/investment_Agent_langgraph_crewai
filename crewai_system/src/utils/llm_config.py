"""
LLM配置模块
提供统一的LLM配置和管理功能
"""

import os
from typing import Optional, Any, Dict
from crewai_system.src.utils.logging_config import get_logger, log_info, log_error, log_success, log_failure
from crewai_system.src.config import config

logger = get_logger("llm")


class LLMConfig:
    """LLM配置管理类"""

    def __init__(self):
        self.config = config
        self._llm_instance = None

    def get_llm_instance(self) -> Optional[Any]:
        """
        获取LLM实例

        Returns:
            配置好的LLM实例或None（如果配置无效）
        """
        if self._llm_instance is not None:
            return self._llm_instance

        try:
            # 优先使用OpenAI兼容API
            if self.config.openai_compatible_api_key and self.config.openai_compatible_base_url:
                return self._create_openai_compatible_llm()

            # 其次使用Gemini API
            if self.config.gemini_api_key:
                return self._create_gemini_llm()

            logger.warning("未配置有效的API密钥，将使用模拟模式")
            return None

        except Exception as e:
            logger.error(f"创建LLM实例失败: {e}")
            return None

    def _create_openai_compatible_llm(self) -> Any:
        """创建OpenAI兼容的LLM实例"""
        try:
            from litellm import completion
            from litellm import OpenAI

            # 确保base_url是字符串类型
            base_url = str(self.config.openai_compatible_base_url)

            # 创建OpenAI兼容客户端 - 确保base_url是字符串
            client = OpenAI(
                api_key=self.config.openai_compatible_api_key,
                base_url=base_url
            )

            # 额外确保base_url属性是字符串（防止pydantic URL对象问题）
            if hasattr(client, 'base_url') and not isinstance(client.base_url, str):
                client.base_url = base_url

            log_success(f"成功创建OpenAI兼容LLM客户端: {base_url}")
            return client

        except ImportError:
            log_error("未安装litellim库，无法创建OpenAI兼容客户端", "llm")
            return None
        except Exception as e:
            log_error(f"创建OpenAI兼容客户端失败: {e}", "llm")
            return None

    def _create_gemini_llm(self) -> Any:
        """创建Gemini LLM实例"""
        try:
            import google.generativeai as genai

            # 配置Gemini
            genai.configure(api_key=self.config.gemini_api_key)

            # 创建模型实例
            model = genai.GenerativeModel(self.config.gemini_model)

            log_success(f"成功创建Gemini LLM实例: {self.config.gemini_model}")
            return model

        except ImportError:
            log_error("未安装google-generativeai库，无法创建Gemini客户端", "llm")
            return None
        except Exception as e:
            log_error(f"创建Gemini客户端失败: {e}", "llm")
            return None

    def chat_completion(self, messages: list, **kwargs) -> Dict[str, Any]:
        """
        执行聊天补全

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            响应结果
        """
        llm_instance = self.get_llm_instance()
        if llm_instance is None:
            return {
                "error": "LLM实例未配置",
                "content": "模拟响应：由于未配置API密钥，返回模拟分析结果"
            }

        try:
            # 使用OpenAI兼容API
            if hasattr(llm_instance, 'chat') and hasattr(llm_instance.chat, 'completions'):
                response = llm_instance.chat.completions.create(
                    model=self.config.openai_compatible_model,
                    messages=messages,
                    **kwargs
                )
                return {
                    "content": response.choices[0].message.content,
                    "usage": getattr(response, 'usage', None),
                    "model": response.model,
                    "provider": "openai_compatible"
                }

            # 使用litellm直接调用
            else:
                from litellm import completion

                # 确保所有参数都是正确的类型
                model = f"openai/{self.config.openai_compatible_model}"
                api_key = str(self.config.openai_compatible_api_key)
                api_base = str(self.config.openai_compatible_base_url)

                response = completion(
                    model=model,
                    messages=messages,
                    api_key=api_key,
                    api_base=api_base,
                    **kwargs
                )

                return {
                    "content": response.choices[0].message.content,
                    "usage": getattr(response, 'usage', None),
                    "model": response.model,
                    "provider": "litellm_openai_compatible"
                }

        except Exception as e:
            log_error(f"LLM调用失败: {e}", "llm")
            return {
                "error": str(e),
                "content": "模拟响应：由于API调用失败，返回模拟分析结果"
            }

    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "has_openai_compatible": bool(self.config.openai_compatible_api_key),
            "has_gemini": bool(self.config.gemini_api_key),
            "openai_model": self.config.openai_compatible_model,
            "gemini_model": self.config.gemini_model,
            "llm_available": self.get_llm_instance() is not None
        }


# 全局LLM配置实例
llm_config = LLMConfig()


def get_llm_config() -> LLMConfig:
    """获取全局LLM配置实例"""
    return llm_config


def chat_completion(messages: list, **kwargs) -> Dict[str, Any]:
    """
    便捷的聊天补全函数

    Args:
        messages: 消息列表
        **kwargs: 其他参数

    Returns:
        响应结果
    """
    return llm_config.chat_completion(messages, **kwargs)