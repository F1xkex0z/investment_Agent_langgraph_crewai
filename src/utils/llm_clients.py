import os
import time
import backoff
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# 设置日志记录
logger = setup_logger('llm_clients')


class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    def get_completion(self, messages, **kwargs):
        """获取模型回答"""
        pass


class GeminiClient(LLMClient):
    """Google Gemini API 客户端"""

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        if not self.api_key:
            logger.error(f"{ERROR_ICON} 未找到 GEMINI_API_KEY 环境变量")
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables")

        # 初始化 Gemini 客户端
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"{SUCCESS_ICON} Gemini 客户端初始化成功")

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        max_time=300,
        giveup=lambda e: "AFC is enabled" not in str(e)
    )
    def generate_content_with_retry(self, contents, config=None, max_tokens=14096):
        """带重试机制的内容生成函数"""
        try:
            logger.info(f"{WAIT_ICON} 正在调用 Gemini API...")
            logger.debug(f"请求内容: {contents}")
            logger.debug(f"请求配置: {config}, max_tokens: {max_tokens}")

            # 如果配置不存在，创建一个新的配置字典
            if config is None:
                config = {}
            
            # 如果提供了max_tokens参数，将其添加到配置中
            if max_tokens is not None:
                config['generation_config'] = config.get('generation_config', {})
                config['generation_config']['max_output_tokens'] = max_tokens

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            logger.info(f"{SUCCESS_ICON} API 调用成功")
            logger.debug(f"响应内容: {response.text[:500]}...")
            return response
        except Exception as e:
            error_msg = str(e)
            if "location" in error_msg.lower():
                logger.info(
                    f"\033[91m❗ Gemini API 地理位置限制错误: 请使用美国节点VPN后重试\033[0m")
                logger.error(f"详细错误: {error_msg}")
            elif "AFC is enabled" in error_msg:
                logger.warning(
                    f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {error_msg}")
                time.sleep(5)
            else:
                logger.error(f"{ERROR_ICON} API 调用失败: {error_msg}")
            raise e

    def get_completion(self, messages, max_retries=3, initial_retry_delay=1, **kwargs):
        """获取聊天完成结果，包含重试逻辑"""
        try:
            logger.info(f"{WAIT_ICON} 使用 Gemini 模型: {self.model}")
            logger.debug(f"消息内容: {messages}")

            # 从kwargs中提取max_tokens参数
            max_tokens = kwargs.get("max_tokens")

            for attempt in range(max_retries):
                try:
                    # 转换消息格式
                    prompt = ""
                    system_instruction = None

                    for message in messages:
                        role = message["role"]
                        content = message["content"]
                        if role == "system":
                            system_instruction = content
                        elif role == "user":
                            prompt += f"User: {content}\n"
                        elif role == "assistant":
                            prompt += f"Assistant: {content}\n"

                    # 准备配置
                    config = {}
                    if system_instruction:
                        config['system_instruction'] = system_instruction

                    # 调用 API，传递max_tokens参数
                    response = self.generate_content_with_retry(
                        contents=prompt.strip(),
                        config=config,
                        max_tokens=max_tokens
                    )

                    if response is None:
                        logger.warning(
                            f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(
                                f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                            time.sleep(retry_delay)
                            continue
                        return None

                    logger.debug(f"API 原始响应: {response.text}")
                    logger.info(f"{SUCCESS_ICON} 成功获取 Gemini 响应")

                    # 直接返回文本内容
                    return response.text

                except Exception as e:
                    logger.error(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                        return None

        except Exception as e:
            logger.error(f"{ERROR_ICON} get_completion 发生错误: {str(e)}")
            return None


class OpenAICompatibleClient(LLMClient):
    """OpenAI 兼容 API 客户端"""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        self.model = model or os.getenv("OPENAI_COMPATIBLE_MODEL")

        if not self.api_key:
            logger.error(f"{ERROR_ICON} 未找到 OPENAI_COMPATIBLE_API_KEY 环境变量")
            raise ValueError(
                "OPENAI_COMPATIBLE_API_KEY not found in environment variables")

        if not self.base_url:
            logger.error(f"{ERROR_ICON} 未找到 OPENAI_COMPATIBLE_BASE_URL 环境变量")
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL not found in environment variables")

        if not self.model:
            logger.error(f"{ERROR_ICON} 未找到 OPENAI_COMPATIBLE_MODEL 环境变量")
            raise ValueError(
                "OPENAI_COMPATIBLE_MODEL not found in environment variables")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        logger.info(f"{SUCCESS_ICON} OpenAI Compatible 客户端初始化成功")

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        max_time=300
    )
    def call_api_with_retry(self, messages, stream=False, max_tokens=None):
        """带重试机制的 API 调用函数"""
        try:
            logger.info(f"{WAIT_ICON} 正在调用 OpenAI Compatible API...")
            logger.debug(f"请求内容: {messages}")
            logger.debug(f"模型: {self.model}, 流式: {stream}, max_tokens: {max_tokens}")

            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": messages,
                "stream": stream
            }
            
            # 如果提供了max_tokens参数，则添加到请求中
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(
                **request_params
            )

            logger.info(f"{SUCCESS_ICON} API 调用成功")
            return response
        except Exception as e:
            error_msg = str(e)
            logger.error(f"{ERROR_ICON} API 调用失败: {error_msg}")
            raise e

    def get_completion(self, messages, max_retries=3, initial_retry_delay=1, **kwargs):
        """获取聊天完成结果，包含重试逻辑"""
        try:
            logger.info(f"{WAIT_ICON} 使用 OpenAI Compatible 模型: {self.model}")
            logger.debug(f"消息内容: {messages}")

            # 从kwargs中提取max_tokens参数
            max_tokens = kwargs.get("max_tokens")

            for attempt in range(max_retries):
                try:
                    # 调用 API，传递max_tokens参数
                    response = self.call_api_with_retry(messages, max_tokens=max_tokens)

                    if response is None:
                        logger.warning(
                            f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(
                                f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                            time.sleep(retry_delay)
                            continue
                        return None

                    # 打印调试信息
                    content = response.choices[0].message.content
                    ## logger.debug(f"API 原始响应: {content[:500]}...")
                    logger.debug(f"API 原始响应: {content}")
                    logger.info(f"{SUCCESS_ICON} 成功获取 OpenAI Compatible 响应")

                    # 直接返回文本内容
                    return content

                except Exception as e:
                    logger.error(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                        return None

        except Exception as e:
            logger.error(f"{ERROR_ICON} get_completion 发生错误: {str(e)}")
            return None


class LLMClientFactory:
    """LLM 客户端工厂类"""

    @staticmethod
    def create_client(client_type="auto", **kwargs):
        """
        创建 LLM 客户端

        Args:
            client_type: 客户端类型 ("auto", "gemini", "openai_compatible")
            **kwargs: 特定客户端的配置参数

        Returns:
            LLMClient: 实例化的 LLM 客户端
        """
        # 打印当前客户端选择和环境变量状态
        logger.info(f"开始创建 LLM 客户端，类型: {client_type}")
        logger.info(f"OpenAI Compatible API 配置检查 - API_KEY: {'存在' if kwargs.get('api_key') or os.getenv('OPENAI_COMPATIBLE_API_KEY') else '不存在'}")
        logger.info(f"OpenAI Compatible API 配置检查 - BASE_URL: {'存在' if kwargs.get('base_url') or os.getenv('OPENAI_COMPATIBLE_BASE_URL') else '不存在'}")
        logger.info(f"OpenAI Compatible API 配置检查 - MODEL: {'存在' if kwargs.get('model') or os.getenv('OPENAI_COMPATIBLE_MODEL') else '不存在'}")
        logger.info(f"Gemini API 配置检查 - API_KEY: {'存在' if kwargs.get('api_key') or os.getenv('GEMINI_API_KEY') and os.getenv('GEMINI_API_KEY') != 'your_gemini_api_key_here' else '不存在或为占位符'}")
        
        # 如果设置为 auto，自动检测可用的客户端
        if client_type == "auto":
            # 检查是否提供了 OpenAI Compatible API 相关配置
            if (kwargs.get("api_key") and kwargs.get("base_url") and kwargs.get("model")) or \
               (os.getenv("OPENAI_COMPATIBLE_API_KEY") and os.getenv("OPENAI_COMPATIBLE_BASE_URL") and os.getenv("OPENAI_COMPATIBLE_MODEL")):
                client_type = "openai_compatible"
                logger.info(f"{WAIT_ICON} 自动选择 OpenAI Compatible API")
            else:
                client_type = "gemini"
                logger.info(f"{WAIT_ICON} 自动选择 Gemini API")

        if client_type == "gemini":
            logger.info("正在创建 Gemini 客户端")
            client = GeminiClient(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model")
            )
            logger.info("Gemini 客户端创建成功")
            return client
        elif client_type == "openai_compatible":
            logger.info("正在创建 OpenAI Compatible 客户端")
            client = OpenAICompatibleClient(
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url"),
                model=kwargs.get("model")
            )
            logger.info("OpenAI Compatible 客户端创建成功")
            return client
        else:
            raise ValueError(f"不支持的客户端类型: {client_type}")
