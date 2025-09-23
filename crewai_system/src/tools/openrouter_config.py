import time
import os
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from crewai_system.src.utils.logging_config import get_logger
from crewai_system.src.utils.llm_clients import LLMClientFactory

# 设置日志记录
logger = get_logger('api_calls')


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info("✅ 已加载环境变量: {env_path}")
else:
    logger.warning("❌ 未找到环境变量文件: {env_path}")

# 验证OpenAI兼容API环境变量
api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
model = os.getenv("OPENAI_COMPATIBLE_MODEL")

if not api_key:
    logger.error("❌ 未找到 OPENAI_COMPATIBLE_API_KEY 环境变量")
    raise ValueError("OPENAI_COMPATIBLE_API_KEY not found in environment variables")
if not model:
    model = "gpt-4o"
    logger.info("⏳ 使用默认模型: {model}")


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """带重试机制的内容生成函数"""
    # 注意：此函数保留用于向后兼容，但实际上不会被调用，因为LLMClientFactory会创建合适的客户端
    try:
        logger.info("⏳ 正在调用 API...")
        # 实际API调用逻辑已移至LLMClientFactory和相关客户端类
        raise NotImplementedError("generate_content_with_retry已被弃用，请使用get_chat_completion函数")
    except Exception as e:
        error_msg = str(e)
        logger.error("❌ API 调用失败: {error_msg}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1,
                        client_type="openai_compatible", api_key=None, base_url=None, max_tokens=14096):
    """
    获取聊天完成结果，包含重试逻辑

    Args:
        messages: 消息列表，OpenAI 格式
        model: 模型名称（可选）
        max_retries: 最大重试次数
        initial_retry_delay: 初始重试延迟（秒）
        client_type: 客户端类型 ("auto", "gemini", "openai_compatible")
        api_key: API 密钥（可选，仅用于 OpenAI Compatible API）
        base_url: API 基础 URL（可选，仅用于 OpenAI Compatible API）
        max_tokens: 最大生成的token数量（可选）

    Returns:
        str: 模型回答内容或 None（如果出错）
    """
    try:
        # 创建客户端
        client = LLMClientFactory.create_client(
            client_type=client_type,
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        # 获取回答
        return client.get_completion(
            messages=messages,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error("❌ get_chat_completion 发生错误: {str(e)}")
        return None