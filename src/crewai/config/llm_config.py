"""
CrewAI LLM Configuration

This module handles LLM configuration for CrewAI agents.
"""

import os
from typing import Dict, Any, Optional
from crewai import LLM
from src.utils.logging_config import setup_logger

logger = setup_logger('crewai_llm_config')


class LLMConfig:
    """Configuration for LLM models in CrewAI system"""

    def __init__(self):
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.temperature = 0.1
        self.max_tokens = 4000

        # OpenAI兼容配置
        self.openai_compatible_base_url = os.getenv('OPENAI_COMPATIBLE_BASE_URL')
        self.openai_compatible_api_key = os.getenv('OPENAI_COMPATIBLE_API_KEY')
        self.openai_compatible_model = os.getenv('OPENAI_COMPATIBLE_MODEL')

    def get_llm(self) -> LLM:
        """
        Create and configure LLM instance for CrewAI
        Priority: OpenAI Compatible > Gemini
        """
        try:
            # Try OpenAI Compatible API first
            if self.openai_compatible_base_url and self.openai_compatible_api_key:
                logger.info(f"Using OpenAI Compatible API: {self.openai_compatible_model or 'default'}")
                return LLM(
                    model=self.openai_compatible_model or "gpt-3.5-turbo",
                    base_url=self.openai_compatible_base_url,
                    api_key=self.openai_compatible_api_key,
                    temperature=self.temperature
                )

            # Fallback to Gemini
            if self.api_key:
                logger.info(f"Using Gemini model: {self.model_name}")
                return LLM(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=self.temperature
                )

            raise ValueError("No LLM configuration found. Please set GEMINI_API_KEY or OPENAI_COMPATIBLE_API_KEY")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        config = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        if self.openai_compatible_base_url:
            config.update({
                "model": self.openai_compatible_model or "gpt-3.5-turbo",
                "base_url": self.openai_compatible_base_url,
                "api_key": self.openai_compatible_api_key
            })
        else:
            config.update({
                "model": self.model_name,
                "api_key": self.api_key
            })

        return config


def get_default_llm_config() -> LLMConfig:
    """Get default LLM configuration"""
    return LLMConfig()


def create_crewai_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> LLM:
    """
    Create CrewAI LLM instance with custom parameters

    Args:
        model: Model name (overrides environment config)
        temperature: Temperature setting
        api_key: API key (overrides environment config)
        base_url: Base URL for OpenAI compatible APIs

    Returns:
        Configured LLM instance
    """
    config = LLMConfig()

    if model:
        config.model_name = model
    if temperature is not None:
        config.temperature = temperature
    if api_key:
        config.api_key = api_key
    if base_url:
        config.openai_compatible_base_url = base_url

    return config.get_llm()


# Compatibility with existing system
def get_chat_completion(messages, model=None, temperature=None, **kwargs):
    """
    Compatibility function for existing chat completion calls
    This will be gradually replaced by direct CrewAI LLM usage
    """
    try:
        llm = create_crewai_llm(model=model, temperature=temperature)
        response = llm.call(messages=messages, **kwargs)
        return response
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        # Fallback to original implementation
        from src.tools.openrouter_config import get_chat_completion as original_completion
        return original_completion(messages, model=model, temperature=temperature, **kwargs)