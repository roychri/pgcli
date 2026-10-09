"""LLM configuration management for pgcli.

Handles loading and managing configuration for LLM providers,
API keys, and behavior settings from environment variables.
"""
import os
from typing import Optional


class LLMConfig:
    """LLM configuration loaded from environment variables.

    Supports multiple providers (Anthropic, OpenAI, Ollama) with
    configurable models and behavior settings.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        log_dir: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
    ):
        """Initialize LLM configuration.

        :param provider: LLM provider (anthropic, openai, ollama)
        :param model: Model name (e.g., claude-sonnet-4-0, gpt-4o)
        :param api_key: API key (provider-specific)
        :param max_retries: Maximum retries for tool calls
        :param log_dir: Directory for LLM interaction logs
        :param max_context_tokens: Maximum context tokens (defaults to 90% of model limit)
        """
        # Provider selection
        self.provider = provider or os.getenv("PGCLI_LLM_PROVIDER", "anthropic")

        # Model selection (with provider-specific defaults)
        if model:
            self.model = model
        elif self.provider == "anthropic":
            self.model = os.getenv("PGCLI_LLM_MODEL", "claude-sonnet-4-5-20250929")
        elif self.provider == "openai":
            self.model = os.getenv("PGCLI_LLM_MODEL", "gpt-4o-mini")
        elif self.provider == "ollama":
            self.model = os.getenv("PGCLI_LLM_MODEL", "llama3.2")
        else:
            self.model = os.getenv("PGCLI_LLM_MODEL", "claude-sonnet-4-0")

        # API key (provider-specific env vars)
        if api_key:
            self.api_key = api_key
        elif self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            # Ollama doesn't need API key, but check generic var
            self.api_key = os.getenv("PGCLI_LLM_API_KEY")

        # Behavior settings
        self.max_retries = max_retries
        self.log_dir = log_dir
        self.max_context_tokens = max_context_tokens

    def get_model_string(self) -> str:
        """Get full model string for Pydantic AI.

        :return: Model string (e.g., 'anthropic:claude-sonnet-4-0')
        """
        return f"{self.provider}:{self.model}"

    def get_context_limit(self) -> int:
        """Get context limit for the current provider/model.

        Returns 90% of the model's context window to allow for response.

        :return: Maximum context tokens to use
        """
        if self.max_context_tokens:
            return self.max_context_tokens

        # Model-specific context windows (90% of max)
        # Based on research findings
        if self.provider == "anthropic":
            if "claude-sonnet-4-5" in self.model or "claude-4-5" in self.model:
                return 180000  # 90% of 200K
            elif "claude-sonnet-4" in self.model or "claude-4" in self.model:
                return 180000  # 90% of 200K
            else:
                return 180000  # Default for Anthropic

        elif self.provider == "openai":
            if "gpt-5" in self.model.lower():
                return 360000  # 90% of 400K
            elif "gpt-4o" in self.model.lower():
                return 115000  # 90% of 128K
            elif "gpt-4" in self.model.lower():
                return 115000  # 90% of 128K
            else:
                return 115000  # Default for OpenAI

        elif self.provider == "ollama":
            # Ollama default context is usually 2K, but can be configured
            # Use conservative estimate
            return 1800  # 90% of 2K

        else:
            # Generic fallback
            return 100000  # Conservative default

    def is_configured(self) -> bool:
        """Check if LLM is properly configured.

        :return: True if provider has required API key or doesn't need one
        """
        # Ollama doesn't require API key
        if self.provider == "ollama":
            return True

        # Other providers need API keys
        return self.api_key is not None

    def get_missing_config(self) -> Optional[str]:
        """Get description of missing configuration.

        :return: Error message if config incomplete, None if OK
        """
        if not self.is_configured():
            if self.provider == "anthropic":
                return "ANTHROPIC_API_KEY environment variable not set"
            elif self.provider == "openai":
                return "OPENAI_API_KEY environment variable not set"
            else:
                return f"API key not configured for provider: {self.provider}"
        return None


def load_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> LLMConfig:
    """Load LLM configuration from environment variables.

    :param provider: Override provider (defaults to env var or 'anthropic')
    :param model: Override model (defaults to env var or provider default)
    :param log_dir: Log directory (defaults to env var or None)
    :return: LLMConfig instance
    """
    return LLMConfig(
        provider=provider,
        model=model,
        log_dir=log_dir or os.getenv("PGCLI_LLM_LOG_DIR"),
    )
