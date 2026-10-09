"""LLM configuration management for pgcli.

Handles loading and managing configuration for LLM providers,
API keys, and behavior settings from environment variables.
"""
import logging
import os
from typing import Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-6-luna",
    "ollama": "llama3.2",
}

# Unified pydantic-ai thinking level: False (off), True (provider default) or an effort level.
ThinkingLevel = Union[bool, str]
THINKING_LEVELS = ("minimal", "low", "medium", "high", "xhigh")
DEFAULT_THINKING = "low"

# Message history sent with each request is trimmed to this many tokens, even when the
# model's context window is larger: resending ~1M tokens on every question is slow and
# costly. Override with PGCLI_LLM_MAX_CONTEXT_TOKENS.
DEFAULT_HISTORY_TOKEN_BUDGET = 180000

# Context windows used when the model is unknown to genai-prices.
FALLBACK_CONTEXT_WINDOWS = {
    "anthropic": 200000,
    "openai": 128000,
    "ollama": 2000,  # Ollama's default num_ctx
}


def parse_thinking(value: Optional[str]) -> Optional[ThinkingLevel]:
    """Parse PGCLI_LLM_THINKING into a pydantic-ai ``thinking`` setting.

    Returns None to leave thinking at the model's own default.
    """
    if value is None or value.strip() == "":
        return DEFAULT_THINKING
    value = value.strip().lower()
    if value in ("off", "false", "no", "0", "none"):
        return False
    if value in ("on", "true", "yes", "1"):
        return True
    if value == "default":
        return None
    if value in THINKING_LEVELS:
        return value
    logger.warning("Ignoring invalid PGCLI_LLM_THINKING=%r, using %r", value, DEFAULT_THINKING)
    return DEFAULT_THINKING


def lookup_context_window(provider: str, model: str) -> Optional[int]:
    """Return the model's context window from genai-prices, or None when unknown."""
    try:
        from genai_prices import data_snapshot

        _, model_info = data_snapshot.get_snapshot().find_provider_model(model, None, provider, None)
    except (ImportError, LookupError):
        return None
    return model_info.context_window


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
        thinking: Optional[ThinkingLevel] = None,
    ):
        """Initialize LLM configuration.

        :param provider: LLM provider (anthropic, openai, ollama)
        :param model: Model name (e.g., claude-sonnet-5-5, gpt-6-luna)
        :param api_key: API key (provider-specific)
        :param max_retries: Maximum retries for tool calls
        :param log_dir: Directory for LLM interaction logs
        :param max_context_tokens: Maximum context tokens (defaults to 90% of model limit)
        :param thinking: Thinking level: False, True or one of THINKING_LEVELS
                         (defaults to PGCLI_LLM_THINKING, else DEFAULT_THINKING)
        """
        # Provider selection
        self.provider = provider or os.getenv("PGCLI_LLM_PROVIDER", "anthropic")

        # Model selection (with provider-specific defaults)
        if model:
            self.model = model
        else:
            self.model = os.getenv("PGCLI_LLM_MODEL", DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS["anthropic"]))

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
        if max_context_tokens is None and os.getenv("PGCLI_LLM_MAX_CONTEXT_TOKENS"):
            try:
                max_context_tokens = int(os.environ["PGCLI_LLM_MAX_CONTEXT_TOKENS"])
            except ValueError:
                logger.warning("Ignoring invalid PGCLI_LLM_MAX_CONTEXT_TOKENS=%r", os.environ["PGCLI_LLM_MAX_CONTEXT_TOKENS"])
        self.max_context_tokens = max_context_tokens
        self.thinking = thinking if thinking is not None else parse_thinking(os.getenv("PGCLI_LLM_THINKING"))

    def get_model_string(self) -> str:
        """Get full model string for Pydantic AI.

        :return: Model string (e.g., 'anthropic:claude-sonnet-5-5')
        """
        return f"{self.provider}:{self.model}"

    def get_context_limit(self) -> int:
        """Get the token budget for message history sent with each request.

        90% of the model's context window (leaving room for the response), capped at
        DEFAULT_HISTORY_TOKEN_BUDGET unless max_context_tokens is set explicitly.

        :return: Maximum context tokens to use
        """
        if self.max_context_tokens:
            return self.max_context_tokens

        window = lookup_context_window(self.provider, self.model)
        if window is None:
            window = FALLBACK_CONTEXT_WINDOWS.get(self.provider, 100000)
        return min(int(window * 0.9), DEFAULT_HISTORY_TOKEN_BUDGET)

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
