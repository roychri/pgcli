"""Token counting utilities for different LLM providers.

Provides token counting functionality with provider-specific implementations.
Falls back to character-based estimation when specific libraries unavailable.
"""
from abc import ABC, abstractmethod
from typing import Optional


class TokenCounter(ABC):
    """Abstract base class for token counting."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        :param text: Text to count tokens for
        :return: Estimated token count
        """
        pass


class CharacterBasedTokenCounter(TokenCounter):
    """Fallback token counter using character estimation.

    Uses the rule of thumb: 1 token ≈ 4 characters.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens using character-based estimation.

        :param text: Text to count tokens for
        :return: Estimated token count (chars / 4)
        """
        return len(text) // 4


class OpenAITokenCounter(TokenCounter):
    """Token counter for OpenAI models using tiktoken library."""

    def __init__(self, model: str = "gpt-4o"):
        """Initialize OpenAI token counter.

        :param model: Model name for encoding selection
        """
        self.model = model
        self._encoding = None
        self._init_encoding()

    def _init_encoding(self):
        """Initialize tiktoken encoding with graceful fallback."""
        try:
            import tiktoken

            try:
                # Try to get encoding for specific model
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to o200k_base (used by gpt-4o and later), for models tiktoken doesn't know yet
                self._encoding = tiktoken.get_encoding("o200k_base")
        except ImportError:
            # tiktoken not installed, will use character fallback
            self._encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken or character fallback.

        :param text: Text to count tokens for
        :return: Token count
        """
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        else:
            # Fallback to character estimation
            return len(text) // 4


class AnthropicTokenCounter(TokenCounter):
    """Token counter for Anthropic models.

    Uses character-based estimation as Anthropic's token counting
    requires API calls which are too slow for real-time use.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens using character-based estimation.

        :param text: Text to count tokens for
        :return: Estimated token count (chars / 4)
        """
        return len(text) // 4


class OllamaTokenCounter(TokenCounter):
    """Token counter for Ollama models.

    Uses character-based estimation.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens using character-based estimation.

        :param text: Text to count tokens for
        :return: Estimated token count (chars / 4)
        """
        return len(text) // 4


def get_token_counter(provider: str, model: Optional[str] = None) -> TokenCounter:
    """Get appropriate token counter for the given provider.

    :param provider: LLM provider (anthropic, openai, ollama)
    :param model: Optional model name (used for OpenAI)
    :return: TokenCounter instance
    """
    if provider == "openai":
        return OpenAITokenCounter(model=model or "gpt-4o")
    elif provider == "anthropic":
        return AnthropicTokenCounter()
    elif provider == "ollama":
        return OllamaTokenCounter()
    else:
        # Default fallback
        return CharacterBasedTokenCounter()
