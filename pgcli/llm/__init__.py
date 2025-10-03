"""LLM integration for pgcli using Pydantic AI.

This module provides SQL query generation from natural language using
multiple LLM providers (Anthropic, OpenAI, Ollama, etc.).

Public API:
    generate_sql(pgcli, user_input, message_history, config) -> (query, history)
"""

from .agent import generate_sql, create_agent, llm_append_output, add_sql_result_to_history
from .config import load_config, LLMConfig
from .models import PgcliDeps

__all__ = [
    "generate_sql",
    "create_agent",
    "llm_append_output",
    "add_sql_result_to_history",
    "load_config",
    "LLMConfig",
    "PgcliDeps",
]
