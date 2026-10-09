"""Data models for Pydantic AI LLM integration."""
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class PgcliDeps:
    """Dependencies injected into agent tools via RunContext.

    This dataclass holds the pgcli instance and any other dependencies
    that tools need to access the database.
    """
    pgcli: Any  # Using Any to avoid circular import with pgcli.main


class SQLQuery(BaseModel):
    """The final answer the LLM submits through the ``submit_sql`` output tool."""

    sql: str = Field(
        description=(
            "The PostgreSQL query (or \\copy command) to execute, exactly as it should be run. "
            "Only SQL: explanations go in SQL comments (-- ...). No markdown fences, no prose."
        )
    )
