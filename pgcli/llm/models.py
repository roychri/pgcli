"""Data models for Pydantic AI LLM integration."""
from dataclasses import dataclass
from typing import Any


@dataclass
class PgcliDeps:
    """Dependencies injected into agent tools via RunContext.

    This dataclass holds the pgcli instance and any other dependencies
    that tools need to access the database.
    """
    pgcli: Any  # Using Any to avoid circular import with pgcli.main
