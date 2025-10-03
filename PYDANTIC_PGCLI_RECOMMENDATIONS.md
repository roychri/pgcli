# Pydantic for pgcli LLM Integration - Specific Recommendations

**Date:** 2025-10-01 (Updated with Pydantic AI recommendation)
**Context:** pgcli LLM migration from OpenAI-only to multi-provider architecture
**Related Docs:**
- PYDANTIC_AI_EXECUTIVE_SUMMARY.md **(NEW - Start here!)**
- PYDANTIC_AI_IMPLEMENTATION_GUIDE.md **(NEW - Complete guide)**
- PYDANTIC_LLM_RESEARCH.md (full Pydantic analysis)
- LLM_IMPLEMENTATION_FINDINGS.md (current implementation)

---

## 📌 UPDATE: Recommendation Changed to Pydantic AI

After clarification, **we're using Pydantic AI framework** (not just Pydantic validation library).

**See comprehensive Pydantic AI documentation:**
- PYDANTIC_AI_EXECUTIVE_SUMMARY.md - Decision overview
- PYDANTIC_AI_IMPLEMENTATION_GUIDE.md - Complete implementation guide
- PYDANTIC_AI_VS_ALTERNATIVES.md - Scores 8.7/10 vs alternatives

**This document contains the original analysis of three approaches.** The approved approach is now Pydantic AI (similar to "Option 1" but with updated recommendation).

---

## Critical Clarification: What Pydantic IS and ISN'T

### ❌ Pydantic is NOT:
- **NOT** a replacement for the Pydantic AI framework (different library!)
- **NOT** an LLM client library
- **NOT** a multi-provider abstraction layer
- **NOT** a streaming/async framework
- **NOT** a function calling orchestrator

### ✅ Pydantic IS:
- A **data validation** library
- A **JSON schema generator**
- A **type-safe model** system
- A **configuration management** tool (with pydantic-settings)

---

## The Confusion: Pydantic vs Pydantic AI

There are **TWO different libraries** that are often confused:

### 1. **Pydantic** (https://docs.pydantic.dev)
- Pure validation/schema library
- What this research document is about
- Version 2.x is current
- Does NOT do LLM calls

### 2. **Pydantic AI** (https://ai.pydantic.dev)
- Built ON TOP of Pydantic
- Provides LLM abstraction, Agents, Tools
- Multi-provider support (OpenAI, Anthropic, Gemini, etc.)
- Streaming, function calling, etc.
- **This might be what you're looking for!**

---

## Recommendation for pgcli

Based on your needs in LLM_IMPLEMENTATION_FINDINGS.md and ARCHITECTURE_DECISIONS.md, here's what you should use:

### Option 1: Pydantic AI Framework (RECOMMENDED)
**Use if you want**: Complete LLM abstraction with minimal custom code

```python
# This is Pydantic AI (separate library!)
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

class TableInfo(BaseModel):
    table_name: str

# Define an agent with tools
agent = Agent(
    'anthropic:claude-sonnet-4-20250514',  # Your default model
    system_prompt='You are a SQL query assistant...',
)

@agent.tool
async def get_tables(ctx: RunContext[PGCli]) -> list[str]:
    """List all tables in the database."""
    return ctx.deps.get_all_tables()

# Use the agent
result = await agent.run('Show me all users', deps=pgcli_instance)
```

**Pros:**
- ✅ Multi-provider support built-in (OpenAI, Anthropic, Ollama, etc.)
- ✅ Function calling handled automatically
- ✅ Streaming support
- ✅ Type-safe tools with Pydantic validation
- ✅ Session management
- ✅ Async/await support
- ✅ Well-documented

**Cons:**
- ❌ Another dependency (pydantic-ai package)
- ❌ Different from your current implementation
- ❌ Learning curve for the framework

### Option 2: Pure Pydantic + Provider SDKs (CURRENT APPROACH)
**Use if you want**: Maximum control with manual provider abstraction

```python
# Using Pydantic for validation only
from pydantic import BaseModel, Field
from anthropic import Anthropic
from openai import OpenAI

# Define function schemas with Pydantic
class GetTablesArgs(BaseModel):
    """Get all tables in the database."""
    pass  # No arguments

class DescribeTableArgs(BaseModel):
    """Describe a specific table."""
    table_name: str = Field(description="Name of table to describe")

# Build your own provider abstraction
class LLMProvider:
    def generate(self, messages, tools): ...
    def parse_tool_calls(self, response): ...

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)

    def generate(self, messages, tools):
        # Convert Pydantic schemas to Anthropic format
        anthropic_tools = [
            {
                "name": "get_tables",
                "description": GetTablesArgs.__doc__,
                "input_schema": GetTablesArgs.model_json_schema()
            }
        ]
        return self.client.messages.create(...)

# Similar for OpenAI, Ollama, etc.
```

**Pros:**
- ✅ Maximum control over implementation
- ✅ Minimal dependencies
- ✅ Pydantic provides excellent validation
- ✅ Can customize everything

**Cons:**
- ❌ Must build all provider abstraction yourself
- ❌ Must handle streaming yourself
- ❌ Must manage function calling loop yourself
- ❌ More code to maintain
- ❌ More testing required

### Option 3: Hybrid Approach (BEST OF BOTH)
**Use if you want**: Pydantic validation + lightweight provider abstraction

```python
# Use Pydantic for schemas/validation
from pydantic import BaseModel
from litellm import completion  # Unified LLM interface

class SQLFunction(BaseModel):
    """Base class for SQL assistant functions."""
    pass

class GetTables(SQLFunction):
    """List all tables."""
    pass

# Use LiteLLM for provider abstraction (lightweight!)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_tables",
            "description": GetTables.__doc__,
            "parameters": GetTables.model_json_schema()
        }
    }
]

# Works with any provider
response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=messages,
    tools=tools
)
```

**Pros:**
- ✅ Pydantic for validation (what you wanted)
- ✅ LiteLLM provides multi-provider support
- ✅ Less code than Option 2
- ✅ More control than Option 1
- ✅ Lightweight (LiteLLM is just a wrapper)

**Cons:**
- ❌ Two dependencies (pydantic + litellm)
- ❌ LiteLLM is another abstraction layer

---

## My Recommendation for pgcli

Based on your architecture decisions and current implementation:

### 🎯 Use **Option 3: Hybrid Approach**

**Why:**
1. You want Pydantic (for validation and schemas) ✅
2. You need multi-provider support (Anthropic default, but also OpenAI, Ollama) ✅
3. You want to keep control of the implementation ✅
4. You don't want to rebuild everything from scratch ✅

**Implementation Plan:**

```python
# pgcli/llm/schemas.py
from pydantic import BaseModel, Field

class DatabaseFunction(BaseModel):
    """Base class for database introspection functions."""
    class Config:
        json_schema_extra = {"strict": True}

class GetTablesParams(DatabaseFunction):
    """List all tables in the current database."""
    pass

class DescribeTableParams(DatabaseFunction):
    """Get detailed schema information for a table."""
    table_name: str = Field(description="Name of the table to describe")
    include_indexes: bool = Field(default=False, description="Include index information")

# pgcli/llm/providers.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list, tools: list[BaseModel]) -> dict:
        """Generate a response with function calling support."""
        pass

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def generate(self, messages, tools):
        # Convert Pydantic models to Anthropic tool format
        anthropic_tools = [
            {
                "name": tool.__name__.lower(),
                "description": tool.__doc__,
                "input_schema": tool.model_json_schema()
            }
            for tool in tools
        ]

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            tools=anthropic_tools,
            max_tokens=2000
        )
        return response

# Or use LiteLLM
class UnifiedProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        from litellm import completion
        self.model = model
        self.completion = completion

    def generate(self, messages, tools):
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.__name__.lower(),
                    "description": tool.__doc__,
                    "parameters": tool.model_json_schema()
                }
            }
            for tool in tools
        ]

        return self.completion(
            model=self.model,
            messages=messages,
            tools=tool_schemas
        )

# pgcli/llm/agent.py
from pydantic import ValidationError
from typing import Type

class SQLAgent:
    def __init__(self, provider: LLMProvider, pgcli):
        self.provider = provider
        self.pgcli = pgcli
        self.functions = self._register_functions()

    def _register_functions(self):
        return {
            "get_tables": (GetTablesParams, self.pgcli.get_tables),
            "describe_table": (DescribeTableParams, self.pgcli.describe_table),
            # ... other functions
        }

    def execute_function(self, name: str, args_json: str):
        if name not in self.functions:
            raise ValueError(f"Unknown function: {name}")

        params_class, func = self.functions[name]

        try:
            # Validate with Pydantic
            params = params_class.model_validate_json(args_json)
            # Execute function
            return func(**params.model_dump())
        except ValidationError as e:
            return f"Invalid arguments: {e}"

    async def run(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]

        for _ in range(10):  # Max iterations
            response = self.provider.generate(
                messages,
                tools=list(self.functions.keys())
            )

            if response.has_tool_calls():
                for tool_call in response.tool_calls:
                    result = self.execute_function(
                        tool_call.name,
                        tool_call.arguments
                    )
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
            else:
                return response.content
```

### Configuration

```python
# pgcli/llm/config.py
from pydantic_settings import BaseSettings

class LLMConfig(BaseSettings):
    """LLM configuration using pydantic-settings."""

    # Provider settings
    provider: str = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Model settings
    default_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 2000

    # Safety settings
    allow_destructive: bool = False
    confirm_before_execute: bool = True

    # Performance settings
    max_function_calls: int = 10
    cache_schema: bool = True

    class Config:
        env_prefix = "PGCLI_LLM_"
        env_file = ".env"

# Usage
config = LLMConfig()  # Auto-loads from .env and env vars
```

---

## Migration Path from Current Implementation

### Phase 1: Add Pydantic Validation (Week 1)
1. Install: `pip install pydantic pydantic-settings`
2. Create Pydantic models for all 7 function parameters
3. Replace manual validation with Pydantic validation
4. Add configuration management with pydantic-settings
5. Keep OpenAI provider as-is

### Phase 2: Add Provider Abstraction (Week 2)
1. Install: `pip install litellm` OR `pip install anthropic`
2. Create provider abstraction layer
3. Implement Anthropic provider (your default)
4. Add provider selection in config
5. Test with both providers

### Phase 3: Improve Safety (Week 3)
1. Add destructive query detection (Pydantic validator)
2. Implement confirmation workflow
3. Add query preview/dry-run mode
4. Improve error messages
5. Add user-friendly validation errors

### Phase 4: Add Advanced Features (Week 4+)
1. Implement caching with Pydantic
2. Add streaming support
3. Improve context management
4. Add metrics/logging
5. Write tests

---

## Key Takeaways

1. **Pydantic** (the validation library) is PERFECT for:
   - Validating function call arguments ⭐⭐⭐⭐⭐
   - Generating JSON schemas for tools ⭐⭐⭐⭐⭐
   - Configuration management ⭐⭐⭐⭐⭐
   - Type-safe models ⭐⭐⭐⭐⭐

2. **Pydantic does NOT provide**:
   - Multi-provider LLM support (use LiteLLM or provider SDKs)
   - Function calling orchestration (build yourself or use Pydantic AI)
   - Streaming support (use provider SDKs)

3. **Best approach for pgcli**:
   - Use **Pydantic** for validation and schemas
   - Use **LiteLLM** or provider SDKs for multi-provider support
   - Build lightweight abstraction layer yourself
   - This gives you control + power of Pydantic

4. **Alternative**: If you want a complete framework, consider **Pydantic AI** (different library!), but it's a bigger change from your current implementation.

---

## Quick Start

```bash
# Install dependencies
pip install pydantic pydantic-settings litellm

# Create schemas
# See pgcli/llm/schemas.py example above

# Create provider abstraction
# See pgcli/llm/providers.py example above

# Create agent
# See pgcli/llm/agent.py example above

# Use in main.py
from pgcli.llm.agent import SQLAgent
from pgcli.llm.providers import AnthropicProvider
from pgcli.llm.config import LLMConfig

config = LLMConfig()
provider = AnthropicProvider(config.anthropic_api_key)
agent = SQLAgent(provider, pgcli)

sql = await agent.run("Show me all active users")
```

---

## Resources

**Pydantic (Validation Library):**
- Docs: https://docs.pydantic.dev
- Settings: https://docs.pydantic-settings.helpmanual.io/

**Pydantic AI (LLM Framework - different library!):**
- Docs: https://ai.pydantic.dev
- GitHub: https://github.com/pydantic/pydantic-ai

**LiteLLM (Multi-Provider Wrapper):**
- Docs: https://docs.litellm.ai
- GitHub: https://github.com/BerriAI/litellm

**Provider SDKs:**
- Anthropic: https://github.com/anthropics/anthropic-sdk-python
- OpenAI: https://github.com/openai/openai-python
- Ollama: https://github.com/ollama/ollama-python

---

## Decision Required

**Which approach do you prefer?**

1. ⭐ **Option 3 (Hybrid)** - Pydantic + LiteLLM (my recommendation)
2. Option 1 (Framework) - Pydantic AI (complete framework, big change)
3. Option 2 (Manual) - Pure Pydantic + provider SDKs (most control, most work)

Let me know and I can provide detailed implementation guides for your choice!
