# Pydantic AI Quick Reference for pgcli

**One-page reference for implementing LLM features with Pydantic AI**

---

## Installation

```bash
# Slim installation (recommended)
pip install "pydantic-ai-slim[anthropic,openai]"

# With observability
pip install "pydantic-ai-slim[anthropic,logfire]"

# Full installation
pip install pydantic-ai
```

**Requirements:** Python 3.10+

---

## Basic Agent Setup

```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from dataclasses import dataclass

# 1. Define dependencies
@dataclass
class Deps:
    pgcli: Any

# 2. Define output model
class SQLQuery(BaseModel):
    query: str
    explanation: str

# 3. Create agent
agent = Agent(
    'anthropic:claude-sonnet-4-0',
    output_type=SQLQuery,
    deps_type=Deps,
    retries=2
)

# 4. Define tools
@agent.tool
def get_tables(ctx: RunContext[Deps]) -> str:
    """List all tables."""
    return ctx.deps.pgcli.execute_query(...)

# 5. Run agent
result = agent.run_sync(
    'show me users',
    deps=Deps(pgcli=my_pgcli)
)
print(result.output.query)
```

---

## Tool Definition Patterns

### Simple Tool (No Parameters)
```python
@agent.tool
def get_tables(ctx: RunContext[Deps]) -> str:
    """Get all table names."""
    query = "SELECT tablename FROM pg_tables WHERE ..."
    return ctx.deps.pgcli.execute(query)
```

### Tool with Parameters
```python
from pydantic import Field

@agent.tool
def describe_table(
    ctx: RunContext[Deps],
    table_name: str = Field(..., min_length=1, max_length=63),
    schema: str = Field(default='public', pattern=r'^[a-zA-Z_]\w*$')
) -> str:
    """Get columns for a table."""
    # Pydantic validates parameters automatically
    query = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    """
    return ctx.deps.pgcli.execute_with_params(query, (schema, table_name))
```

### Tool Without Context
```python
@agent.tool_plain
def get_current_date() -> str:
    """Get today's date."""
    return str(date.today())
```

---

## Output Validation

```python
from pydantic_ai import ModelRetry

@agent.output_validator
async def validate_sql(
    ctx: RunContext[Deps],
    output: SQLQuery
) -> SQLQuery:
    """Validate SQL before returning."""

    # Check it's a SELECT query
    if not output.query.upper().startswith('SELECT'):
        raise ModelRetry('Only SELECT queries allowed')

    # Validate with EXPLAIN
    try:
        ctx.deps.pgcli.execute(f'EXPLAIN {output.query}')
    except Exception as e:
        raise ModelRetry(f'Invalid SQL: {e}')

    return output
```

---

## Provider Configuration

### Anthropic (Recommended)
```python
# Method 1: String
agent = Agent('anthropic:claude-sonnet-4-0')

# Method 2: Environment variable
# export ANTHROPIC_API_KEY=sk-ant-...
agent = Agent('anthropic:claude-sonnet-4-0')

# Method 3: Explicit
from pydantic_ai.models.anthropic import AnthropicModel
model = AnthropicModel('claude-sonnet-4-0', api_key='sk-ant-...')
agent = Agent(model)
```

### OpenAI
```python
agent = Agent('openai:gpt-4o')
# Requires: export OPENAI_API_KEY=sk-...
```

### Ollama (Local)
```python
agent = Agent('ollama:llama3.2')
# Requires Ollama running on http://localhost:11434
```

### Fallback Model
```python
from pydantic_ai.models import FallbackModel

agent = Agent(
    FallbackModel(
        'anthropic:claude-sonnet-4-0',  # Try first
        'openai:gpt-4o'                 # Fallback
    )
)
```

---

## Message History (Multi-turn)

```python
# First query
result1 = agent.run_sync('show me users', deps=deps)

# Follow-up with history
result2 = agent.run_sync(
    'filter by active users',
    deps=deps,
    message_history=result1.new_messages()
)

# Access all messages
all_msgs = result2.all_messages()

# Serialize for persistence
import json
json.dumps([msg.model_dump() for msg in all_msgs])
```

---

## Testing with Mocks

### TestModel (Simple)
```python
from pydantic_ai.models.test import TestModel

def test_agent():
    test_model = TestModel()

    with agent.override(model=test_model):
        result = agent.run_sync('test', deps=deps)

    # TestModel generates random valid SQLQuery
    assert isinstance(result.output, SQLQuery)
```

### FunctionModel (Custom Logic)
```python
from pydantic_ai.models.function import FunctionModel
import json

def custom_response(messages, tools):
    user_msg = messages[-1]['content']

    if 'show tables' in user_msg:
        # Simulate tool call
        return json.dumps({
            'tool_calls': [{
                'id': 'call_1',
                'name': 'get_tables',
                'arguments': '{}'
            }]
        })

    # Return SQL
    return json.dumps({
        'query': 'SELECT * FROM users',
        'explanation': 'Get all users'
    })

func_model = FunctionModel(custom_response)

with agent.override(model=func_model):
    result = agent.run_sync('show tables', deps=deps)
```

---

## Configuration with pydantic-settings

```python
from pydantic_settings import BaseSettings

class LLMConfig(BaseSettings):
    provider: str = 'anthropic'
    model: str = 'claude-sonnet-4-0'
    anthropic_api_key: str | None = None
    max_retries: int = 2

    class Config:
        env_file = '.env'
        env_prefix = 'PGCLI_LLM_'

# Usage
config = LLMConfig()
agent = Agent(f'{config.provider}:{config.model}')
```

**.env file:**
```bash
PGCLI_LLM_PROVIDER=anthropic
PGCLI_LLM_MODEL=claude-sonnet-4-0
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Structured Output Types

### Union Type (Success/Error)
```python
from pydantic import BaseModel

class Success(BaseModel):
    query: str
    explanation: str

class Error(BaseModel):
    error_message: str
    suggestion: str | None = None

Response = Success | Error

agent = Agent('anthropic:...', output_type=Response)

result = agent.run_sync('query', deps=deps)

if isinstance(result.output, Success):
    print(result.output.query)
else:
    print(result.output.error_message)
```

---

## Common Patterns

### SQL Injection Protection
```python
# ❌ WRONG
query = f"SELECT * FROM {table_name}"

# ✅ RIGHT
from pydantic import Field

@agent.tool
def describe_table(
    ctx: RunContext[Deps],
    table_name: str = Field(..., pattern=r'^[a-zA-Z_]\w*$')
) -> str:
    # Validated by Pydantic
    query = "SELECT * FROM information_schema.columns WHERE table_name = %s"
    return ctx.deps.pgcli.execute_with_params(query, (table_name,))
```

### Caching Tool Results
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def _get_tables_cached(conn_string: str):
    # Expensive query
    return result

@agent.tool
def get_tables(ctx: RunContext[Deps]) -> str:
    """Get all tables (cached)."""
    return _get_tables_cached(ctx.deps.pgcli.connection_string)
```

### Error Recovery
```python
@agent.tool
def describe_table(ctx: RunContext[Deps], table_name: str) -> str:
    """Describe a table."""
    try:
        return ctx.deps.pgcli.execute(f"SELECT * FROM {table_name}")
    except Exception as e:
        # Return error message, LLM will retry
        return f"Error: {e}. Try a different table."
```

---

## Observability

```python
import logfire

# Enable Logfire
logfire.configure()
logfire.instrument_pydantic_ai()

# Automatic logging of:
# - Agent runs
# - Tool calls
# - Token usage
# - Costs
# - Errors
```

---

## Result Object Properties

```python
result = agent.run_sync('query', deps=deps)

# Output
result.output              # Your output_type (SQLQuery)

# Messages
result.all_messages()      # All messages in conversation
result.new_messages()      # Just from this run

# Usage & Cost
result.usage()             # Token counts
result.cost()              # Estimated cost

# Timestamp
result.timestamp()         # When response completed

# Data
result.data                # Output as Pydantic model
```

---

## Common Gotchas

1. **Type hints required** - Tools need full type annotations
2. **ModelRetry counts as retry** - Set retries higher if using validators
3. **Message history grows** - Manually trim if needed
4. **Tool names = function names** - Use valid Python identifiers
5. **Dependencies must match deps_type** - Type checking enforced

---

## Cheat Sheet

| Task | Code |
|------|------|
| Create agent | `Agent('anthropic:claude-sonnet-4-0')` |
| Define tool | `@agent.tool` decorator |
| Tool without context | `@agent.tool_plain` decorator |
| Access pgcli | `ctx.deps.pgcli` |
| Validate output | `@agent.output_validator` decorator |
| Retry on error | `raise ModelRetry('message')` |
| Run agent | `agent.run_sync('query', deps=deps)` |
| Multi-turn | Pass `message_history=result.new_messages()` |
| Test with mock | `with agent.override(model=TestModel())` |
| Fallback model | `FallbackModel('primary', 'fallback')` |

---

## Quick Links

- **Docs:** https://ai.pydantic.dev
- **GitHub:** https://github.com/pydantic/pydantic-ai
- **SQL Example:** https://ai.pydantic.dev/examples/sql-gen/
- **Full Guide:** PYDANTIC_AI_IMPLEMENTATION_GUIDE.md

---

**Version:** 1.0 | **Date:** 2025-10-01
