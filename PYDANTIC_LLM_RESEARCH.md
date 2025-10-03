# Pydantic LLM Integration Research

## Executive Summary

Pydantic v2 is a powerful data validation library with **excellent built-in support for LLM use cases**, particularly for:
- Validating and parsing LLM JSON outputs (including partial/streaming responses)
- Generating JSON schemas for function calling
- Type-safe model definitions
- Flexible serialization/deserialization

**Key Finding**: Pydantic is NOT a multi-provider LLM client library. It's a validation and schema generation library that works WITH LLM clients (OpenAI, Anthropic, etc.) but does NOT provide provider abstraction, API calls, or streaming capabilities itself.

---

## Documentation Map

### Core Concepts (Most Relevant to LLM Integration)

| Topic | Location | Purpose |
|-------|----------|---------|
| **JSON Parsing** | `concepts/json/` | Parse JSON with partial support for streaming LLM outputs |
| **JSON Schema** | `concepts/json_schema/` | Generate schemas for function calling (OpenAI, Anthropic tools) |
| **Models** | `concepts/models/` | Define structured data models |
| **Serialization** | `concepts/serialization/` | Convert models to/from JSON |
| **Validators** | `concepts/validators/` | Custom validation logic |
| **Fields** | `concepts/fields/` | Field definitions, constraints, metadata |
| **Type Adapter** | `concepts/type_adapter/` | Validate arbitrary types without BaseModel |
| **Strict Mode** | `concepts/strict_mode/` | Enforce exact type matching |
| **Configuration** | `concepts/config/` | Model-level settings |

### API Documentation

| Class/Module | Purpose | Key Methods |
|--------------|---------|-------------|
| `BaseModel` | Core model class | `model_validate()`, `model_validate_json()`, `model_dump()`, `model_dump_json()`, `model_json_schema()` |
| `TypeAdapter` | Validate non-model types | `validate_python()`, `validate_json()`, `dump_python()`, `dump_json()`, `json_schema()` |
| `Field()` | Field customization | Constraints, aliases, metadata, defaults |
| `pydantic_core.from_json()` | Low-level JSON parser | Partial JSON parsing with `allow_partial=True` |
| `ConfigDict` | Model configuration | `strict`, `extra`, `validate_default`, etc. |

### LLM-Specific Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Partial JSON Parsing** | Line 3126-3255 | Parse incomplete JSON from streaming LLMs |
| **JSON Schema Generation** | Line 3293+ | OpenAPI 3.1.0 & JSON Schema Draft 2020-12 compliant |
| **LLM Output Validation** | Line 3163-3189 | Specific guidance for validating LLM responses |
| **Experimental Partial Validation** | Line 1302-1607 | TypeAdapter-based partial validation |

---

## Cheatsheet: Common Patterns

### 1. Basic Model Definition
```python
from pydantic import BaseModel, Field

class User(BaseModel):
    id: int
    name: str = Field(description="User's full name")
    email: str
    age: int | None = None
```

### 2. Generate JSON Schema (for LLM function calling)
```python
# For OpenAI/Anthropic function/tool definitions
schema = User.model_json_schema()
# Returns JSON Schema Draft 2020-12 compliant dict

# With TypeAdapter for non-model types
from pydantic import TypeAdapter
adapter = TypeAdapter(list[User])
schema = adapter.json_schema()
```

### 3. Validate JSON Response
```python
# Complete JSON
user = User.model_validate_json('{"id": 1, "name": "John", "email": "john@example.com"}')

# With Python dict
data = {"id": 1, "name": "John", "email": "john@example.com"}
user = User.model_validate(data)
```

### 4. Partial JSON Parsing (Streaming LLM Responses)
```python
from pydantic_core import from_json
from pydantic import BaseModel

class Response(BaseModel):
    content: str = ""
    metadata: dict = {}

# Parse incomplete JSON from streaming
partial_json = '{"content": "Hello world", "metadata": {"source'
data = from_json(partial_json, allow_partial=True)
response = Response.model_validate(data)  # Works with partial data!
```

### 5. Handle Partial Validation with Defaults
```python
from typing import Annotated, Optional, Any
from pydantic import BaseModel, ValidationError, WrapValidator
import pydantic_core

def default_on_error(v, handler) -> Any:
    try:
        return handler(v)
    except ValidationError as exc:
        if all(e['type'] == 'missing' for e in exc.errors()):
            raise pydantic_core.PydanticUseDefault()
        raise

class SafeModel(BaseModel):
    required_field: str
    optional_field: Annotated[Optional[dict], WrapValidator(default_on_error)] = None
```

### 6. Serialize Models
```python
# To dict
data = user.model_dump()

# To JSON string
json_str = user.model_dump_json()

# With aliases
class APIModel(BaseModel):
    internal_id: int = Field(serialization_alias="id")

model.model_dump(by_alias=True)  # Uses "id" instead of "internal_id"
```

### 7. Strict Mode (Exact Type Matching)
```python
from pydantic import ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True)
    count: int  # Will reject "123" string, requires actual int

# Or per-field
class MixedModel(BaseModel):
    strict_field: int = Field(strict=True)
    flexible_field: int  # Normal coercion
```

### 8. Custom Validators
```python
from pydantic import field_validator, model_validator

class ValidatedModel(BaseModel):
    email: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()

    @model_validator(mode='after')
    def check_consistency(self) -> 'ValidatedModel':
        # Cross-field validation
        return self
```

### 9. TypeAdapter for Non-Models
```python
from pydantic import TypeAdapter

# Validate lists, dicts, unions without BaseModel
adapter = TypeAdapter(list[int])
result = adapter.validate_python([1, "2", 3.0])  # [1, 2, 3]

# Generate schema for primitive types
schema = TypeAdapter(dict[str, int]).json_schema()
```

### 10. Field Constraints & Metadata
```python
from pydantic import Field
from typing import Annotated

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Price in USD")
    tags: list[str] = Field(default_factory=list, max_length=10)

    # Using Annotated pattern
    quantity: Annotated[int, Field(ge=0, le=1000)]
```

---

## DOs and DONTs

### ✅ DOs

1. **DO use `model_validate_json()` instead of `model_validate(json.loads())`**
   - Direct JSON parsing is faster and more efficient
   - Built on Rust-based `jiter` parser

2. **DO use concrete types (`list`, `dict`) over abstract (`Sequence`, `Mapping`)**
   - Better performance
   - Pydantic handles conversion automatically

3. **DO use `TypeAdapter` for non-model validation**
   - Perfect for validating function arguments, API responses
   - No need to create BaseModel subclass for every type

4. **DO use partial JSON parsing for streaming LLM outputs**
   - `pydantic_core.from_json(json_str, allow_partial=True)`
   - Requires all fields to have defaults

5. **DO generate JSON schemas from models for function calling**
   - `model_json_schema()` creates OpenAPI 3.1.0 compliant schemas
   - Works directly with OpenAI/Anthropic tool/function definitions

6. **DO use `Field()` for rich metadata**
   - Descriptions appear in JSON schema
   - Constraints (gt, lt, min_length, etc.) are enforced and documented

7. **DO use validators for complex business logic**
   - `@field_validator` for single-field validation
   - `@model_validator(mode='after')` for cross-field validation

8. **DO use `ConfigDict` for model-wide settings**
   - `strict=True` for exact type matching
   - `extra='forbid'` to reject unknown fields
   - `validate_default=True` to validate default values

9. **DO use aliases for API compatibility**
   - `validation_alias` for input
   - `serialization_alias` for output
   - `alias` for both

10. **DO instantiate `TypeAdapter` once and reuse**
    - TypeAdapter creation has overhead
    - Cache and reuse for better performance

### ❌ DONTs

1. **DON'T expect Pydantic to make LLM API calls**
   - Pydantic is validation/schema only
   - You need a separate client (openai, anthropic, httpx, etc.)

2. **DON'T expect multi-provider LLM abstraction**
   - Pydantic doesn't know about OpenAI vs Anthropic vs others
   - You must handle provider-specific APIs yourself

3. **DON'T use Pydantic for streaming/async LLM communication**
   - Pydantic validates data, doesn't stream it
   - Use async HTTP clients for streaming
   - Validate chunks with partial JSON parsing

4. **DON'T use `dict(model)` for serialization**
   - Use `model_dump()` instead
   - `dict()` doesn't recursively convert nested models

5. **DON'T use `validate()` in tight loops without caching**
   - Validation has overhead
   - Cache TypeAdapter instances
   - Consider `model_construct()` if you trust the data

6. **DON'T mix field and type metadata in `Annotated`**
   ```python
   # WRONG: Field applied to int, not the union
   field: Annotated[int, Field(deprecated=True)] | None = None

   # RIGHT: Field applied to union
   field: Annotated[int | None, Field(deprecated=True)] = None
   ```

7. **DON'T use abstract types like `Sequence` unless necessary**
   - Performance penalty
   - Pydantic auto-converts anyway

8. **DON'T forget `by_alias=True` when serializing with aliases**
   - Default is `False`
   - Use `ConfigDict(serialize_by_alias=True)` for always-on

9. **DON'T use mutable defaults without `default_factory`**
   - Pydantic handles this better than stdlib, but still use factories
   - `Field(default_factory=list)` not `Field(default=[])`

10. **DON'T use wrap validators if you care about performance**
    - Slowest validator type
    - Use `before` or `after` validators when possible

11. **DON'T forget that partial JSON requires all fields to have defaults**
    - `allow_partial=True` skips incomplete fields
    - Fields without defaults will cause errors

12. **DON'T use Pydantic for LLM prompt management**
    - Not designed for prompt templates
    - Use LangChain, LlamaIndex, or custom solution

---

## Limitations & Constraints

### What Pydantic DOES NOT Provide

1. **No LLM Provider Clients**
   - No OpenAI client
   - No Anthropic client
   - No Azure/AWS/GCP clients
   - Must use provider SDKs separately

2. **No Streaming Infrastructure**
   - Can parse partial JSON chunks
   - But doesn't handle SSE/streaming protocols
   - Use httpx, aiohttp, or provider SDKs

3. **No Provider Abstraction**
   - No unified interface across providers
   - Each provider's API is different
   - You must handle differences manually

4. **No Prompt Management**
   - No template system
   - No prompt versioning
   - No prompt optimization

5. **No Token Counting**
   - No tokenization utilities
   - Use tiktoken, anthropic, etc.

6. **No Response Parsing Helpers**
   - No markdown extraction
   - No code block parsing
   - Just JSON validation

7. **No Retry/Rate Limiting**
   - No built-in retry logic
   - No rate limit handling
   - Use tenacity or provider SDKs

8. **No Cost Tracking**
   - No usage monitoring
   - No cost calculation
   - Build separately

### Pydantic-Specific Limitations

1. **Partial JSON Parsing Constraints**
   - TypeAdapter only (not BaseModel.model_validate_json yet)
   - All fields must have defaults
   - Last field errors are ignored
   - Some invalid-but-complete JSON accepted

2. **Performance Considerations**
   - Wrap validators are slow (use sparingly)
   - Abstract types (Sequence, Mapping) slower than concrete
   - TypeAdapter instantiation has overhead (cache it)
   - Validation is slower than `model_construct()` (use when data trusted)

3. **JSON Schema Limitations**
   - Draft 2020-12 only (some tools use older drafts)
   - No built-in OpenAPI parameter location (query, header, etc.)
   - Custom types may need manual schema customization

4. **Strict Mode Gotchas**
   - JSON parsing allows some coercion even in strict mode
   - Date strings → date objects (JSON has no date type)
   - Arrays → tuples (JSON has no tuple type)

5. **Experimental Features**
   - Partial validation is experimental (API may change)
   - Pipeline API is experimental
   - Check version compatibility

---

## Relevance Analysis: Pydantic for Multi-Provider LLM CLI

### ✅ What Pydantic EXCELS At (Use These!)

1. **JSON Schema Generation for Function Calling** ⭐⭐⭐⭐⭐
   - Perfect for OpenAI function calling
   - Perfect for Anthropic tool use
   - Auto-generates from Python types
   - **USE**: `BaseModel.model_json_schema()` for all function definitions

2. **Validating LLM JSON Outputs** ⭐⭐⭐⭐⭐
   - Validate structured responses
   - Partial parsing for streaming
   - Type coercion for flexibility
   - **USE**: `model_validate_json()` for all structured responses

3. **Type-Safe Function Definitions** ⭐⭐⭐⭐⭐
   - Define functions once in Python
   - Auto-generate schemas
   - Type hints for IDE support
   - **USE**: BaseModel for all tool/function definitions

4. **Response Parsing & Validation** ⭐⭐⭐⭐
   - Parse JSON tool calls
   - Validate arguments
   - Handle errors gracefully
   - **USE**: TypeAdapter for argument validation

5. **Configuration Validation** ⭐⭐⭐⭐
   - Validate CLI config files
   - Environment variable parsing (with pydantic-settings)
   - **USE**: BaseModel for app configuration

### ❌ What Pydantic CANNOT Do (Build Separately!)

1. **Provider Abstraction** ⭐
   - Must build: Unified provider interface
   - Must build: Provider registry/factory
   - Must build: Provider-specific adapters

2. **API Communication** ⭐
   - Must build: HTTP client wrapper
   - Must build: Streaming response handling
   - Must build: Authentication/headers

3. **Prompt Management** ⭐
   - Must build: Template system
   - Must build: Prompt storage
   - Must build: Variable interpolation

4. **Cost & Usage Tracking** ⭐
   - Must build: Token counting
   - Must build: Cost calculation
   - Must build: Usage analytics

5. **Retry & Error Handling** ⭐
   - Must build: Retry logic
   - Must build: Rate limiting
   - Must build: Fallback providers

### 📊 Architecture Recommendation for pgcli LLM

```
┌─────────────────────────────────────────┐
│         CLI Layer (Click/Typer)         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│     LLM Abstraction Layer (Custom)      │
│  - Provider registry                     │
│  - Unified interface                     │
│  - Streaming coordination                │
└─────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌───────────────┐      ┌────────────────┐
│   Providers   │      │    Pydantic    │
│  (Custom)     │      │   (Validation) │
│               │      │                │
│ - OpenAI SDK  │◄────►│ - Schemas      │
│ - Anthropic   │      │ - Validation   │
│ - Azure       │      │ - Models       │
└───────────────┘      └────────────────┘
```

**Pydantic's Role**:
1. Define function schemas (BaseModel)
2. Generate JSON schemas (model_json_schema)
3. Validate responses (model_validate_json)
4. Parse streaming chunks (pydantic_core.from_json)
5. Validate config (pydantic-settings)

**Must Build Separately**:
1. Provider abstraction (factory pattern)
2. HTTP client (httpx/aiohttp)
3. Streaming handler (SSE parsing)
4. Retry logic (tenacity)
5. Prompt templates (jinja2 or custom)
6. Cost tracking (custom)

---

## Code Examples for pgcli Integration

### 1. Define SQL Functions for LLM
```python
from pydantic import BaseModel, Field

class GetTableSchema(BaseModel):
    """Get the schema of a table in the database."""
    table_name: str = Field(description="Name of the table to describe")

class ExecuteQuery(BaseModel):
    """Execute a SQL query and return results."""
    query: str = Field(description="SQL query to execute")
    explain: bool = Field(default=False, description="Return EXPLAIN plan instead of results")

class ListTables(BaseModel):
    """List all tables in the current database."""
    schema: str | None = Field(default=None, description="Optional schema name to filter tables")

# Generate schemas for OpenAI/Anthropic
get_table_schema = GetTableSchema.model_json_schema()
execute_query_schema = ExecuteQuery.model_json_schema()
list_tables_schema = ListTables.model_json_schema()
```

### 2. Validate LLM Function Calls
```python
from pydantic import TypeAdapter, ValidationError

# When LLM returns a function call
def handle_function_call(function_name: str, arguments_json: str):
    function_map = {
        "get_table_schema": GetTableSchema,
        "execute_query": ExecuteQuery,
        "list_tables": ListTables,
    }

    if function_name not in function_map:
        raise ValueError(f"Unknown function: {function_name}")

    model_class = function_map[function_name]

    try:
        # Validate and parse arguments
        args = model_class.model_validate_json(arguments_json)

        # Execute function
        return execute_function(function_name, args)
    except ValidationError as e:
        return f"Invalid arguments: {e}"
```

### 3. Parse Streaming Responses
```python
from pydantic_core import from_json
import pydantic_core

class StreamingResponse(BaseModel):
    content: str = ""
    function_call: dict | None = None
    finish_reason: str | None = None

async def parse_stream_chunk(partial_json: str) -> StreamingResponse:
    """Parse incomplete JSON from streaming LLM response."""
    try:
        data = from_json(partial_json, allow_partial=True)
        return StreamingResponse.model_validate(data)
    except Exception as e:
        # Return empty response on error
        return StreamingResponse()
```

### 4. Configuration with pydantic-settings
```python
from pydantic_settings import BaseSettings

class LLMConfig(BaseSettings):
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    default_provider: str = "openai"
    default_model: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.7

    class Config:
        env_file = ".env"
        env_prefix = "PGCLI_LLM_"

# Usage
config = LLMConfig()  # Auto-loads from .env and env vars
```

---

## Best Practices for pgcli

1. **Define all SQL functions as Pydantic models**
   - Use Field() descriptions for better LLM understanding
   - Include examples in docstrings
   - Add constraints (min_length, regex, etc.)

2. **Generate schemas once at startup**
   - Cache model_json_schema() results
   - Don't regenerate on every request

3. **Use TypeAdapter for dynamic validation**
   - Create adapters for each function
   - Reuse them across requests

4. **Handle partial responses gracefully**
   - Use pydantic_core.from_json with allow_partial=True
   - Provide defaults for all fields
   - Use WrapValidator for fallbacks

5. **Validate config early**
   - Use pydantic-settings for env vars
   - Fail fast on invalid config
   - Provide clear error messages

6. **Separate concerns**
   - Pydantic for validation/schemas
   - Custom code for provider logic
   - HTTP client for communication

7. **Error handling**
   - Catch ValidationError specifically
   - Provide user-friendly messages
   - Log full errors for debugging

---

## Resources

- **Official Docs**: https://docs.pydantic.dev/latest/
- **LLM Articles**: https://pydantic.dev/articles (partial JSON parsing, LLM validation)
- **JSON Schema**: https://docs.pydantic.dev/latest/concepts/json_schema/
- **Type Adapter**: https://docs.pydantic.dev/latest/concepts/type_adapter/
- **Settings**: https://docs.pydantic.dev/latest/concepts/pydantic_settings/ (pydantic-settings package)
- **GitHub**: https://github.com/pydantic/pydantic

---

## Conclusion

**Pydantic is PERFECT for**:
- ✅ JSON schema generation (function calling)
- ✅ Response validation
- ✅ Type-safe models
- ✅ Partial JSON parsing (streaming)
- ✅ Configuration management

**Pydantic is NOT**:
- ❌ An LLM client library
- ❌ A multi-provider abstraction
- ❌ A streaming framework
- ❌ A prompt manager

**For pgcli LLM integration**:
- Use Pydantic for all validation and schema generation
- Build provider abstraction separately (factory pattern)
- Use provider SDKs (openai, anthropic) or httpx for API calls
- Combine Pydantic's partial JSON parsing with streaming responses
- Keep provider logic separate from Pydantic models

Pydantic solves the "schema and validation" problem beautifully. The "multi-provider LLM client" problem requires custom architecture built ON TOP of Pydantic + provider SDKs.
