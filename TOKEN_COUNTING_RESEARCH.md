# Token Counting and Context Window Research for Multi-Provider LLM Integration

**Date:** 2025-10-02
**Purpose:** Research token counting approaches and context window limits for pgcli's multi-provider LLM support (Anthropic, OpenAI, Ollama)

---

## Executive Summary

### Key Findings

1. **Pydantic AI** does NOT provide built-in token counting utilities across providers
2. **Token counting is provider-specific** and requires different approaches for each
3. **Context window limits vary significantly** by model and provider
4. **Minimal dependency approach:** Use provider SDKs directly (already installed)

### Recommendations

- **Anthropic:** Use built-in `client.messages.count_tokens()` API (free, accurate)
- **OpenAI:** Install `tiktoken` for pre-request counting (lightweight, ~1MB)
- **Ollama:** Use post-request token counts from response metadata
- **Context limits:** Hardcode a lookup table with conservative defaults

---

## 1. Pydantic AI Token Counting Support

### Investigation Results

**Location:** `/home/christian/dev/oss/pgcli/.venv/lib/python3.9/site-packages/pydantic_ai/`

**Key Files Examined:**
- `usage.py` (lines 1-316)
- `models/__init__.py` (lines 425-433)
- `models/google.py` (lines 201-244)

### Findings

1. **Usage Tracking:** Pydantic AI tracks token usage AFTER requests via `RequestUsage` and `RunUsage` classes
   - `input_tokens`, `output_tokens`, `total_tokens`
   - Populated from provider responses

2. **Pre-Request Counting:** Only Google/Gemini models support `count_tokens()` method
   ```python
   async def count_tokens(
       self,
       messages: list[ModelMessage],
       model_settings: ModelSettings | None,
       model_request_parameters: ModelRequestParameters,
   ) -> RequestUsage:
       # Calls Google's count_tokens API
   ```

3. **OpenAI/Anthropic/Ollama:** Do NOT implement `count_tokens()` in Pydantic AI
   - Method exists in abstract `Model` class but raises `NotImplementedError`
   - Line 432-433: "Token counting ahead of the request is not supported by {model}"

4. **No tiktoken dependency:** Pydantic AI does not bundle token counting libraries

**Conclusion:** We must implement provider-specific token counting ourselves.

---

## 2. Provider-Specific Token Counting

### 2.1 Anthropic Claude

**SDK Version:** `anthropic==0.69.0` (already installed in pgcli)

#### Built-in Token Counting API

Anthropic provides a dedicated token counting endpoint (introduced Nov 2024):

```python
import anthropic

client = anthropic.Anthropic(api_key="...")

# Count tokens in messages
response = client.messages.count_tokens(
    model="claude-sonnet-4-5-20250929",
    system="You are a PostgreSQL expert.",
    messages=[
        {"role": "user", "content": "List all tables"},
        {"role": "assistant", "content": "SELECT * FROM information_schema.tables;"},
        {"role": "user", "content": "Show indexes"}
    ],
    tools=[{  # Optional: count tool definitions too
        "name": "get_tables",
        "description": "List database tables",
        "input_schema": {"type": "object", "properties": {...}}
    }]
)

print(response.input_tokens)  # Total input tokens
```

**API Details:**
- **Endpoint:** `POST /v1/messages/count_tokens`
- **Free to use** (rate limits apply based on tier)
- **Supports:** Text, PDFs, images, tools, system prompts
- **SDK Methods:**
  - `client.messages.count_tokens()` (sync)
  - `await client.messages.count_tokens()` (async - use this)

**File Reference:**
Source: Web search (Anthropic docs), confirmed with Python SDK inspection

#### Implementation Example for pgcli

```python
async def count_anthropic_tokens(
    messages: list[dict],  # Pydantic AI format
    system_prompt: str,
    tools: list[dict],
    model: str = "claude-sonnet-4-5-20250929"
) -> int:
    """Count tokens for Anthropic models using official API."""
    import anthropic

    client = anthropic.AsyncAnthropic()

    # Convert Pydantic AI messages to Anthropic format
    anthropic_messages = convert_to_anthropic_format(messages)

    response = await client.messages.count_tokens(
        model=model,
        system=system_prompt,
        messages=anthropic_messages,
        tools=tools if tools else anthropic.NOT_GIVEN
    )

    return response.input_tokens
```

**Pros:**
- Official, accurate token counting
- Includes tool definitions, images, etc.
- No external dependencies

**Cons:**
- Requires async API call (network latency ~100-300ms)
- Subject to rate limits

---

### 2.2 OpenAI (GPT-4o, GPT-5)

**SDK Version:** `openai==2.0.0` (already installed in pgcli)

#### Official Approach: Tiktoken Library

OpenAI does NOT provide a token counting API. The recommended approach is `tiktoken`:

```bash
pip install tiktoken  # ~1MB package
```

**Model Encodings (2025):**
- **GPT-5 family:** `o200k_base` encoding
- **GPT-4.1 family:** `o200k_base` encoding
- **GPT-4o family:** `o200k_base` encoding
- **GPT-4, GPT-3.5:** `cl100k_base` encoding

**File Reference:**
Source: `github.com/openai/tiktoken/blob/main/tiktoken/model.py`

#### Implementation Example

```python
def count_openai_tokens(
    messages: list[dict],  # OpenAI chat format
    model: str = "gpt-4o"
) -> int:
    """Count tokens for OpenAI models using tiktoken."""
    import tiktoken

    # Get encoding for model
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("o200k_base")

    # Count tokens in messages
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Every message has 4 tokens overhead
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += -1  # Role is always 1 token, name varies

    num_tokens += 2  # Every reply is primed with assistant tokens
    return num_tokens
```

**Accurate Token Counting for Chat:**

For precise counting including function/tool definitions:

```python
def count_tokens_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str = "gpt-4o"
) -> int:
    """Accurate token counting including tools."""
    import tiktoken
    import json

    encoding = tiktoken.encoding_for_model(model)

    # Count message tokens
    tokens = 0
    for msg in messages:
        tokens += 4  # Message overhead
        for key, val in msg.items():
            tokens += len(encoding.encode(str(val)))

    # Count tool tokens (tools are serialized as JSON)
    if tools:
        tools_json = json.dumps(tools)
        tokens += len(encoding.encode(tools_json))

    tokens += 2  # Completion priming
    return tokens
```

**Pros:**
- Fast, local computation (no API calls)
- Official OpenAI library
- Accurate for text

**Cons:**
- Additional dependency (~1MB)
- Complex for images/multimodal (approximations needed)
- Model mapping needs maintenance

---

### 2.3 Ollama (Local Models)

**Provider:** Ollama (uses OpenAI-compatible API via Pydantic AI)

#### Approach: Post-Request Token Counts

Ollama does NOT provide pre-request token counting. Instead, use response metadata:

```python
# After making a request to Ollama
response = await ollama_client.chat.completions.create(
    model="llama3.2",
    messages=messages
)

# Response includes token counts
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens
total_tokens = response.usage.total_tokens
```

**Alternative:** Approximate with tiktoken (rough estimate)

```python
def estimate_ollama_tokens(text: str, model: str = "llama3.2") -> int:
    """Rough token estimate for Ollama models."""
    import tiktoken

    # Most Ollama models use similar tokenization to GPT-3.5/4
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
```

**Pros:**
- Accurate counts in response
- No extra dependencies

**Cons:**
- Only available AFTER request (can't trim beforehand)
- Estimation for pre-request checks is imprecise

**Recommendation:** For conversation history trimming with Ollama:
1. Track cumulative tokens from previous responses
2. Use rough estimation for new user messages
3. Trim when approaching context limit with buffer

---

## 3. Context Window Limits (2025)

### 3.1 Anthropic Claude

| Model | Context Window | Output Limit | API Tier Notes |
|-------|---------------|--------------|----------------|
| Claude Sonnet 4.5 | 200K tokens | 64K tokens | Standard plans |
| Claude Sonnet 4.5 | 500K tokens | 64K tokens | Enterprise plans |
| Claude Sonnet 4.0 | 1M tokens | 64K tokens | Public beta |
| Claude Haiku 3.5 | 200K tokens | 16K tokens | Standard |

**Source:** Web search (Anthropic docs, announcements Aug 2025)

**Recommended Conservative Limit:** 180K tokens (leave 20K buffer for tools/system)

---

### 3.2 OpenAI Models

| Model | Context Window | Output Limit | Notes |
|-------|---------------|--------------|-------|
| GPT-5 | 400K tokens | 128K tokens | API only, 272K input max |
| GPT-4o | 128K tokens | 16K tokens | Default choice |
| GPT-4o-mini | 128K tokens | 16K tokens | Cost-effective |
| GPT-4.1 | 1M tokens | varies | Azure/specific plans |
| o1, o3 (reasoning) | 128K-400K | varies | Model-specific |

**Source:** Web search (OpenAI docs, community forums)

**ChatGPT Interface Limits (NOT API):**
- Free: 8K tokens
- Plus: 32K tokens
- Pro/Enterprise: 128K tokens

**Recommended Conservative Limit:**
- GPT-4o family: 120K tokens
- GPT-5 family: 350K tokens (for API users)

---

### 3.3 Ollama Local Models

| Model Family | Context Window | Default Ollama | Configurable? |
|--------------|---------------|----------------|---------------|
| Llama 3.2 | 128K tokens | 2048 tokens | Yes (`num_ctx`) |
| Llama 3.3 | 128K tokens | 2048 tokens | Yes |
| Qwen 2.5 | varies | 2048 tokens | Yes |

**Important:** Ollama defaults to 2048 tokens regardless of model capability!

**Configure Context Window:**

```python
# Via Modelfile
"""
FROM llama3.2
PARAMETER num_ctx 32768
"""

# Via API
response = client.chat.completions.create(
    model="llama3.2",
    messages=messages,
    options={"num_ctx": 32768}  # Up to 128K for llama3.2
)
```

**Memory Considerations:** Larger context = more VRAM. Ensure hardware supports it.

**Recommended Conservative Limit:** 30K tokens (safe default, increase if hardware allows)

---

## 4. Implementation Architecture

### 4.1 Proposed Token Counter Module

**File:** `/home/christian/dev/oss/pgcli/pgcli/llm/token_counter.py`

```python
"""Token counting for multi-provider LLM support."""
from typing import Optional, List, Dict
from enum import Enum


class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


# Context window limits (conservative, includes buffer)
CONTEXT_LIMITS = {
    # Anthropic
    "claude-sonnet-4-5-20250929": 180_000,
    "claude-sonnet-4-0": 180_000,
    "claude-haiku-3-5": 180_000,

    # OpenAI
    "gpt-4o": 120_000,
    "gpt-4o-mini": 120_000,
    "gpt-5": 350_000,
    "gpt-5-mini": 350_000,

    # Ollama (default conservative)
    "llama3.2": 30_000,
    "llama3.3": 30_000,
}

DEFAULT_CONTEXT_LIMIT = 30_000  # Fallback for unknown models


async def count_tokens(
    messages: List[Dict],
    system_prompt: str,
    tools: List[Dict],
    provider: Provider,
    model: str
) -> int:
    """Count tokens for conversation history.

    Args:
        messages: Conversation history
        system_prompt: System prompt text
        tools: Tool definitions
        provider: LLM provider
        model: Model name

    Returns:
        Estimated token count
    """
    if provider == Provider.ANTHROPIC:
        return await _count_anthropic_tokens(messages, system_prompt, tools, model)
    elif provider == Provider.OPENAI:
        return _count_openai_tokens(messages, system_prompt, tools, model)
    elif provider == Provider.OLLAMA:
        return _estimate_ollama_tokens(messages, system_prompt, tools)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def _count_anthropic_tokens(
    messages: List[Dict],
    system_prompt: str,
    tools: List[Dict],
    model: str
) -> int:
    """Use Anthropic's official token counting API."""
    import anthropic
    from os import getenv

    client = anthropic.AsyncAnthropic(api_key=getenv("ANTHROPIC_API_KEY"))

    # Convert to Anthropic format
    anthropic_messages = _convert_to_anthropic_format(messages)

    response = await client.messages.count_tokens(
        model=model,
        system=system_prompt,
        messages=anthropic_messages,
        tools=tools if tools else anthropic.NOT_GIVEN
    )

    return response.input_tokens


def _count_openai_tokens(
    messages: List[Dict],
    system_prompt: str,
    tools: List[Dict],
    model: str
) -> int:
    """Use tiktoken for OpenAI token counting."""
    try:
        import tiktoken
    except ImportError:
        # Fallback to rough estimate if tiktoken not installed
        return _estimate_tokens_by_chars(messages, system_prompt, tools)

    # Get encoding
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to o200k_base for GPT-4o/5
        encoding = tiktoken.get_encoding("o200k_base")

    # Count system prompt
    tokens = len(encoding.encode(system_prompt)) + 4

    # Count messages
    for msg in messages:
        tokens += 4  # Message overhead
        for key, val in msg.items():
            tokens += len(encoding.encode(str(val)))

    # Count tools (approximation)
    if tools:
        import json
        tools_str = json.dumps(tools)
        tokens += len(encoding.encode(tools_str))

    tokens += 2  # Completion priming
    return tokens


def _estimate_ollama_tokens(
    messages: List[Dict],
    system_prompt: str,
    tools: List[Dict]
) -> int:
    """Rough token estimate for Ollama (no tiktoken dependency)."""
    return _estimate_tokens_by_chars(messages, system_prompt, tools)


def _estimate_tokens_by_chars(
    messages: List[Dict],
    system_prompt: str,
    tools: List[Dict]
) -> int:
    """Fallback: Rough estimate using character count.

    Rule of thumb: 1 token ≈ 4 characters in English.
    """
    import json

    total_chars = len(system_prompt)

    for msg in messages:
        for val in msg.values():
            total_chars += len(str(val))

    if tools:
        total_chars += len(json.dumps(tools))

    # 1 token per 4 chars (conservative)
    return total_chars // 4 + 100  # +100 overhead


def get_context_limit(model: str) -> int:
    """Get context window limit for model."""
    return CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)


def _convert_to_anthropic_format(messages: List[Dict]) -> List[Dict]:
    """Convert Pydantic AI messages to Anthropic format."""
    # TODO: Implement conversion logic
    return messages
```

---

### 4.2 Conversation History Trimming

**Strategy:** Trim oldest messages first, preserving recent context

```python
def trim_conversation_history(
    messages: List[Dict],
    max_tokens: int,
    current_tokens: int
) -> List[Dict]:
    """Trim conversation history to fit within context limit.

    Args:
        messages: Full conversation history
        max_tokens: Maximum allowed tokens
        current_tokens: Current token count

    Returns:
        Trimmed message list
    """
    if current_tokens <= max_tokens:
        return messages

    # Keep last N messages that fit
    trimmed = []
    tokens = 0

    # Iterate from most recent
    for msg in reversed(messages):
        msg_tokens = estimate_message_tokens(msg)
        if tokens + msg_tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        tokens += msg_tokens

    return trimmed


def estimate_message_tokens(message: Dict) -> int:
    """Rough estimate of single message tokens."""
    chars = sum(len(str(v)) for v in message.values())
    return chars // 4 + 4  # +4 message overhead
```

---

## 5. Dependencies Analysis

### Current Dependencies (from setup.py)

```python
install_requirements = [
    "openai >= 1.52.0",  # Already included!
    # ... other deps
]
```

### Proposed Additional Dependencies

```python
# Option 1: Add tiktoken for OpenAI (recommended)
"tiktoken >= 0.5.0",  # ~1MB, fast, official

# Option 2: No new dependencies
# - Use Anthropic's API for Anthropic
# - Use rough estimation for OpenAI/Ollama
# - Trade accuracy for simplicity
```

**Recommendation:** Add `tiktoken` as optional dependency

```python
extras_require={
    "keyring": ["keyring >= 12.2.0"],
    "sshtunnel": ["sshtunnel >= 0.4.0"],
    "llm": ["tiktoken >= 0.5.0"],  # New: for accurate OpenAI token counting
}
```

**Justification:**
- Small package (~1MB)
- Official OpenAI library
- Significantly more accurate than character-based estimation
- Optional: users without tiktoken get rough estimates

---

## 6. Recommended Implementation Plan

### Phase 1: Basic Token Tracking (No Trimming)

1. Track tokens from response metadata
2. Store cumulative count in `pgcli_instance.llm_token_count`
3. Display warning when approaching limit

### Phase 2: Pre-Request Token Counting

1. Implement `token_counter.py` module
2. Add Anthropic API counting
3. Add tiktoken for OpenAI (optional dependency)
4. Add character-based fallback

### Phase 3: Conversation History Trimming

1. Implement trimming logic
2. Trim before each request if needed
3. Preserve last N messages + system prompt

### Phase 4: User Configuration

1. Add config option: `llm_max_context_tokens`
2. Allow per-provider overrides
3. Add command to show current token usage: `\llm-tokens`

---

## 7. Code Examples

### Example 1: Anthropic Token Counting

**File:** `/home/christian/dev/oss/pgcli/pgcli/llm/agent.py` (potential addition)

```python
async def count_conversation_tokens(
    pgcli_instance,
    message_history: List,
    config: LLMConfig
) -> int:
    """Count tokens in conversation history."""
    if config.provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=config.api_key)

        response = await client.messages.count_tokens(
            model=config.model,
            system=SYSTEM_PROMPT,
            messages=message_history,
            tools=[/* tool definitions */]
        )
        return response.input_tokens

    elif config.provider == "openai":
        # Use tiktoken
        return count_openai_tokens_tiktoken(
            message_history,
            SYSTEM_PROMPT,
            config.model
        )

    else:  # ollama
        # Rough estimate
        return estimate_tokens_by_chars(message_history, SYSTEM_PROMPT)
```

### Example 2: Integration with generate_sql()

```python
def generate_sql(
    pgcli_instance: Any,
    user_input: str,
    message_history: Optional[List] = None,
    config: Optional[LLMConfig] = None,
) -> str:
    """Generate SQL with token-aware history trimming."""

    # Get config
    if config is None:
        config = load_config()

    # Get message history
    if message_history is None and hasattr(pgcli_instance, 'llm_message_history'):
        message_history = pgcli_instance.llm_message_history

    # Check token count
    if message_history:
        current_tokens = await count_conversation_tokens(
            pgcli_instance,
            message_history,
            config
        )

        context_limit = get_context_limit(config.model)

        # Trim if needed (leave 20% buffer for response)
        max_input_tokens = int(context_limit * 0.8)
        if current_tokens > max_input_tokens:
            message_history = trim_conversation_history(
                message_history,
                max_input_tokens,
                current_tokens
            )

    # Create agent and run
    agent = create_agent(config=config)
    result = agent.run_sync(user_input, deps=deps, message_history=message_history)

    # Update history
    if hasattr(pgcli_instance, 'llm_message_history'):
        pgcli_instance.llm_message_history = result.all_messages()

    return result.output.strip()
```

---

## 8. Testing Strategy

### Test Cases

1. **Token Counting Accuracy**
   - Compare tiktoken counts to actual API usage
   - Verify Anthropic API counts match response metadata
   - Test fallback estimation accuracy

2. **Conversation Trimming**
   - Test trimming removes oldest messages first
   - Verify trimmed history fits in context
   - Test edge case: single message > context limit

3. **Provider-Specific Behavior**
   - Test Anthropic with tools
   - Test OpenAI with/without tiktoken
   - Test Ollama estimation

### Example Test

```python
def test_anthropic_token_counting():
    """Test Anthropic token counting matches API."""
    import anthropic

    client = anthropic.Anthropic()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    # Count using our module
    our_count = count_tokens(
        messages,
        "You are helpful",
        [],
        Provider.ANTHROPIC,
        "claude-sonnet-4-5-20250929"
    )

    # Count using API directly
    api_response = client.messages.count_tokens(
        model="claude-sonnet-4-5-20250929",
        system="You are helpful",
        messages=messages
    )

    assert our_count == api_response.input_tokens
```

---

## 9. References

### Documentation
- Anthropic Token Counting: https://docs.anthropic.com/en/docs/build-with-claude/token-counting
- OpenAI Tiktoken: https://github.com/openai/tiktoken
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md

### File Locations in pgcli Project
- Pydantic AI agent: `/home/christian/dev/oss/pgcli/pgcli/llm/agent.py`
- Config: `/home/christian/dev/oss/pgcli/pgcli/llm/config.py`
- Setup: `/home/christian/dev/oss/pgcli/setup.py`

### Pydantic AI Source (inspected)
- Usage tracking: `.venv/lib/python3.9/site-packages/pydantic_ai/usage.py`
- Model interface: `.venv/lib/python3.9/site-packages/pydantic_ai/models/__init__.py`
- Google count_tokens: `.venv/lib/python3.9/site-packages/pydantic_ai/models/google.py`

---

## 10. Decision Matrix

| Approach | Accuracy | Speed | Dependencies | Complexity |
|----------|----------|-------|--------------|------------|
| **Anthropic API** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (network) | ✅ None (SDK installed) | ⭐⭐ |
| **Tiktoken (OpenAI)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (local) | ⚠︎  +1MB | ⭐⭐ |
| **Character estimation** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ None | ⭐ |
| **Pydantic AI built-in** | N/A | N/A | ❌ Not available | N/A |

**Recommended Hybrid Approach:**
1. Anthropic: Use API (already have SDK)
2. OpenAI: Use tiktoken (add as optional dep)
3. Ollama: Use character estimation (no deps)
4. Fallback: Character estimation if tiktoken missing



---

### Token Counting

  Pydantic AI: ❌ No built-in token counting for Anthropic/OpenAI/Ollama

  Solutions:
  - Anthropic: Use API client.messages.count_tokens() (network call, exact)
  - OpenAI: Use tiktoken library (local, instant, +1MB dependency)
  - Ollama: Character estimation (1 token ≈ 4 chars)

  Recommendation: Add tiktoken as optional dependency:
  # setup.py
  extras_require={
      "llm": ["tiktoken >= 0.5.0"],
  }

  2. Context Windows & Limits

  | Model      | Context | Recommended Max History | SQL Result Limit |
  |------------|---------|-------------------------|------------------|
  | Claude 4.5 | 200K    | 180K (90%)              | 5K tokens        |
  | GPT-4o     | 128K    | 115K (90%)              | 4K tokens        |
  | GPT-5      | 400K    | 360K (90%)              | 8K tokens        |
  | Ollama     | 128K*   | 30K (safe)              | 2K tokens        |

  *Ollama defaults to 2K regardless of model capability!

  3. Smart Trimming Strategy

  For SQL results:
  If < 2K tokens: keep all
  If single line: token truncate (preserve JSON structure)
  If < 20 lines: truncate each line individually
  If many short lines: head (50) + tail (50) + omission marker
  If mixed: hybrid approach

  For conversation:
  Always keep: system messages
  Always keep: last 10 messages
  Trim aggressively: tool outputs (SQL results)
  Trim from oldest: middle messages

---

## Conclusion

**Token counting for pgcli's multi-provider LLM support requires provider-specific implementations:**

1. **Pydantic AI does not provide unified token counting** - we must handle it ourselves
2. **Best approaches:**
   - Anthropic: Built-in API (free, accurate)
   - OpenAI: Tiktoken library (1MB dependency, highly accurate)
   - Ollama: Post-request metadata + estimation
3. **Context limits:** Hardcode lookup table with conservative defaults
4. **Implementation:** Create `token_counter.py` module with provider-specific logic
5. **Dependencies:** Add tiktoken as optional dependency in `extras_require`

**Next Steps:**
1. Implement basic token tracking from responses
2. Add token counting module with provider-specific logic
3. Implement conversation history trimming
4. Add user-facing token usage commands
