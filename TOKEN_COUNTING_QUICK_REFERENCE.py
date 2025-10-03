"""
Token Counting Quick Reference for pgcli Multi-Provider LLM Support
====================================================================

Quick copy-paste code snippets for implementing token counting.
"""

# ==============================================================================
# 1. ANTHROPIC TOKEN COUNTING (Recommended: Use built-in API)
# ==============================================================================

async def count_anthropic_tokens_example():
    """Example: Count tokens using Anthropic's official API."""
    import anthropic
    import os

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = await client.messages.count_tokens(
        model="claude-sonnet-4-5-20250929",
        system="You are a PostgreSQL expert.",
        messages=[
            {"role": "user", "content": "SELECT * FROM users"},
            {"role": "assistant", "content": "This query selects all columns..."},
        ],
        tools=[
            {
                "name": "get_tables",
                "description": "List tables",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
    )

    print(f"Input tokens: {response.input_tokens}")
    return response.input_tokens


# ==============================================================================
# 2. OPENAI TOKEN COUNTING (Recommended: Use tiktoken)
# ==============================================================================

def count_openai_tokens_example():
    """Example: Count tokens using tiktoken."""
    import tiktoken

    # Get encoding for model
    encoding = tiktoken.encoding_for_model("gpt-4o")
    # Or fallback: encoding = tiktoken.get_encoding("o200k_base")

    messages = [
        {"role": "system", "content": "You are a PostgreSQL expert."},
        {"role": "user", "content": "List tables"},
        {"role": "assistant", "content": "SELECT * FROM information_schema.tables;"}
    ]

    # Count tokens
    num_tokens = 2  # Priming tokens
    for message in messages:
        num_tokens += 4  # Message overhead
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))

    return num_tokens


def count_openai_with_tools_example():
    """Example: Count tokens including tool definitions."""
    import tiktoken
    import json

    encoding = tiktoken.encoding_for_model("gpt-4o")

    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"name": "get_tables", "description": "List tables"}]

    # Count messages
    tokens = 2
    for msg in messages:
        tokens += 4
        for val in msg.values():
            tokens += len(encoding.encode(str(val)))

    # Count tools (approximate)
    if tools:
        tokens += len(encoding.encode(json.dumps(tools)))

    return tokens


# ==============================================================================
# 3. OLLAMA TOKEN COUNTING (Use character estimation or post-response)
# ==============================================================================

def estimate_ollama_tokens_example():
    """Example: Rough estimation for Ollama models."""
    messages = [
        {"role": "user", "content": "List tables"},
        {"role": "assistant", "content": "SELECT * FROM information_schema.tables;"}
    ]
    system_prompt = "You are a PostgreSQL expert."

    # Count characters
    total_chars = len(system_prompt)
    for msg in messages:
        for val in msg.values():
            total_chars += len(str(val))

    # Rule of thumb: 1 token ≈ 4 chars
    tokens = (total_chars // 4) + 100  # +100 overhead

    return tokens


def get_ollama_tokens_from_response():
    """Example: Get actual token counts from Ollama response."""
    # After making request:
    # response = await client.chat.completions.create(...)

    # Access token counts from response
    # prompt_tokens = response.usage.prompt_tokens
    # completion_tokens = response.usage.completion_tokens
    # total_tokens = response.usage.total_tokens

    pass  # Placeholder


# ==============================================================================
# 4. CONTEXT WINDOW LIMITS LOOKUP TABLE
# ==============================================================================

CONTEXT_LIMITS = {
    # Anthropic (conservative, includes 20K buffer)
    "claude-sonnet-4-5-20250929": 180_000,
    "claude-sonnet-4-0": 180_000,
    "claude-haiku-3-5": 180_000,

    # OpenAI (conservative, includes buffer)
    "gpt-4o": 120_000,
    "gpt-4o-mini": 120_000,
    "gpt-5": 350_000,
    "gpt-5-mini": 350_000,

    # Ollama (conservative defaults)
    "llama3.2": 30_000,
    "llama3.3": 30_000,
    "qwen2.5": 30_000,
}

DEFAULT_CONTEXT_LIMIT = 30_000


def get_context_limit(model_name: str) -> int:
    """Get context window limit for a model."""
    return CONTEXT_LIMITS.get(model_name, DEFAULT_CONTEXT_LIMIT)


# ==============================================================================
# 5. MODEL ENCODING MAPPINGS (for tiktoken)
# ==============================================================================

MODEL_TO_ENCODING = {
    # GPT-5 family
    "gpt-5": "o200k_base",
    "gpt-5-mini": "o200k_base",
    "gpt-5-nano": "o200k_base",

    # GPT-4.1 family
    "gpt-4.1": "o200k_base",
    "gpt-4.1-mini": "o200k_base",

    # GPT-4o family
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",

    # Older models
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


def get_encoding_for_model(model_name: str):
    """Get tiktoken encoding for a model."""
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to o200k_base for newer models
        encoding_name = MODEL_TO_ENCODING.get(model_name, "o200k_base")
        return tiktoken.get_encoding(encoding_name)


# ==============================================================================
# 6. CONVERSATION HISTORY TRIMMING
# ==============================================================================

def trim_conversation_history(
    messages: list,
    max_tokens: int,
    token_estimator_func
) -> list:
    """
    Trim conversation history to fit within token limit.

    Args:
        messages: Full conversation history
        max_tokens: Maximum allowed tokens
        token_estimator_func: Function to estimate tokens for a message

    Returns:
        Trimmed message list (oldest messages removed first)
    """
    if not messages:
        return messages

    # Calculate current tokens
    current_tokens = sum(token_estimator_func(msg) for msg in messages)

    if current_tokens <= max_tokens:
        return messages

    # Trim from oldest
    trimmed = []
    tokens_used = 0

    for msg in reversed(messages):
        msg_tokens = token_estimator_func(msg)
        if tokens_used + msg_tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        tokens_used += msg_tokens

    return trimmed


def estimate_message_tokens_simple(message: dict) -> int:
    """Simple character-based token estimation for a single message."""
    chars = sum(len(str(v)) for v in message.values())
    return (chars // 4) + 4  # +4 message overhead


# ==============================================================================
# 7. MINIMAL IMPLEMENTATION (No External Dependencies)
# ==============================================================================

def minimal_token_counter(text: str) -> int:
    """
    Simplest possible token estimator (no dependencies).

    Rule: 1 token ≈ 4 characters
    Overhead: +10% for message structure
    """
    return int(len(text) / 4 * 1.1)


def minimal_conversation_counter(messages: list, system_prompt: str = "") -> int:
    """Count tokens for entire conversation (no dependencies)."""
    total_chars = len(system_prompt)

    for msg in messages:
        if isinstance(msg, dict):
            for val in msg.values():
                total_chars += len(str(val))
        else:
            total_chars += len(str(msg))

    return int(total_chars / 4 * 1.1) + 50  # +50 overhead


# ==============================================================================
# 8. INTEGRATION WITH PYDANTIC AI
# ==============================================================================

async def count_tokens_for_pydantic_ai(
    provider: str,
    model: str,
    message_history: list,
    system_prompt: str,
    tools: list = None
) -> int:
    """
    Count tokens for Pydantic AI message history.

    Args:
        provider: "anthropic", "openai", or "ollama"
        model: Model name
        message_history: Pydantic AI message history from result.all_messages()
        system_prompt: System prompt text
        tools: Tool definitions (optional)

    Returns:
        Token count
    """
    if provider == "anthropic":
        import anthropic
        import os

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Convert Pydantic AI messages to Anthropic format
        anthropic_messages = []
        for msg in message_history:
            # TODO: Implement actual conversion
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        response = await client.messages.count_tokens(
            model=model,
            system=system_prompt,
            messages=anthropic_messages,
            tools=tools if tools else anthropic.NOT_GIVEN
        )
        return response.input_tokens

    elif provider == "openai":
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)

            tokens = len(encoding.encode(system_prompt)) + 4
            for msg in message_history:
                # TODO: Implement proper conversion
                tokens += 4
                content = str(getattr(msg, 'content', ''))
                tokens += len(encoding.encode(content))

            if tools:
                import json
                tokens += len(encoding.encode(json.dumps(tools)))

            return tokens + 2

        except ImportError:
            # Fallback
            return minimal_conversation_counter(message_history, system_prompt)

    else:  # ollama
        return minimal_conversation_counter(message_history, system_prompt)


# ==============================================================================
# 9. EXAMPLE: FULL INTEGRATION
# ==============================================================================

async def generate_sql_with_token_management(
    provider: str,
    model: str,
    user_input: str,
    message_history: list,
    system_prompt: str,
    tools: list
):
    """
    Example showing full token management in SQL generation.
    """
    # Get context limit
    context_limit = get_context_limit(model)

    # Count current tokens
    current_tokens = await count_tokens_for_pydantic_ai(
        provider,
        model,
        message_history,
        system_prompt,
        tools
    )

    # Trim if needed (leave 20% for response)
    max_input_tokens = int(context_limit * 0.8)

    if current_tokens > max_input_tokens:
        print(f"⚠️  Trimming conversation history ({current_tokens} -> {max_input_tokens} tokens)")

        # Simple trimming: remove oldest messages
        def estimate(msg):
            return minimal_token_counter(str(getattr(msg, 'content', '')))

        message_history = trim_conversation_history(
            message_history,
            max_input_tokens,
            estimate
        )

        # Recount
        current_tokens = await count_tokens_for_pydantic_ai(
            provider,
            model,
            message_history,
            system_prompt,
            tools
        )

    print(f"📊 Using {current_tokens} / {context_limit} tokens ({current_tokens/context_limit*100:.1f}%)")

    # Now generate SQL with trimmed history
    # result = await agent.run(user_input, message_history=message_history)
    # return result.output


# ==============================================================================
# 10. DEPENDENCY INSTALLATION COMMANDS
# ==============================================================================

"""
# Install tiktoken for OpenAI (recommended)
pip install tiktoken

# Already installed in pgcli:
# - anthropic >= 0.69.0
# - openai >= 2.0.0

# Add to setup.py extras_require:
extras_require={
    "llm": ["tiktoken >= 0.5.0"],
}

# Install with:
pip install pgcli[llm]
"""


# ==============================================================================
# 11. HELPFUL CONSTANTS
# ==============================================================================

# Anthropic models
ANTHROPIC_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-0",
    "claude-haiku-3-5",
]

# OpenAI models
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
]

# Ollama common models
OLLAMA_MODELS = [
    "llama3.2",
    "llama3.3",
    "qwen2.5",
    "deepseek-r1",
]


if __name__ == "__main__":
    # Quick test examples
    print("Token Counting Quick Reference")
    print("=" * 60)

    # Test minimal counter
    text = "SELECT * FROM users WHERE id = 1;"
    tokens = minimal_token_counter(text)
    print(f"Minimal estimate for '{text}': {tokens} tokens")

    # Test context limits
    for model in ["gpt-4o", "claude-sonnet-4-5-20250929", "llama3.2"]:
        limit = get_context_limit(model)
        print(f"{model}: {limit:,} tokens")
