"""SQL result and message history trimming utilities.

Provides intelligent trimming strategies to fit content within token limits.
"""
from typing import List, Optional, Tuple
from .token_counter import TokenCounter


def detect_content_structure(content: str) -> str:
    """Detect the structure of content for appropriate trimming strategy.

    :param content: Content to analyze
    :return: Structure type ('single_line', 'few_lines', 'tabular', 'empty')
    """
    if not content or not content.strip():
        return "empty"

    lines = content.split("\n")
    line_count = len(lines)

    if line_count == 1:
        return "single_line"
    elif line_count <= 10:
        return "few_lines"
    else:
        # Check if it looks tabular (multiple lines with consistent separators)
        # Common separators: |, \t, multiple spaces
        first_line = lines[0]
        if "|" in first_line or "\t" in first_line:
            return "tabular"
        return "few_lines"


def trim_single_line(
    content: str, max_tokens: int, counter: TokenCounter
) -> Tuple[str, bool]:
    """Trim a single long line to fit within token limit.

    :param content: Single line content
    :param max_tokens: Maximum tokens allowed
    :param counter: Token counter instance
    :return: Tuple of (trimmed_content, was_trimmed)
    """
    current_tokens = counter.count_tokens(content)

    if current_tokens <= max_tokens:
        return content, False

    # Binary search to find the right length
    # Start with character ratio estimation
    char_ratio = max_tokens / current_tokens
    estimated_chars = int(len(content) * char_ratio * 0.9)  # 90% to be safe

    trimmed = content[:estimated_chars]
    while counter.count_tokens(trimmed) > max_tokens and len(trimmed) > 0:
        # Reduce by 10% each iteration
        estimated_chars = int(estimated_chars * 0.9)
        trimmed = content[:estimated_chars]

    return f"{trimmed}\n... [content truncated, {current_tokens} tokens total]", True


def trim_tabular_data(
    content: str, max_tokens: int, counter: TokenCounter, head_lines: int = 50
) -> Tuple[str, bool]:
    """Trim tabular data using head+tail strategy.

    :param content: Tabular content
    :param max_tokens: Maximum tokens allowed
    :param counter: Token counter instance
    :param head_lines: Number of lines to keep from head
    :return: Tuple of (trimmed_content, was_trimmed)
    """
    lines = content.split("\n")
    total_lines = len(lines)

    current_tokens = counter.count_tokens(content)
    if current_tokens <= max_tokens:
        return content, False

    # Always keep header if present (first line)
    header = [lines[0]] if lines else []
    remaining_lines = lines[1:] if len(lines) > 1 else []

    # Start with head_lines, adjust if needed
    tail_lines = head_lines

    while head_lines + tail_lines > len(remaining_lines):
        head_lines = max(1, head_lines - 10)
        tail_lines = max(1, tail_lines - 10)

    head = remaining_lines[:head_lines]
    tail = remaining_lines[-tail_lines:] if tail_lines > 0 else []

    omitted_count = total_lines - head_lines - tail_lines - 1  # -1 for header

    trimmed_lines = (
        header
        + head
        + [f"... [{omitted_count} rows omitted, {current_tokens} tokens total] ..."]
        + tail
    )

    trimmed = "\n".join(trimmed_lines)

    # Verify we're within limits, reduce further if needed
    while counter.count_tokens(trimmed) > max_tokens and (head_lines > 1 or tail_lines > 1):
        head_lines = max(1, head_lines - 5)
        tail_lines = max(1, tail_lines - 5)

        head = remaining_lines[:head_lines]
        tail = remaining_lines[-tail_lines:] if tail_lines > 0 else []
        omitted_count = total_lines - head_lines - tail_lines - 1

        trimmed_lines = (
            header
            + head
            + [f"... [{omitted_count} rows omitted, {current_tokens} tokens total] ..."]
            + tail
        )
        trimmed = "\n".join(trimmed_lines)

    return trimmed, True


def trim_sql_result(
    content: str, max_tokens: int, counter: TokenCounter
) -> Tuple[str, bool]:
    """Trim SQL result using appropriate strategy based on structure.

    :param content: SQL result content
    :param max_tokens: Maximum tokens allowed
    :param counter: Token counter instance
    :return: Tuple of (trimmed_content, was_trimmed)
    """
    if not content or not content.strip():
        return content, False

    structure = detect_content_structure(content)

    if structure == "empty":
        return content, False
    elif structure == "single_line":
        return trim_single_line(content, max_tokens, counter)
    elif structure == "tabular" or structure == "few_lines":
        return trim_tabular_data(content, max_tokens, counter)
    else:
        # Fallback to single line trimming
        return trim_single_line(content, max_tokens, counter)


def trim_message_history(
    messages: List, max_tokens: int, counter: TokenCounter
) -> List:
    """Trim message history to fit within token limit.

    Strategy:
    - Keep all system messages
    - Keep last 10 user/assistant exchanges
    - Trim middle messages if needed

    :param messages: List of Pydantic AI messages
    :param max_tokens: Maximum tokens allowed
    :param counter: Token counter instance
    :return: Trimmed message list
    """
    if not messages:
        return messages

    # Quick check: count total tokens
    total_text = ""
    for msg in messages:
        # Extract text from various message types
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if hasattr(part, "content"):
                    total_text += str(part.content) + " "

    current_tokens = counter.count_tokens(total_text)

    if current_tokens <= max_tokens:
        return messages

    # Need to trim - keep last N messages
    # Start with last 20 messages (10 exchanges)
    keep_count = min(20, len(messages))

    while keep_count > 0:
        trimmed = messages[-keep_count:]

        # Recalculate tokens
        trimmed_text = ""
        for msg in trimmed:
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "content"):
                        trimmed_text += str(part.content) + " "

        if counter.count_tokens(trimmed_text) <= max_tokens:
            return trimmed

        # Reduce by 2 messages at a time
        keep_count -= 2

    # If we still can't fit, return last message only
    return messages[-1:] if messages else []
