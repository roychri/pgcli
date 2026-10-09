"""
Natural language detection for SQL input.

Heuristic to detect if user input looks like natural language rather than SQL.
"""

import re
from typing import List


# Common English words that suggest natural language
NATURAL_LANGUAGE_INDICATORS = {
    # Question words
    'show', 'get', 'find', 'list', 'give', 'tell', 'display', 'fetch',
    'what', 'which', 'who', 'when', 'where', 'why', 'how',
    # Action words
    'please', 'can', 'could', 'would', 'should', 'want', 'need',
    # Common words
    'all', 'me', 'the', 'my', 'our', 'latest', 'recent', 'new', 'old',
    'first', 'last', 'top', 'bottom', 'best', 'worst',
}

# SQL keywords that should be at the start of valid SQL
SQL_KEYWORDS = {
    'select', 'insert', 'update', 'delete', 'create', 'alter', 'drop',
    'truncate', 'grant', 'revoke', 'begin', 'commit', 'rollback',
    'explain', 'analyze', 'vacuum', 'copy', 'set', 'show', 'describe',
    'use', 'with',
}

# PostgreSQL special commands (backslash commands)
PSQL_COMMANDS = re.compile(r'^\s*\\[a-zA-Z]')


def looks_like_natural_language(text: str) -> bool:
    """
    Heuristic to detect if input looks like natural language rather than SQL.

    This is used to prompt users who may have forgotten to enable LLM mode.

    :param text: User input text
    :return: True if text looks like natural language, False if it looks like SQL
    """
    if not text or not text.strip():
        return False

    text_lower = text.lower().strip()

    # Check for psql special commands (\\d, \\dt, etc.)
    if PSQL_COMMANDS.match(text_lower):
        return False

    # Split into words
    words = re.findall(r'\b\w+\b', text_lower)

    if len(words) == 0:
        return False

    first_word = words[0]

    # Check if starts with SQL keyword followed by proper syntax
    # Example: "SELECT * FROM" is SQL, but "select me all users" is natural language
    if first_word in SQL_KEYWORDS:
        # If it starts with a SQL keyword, check if it follows SQL syntax
        if first_word in ('select', 'delete', 'insert', 'update'):
            # Look for typical SQL structure
            # SELECT should be followed by columns or *
            # DELETE should be followed by FROM
            # INSERT should be followed by INTO
            # UPDATE should be followed by table name

            if first_word == 'select' and len(words) >= 2:
                second_word = words[1]
                # "SELECT *" or "SELECT col" is SQL
                # "SELECT me" or "SELECT all" is more likely natural language
                if second_word in ('*', 'distinct', 'all', 'count', 'sum', 'avg', 'min', 'max'):
                    return False  # Looks like SQL
                # Check if second word is followed by FROM or comma (column list)
                if 'from' in words or ',' in text or 'where' in words:
                    return False  # Looks like SQL

            elif first_word == 'delete' and len(words) >= 2:
                second_word = words[1]
                if second_word == 'from':
                    return False  # "DELETE FROM" is SQL

            elif first_word == 'insert' and len(words) >= 2:
                second_word = words[1]
                if second_word == 'into':
                    return False  # "INSERT INTO" is SQL

            elif first_word == 'update' and len(words) >= 2:
                # UPDATE should be followed by table name then SET
                if 'set' in words:
                    return False  # Looks like SQL

        # If it starts with SQL keyword but doesn't follow SQL syntax,
        # it might be natural language like "show me all users"
        # Let's continue checking for natural language indicators

    # Count natural language indicators
    natural_word_count = sum(1 for word in words if word in NATURAL_LANGUAGE_INDICATORS)

    # If we have 2+ natural language words, it's likely natural language
    if natural_word_count >= 2:
        return True

    # Check for common natural language patterns
    # "show me", "get me", "find all", "list all", etc.
    if len(words) >= 2:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        natural_bigrams = {
            'show me', 'get me', 'find all', 'list all', 'give me',
            'tell me', 'how many', 'how much', 'what are', 'what is',
            'which are', 'which is', 'can you', 'could you', 'please show',
            'please get', 'please list', 'please find',
        }

        for bigram in bigrams:
            if bigram in natural_bigrams:
                return True

    # Check if it contains a question mark (natural language)
    if '?' in text:
        return True

    # If it's a single SQL keyword, it's probably a typo or incomplete SQL
    if len(words) == 1 and first_word in SQL_KEYWORDS:
        return False

    # If it has multiple words with spaces and no semicolon, check ratio
    if len(words) >= 3:
        # If 30%+ of words are natural language indicators, likely natural language
        ratio = natural_word_count / len(words)
        if ratio >= 0.3:
            return True

    # Default: if we have any natural language indicators and it doesn't look like SQL, say yes
    if natural_word_count > 0 and first_word not in SQL_KEYWORDS:
        return True

    return False


def is_syntax_error(exception: Exception) -> bool:
    """
    Check if exception is a SQL syntax error.

    :param exception: Exception to check
    :return: True if it's a syntax error, False otherwise
    """
    error_str = str(exception).lower()

    # Common syntax error patterns in PostgreSQL
    syntax_error_patterns = [
        'syntax error',
        'at or near',
        'unexpected',
        'unrecognized',
        'invalid syntax',
        'cannot parse',
    ]

    return any(pattern in error_str for pattern in syntax_error_patterns)
