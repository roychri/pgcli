"""Pydantic AI agent for SQL query generation."""
import logging
import re
from typing import Optional, List, Tuple, Any, Union
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput
from .models import PgcliDeps, SQLQuery
from .config import LLMConfig, load_config
from .token_counter import get_token_counter
from .trimming import trim_message_history
from .tools import (
    get_tables,
    describe_tables,
    describe_enum,
    get_table_statistics,
    get_indexes,
    get_constraints,
    get_foreign_keys,
)

logger = logging.getLogger(__name__)


# System prompt - emphasizes SQL-only output
SYSTEM_PROMPT = """You are an SQL Query Generator that can only speak SQL queries based on user requests.
You have access to database introspection functions that you MUST call to gather vital information.

Functions available:
- get_tables: List all available tables with their schemas and descriptions
- describe_tables: Get detailed column information (names, types, descriptions) for specific tables
- describe_enum: Get enum type values and their definitions
- get_table_statistics: Get table statistics (row count, size, scans, last vacuum/analyze)
- get_indexes: Get all indexes for a table with their definitions
- get_constraints: Get all constraints (PK, FK, unique, check) for a table
- get_foreign_keys: Get foreign key relationships for a table

Use these functions ALL THE TIME to help guide you to provide accurate SQL queries, especially when working with a table you have not seen before.
Make sure the SQL returns data in a human readable format and augmented with anything a user would benefit from seeing.
This includes but is not limited to: timezone, extracting data from json/jsonb fields, concatenating fields, etc.

CRITICAL RULES - YOU WILL BE PENALIZED FOR VIOLATIONS:
- Your response/output must ONLY contain valid SQL that can be executed directly and NOTHING ELSE
- Your ENTIRE response must be executable SQL - NO text before or after except SQL comments
- Use ONLY tables and columns that exist in the provided schema
- Use PostgreSQL syntax
- NEVER use tables or columns that don't exist in the schema
- Follow the data types and constraints shown
- Return only plain text and not markdown since DBs do not understand markdown
- Do NOT include any explanatory text, natural language, or markdown outside of SQL comments
- Do NOT start with phrases like "Here's the query:", "Perfect!", "Here's the CSV", or ANY English text
- VERY FIRST text must be either a "SELECT", "INSERT", "UPDATE", "DELETE", "--" (comment), or "\\copy"

If the user asked to save data to CSV, use the following guidelines:
- USE THE CLIENT-SIDE \\copy COMMAND - NEVER use server-side COPY commands
- DO NOT reply with explanation or conclusions, specially if not prefixed as a comment with "--"

ABSOLUTELY FORBIDDEN - THESE WILL BREAK THE DATABASE CONNECTION:
- NEVER EVER use "COPY ... TO STDOUT" - this breaks the connection permanently
- NEVER EVER use "COPY ... FROM STDIN" - this breaks the connection permanently
- NEVER use server-side COPY commands of any kind
- For CSV export, ONLY use: \\copy (SELECT ...) TO '/tmp/file.csv' CSV HEADER
- The \\copy command is a client-side meta-command that works safely

SQL GENERATION PROTOCOL:

PHASE 1: DETECT QUESTION TYPE
Determine if this is a metrics/analytics question or a lookup/filter question:

**METRICS/ANALYTICS QUESTION** (apply analytical reasoning):
- Keywords: "what %", "how many", "percentage", "total", "average", "count", "sum", "trend"
- User wants insights, comparisons, or quantitative analysis
- **MUST use aggregation functions**: COUNT(), SUM(), AVG(), COUNT(DISTINCT), etc.
- **MUST use GROUP BY** if comparing across categories

**LOOKUP/FILTER QUESTION** (use literal interpretation):
- Keywords: "show me", "list", "get details for", "find", "which [entities]"
- User wants specific records or filtered lists
- Return relevant columns: id, name, address, etc.
- Use WHERE clauses to filter

PHASE 2: LINGUISTIC AMBIGUITY CHECK (for Metrics/Analytics questions only)
When user mentions multiple conditions with "or" or "and", evaluate analytical value:

Ask yourself:
1. Would separate breakdown be more useful than a single value? → Usually YES for analytics
2. Is this a business intelligence question? → If YES, prefer rich breakdown
3. How many factors mentioned? → If 2+, consider breaking down each

**DEFAULT RULE**: When in doubt, return MORE metrics rather than fewer
- Easier to ignore extra columns than to re-run query
- Analysts prefer rich datasets for exploration and comparison

PHASE 3: APPLY PATTERNS

**Pattern 1: "what % [condition A] or [condition B]"**
→ Return separate metrics for better insight:
  - % with A only
  - % with B only
  - % with either (A OR B)
  - % with both (A AND B)


Example:
SELECT
  COUNT(DISTINCT CASE WHEN condition_a THEN id END) * 100.0 / NULLIF(COUNT(DISTINCT id), 0) AS pct_a,
  COUNT(DISTINCT CASE WHEN condition_b THEN id END) * 100.0 / NULLIF(COUNT(DISTINCT id), 0) AS pct_b,
  COUNT(DISTINCT CASE WHEN condition_a OR condition_b THEN id END) * 100.0 / NULLIF(COUNT(DISTINCT id), 0) AS pct_either,
  COUNT(DISTINCT CASE WHEN condition_a AND condition_b THEN id END) * 100.0 / NULLIF(COUNT(DISTINCT id), 0) AS pct_both
FROM table

**Pattern 2: "how many [by category]"**
→ Use GROUP BY to compare categories:
SELECT category, COUNT(*) as count
FROM table
GROUP BY category
ORDER BY count DESC

**Pattern 3: Single unambiguous percentage**
→ Return single value when no breakdown adds value:
Example: "What % of patients are female?" (binary attribute, no ambiguity)
SELECT COUNT(CASE WHEN gender = 'female' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0) AS pct_female
FROM patients

**Pattern 4: Lookup with "or" (non-analytical)**
→ Use literal OR for filtering:
Example: "Show me patients with diabetes or hypertension"
SELECT id, name, conditions
FROM patients
WHERE has_diabetes = true OR has_hypertension = true

CRITICAL RULES:
- For analytical questions with multiple conditions: Break down each condition separately
- For percentage calculations: Always use NULLIF to prevent division by zero
- For "or" in analytics context: Provide rich breakdown (A, B, A OR B, A AND B)
- For "or" in lookup context: Use WHERE condition_a OR condition_b

Focus on:
- Using the correct table relationships for joins
- Selecting appropriate columns based on the schema
- Following the data types and constraints shown
- Using the sample data as a guide for expected values
- Providing analytically rich results that enable comparison and decision-making


Example Queries:

-- findUserByCustomerNumber
SELECT u.id user_id, m.id member_id, m.customer_number member_nbr, m.birthday, m.first_name, m.last_name,
       m.email member_email, u.email user_email, u.legacy_logins_count, m.is_active active, m.membership_tier tier,
       m.club_of_access club, m.join_date, u.deleted_at is not null user_deleted
FROM users u
JOIN members m on m.id = u.member_id and m.customer_number in ( 'MBR03015511' )

-- findCheckinsByUserId
SELECT ci.id, m.customer_number, ci.checkin_at at time zone c.timezone checkin_at,
       ci.created_at at time zone c.timezone created_at,
       c.club_number, ci.month, current_timestamp - ci.checkin_at time_ago
FROM checkins ci
JOIN clubs c ON c.id = ci.club_id
JOIN members m ON m.id = ci.member_id JOIN users u ON u.member_id = m.id
WHERE ci.deleted_At IS NULL AND ci.is_dupe is not TRUE AND u.id = 1181
ORDER BY ci.checkin_at DESC LIMIT 15

-- findUserClassReservationsByUserId
SELECT ucr.reservation_id as "reserv_id", ucr.reservation_at at time zone u.timezone reservation_at,
      gx.start_time at time zone COALEASE(c.timezone, u.timezone) start_time, ucr.class_type as "type", c.name as "club",
      gx.instructors->0->>'FirstName', ucr.deleted_at, ucr.is_waitlist as "waitlist", gx.name,
      gx.class_cancelled as "cancelled"
FROM user_class_reservations ucr
JOIN users u on u.id = ucr.user_id
LEFT JOIN group_classes gx on gx.id = ucr.group_class_id join clubs c on c.id = ucr.club_id
LEFT JOIN clubs c on c.id = gx.club_id
WHERE ucr.user_id = 4049771
order by ucr.reservation_at desc limit 35;

Note that:
* We use LIMIT a lot
* Using timezone on date times
* We dont use all fields but we use those that are useful to the user

Required Steps:
1. Gathering content using lookup tools
2. Exploring possibilities
3. Understanding the context
4. validating the assumptions.

Use the functions to do all that. Like describing the enums before using them.

IMPORTANT: Your response MUST ALWAYS be executable SQL with comments (-- ...).
If we ask for the current time, return `select NOW()`.

# REQUIRED OUTPUT FORMAT

You can only speak SQL. SQL is your only language.

When you have the final query, call the `submit_sql` tool with it. That is how your answer is delivered.
The `sql` argument must contain ONLY the SQL (comments with -- are fine): no markdown fences, no prose.

Example of VALID and INVALID responses:

VALID output format (comments inside SQL only):
-- explanation of what this query does
SELECT -- format the data nicely
  LOWER(code) code
FROM ...

INVALID output format (do NOT do this):
Let me create a query for you...
```sql
SELECT ...
"""


# Markdown code fence, optionally tagged as SQL: ```sql ... ```
_FENCE_RE = re.compile(r"```[ \t]*(?:sql|postgres(?:ql)?|psql|pgsql)?[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)

# Start of a plain-text reply that is unambiguously SQL. Deliberately strict: English
# openers such as "Show me", "With the schema...", "Set up" must not be mistaken for SQL.
# Anything else is sent back to the model, which can always use the submit_sql tool.
_SQL_START_RE = re.compile(
    r"""^\s*(?:
        --|/\*|\\|\(
        |select\b|insert\s+into\b|delete\s+from\b|update\s+\S+\s+set\b
        |with\s+(?:recursive\s+)?\S+\s*(?:\(|as\b)
        |explain\b|values\s*\(|create\s|alter\s|drop\s|truncate\s
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_NOT_SQL_RETRY = (
    "Your reply was not executable SQL; it is sent to PostgreSQL as-is. "
    "Call the `submit_sql` tool with only the SQL query (SQL comments are allowed). "
    "Do not reply with prose or markdown."
)


def strip_sql_fences(text: str) -> Optional[str]:
    """Return the SQL inside a single markdown code fence, or the text itself when unfenced.

    Returns None when the text holds several fenced blocks, since there is no
    safe way to pick one.
    """
    text = text.strip()
    blocks = _FENCE_RE.findall(text)
    if len(blocks) > 1:
        return None
    if blocks:
        return blocks[0].strip()
    return text


def extract_sql(text: str) -> Optional[str]:
    """Extract executable SQL from a plain-text model reply.

    Accepts bare SQL, or prose around exactly one fenced SQL block.
    Returns None when no SQL can be identified with confidence.
    """
    candidate = strip_sql_fences(text)
    if candidate and _SQL_START_RE.match(candidate):
        return candidate
    return None


def validate_sql_output(ctx: RunContext[PgcliDeps], output: Union[SQLQuery, str]) -> Union[SQLQuery, str]:
    """Make sure the agent's final output is SQL before it reaches the database.

    Raising ModelRetry sends the message back to the model, which tries again
    within the same run instead of pgcli executing prose.
    """
    if isinstance(output, SQLQuery):
        sql = strip_sql_fences(output.sql)
        if not sql:
            raise ModelRetry("The `sql` argument must contain exactly one SQL query and nothing else.")
        return SQLQuery(sql=sql)

    sql = extract_sql(output)
    if sql is None:
        logger.debug("Rejected non-SQL LLM output: %r", output)
        raise ModelRetry(_NOT_SQL_RETRY)
    return sql


def output_to_sql(output: Union[SQLQuery, str, None]) -> Optional[str]:
    """Return the SQL string from a validated agent output."""
    if isinstance(output, SQLQuery):
        return output.sql.strip()
    return output.strip() if output else None


def create_agent(model: Any = None, config: Optional[LLMConfig] = None) -> Agent[PgcliDeps, Union[SQLQuery, str]]:
    """Create a Pydantic AI agent for SQL query generation.

    :param model: The LLM model to use (e.g., 'openai:gpt-4', TestModel() for testing)
                  If None, uses config to determine model
    :param config: LLMConfig instance. If None, loads from environment
    :return: Configured Agent instance
    """
    # If no model provided, get it from config
    if model is None:
        if config is None:
            config = load_config()
        model = config.get_model_string()

    # Configure model settings for thinking/reasoning (if supported)
    model_settings = None
    if config and isinstance(model, str):
        # Enable thinking for Anthropic Claude Sonnet 4+
        if config.provider == "anthropic" and ("claude-sonnet-4" in config.model or "claude-4" in config.model):
            from pydantic_ai.models.anthropic import AnthropicModelSettings
            model_settings = AnthropicModelSettings(
                anthropic_thinking={'type': 'enabled', 'budget_tokens': 2048}
            )

        # Enable reasoning for OpenAI reasoning models (gpt-5, o3, o4)
        # IMPORTANT: Use OpenAIResponsesModel for these models to expose ThinkingPart objects
        # Note: o1-series models are NOT supported by Responses API
        elif config.provider == "openai":
            reasoning_models = ['gpt-5', 'o3', 'o4']
            if any(rm in config.model.lower() for rm in reasoning_models):
                from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

                # Create model instance (not string!)
                model = OpenAIResponsesModel(config.model)

                # Configure reasoning settings
                model_settings = OpenAIResponsesModelSettings(
                    openai_reasoning_effort='minimal',
                    openai_reasoning_summary='detailed'  # Required to expose thinking
                )

    agent = Agent(
        model,
        deps_type=PgcliDeps,
        # The submit_sql tool is the intended way to answer. Plain text stays allowed
        # because Anthropic rejects forced tool use while extended thinking is on;
        # validate_sql_output keeps prose from reaching the database either way.
        output_type=[
            ToolOutput(SQLQuery, name="submit_sql", description="Submit the final SQL query to execute."),
            str,
        ],
        output_retries=3,
        system_prompt=SYSTEM_PROMPT,
        model_settings=model_settings,
    )
    agent.output_validator(validate_sql_output)

    # Register all 7 database introspection tools
    agent.tool(get_tables)
    agent.tool(describe_tables)
    agent.tool(describe_enum)
    agent.tool(get_table_statistics)
    agent.tool(get_indexes)
    agent.tool(get_constraints)
    agent.tool(get_foreign_keys)

    return agent


def llm_append_output(role: str, output: str) -> None:
    """Legacy compatibility function - no-op for now.

    Message history is now handled via generate_sql's message_history parameter.
    This function exists for backward compatibility during migration.

    :param role: Message role (user/assistant/tool)
    :param output: Message content
    """
    # No-op: Message history now managed by Pydantic AI
    pass


def add_sql_result_to_history(
    pgcli_instance: Any,
    sql_query: str,
    result: str,
    config: Optional[LLMConfig] = None,
    execution_time: Optional[float] = None,
    total_time: Optional[float] = None,
    explain_output: Optional[str] = None,
) -> None:
    """Add SQL query result to message history with intelligent trimming.

    :param pgcli_instance: The pgcli instance with llm_message_history
    :param sql_query: The SQL query that was executed
    :param result: The result from executing the query
    :param config: LLMConfig instance (optional)
    :param execution_time: Query execution time in seconds (optional)
    :param total_time: Total time including formatting in seconds (optional)
    :param explain_output: Parsed EXPLAIN ANALYZE output (optional)
    """
    if not hasattr(pgcli_instance, 'llm_message_history'):
        return

    # Load config if not provided
    if config is None:
        config = load_config()

    # Import Pydantic AI message types
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        UserPromptPart,
        TextPart,
    )

    # Format result for LLM (experimental: CSV format for token efficiency)
    from .formatters import get_result_formatter
    formatter = get_result_formatter()
    formatted_result = formatter(result)

    # Get token counter for trimming
    counter = get_token_counter(config.provider, config.model)
    max_result_tokens = config.get_context_limit() // 10  # Use 10% of context for results

    # Trim result if too large
    from .trimming import trim_sql_result

    original_tokens = counter.count_tokens(formatted_result)
    trimmed_result, was_trimmed = trim_sql_result(formatted_result, max_result_tokens, counter)
    trimmed_tokens = counter.count_tokens(trimmed_result)

    # Debug output showing what's being added to history
    import os
    format_type = os.environ.get('PGCLI_LLM_RESULT_FORMAT', 'tabular')

    if was_trimmed:
        logger.debug(
            "SQL result added to history: %d tokens → %d tokens (trimmed, format=%s)",
            original_tokens, trimmed_tokens, format_type
        )
    else:
        logger.debug(
            "SQL result added to history: %d tokens (not trimmed, format=%s)",
            original_tokens, format_type
        )

    # Create message parts for the SQL execution
    # Add as user message showing what was executed and the result
    result_message = f"SQL Query Executed:\n{sql_query}\n\nResult:\n{trimmed_result}"

    # Add timing information if available
    if execution_time is not None or total_time is not None:
        timing_info = "\n\nPerformance:"
        if total_time is not None:
            timing_info += f"\nTotal Time: {total_time:.3f}s"
        if execution_time is not None:
            timing_info += f"\nExecution Time: {execution_time:.3f}s"
        result_message += timing_info

    # Add EXPLAIN ANALYZE output if available (when both LLM and EXPLAIN modes are on)
    if explain_output:
        result_message += f"\n\n{explain_output}"

    # Append to history
    pgcli_instance.llm_message_history.append(
        ModelRequest(parts=[UserPromptPart(content=result_message)])
    )


async def generate_sql_streaming(
    pgcli_instance: Any,
    user_input: str,
    message_history: Optional[List] = None,
    config: Optional[LLMConfig] = None,
) -> str:
    """Generate SQL query from natural language using Pydantic AI with streaming.

    Provides visual feedback for thinking, tool calling, and SQL generation phases.

    :param pgcli_instance: The pgcli instance (for database introspection)
    :param user_input: Natural language query request
    :param message_history: Previous conversation messages (optional)
    :param config: LLMConfig instance (optional, loads from env if None)
    :return: SQL query string
    """
    import click
    from pydantic_ai.messages import (
        PartStartEvent,
        PartDeltaEvent,
        ThinkingPart,
        ThinkingPartDelta,
    )

    # Load config if not provided
    if config is None:
        config = load_config()

    # Check if LLM is configured
    missing = config.get_missing_config()
    if missing:
        raise Exception(f"LLM not configured: {missing}")

    # Create agent
    agent = create_agent(config=config)

    # Create dependencies
    deps = PgcliDeps(pgcli=pgcli_instance)

    # Use provided message_history or get from pgcli instance
    if message_history is None and hasattr(pgcli_instance, 'llm_message_history'):
        message_history = pgcli_instance.llm_message_history

    # Trim message history to fit within context limit
    if message_history:
        max_tokens = config.get_context_limit()
        counter = get_token_counter(config.provider, config.model)

        tokens_before = 0
        for msg in message_history:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'content'):
                        tokens_before += counter.count_tokens(str(part.content))

        messages_before = len(message_history)
        message_history = trim_message_history(message_history, max_tokens, counter)
        messages_after = len(message_history)

        tokens_after = 0
        for msg in message_history:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'content'):
                        tokens_after += counter.count_tokens(str(part.content))

        if messages_before != messages_after:
            logger.debug(
                "History trimmed: %d msgs (%d tokens) → %d msgs (%d tokens) | Limit: %d tokens",
                messages_before, tokens_before, messages_after, tokens_after, max_tokens
            )

    def show_status(message: str, color: str = "cyan"):
        """Show status message that overwrites the previous one."""
        # Clear line and show new status (without newline)
        click.echo("\r\033[K", nl=False)  # \r = return to start, \033[K = clear to end
        click.secho(message, fg=color, dim=True, nl=False)

    # Log what we're sending to the LLM
    logger.debug("=== Sending to LLM ===")
    logger.debug("User input: %s", user_input)
    logger.debug("Message history: %d messages", len(message_history) if message_history else 0)

    if message_history and len(message_history) > 0:
        # Log the full history sent to the model
        for idx, msg in enumerate(message_history):
            if hasattr(msg, 'parts'):
                for part_idx, part in enumerate(msg.parts):
                    if hasattr(part, 'content'):
                        logger.debug("Message %d (type: %s, part %d: %s): %s",
                                   idx,
                                   type(msg).__name__,
                                   part_idx,
                                   type(part).__name__,
                                   str(part.content))

    try:
        thinking_shown = False
        generating_shown = False

        show_status("Analyzing request...", "cyan")

        # Use iter() to get real-time node streaming
        async with agent.iter(
            user_input,
            deps=deps,
            message_history=message_history,
        ) as run:
            async for node in run:
                # Check if this is a model request node (has streaming responses)
                if agent.is_model_request_node(node):
                    # Stream the model's response to detect thinking
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            # Detect thinking part start
                            if isinstance(event, PartStartEvent):
                                if isinstance(event.part, ThinkingPart) and not thinking_shown:
                                    show_status("Thinking...", "magenta")
                                    thinking_shown = True

                            # Detect thinking deltas (incremental updates)
                            elif isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, ThinkingPartDelta) and not thinking_shown:
                                    show_status("Thinking...", "magenta")
                                    thinking_shown = True

            # Show "Generating SQL..." before getting result
            if not generating_shown:
                show_status("Generating SQL...", "cyan")
                generating_shown = True

            # Get final result
            result = run.result
            sql_query = output_to_sql(result.output)

            # Clear the status line before returning (so final SQL output is clean)
            click.echo("\r\033[K", nl=False)

            # Update message history on pgcli instance
            if hasattr(pgcli_instance, 'llm_message_history'):
                pgcli_instance.llm_message_history = result.all_messages()

            return sql_query

    except Exception as e:
        raise Exception(f"Error generating SQL: {e}")


def generate_sql(
    pgcli_instance: Any,
    user_input: str,
    message_history: Optional[List] = None,
    config: Optional[LLMConfig] = None,
) -> str:
    """Generate SQL query from natural language using Pydantic AI.

    This is the main public API matching the signature of llm_api.generate_sql.

    :param pgcli_instance: The pgcli instance (for database introspection)
    :param user_input: Natural language query request
    :param message_history: Previous conversation messages (optional, not used yet)
    :param config: LLMConfig instance (optional, loads from env if None)
    :return: SQL query string (for backward compatibility with llm_api)
             Returns None if generation fails
    """
    # Use streaming version with visual feedback
    import asyncio

    try:
        # Run the async streaming version in a synchronous context
        return asyncio.run(generate_sql_streaming(
            pgcli_instance,
            user_input,
            message_history,
            config
        ))
    except Exception as e:
        # Raise exception - caller will handle display
        raise Exception(f"Error generating SQL: {e}")


def parse_explain_output(output_text: str) -> Optional[str]:
    """Parse EXPLAIN ANALYZE JSON output and extract key metrics for LLM context.

    :param output_text: Raw output from EXPLAIN ANALYZE query
    :return: Formatted string with key metrics, or None if not EXPLAIN output
    """
    import json
    import re

    try:
        # EXPLAIN (FORMAT JSON) output is a JSON array
        # The output might have table formatting, so we need to extract the JSON

        # Remove table formatting characters (+ at end of lines from pgcli output)
        cleaned_output = re.sub(r'\+$', '', output_text, flags=re.MULTILINE)

        # Try to find JSON in the output
        json_match = re.search(r'\[[\s\S]*?\]', cleaned_output)
        if not json_match:
            return None

        explain_data = json.loads(json_match.group(0))

        if not isinstance(explain_data, list) or len(explain_data) == 0:
            return None

        plan_data = explain_data[0]

        # Extract key metrics
        summary_parts = []

        # Planning and Execution Times
        if 'Planning Time' in plan_data:
            summary_parts.append(f"Planning Time: {plan_data['Planning Time']:.3f} ms")
        if 'Execution Time' in plan_data:
            summary_parts.append(f"Execution Time: {plan_data['Execution Time']:.3f} ms")

        # Plan details
        if 'Plan' in plan_data:
            plan = plan_data['Plan']

            # Node type (Seq Scan, Index Scan, etc.)
            if 'Node Type' in plan:
                node_type = plan['Node Type']
                relation_name = plan.get('Relation Name', '')
                if relation_name:
                    summary_parts.append(f"Primary Operation: {node_type} on {relation_name}")
                else:
                    summary_parts.append(f"Primary Operation: {node_type}")

            # Cost estimates
            if 'Total Cost' in plan:
                summary_parts.append(f"Total Cost: {plan['Total Cost']:.2f}")

            # Actual rows vs estimated
            if 'Actual Rows' in plan and 'Plan Rows' in plan:
                actual = plan['Actual Rows']
                estimated = plan['Plan Rows']
                if estimated > 0:
                    accuracy = (actual / estimated * 100) if estimated != 0 else 0
                    summary_parts.append(f"Rows: {actual} actual vs {estimated} estimated ({accuracy:.0f}% accuracy)")

            # Check for sequential scans (potential optimization opportunity)
            if 'Node Type' in plan and plan['Node Type'] == 'Seq Scan':
                summary_parts.append("⚠️  Sequential scan detected - consider adding an index")

        if summary_parts:
            return "Query Plan Analysis:\n" + "\n".join(f"  • {part}" for part in summary_parts)

        return None

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.debug(f"Could not parse EXPLAIN output: {e}")
        return None
