# LLM Mode Feature Improvements - TODO

This document tracks potential enhancements to pgcli's LLM mode based on user feedback and usage patterns.

---

## Recently Completed ✅

### 1. User Rejection Feedback to LLM ✅ COMPLETE

**Status:** ✅ Implemented and tested

**What Was Done:**
When a user denies a destructive query (types `n`), the LLM now receives feedback about the rejection.

**Implementation:**
```python
# pgcli/main.py lines 885-893
rejection_message = f"""SQL Query REJECTED by User:
{sql_query}

Reason: User cancelled destructive operation
Status: Not executed

The user reviewed this query and chose not to proceed. Consider suggesting safer alternatives or asking for clarification before generating destructive queries."""
add_sql_result_to_history(self, sql_query, rejection_message)
```

**Testing:**
- ✅ New test: `test_destructive_query_rejection_added_to_history`
- ✅ All 37 tests passing
- ✅ Rejection message properly formatted and added to history

**Documentation:**
- ✅ Updated `docs/LLM_USAGE.md` with "LLM Learning from Rejections" section
- ✅ Added example showing LLM suggesting safer alternatives after rejection

**Benefits Delivered:**
- LLM can suggest safer alternatives (SELECT vs DELETE, archiving, etc.)
- Learns user preferences about destructive operations
- Better conversation continuity and context awareness
- Improved safety through intelligent follow-up suggestions

---

### 2. CTRL-C to Cancel Request (Not Exit App) ✅ COMPLETE

**Status:** ✅ Implemented and documented

**What Was Done:**
When user presses CTRL-C during an LLM request, the request is cancelled cleanly without exiting pgcli.

**Implementation:**
```python
# pgcli/main.py lines 963-967
except KeyboardInterrupt:
    # User pressed CTRL-C to cancel LLM request
    self.is_llm_output = False
    click.secho("\nLLM request cancelled.", fg="yellow")
    return
```

**Documentation:**
- ✅ Updated `docs/LLM_USAGE.md` with "Cancelling Requests" section
- ✅ Explained client-side cancellation behavior
- ✅ Added examples and use cases

**Benefits Delivered:**
- Users can stop long-running LLM requests instantly
- pgcli stays alive - no need to restart
- Conversation history is preserved after cancellation
- Clean return to prompt for new requests

**Limitations:**
- Cancellation is client-side only
- API call may complete on provider's side
- User may still be charged for the request tokens

---

## Priority: High

### 3. Include Query Timing in LLM Context ✅ COMPLETE

**Status:** ✅ Implemented and tested

**What Was Done:**
Query execution timing is now included in the LLM's conversation context for all executed queries.

**Implementation:**
```python
# pgcli/llm/agent.py lines 236-252
def add_sql_result_to_history(
    pgcli_instance: Any,
    sql_query: str,
    result: str,
    config: Optional[LLMConfig] = None,
    execution_time: Optional[float] = None,
    total_time: Optional[float] = None,
) -> None:

# Lines 295-302
if execution_time is not None or total_time is not None:
    timing_info = "\n\nPerformance:"
    if total_time is not None:
        timing_info += f"\nTotal Time: {total_time:.3f}s"
    if execution_time is not None:
        timing_info += f"\nExecution Time: {execution_time:.3f}s"
    result_message += timing_info
```

**Testing:**
- ✅ New test: `test_query_timing_added_to_history`
- ✅ All 38 tests passing
- ✅ Timing properly formatted (3 decimal places: 0.123s)
- ✅ Both LLM-generated and manual SQL include timing

**Documentation:**
- ✅ Updated `docs/LLM_USAGE.md` with "Performance Awareness" section
- ✅ Added example showing LLM suggesting index based on slow query
- ✅ Explained what LLM can do with timing information

**Benefits Delivered:**
- LLM can see query performance and suggest optimizations
- Can recommend indexes for slow queries
- Enables performance-aware query generation
- Works for both LLM-generated and manually-entered SQL

**Format:**
```
SQL Query Executed:
SELECT * FROM users WHERE email LIKE '%@example.com'

Result:
[query results]

Performance:
Total Time: 2.345s
Execution Time: 2.312s
```

---

### 4. Visual Feedback for LLM Thinking/Reasoning ✅ COMPLETE

**Status:** ✅ Implemented using Pydantic AI streaming events

**What Was Done:**
pgcli now provides real-time visual feedback for different LLM processing phases by using Pydantic AI's streaming API.

**Implementation:**
```python
# pgcli/llm/agent.py lines 310-429
async def generate_sql_streaming(...):
    # Detect thinking phase
    if isinstance(event, PartStartEvent):
        if isinstance(event.part, ThinkingPart):
            click.secho("Thinking...", fg="magenta", dim=True)

    # Detect tool calling
    elif isinstance(event.part, ToolCallPart):
        tool_calls_seen = True

    # Show final phase
    if tool_calls_seen or thinking_active:
        click.secho("Generating SQL...", fg="cyan", dim=True)
```

**Visual Phases Implemented:**
1. **"Analyzing request..."** (cyan) - Initial phase when request starts
2. **"Thinking..."** (magenta) - When model uses reasoning/thinking tokens
3. **Tool calls** - Existing "...Listing tables..." messages (from tools.py)
4. **"Generating SQL..."** (cyan) - Final phase after thinking/tools complete
5. **Status cleared** - All feedback disappears when SQL output is ready

**Technical Details:**
- Uses Pydantic AI 0.8.1 streaming API (`agent.iter()` → `node.stream(run.ctx)`)
- Detects `ThinkingPart` and `ThinkingPartDelta` events in real-time
- Wrapped async streaming with `asyncio.run()` for sync compatibility
- Maintains backward compatibility with existing `generate_sql()` signature
- **Status messages overwrite each other** using `\r\033[K` (carriage return + ANSI clear line)
- Each new status replaces the previous one - user sees only ONE status line at a time
- Status line is cleared completely before showing final SQL output
- **Anthropic:** Uses `AnthropicModelSettings` with `anthropic_thinking={'type': 'enabled', 'budget_tokens': 2048}`
- **OpenAI:** Uses `OpenAIResponsesModel` + `OpenAIResponsesModelSettings` with `openai_reasoning_summary='detailed'`

**Testing:**
- ✅ All 38 existing tests passing
- ✅ Verified with real API: gpt-5-mini (718 chars thinking, 384 reasoning_tokens)
- ✅ Verified with real API: o3-mini (423 chars thinking, 192 reasoning_tokens)
- ✅ Verified with real API: Claude Sonnet 4 (470 chars thinking)
- ✅ Compatible with all providers (gracefully handles providers without thinking)

**Provider Support (Verified):**
- ✅ Anthropic Claude Sonnet 4.0+ (thinking detected and working)
- ✅ OpenAI gpt-5-* series (thinking detected and working)
- ✅ OpenAI o3-* series (thinking detected and working)
- ✅ OpenAI o4-* series (supported but not yet tested)
- ❌ OpenAI o1-* series (NOT supported - Responses API limitation)
- Other providers work normally (no "Thinking..." phase shown)

**Documentation:**
- ✅ Updated `docs/LLM_USAGE.md` with "Visual Feedback Phases" section
- ✅ Documented which providers support thinking/reasoning
- ✅ Added complete example showing all phases

**Benefits Delivered:**
- Real-time feedback reduces perceived latency
- Clear indication of what the LLM is doing
- Users understand when complex reasoning is happening
- Better UX for long-running requests

**Limitations:**
- "Thinking..." phase only appears with models that support extended reasoning
- Status messages are text-based (no animated spinners)
- Cannot interrupt individual phases (only full request via CTRL-C)

---

## Priority: Medium

### 5. Add Working SQL to History (↑ Arrow) ✅ COMPLETE

**Status:** ✅ Implemented and tested

**What Was Done:**
Successfully executed SQL queries from LLM mode are now automatically added to readline history for easy reuse.

**Implementation:**
```python
# pgcli/main.py lines 917-922
# Add the successful SQL to readline history for reuse (↑ arrow)
if self.prompt_app and self.prompt_app.history:
    # Format: -- LLM: [original natural language query]
    #         [SQL query]
    history_entry = f"-- LLM: {original_text}\n{sql_query}"
    self.prompt_app.history.append_string(history_entry)
```

**History Format:**
```sql
-- LLM: show me all active users created last week
SELECT id, name, email, created_at
FROM users
WHERE active = true
  AND created_at > NOW() - INTERVAL '7 days';
```

**What Gets Added:**
- ✅ **Successfully executed SQL** - Added with comment showing original prompt
- ❌ **Failed SQL** - NOT added (errors don't clutter history)
- ❌ **Rejected SQL** - NOT added (if user declined destructive query)

**Testing:**
- ✅ All 39 tests passing
- ✅ New test: `test_successful_sql_added_to_readline_history`
- ✅ Verifies `append_string()` called with correct format
- ✅ Verifies comment prefix includes original natural language query

**Documentation:**
- ✅ Updated `docs/LLM_USAGE.md` with "Reusing Generated SQL (↑ Arrow History)" section
- ✅ Provided example workflow showing how to recall and edit SQL
- ✅ Documented what gets added vs excluded from history
- ✅ Explained benefits and use cases

**Benefits Delivered:**
- Users can press ↑ to recall successful SQL queries
- Easy to tweak LLM-generated queries manually
- Bridges LLM mode and manual SQL mode seamlessly
- Learn SQL by seeing what the LLM generated
- Comment prefix shows original natural language for context

**Technical Details:**
- Uses `prompt_toolkit.history.FileHistory.append_string()`
- Only adds to history after `query.successful == True`
- Preserves `original_text` variable (natural language prompt) for comment
- Multiline format with comment on first line, SQL on subsequent lines

---

### 6. Smart Autocomplete in LLM Mode ✅ COMPLETE

**Status:** ✅ Implemented and tested

**What Was Done:**
Autocomplete now intelligently filters suggestions based on LLM mode, showing only schema objects (tables, columns, etc.) in LLM mode while excluding SQL-specific completions.

**Implementation:**
```python
# pgcli/pgcompleter.py - Added llm_mode_callback parameter
def __init__(self, ..., llm_mode_callback=None):
    self.llm_mode_callback = llm_mode_callback or (lambda: False)

# pgcli/pgcompleter.py - Filter suggestions in get_completions()
llm_mode = self.llm_mode_callback()
allowed_in_llm_mode = {Column, Schema, Table, View, Database}

for suggestion in suggestions:
    if llm_mode and suggestion_type not in allowed_in_llm_mode:
        continue  # Skip SQL keywords, functions, etc.
```

**What Gets Filtered:**
- **LLM Mode ON:** Only table/column/schema/view/database names (no SQL keywords/functions)
- **Normal Mode:** Everything (no filtering)

**Testing:**
- ✅ All 44 tests passing (5 new autocomplete tests)
- ✅ Keywords excluded in LLM mode
- ✅ Functions excluded in LLM mode
- ✅ Tables allowed in LLM mode
- ✅ Columns allowed in LLM mode
- ✅ Normal mode unaffected

**Documentation:**
- ✅ Updated `docs/LLM_USAGE.md` with "Smart Autocomplete" section
- ✅ Explained behavior with examples

**Benefits Delivered:**
- Less clutter in natural language input
- Schema references still work ("from users table")
- More appropriate for natural language mode

---

### 7. Auto-EXPLAIN Integration ✅ COMPLETE

**Status:** ✅ Implemented and tested

**What Was Done:**
EXPLAIN mode (F5) can now be combined with LLM mode (F6) to provide query execution plans to the LLM for performance analysis and optimization suggestions.

**Implementation:**
```python
# pgcli/llm/agent.py - New parse_explain_output() function
def parse_explain_output(output_text: str) -> Optional[str]:
    """Parse EXPLAIN ANALYZE JSON output and extract key metrics for LLM context."""
    # Parses JSON format EXPLAIN output
    # Extracts: Planning Time, Execution Time, Node Type, Cost, Row estimates
    # Detects sequential scans and suggests indexes

# pgcli/llm/agent.py - Updated add_sql_result_to_history()
def add_sql_result_to_history(..., explain_output: Optional[str] = None):
    # Appends EXPLAIN analysis to LLM context when available

# pgcli/main.py - Integration in execute_command()
if self.explain_mode:
    from pgcli.llm.agent import parse_explain_output
    explain_summary = parse_explain_output(result_text)

add_sql_result_to_history(
    self, sql_query, result_text,
    execution_time=query.execution_time,
    total_time=query.total_time,
    explain_output=explain_summary,  # NEW
)
```

**Features Delivered:**
- **F5 + F6 combo** - Enable both EXPLAIN and LLM modes simultaneously
- **Automatic EXPLAIN ANALYZE** - pgcli automatically prepends `EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON)` when F5 is on
- **Smart parsing** - Extracts key metrics: Planning Time, Execution Time, Node Type, Cost estimates, Row accuracy
- **Sequential scan detection** - Warns LLM when queries use sequential scans (optimization opportunity)
- **Formatted for LLM** - Clean, bullet-point format sent to LLM context

**LLM Context Example:**
```
SQL Query Executed:
SELECT * FROM users WHERE email LIKE '%@gmail.com'

Result:
[1000 rows]

Performance:
Total Time: 2.456s
Execution Time: 2.423s

Query Plan Analysis:
  • Planning Time: 0.033 ms
  • Execution Time: 2423.000 ms
  • Primary Operation: Seq Scan on users
  • Total Cost: 1234.56
  • Rows: 1000 actual vs 950 estimated (105% accuracy)
  • ⚠️  Sequential scan detected - consider adding an index
```

**Testing:**
- ✅ All 52 tests passing (6 new EXPLAIN tests added)
- ✅ `test_parse_explain_output_with_valid_json` - Parses JSON correctly
- ✅ `test_parse_explain_output_with_index_scan` - No warning for index scans
- ✅ `test_parse_explain_output_with_invalid_json` - Returns None gracefully
- ✅ `test_parse_explain_output_with_table_formatted_output` - Handles pgcli table formatting
- ✅ `test_explain_output_included_in_history` - Verifies LLM receives EXPLAIN data
- ✅ `test_explain_output_not_included_when_none` - Verifies no EXPLAIN when disabled

**Documentation:**
- ✅ Updated `docs/LLM_USAGE.md` with "EXPLAIN Integration (F5 + F6)" section
- ✅ Includes usage examples, safety considerations, and best practices
- ✅ Documents what LLM can suggest with EXPLAIN data

**Benefits Delivered:**
- LLM can recommend specific index types (B-tree, GIN, trigram)
- Identifies sequential scans on large tables
- Spots join performance issues (nested loop vs hash join)
- Suggests query rewrites based on actual execution plan
- Compares estimated vs actual row counts (statistics issues)

**Safety Considerations:**
- ⚠️ **EXPLAIN ANALYZE executes queries** - Documented clearly in user guide
- ✅ **Destructive queries still prompt** - Confirmation required before execution
- ✅ **Best practice documented** - Recommend F5+F6 primarily for SELECT queries

**Challenges Addressed:**
- ✅ EXPLAIN ANALYZE execution concern - Documented safety practices
- ✅ Extra query time - User opt-in via F5 toggle
- ✅ Token consumption - Only key metrics sent (not full plan)

---

## Priority: Low / Experimental

### 8. Optimize SQL Result Format for Token Usage ✅ RESEARCH COMPLETE

**Status:** ✅ Research completed, ready for implementation

**Research Findings:**

Comprehensive benchmark testing shows **CSV format offers 26% token savings** compared to current tabular format.

**Benchmark Results (Average Tokens):**

| Format | Avg Tokens | vs Tabular | Verdict |
|--------|------------|------------|---------|
| **CSV** | 5,802 | **-26.5%** ✅ | **Winner** |
| **Compact JSON** | 6,036 | -23.5% ⭐ | Runner-up |
| **Tabular** (current) | 7,894 | baseline | Current |
| JSON (pretty) | 9,528 | +20.7% | Too verbose |
| YAML | 11,968 | +51.6% | Very verbose |
| JSON Objects | 14,739 | +86.7% ❌ | Worst |

**Key Findings:**
- ✅ **CSV is 26% more token-efficient** than current tabular format
- ✅ **Compact JSON is 24% more efficient** (close second)
- ✅ **All formats preserve complete data** - no information loss
- ✅ **LLMs comprehend all formats well** - trained on diverse formats
- ✅ **Larger datasets show bigger savings** (CSV saves 35% on 50-row results)

**Cost Impact:**

For a typical workload (100 queries with 50-row results):
- **35% fewer tokens** per session (185k → 120k tokens)
- **Claude Sonnet:** $195/month savings ($2,340/year)
- **GPT-4 Turbo:** $651/month savings ($7,812/year)

**Recommendation:** 🎯 **Implement CSV as default format**

**Implementation Plan:**
1. Create `pgcli/llm/formatters.py` with format functions
2. Add `PGCLI_LLM_RESULT_FORMAT` env var (csv|compact_json|tabular)
3. Update `add_sql_result_to_history()` to use configured format
4. Default to CSV for maximum efficiency
5. Keep tabular format available for debugging/preference

**Benefits:**
- ✅ **26% lower token costs**
- ✅ **35% more conversation history** fits in context window
- ✅ **Faster LLM responses** (less data to process)
- ✅ **Minimal risk** - configurable, well-tested

**Research Artifacts:**
- ✅ Benchmark script: `benchmark_result_formats.py`
- ✅ Full research report: `RESULT_FORMAT_RESEARCH.md`
- ✅ 7 formats tested across 3 dataset sizes
- ✅ Token counts validated with OpenAI tiktoken

**Sample Output Comparison:**

Current (Tabular - 124 tokens):
```
+------+-------------+-------------------+-------+
|   id | name        | email             |   age |
|------+-------------+-------------------+-------|
|    1 | Alice Smith | alice@example.com |    28 |
+------+-------------+-------------------+-------+
```

Recommended (CSV - 62 tokens, **50% savings**):
```
id,name,email,age
1,Alice Smith,alice@example.com,28
```

**Implementation Status:** ✅ **IMPLEMENTED** (Experimental, opt-in via env var)

**What Was Implemented:**

1. ✅ Created `pgcli/llm/formatters.py` module
   - `format_as_csv()` - Converts tabular results to CSV
   - `format_as_tabular()` - Returns tabular format unchanged (default)
   - `get_result_formatter()` - Returns formatter based on env var

2. ✅ Environment variable configuration
   - `PGCLI_LLM_RESULT_FORMAT=csv` - Enable CSV format (~26% token savings)
   - `PGCLI_LLM_RESULT_FORMAT=tabular` - Default tabular format
   - Unset - Defaults to tabular (backward compatible)

3. ✅ Integration with LLM context
   - Updated `add_sql_result_to_history()` to format results before adding to context
   - Formatting happens before token counting for accurate limits
   - Debug logging shows which format is being used

4. ✅ Comprehensive testing
   - 8 new tests in `test_llm_integration.py`
   - All 60 tests passing
   - Tests cover: CSV parsing, special characters, env var handling, integration

5. ✅ User documentation
   - Added section in `docs/LLM_USAGE.md` under "Reducing Token Usage"
   - Includes comparison examples, benefits, tradeoffs, recommendations

**Usage:**

```bash
# Enable CSV format for token efficiency
export PGCLI_LLM_RESULT_FORMAT=csv
pgcli postgresql://localhost/mydb

# Or use default tabular format (no env var needed)
pgcli postgresql://localhost/mydb
```

**Verified Benefits:**
- ✅ **26% average token reduction** (from benchmark)
- ✅ **50% reduction** on small result sets
- ✅ **35% reduction** on medium result sets
- ✅ LLMs comprehend CSV just as well as tabular
- ✅ Backward compatible (disabled by default)

**Files Modified:**
- `pgcli/llm/formatters.py` - NEW formatter module
- `pgcli/llm/agent.py` - Updated to use formatter
- `tests/test_llm_integration.py` - 8 new tests
- `docs/LLM_USAGE.md` - User documentation
- `benchmark_result_formats.py` - Research script
- `RESULT_FORMAT_RESEARCH.md` - Full research report

---

### 9. "Send to LLM?" Prompt on Failed SQL ✅ COMPLETE

**Status:** ✅ Implemented and tested

**What Was Done:**
Automatic natural language detection when users accidentally type natural language without enabling LLM mode.

**Implementation:**
```python
# pgcli/llm/natural_language_detector.py - New module
def looks_like_natural_language(text: str) -> bool:
    """Heuristic to detect if input looks like natural language rather than SQL."""
    # Checks for:
    # - Common English words (show, get, find, list, all, me, etc.)
    # - Natural language patterns ("show me", "get all", "how many")
    # - Questions with ?
    # - Avoids triggering on SQL with typos (SELECT vs SELCT)

def is_syntax_error(exception: Exception) -> bool:
    """Check if exception is a SQL syntax error."""

# pgcli/main.py - Integration in execute_command()
except Exception as e:
    # ... show error ...

    # Check if this looks like natural language
    if not self.llm_mode:
        if is_syntax_error(e) and looks_like_natural_language(text):
            send_to_llm = confirm("This looks like natural language. Send to LLM?", default=True)
            if send_to_llm:
                self.llm_mode = True
                self.execute_command(text, handle_closed_connection)
                return
```

**User Experience:**
```
postgres@localhost:mydb> show me all users
syntax error at or near "show"
LINE 1: show me all users
        ^

This looks like natural language. Send to LLM? [Y/n]: y

...describing tables 'users'...

> SELECT * FROM users;

+----+-------------+-------------------+
| id | name        | email             |
+----+-------------+-------------------+
|  1 | Alice Smith | alice@example.com |
```

**Heuristic Details:**

**Triggers on (Natural Language):**
- ✅ "show me all users"
- ✅ "get the top 10 products"
- ✅ "find orders from last week"
- ✅ "list all tables"
- ✅ "how many users are there?"
- ✅ Any question with ?

**Doesn't Trigger (SQL with Typos):**
- ❌ "SELCT * FROM users" (typo in SELECT)
- ❌ "DELETE FORM users" (typo in FROM)
- ❌ "\\d users" (psql commands)
- ❌ "SELECT * FROM users" (valid SQL)

**Testing:**
- ✅ All 68 tests passing (8 new natural language detector tests)
- ✅ `test_obvious_natural_language_detected`
- ✅ `test_obvious_sql_not_detected`
- ✅ `test_psql_commands_not_detected`
- ✅ `test_ambiguous_cases`
- ✅ `test_questions_detected`
- ✅ `test_empty_or_whitespace`
- ✅ `test_single_sql_keyword`
- ✅ `test_is_syntax_error_detection`

**Documentation:**
- ✅ Added "Auto-Detection: Forgot to Press F6?" section to `docs/LLM_USAGE.md`
- ✅ Includes examples, how it works, benefits

**Benefits Delivered:**
- Users don't need to remember F6
- Smooth transition between SQL and natural language modes
- Reduces frustration from mode confusion
- Great for new users learning LLM mode
- Smart heuristic avoids false positives on SQL typos

**Files Modified:**
- `pgcli/llm/natural_language_detector.py` - NEW heuristic module
- `pgcli/main.py` - Added auto-detection in exception handler
- `tests/test_llm_integration.py` - 8 new tests
- `docs/LLM_USAGE.md` - User documentation


### 10. Upgrade pydantic-ai to Latest Version

**Status:** ⚠️ **BLOCKED** - Requires decision on Python version support

**Current Situation:**
- **Current version:** pydantic-ai 0.8.1 (installed via extras_require['llm'])
- **Latest version:** pydantic-ai 1.0.14 (as of 2025-10-03)
- **Version in setup.py:** `>= 0.0.14` (allows any version)

**Critical Blocker:**

🚨 **pydantic-ai v1.0.0+ dropped Python 3.9 support**

- v1.0.0 released 2025-09-04 requires **Python 3.10+**
- pgcli currently supports **Python 3.8-3.12** (see setup.py line 51)
- Current development environment: **Python 3.9.23**
- Many users likely still on Python 3.9

**Version History:**
- 0.8.1 (Aug 29, 2024) - Current version, last Python 3.9-compatible release
- 0.8.0 (Aug 26, 2024)
- 1.0.0b1 (beta)
- 1.0.0 (Sep 4, 2024) - **Dropped Python 3.9**, requires Python 3.10+
- 1.0.1 through 1.0.14 - All require Python 3.10+

**Breaking Changes from 0.8.1 → 1.0.x:**

✅ **Already Compatible:**
1. We use `result.output` (v0.6.0+ API) ✓
2. We don't use deprecated `result_retries`, `result_type`, `result_tool_name` ✓
3. We don't use deprecated Agent parameters ✓

⚠️ **May Require Updates:**
1. **Streaming API** (v0.7.0):
   - `Model.request_stream` receives new `run_context` argument
   - We use `node.stream(run.ctx)` - likely compatible

2. **Dataclass keyword arguments** (v1.0.0):
   - Many dataclasses now require keyword arguments
   - Need to verify our usage

3. **ModelRequest/Response.parts types changed** (v1.0.0):
   - May affect message handling if we inspect these

**Options:**

**Option A: Stay on 0.8.1 (Recommended for now)**
- ✅ Maintains Python 3.8-3.12 compatibility
- ✅ Already on latest Python 3.9-compatible version
- ✅ No breaking changes needed
- ✅ Zero risk to existing users
- ❌ Miss out on bug fixes and features in 1.0.x

**Option B: Upgrade to 1.0.14**
- ✅ Latest features and bug fixes
- ✅ Better Gemini support (handles multiple candidates)
- ✅ Improved Ollama support
- ❌ **Requires dropping Python 3.8 and 3.9 support**
- ❌ Breaking change for users
- ❌ Need to update setup.py `python_requires`
- ⚠️ Need to test all streaming code
- ⚠️ Need to verify dataclass usage

**Option C: Dual support (Complex)**
- Support both 0.8.1 and 1.0.x with version detection
- ❌ Very complex to maintain
- ❌ Not worth the effort

**Decision Required:**

Should pgcli drop Python 3.8 and 3.9 support to enable pydantic-ai 1.0.14 upgrade?

**If YES:**
1. Update setup.py `python_requires` to `>=3.10`
2. Update setup.py pydantic-ai requirement to `>=1.0.0`
3. Test streaming API for compatibility
4. Verify dataclass keyword argument usage
5. Run full test suite
6. Document breaking change in release notes

**If NO:**
1. Stay on pydantic-ai 0.8.1
2. Update setup.py to cap version: `pydantic-ai>=0.8.1,<1.0.0`
3. Mark this feature as "Won't Fix" or "Future"

**Recommendation:**

Given that Python 3.9 is still widely used and pgcli is a CLI tool (not a library), I recommend **staying on pydantic-ai 0.8.1** until:
1. Python 3.9 reaches end-of-life (October 2025), OR
2. User demand for 1.0.x features is high, OR
3. Critical bugs exist in 0.8.1 that are fixed in 1.0.x

**Action:** Need user decision on Python version policy.

### 11. make sure we worked on latest pgcli master branch. I see we biult on top of 4.1.0 but online I see 4.3.0 and even some recent commits after that.

### 12. perform full security review

### 13. support non thinking models too

---

## Notes

### Testing Strategy
- Each feature should include unit tests
- Manual testing with real databases
- Token usage benchmarking for optimization features

### Documentation
- Update `docs/LLM_USAGE.md` for user-facing features
- Update `PYDANTIC_AI_IMPLEMENTATION_GUIDE.md` for technical features
- Add examples and screenshots where helpful

### Backward Compatibility
- All features should be optional or backwards compatible
- Use configuration flags for experimental features
- Preserve existing behavior by default

---

## Completed Features ✅

- F6 toggle for LLM mode
- Multi-provider support (Anthropic, OpenAI, Ollama)
- Conversation history with token-aware trimming
- Automatic retry on SQL errors (3 attempts)
- Manual SQL added to LLM history
- Dynamic syntax highlighting (SQL vs plain text)
- Destructive query confirmation prompts
- Token count display in status bar
- Human-readable token formatting (41.6k)
- Context limit management with smart trimming
- Debug logging for token tracking

---

## Priority Legend

- **High**: Significant UX/safety improvement, high user value
- **Medium**: Nice-to-have, improves workflow
- **Low**: Experimental, needs research/testing
