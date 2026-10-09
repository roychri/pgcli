# LLM Mode User Guide for pgcli

This guide shows you how to use pgcli's natural language SQL generation feature.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Understanding the Interface](#understanding-the-interface)
4. [Writing Effective Prompts](#writing-effective-prompts)
5. [Conversation History](#conversation-history)
6. [Token Management](#token-management)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Getting Started

### Prerequisites

1. **Install pgcli with LLM support:**
   ```bash
   pip install -U "pgcli[llm]"
   ```

2. **Set up your API key** (see [LLM_SETUP.md](LLM_SETUP.md)):
   ```bash
   # For Anthropic (default)
   export ANTHROPIC_API_KEY="sk-ant-..."

   # For OpenAI
   export PGCLI_LLM_PROVIDER=openai
   export OPENAI_API_KEY="sk-..."
   ```

3. **Connect to your database:**
   ```bash
   pgcli postgresql://localhost/mydb
   ```

### Quick Start

```
postgres@localhost:mydb> [Press F6]
# Status bar changes: [F6] LLM Mode: ON (0 tokens)

postgres@localhost:mydb> show me all tables
...Listing tables...
> -- List all tables in the database
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public';

+-------------+
| tablename   |
|-------------|
| users       |
| orders      |
| products    |
+-------------+
SELECT 3
Time: 0.023s
```

That's it! The LLM generates SQL and executes it automatically.

---

## Basic Usage

### Enable/Disable LLM Mode

**Press F6** to toggle LLM mode on or off.

- **LLM Mode: ON** - Type natural language queries
  - Plain text input (no SQL syntax highlighting)
  - Natural language is not highlighted as SQL keywords
- **LLM Mode: OFF** - Type SQL directly (normal mode)
  - SQL syntax highlighting active
  - Keywords, strings, and operators are colored

### Auto-Detection: Forgot to Press F6?

**pgcli automatically detects when you accidentally type natural language!**

If you forget to enable LLM mode and type a natural language query, pgcli will offer to send it to the LLM:

```
postgres@localhost:mydb> show me all users
syntax error at or near "show"
LINE 1: show me all users
        ^

This looks like natural language. Send to LLM? [Y/n]: y
```

When you press **y** (or just hit Enter for yes), pgcli will:
1. Automatically enable LLM mode
2. Retry your query with the LLM
3. Keep LLM mode enabled for your next queries

**How it works:**
- Detects SQL syntax errors in normal mode
- Uses a heuristic to identify natural language (vs typos in SQL)
- Looks for common English words: "show", "get", "find", "list", "all", "me", etc.
- Avoids false positives on actual SQL with typos

**Examples that trigger auto-detection:**
- ✅ "show me all users"
- ✅ "get the top 10 products"
- ✅ "find orders from last week"
- ✅ "list all tables"
- ✅ "how many users are there?"

**Examples that DON'T trigger (these are SQL, just with typos):**
- ❌ "SELCT * FROM users" (typo in SELECT)
- ❌ "DELETE FORM users" (typo in FROM)

**Benefits:**
- No need to remember F6
- Smooth transition between SQL and natural language
- Helpful for new users learning LLM mode

### Your First Query

```
postgres@localhost:mydb> find all users who signed up last week
```

The LLM will:
1. Call database introspection functions to understand your schema
2. Generate appropriate SQL
3. Execute the query automatically
4. Show you the results

### Multi-Turn Conversations

The LLM remembers previous queries and results:

```
# First query
postgres@localhost:mydb> find one random club from the clubs table

> SELECT * FROM clubs ORDER BY random() LIMIT 1;

-[ RECORD 1 ]-----
id           | 39
name         | Costa Mesa
...

# Second query (references previous result!)
postgres@localhost:mydb> now show me just the name for that club

> -- Use the club ID from the previous result
SELECT name FROM clubs WHERE id = 39;

+-----------+
| name      |
|-----------|
| Costa Mesa|
+-----------+
```

The LLM can see:
- Your natural language queries
- The SQL it generated
- The results from previous queries
- Your database schema (learned via function calls)

---

## Understanding the Interface

### Status Bar

The bottom status bar shows:

```
[F6] LLM Mode: ON (41.6k tokens)  [F5] Explain: OFF  Transaction
```

**What this means:**
- `[F6] LLM Mode: ON` - LLM mode is active
- `(41.6k tokens)` - Current conversation uses 41,600 tokens
- This updates in real-time as you interact

### Input Area

The input prompt changes behavior based on LLM mode:

**LLM Mode: OFF (SQL mode)**
```sql
postgres@localhost:mydb> SELECT * FROM users
```
- Words like `SELECT`, `FROM` are highlighted in green (SQL keywords)
- Strings in quotes are highlighted in red
- Special characters are colored

**LLM Mode: ON (Natural language mode)**
```
postgres@localhost:mydb> show me all users from last week
```
- No syntax highlighting (plain text)
- All text appears in default color
- Prevents confusion when typing natural language

### Smart Autocomplete

Autocomplete behavior changes based on LLM mode:

**Normal Mode (LLM OFF):**
- Full SQL autocomplete active
- Keywords: `SELECT`, `FROM`, `WHERE`, etc.
- Functions: `COUNT`, `SUM`, `AVG`, etc.
- Table names, column names, schemas

**LLM Mode (LLM ON):**
- **Only schema object completions** (tables, columns, schemas, views, databases)
- **SQL keywords excluded** (SELECT, FROM, WHERE, etc.)
- **Functions excluded** (COUNT, SUM, etc.)

**Why?** In LLM mode, you're typing natural language, not SQL. SQL keywords would be noise. But you still want to reference actual table/column names from your schema.

**Example in LLM mode:**
```
postgres@localhost:mydb> show me data from use[TAB]
# Autocomplete suggests: users, user_roles (table names)
# Does NOT suggest: USER (SQL keyword)

postgres@localhost:mydb> show me data from users ta[TAB]
# Autocomplete suggests: table names starting with "ta"
# Does NOT suggest: TABLE (SQL keyword)
```

**History:** Press ↑ arrow to recall previous queries in both modes!

### Token Display

Tokens are the "cost units" for LLM providers:
- `500` - Less than 1,000 tokens
- `41.6k` - 41,600 tokens (thousands)
- `1.5m` - 1.5 million tokens

**Context limits by provider:**
- Claude Sonnet: 180,000 tokens max
- GPT-4o-mini: 115,000 tokens max
- Ollama: ~1,800 tokens max (very small!)

### Visual Feedback Phases

pgcli provides **real-time visual feedback** that updates as the LLM processes your request. Each status message **overwrites the previous one**, showing only the current phase. When the SQL is ready, the status line clears completely.

**How It Works:**

While processing, you'll see **only one status line** at a time:

1. **Analyzing request...** - Initial phase when request starts
2. **Thinking...** - When using reasoning models (replaces previous status)
3. **...Listing tables...** or other tool messages - During schema introspection (replaces previous status)
4. **Generating SQL...** - Final phase before output (replaces previous status)
5. Status clears, final SQL appears

**Example - What You See:**

During processing (only ONE line visible, updating in place):
```
postgres@localhost:mydb> how many deleted clubs we have?
Thinking...
```

When complete (status cleared, only SQL shown):
```
postgres@localhost:mydb> how many deleted clubs we have?
> -- Count deleted clubs (where deleted_at is not null)
SELECT COUNT(*) AS deleted_clubs_count
FROM clubs
WHERE deleted_at IS NOT NULL;
```

**Reasoning Model Support:**

The **"Thinking..."** phase only appears with models that support thinking/reasoning,
such as Claude Opus/Sonnet/Haiku 5.5 and OpenAI's GPT-6 models (Luna, Sol, Astra).
Thinking is enabled at `low` effort by default and is ignored by models that can't think.
Change it with `PGCLI_LLM_THINKING` (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`,
or `default` for the model's own default).

**Complete Flow Example:**

```
# Step 1: Request submitted
postgres@localhost:mydb> show me slow queries from last week
Analyzing request...

# Step 2: Model is thinking (REPLACES "Analyzing request...")
postgres@localhost:mydb> show me slow queries from last week
Thinking...

# Step 3: Calling database tools (REPLACES "Thinking...")
postgres@localhost:mydb> show me slow queries from last week
...Listing tables...

# Step 4: Generating final SQL (REPLACES tool message)
postgres@localhost:mydb> show me slow queries from last week
Generating SQL...

# Step 5: Status cleared, SQL output shown
postgres@localhost:mydb> show me slow queries from last week
> -- Get slow queries from the last 7 days
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 1000
  AND last_exec > NOW() - INTERVAL '7 days'
ORDER BY mean_exec_time DESC
LIMIT 10;

+------------------+------------------+-------+
| query            | mean_exec_time   | calls |
|------------------+------------------+-------|
| ...              | ...              | ...   |
+------------------+------------------+-------+
```

### Cancelling Requests

**Press CTRL-C** to cancel an in-progress LLM request without exiting pgcli.

```
postgres@localhost:mydb> analyze all my users and find patterns
...Listing tables...
...describing tables 'users' in public...
^C
LLM request cancelled.

postgres@localhost:mydb>
```

**What happens when you cancel:**
- The LLM request stops immediately
- pgcli returns to the prompt (doesn't exit)
- Conversation history is preserved
- You can continue with a new request

**Note:** Due to how LLM APIs work, the cancellation happens on the client side. The API call may complete on the provider's side, but the response is discarded. You may still be charged for the request.

**When to use CTRL-C:**
- Accidentally asked the wrong question
- Request is taking too long
- Want to rephrase your query
- Need to stop and think

---

## Writing Effective Prompts

### Good Prompts ✅

**Be specific and clear:**
```
✅ "show me all users who signed up in the last 7 days"
✅ "find the top 10 products by revenue this month"
✅ "count how many orders each user has placed"
✅ "list all tables in the database"
```

**Reference previous results:**
```
✅ "now show me just the name for that club"
✅ "use the same ID from the previous query"
✅ "what are the columns in that table?"
```

**Ask for specific formats:**
```
✅ "show user email and name, sorted by creation date"
✅ "group by status and count"
✅ "return only the first 5 results"
```

### Bad Prompts ❌

**Too vague:**
```
❌ "show me data"  (which table? what columns?)
❌ "find stuff"    (too ambiguous)
❌ "query"         (not descriptive)
```

**Asking for what pgcli can't do:**
```
❌ "create a new database"  (pgcli connects to existing databases)
❌ "backup the database"    (use pg_dump instead)
❌ "optimize my queries"    (LLM generates queries, doesn't optimize)
```

**Expecting conversation about SQL:**
```
❌ "how do I write a JOIN?"  (LLM generates SQL, not tutorials)
❌ "explain window functions" (not a SQL generation task)
```

### Tips for Better Results

1. **Start simple, then refine:**
   ```
   First:  "show me users"
   Then:   "only users from last week"
   Finally: "sorted by email"
   ```

2. **Be explicit about joins:**
   ```
   ✅ "show me orders with user names"
   ❌ "show me orders and users" (ambiguous)
   ```

3. **Mention specific columns when needed:**
   ```
   ✅ "show user email, name, and created_at"
   ❌ "show user stuff" (vague)
   ```

4. **Use business terms, not implementation details:**
   ```
   ✅ "show me active subscriptions"
   ❌ "show me rows where status = 1"
   ```

---

## Conversation History

### How It Works

pgcli maintains conversation history automatically:
- Your natural language queries
- Generated SQL
- Query results (trimmed if large)
- Query performance metrics (execution time)
- Database schema information

**History is stored** until:
- You exit pgcli
- Context limit is reached (automatic trimming)

### Performance Awareness

The LLM can see query execution times, enabling performance-aware suggestions:

**Example:**
```
postgres@localhost:mydb> SELECT * FROM users WHERE email LIKE '%@example.com'

> -- Get users with example.com emails
SELECT * FROM users WHERE email LIKE '%@example.com';

+----+------------------+
| id | email            |
|----+------------------|
| 1  | alice@example.com|
| 2  | bob@example.com  |
+----+------------------+
SELECT 2
Time: 2.345s

# The LLM sees in its context:
# SQL Query Executed:
# SELECT * FROM users WHERE email LIKE '%@example.com'
#
# Result: [...]
#
# Performance:
# Total Time: 2.345s
# Execution Time: 2.312s

postgres@localhost:mydb> that was slow, can you optimize it?

...describing tables 'users' in public...
...get_indexes for 'users'...

> -- Use index on email for faster lookup
> -- Consider adding: CREATE INDEX idx_users_email ON users(email)
SELECT * FROM users WHERE email LIKE '%@example.com';
```

**What the LLM can do with timing:**
- Suggest adding indexes for slow queries
- Recommend query rewrites for better performance
- Identify missing constraints
- Warn about full table scans

### EXPLAIN Integration (F5 + F6)

**Combine EXPLAIN mode with LLM mode to get query performance insights!**

pgcli's EXPLAIN mode (F5) can be combined with LLM mode (F6) to provide the AI with detailed query execution plans. This enables the LLM to make highly informed optimization suggestions.

#### How It Works

1. **Enable both modes:**
   - Press **F5** to enable EXPLAIN mode
   - Press **F6** to enable LLM mode
   - Status bar shows both: `[F5] Explain: ON` and `[F6] LLM Mode: ON`

2. **Ask the LLM to generate a query:**
   - The query automatically gets `EXPLAIN ANALYZE` prepended
   - PostgreSQL executes the query and returns the execution plan
   - pgcli parses the plan and sends key metrics to the LLM

3. **The LLM sees:**
   - Execution time and planning time
   - Query plan type (Seq Scan, Index Scan, etc.)
   - Cost estimates vs actual rows
   - Performance warnings (e.g., sequential scans)

#### Example Usage

```
postgres@localhost:mydb> [Press F5 then F6]
# Status bar: [F5] Explain: ON  [F6] LLM Mode: ON (0 tokens)

postgres@localhost:mydb> find all users with gmail addresses

...describing tables 'users' in public...

> SELECT email, name, created_at
> FROM users
> WHERE email LIKE '%@gmail.com';

[Query executes with EXPLAIN ANALYZE automatically]

# The LLM sees in its context:
# SQL Query Executed:
# SELECT email, name, created_at FROM users WHERE email LIKE '%@gmail.com'
#
# Result: [1000 rows returned]
#
# Performance:
# Total Time: 2.456s
# Execution Time: 2.423s
#
# Query Plan Analysis:
#   • Planning Time: 0.033 ms
#   • Execution Time: 2423.000 ms
#   • Primary Operation: Seq Scan on users
#   • Total Cost: 1234.56
#   • Rows: 1000 actual vs 950 estimated (105% accuracy)
#   • ⚠️  Sequential scan detected - consider adding an index

postgres@localhost:mydb> can you optimize this?

...get_indexes for 'users'...

> -- Add index on email for pattern matching
> -- Note: For LIKE '%@gmail.com', consider using trigram index:
> -- CREATE INDEX idx_users_email_trgm ON users USING gin(email gin_trgm_ops);
>
> SELECT email, name, created_at
> FROM users
> WHERE email LIKE '%@gmail.com';
```

#### What the LLM Can Suggest

With EXPLAIN data, the LLM can:
- **Recommend specific index types** (B-tree, GIN, trigram)
- **Identify sequential scans** on large tables
- **Spot join performance issues** (nested loop vs hash join)
- **Suggest query rewrites** based on actual execution plan
- **Compare estimated vs actual row counts** (statistics issues)

#### Safety Considerations

**Important:** `EXPLAIN ANALYZE` actually executes the query! This means:
- ✅ **Safe for SELECT queries** - EXPLAIN just adds execution plan data
- ⚠️ **Caution with DML** - `EXPLAIN ANALYZE UPDATE/DELETE` will execute!
- 🛑 **Destructive queries still prompt** - You'll get confirmation before execution

**Best Practice:**
- Use F5 + F6 primarily for **SELECT queries** you want to optimize
- For INSERT/UPDATE/DELETE, use F6 alone to avoid automatic execution via EXPLAIN
- Or press F5 to disable EXPLAIN mode before asking for destructive queries

#### Disabling EXPLAIN Mode

Press **F5** again to toggle EXPLAIN mode off. The status bar will show `[F5] Explain: OFF`.

The LLM will no longer receive query plan analysis, but will still see execution timing.

### Reusing Generated SQL (↑ Arrow History)

**All successful SQL queries are automatically added to your readline history!**

After the LLM generates and executes SQL successfully, you can press **↑** (up arrow) to recall and reuse it:

**Example:**
```
# Step 1: Use LLM to generate SQL
postgres@localhost:mydb> [F6 to enable LLM mode]
postgres@localhost:mydb> show me all active users created last week

> -- Get active users from the last 7 days
SELECT id, name, email, created_at
FROM users
WHERE active = true
  AND created_at > NOW() - INTERVAL '7 days';

+----+-------+------------------+---------------------+
| id | name  | email            | created_at          |
|----+-------+------------------+---------------------|
| 42 | Alice | alice@domain.com | 2024-01-15 10:23:11 |
+----+-------+------------------+---------------------+

# Step 2: Press ↑ to recall the SQL
postgres@localhost:mydb> [Press ↑ arrow]

# You'll see:
postgres@localhost:mydb> -- LLM: show me all active users created last week
SELECT id, name, email, created_at
FROM users
WHERE active = true
  AND created_at > NOW() - INTERVAL '7 days';

# Now you can:
# - Press Enter to run it again
# - Edit it manually (maybe change the interval to '14 days')
# - Copy/paste it elsewhere
```

**What gets added to history:**
- ✅ **Successfully executed SQL** - Added with comment showing original prompt
- ❌ **Failed SQL** - NOT added (errors don't clutter your history)
- ❌ **Rejected SQL** - NOT added (if you declined a destructive query)

**History Format:**
```sql
-- LLM: [your original natural language query]
[generated SQL query]
```

**Benefits:**
- **Reuse working queries** - No need to regenerate
- **Tweak and customize** - Edit the SQL manually for variations
- **Learn SQL** - See what the LLM generated
- **Bridge LLM and manual modes** - Seamlessly switch between natural language and SQL

**Pro Tip:** Press ↑ multiple times to browse through all your previous queries (both manual SQL and LLM-generated).

### Context Limits

Each provider has a maximum context window:

| Provider | Total Limit | Usable (90%) | Per Result |
|----------|-------------|--------------|------------|
| Claude | 200K tokens | 180K tokens | 18K tokens |
| GPT-4o-mini | 128K tokens | 115K tokens | 11.5K tokens |
| Ollama | 2-8K tokens | 1.8-7.2K tokens | 180-720 tokens |

### Automatic Trimming

When your conversation exceeds 90% of the context limit:
1. **Oldest messages are removed**
2. **Last 10-20 exchanges are kept**
3. **System messages are preserved**

**You won't notice trimming** - it happens automatically.

**Debug logs show trimming** (if logging enabled):
```
DEBUG: History trimmed: 30 msgs (185000 tokens) → 22 msgs (160000 tokens)
```

### Large Result Handling

If a SQL query returns massive results:
- **< 18K tokens** (Claude): Full results kept
- **> 18K tokens**: Results are trimmed using smart strategies:
  - **Tabular data**: Keep first 50 + last 50 rows, show "... [rows omitted] ..."
  - **Long text**: Truncate with "... [content truncated, N tokens total]"

**Example of trimmed result:**
```
id | name
1  | Alice
2  | Bob
...
... [150 rows omitted, 25000 tokens total] ...
...
199 | Zara
200 | Zeke
```

---

## Token Management

### Understanding Token Costs

Tokens are charged by providers:
- **Input tokens**: Your prompts + history + database schema
- **Output tokens**: Generated SQL + explanations

**Approximate costs:**
- Claude Sonnet: $3/$15 per million tokens (in/out)
- GPT-4o-mini: $0.15/$0.60 per million tokens (in/out)
- Ollama: Free (local)

### Monitoring Token Usage

Watch the status bar:
```
[F6] LLM Mode: ON (41.6k tokens)
```

This shows **current conversation tokens** (input only).

### Reducing Token Usage

**1. Keep conversations focused:**
```bash
# Exit and restart pgcli for new conversations
exit
pgcli
```

**2. Avoid querying huge result sets:**
```
❌ "show me all users"  (could be millions of rows!)
✅ "show me first 10 users"
✅ "count users"
```

**3. Be specific to avoid re-querying:**
```
❌ "show data"  (LLM has to guess, calls many functions)
✅ "show users table"  (LLM knows exactly what to do)
```

**4. Use limits:**
```
✅ "show me 5 examples of..."
✅ "limit to 10 rows"
```

**5. Enable CSV format for results (Experimental):**

By default, SQL results are sent to the LLM in tabular format (ASCII tables with borders). You can enable CSV format for ~26% token savings:

```bash
export PGCLI_LLM_RESULT_FORMAT=csv
```

**Formats Available:**
- `tabular` (default) - ASCII tables with borders, more readable
- `csv` - Comma-separated values, **26% fewer tokens**

**Example comparison:**

Tabular format (124 tokens):
```
+------+-------------+-------------------+-------+
|   id | name        | email             |   age |
|------+-------------+-------------------+-------|
|    1 | Alice Smith | alice@example.com |    28 |
|    2 | Bob Johnson | bob@example.com   |    35 |
+------+-------------+-------------------+-------+
```

CSV format (62 tokens, 50% savings):
```
id,name,email,age
1,Alice Smith,alice@example.com,28
2,Bob Johnson,bob@example.com,35
```

**Benefits:**
- **26% lower token costs** on average
- **35% more conversation history** fits in context window
- **Faster LLM responses** (less data to process)
- LLMs comprehend CSV format just as well as tabular

**Tradeoffs:**
- CSV is less readable in debug logs (but you don't see the LLM context anyway)
- Results displayed to YOU are still in tabular format - this only affects what's sent to the LLM

**Recommendation:** Enable CSV format for production use to reduce costs. Keep tabular for local development/debugging if you prefer.

---

## Best Practices

### Database Operations

**Safe Operations (Recommended):**
- SELECT queries (reading data)
- Table introspection
- Schema exploration
- Aggregate queries (COUNT, SUM, AVG)

**Destructive Operations (Protected):**
- INSERT, UPDATE, DELETE
- CREATE, ALTER, DROP
- Transaction management

**Safety Feature: Confirmation Prompts**

When the LLM generates a destructive query, pgcli will **automatically prompt you for confirmation** before execution:

```
postgres@localhost:mydb> delete the youngest user
> DELETE FROM users WHERE age = (SELECT MIN(age) FROM users);

You're about to run a destructive command.
Do you want to proceed? (y/n):
```

- Type `y` to execute
- Type `n` to cancel (recommended if unsure!)

This protection works exactly like manual SQL mode - **all destructive queries require confirmation**.

**Tip:** For frequent destructive operations, consider using pgcli's normal mode (F6 OFF) for more control.

### Performance

**LLM mode is slower than normal SQL:**
- LLM needs to think (1-3 seconds)
- Function calls to database (0.1-0.5 seconds each)
- Network latency (cloud providers)

**When to use LLM mode:**
- Exploring unfamiliar databases
- Complex queries you don't remember syntax for
- Ad-hoc analysis
- Learning SQL patterns

**When to use normal mode:**
- Known queries (faster)
- Repetitive operations
- Production scripts
- Performance-critical tasks

### Security

**Destructive Query Protection:**

All destructive queries generated by the LLM require **mandatory confirmation** before execution:

```
postgres@localhost:mydb> delete the test data
> DELETE FROM test_table WHERE is_test = true;

You're about to run a destructive command.
Do you want to proceed? (y/n):
```

**Protected operations:**
- DELETE, UPDATE, DROP
- INSERT (if configured)
- CREATE, ALTER
- TRUNCATE

This protection prevents accidental data loss. You must explicitly type `y` to proceed.

**LLM Learning from Rejections:**

When you reject a destructive query, the LLM is informed about your decision:

```
postgres@localhost:mydb> delete old users
> DELETE FROM users WHERE last_login < NOW() - INTERVAL '2 years';

You're about to run a destructive command.
Do you want to proceed? (y/n): n
Wise choice! Query cancelled.

[LLM receives: "User rejected destructive operation. Consider safer alternatives."]

postgres@localhost:mydb> can you suggest a safer approach?
> -- Archive old users instead of deleting
SELECT id, email, last_login
INTO old_users_archive
FROM users
WHERE last_login < NOW() - INTERVAL '2 years';
```

The LLM can then:
- Learn your preferences about destructive operations
- Suggest safer alternatives (SELECT instead of DELETE, archiving, etc.)
- Ask for clarification before generating risky queries
- Provide better context-aware assistance

**Never put sensitive data in prompts:**
```
❌ "find user with password 'secretpass123'"
✅ "find user with id 42"
```

**API keys are sent to providers** - they see:
- Your natural language prompts
- Your database schema
- Query results

**If this is a concern:** Use Ollama (local, private).

---

## Troubleshooting

### Problem: "LLM not configured: ANTHROPIC_API_KEY environment variable not set"

**Cause:** No API key set

**Solution:**
```bash
# Check your API key is set
echo $ANTHROPIC_API_KEY

# If empty, set it
export ANTHROPIC_API_KEY="sk-ant-..."

# Make permanent
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

### Problem: "Error communicating with LLM: ..."

**Cause:** Network issues, invalid API key, or rate limits

**Solutions:**

1. **Check API key is valid:**
   ```bash
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -H "content-type: application/json" \
     -d '{"model":"claude-sonnet-5-5","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}'
   ```

2. **Check internet connection**

3. **Check rate limits:** You might be hitting API rate limits

4. **Try a different model:**
   ```bash
   export PGCLI_LLM_MODEL="claude-haiku-5-5"  # Cheaper, faster
   ```

### Problem: LLM generates wrong SQL

**Cause:** Ambiguous prompt or incorrect schema understanding

**Solutions:**

1. **Be more specific:**
   ```
   Instead of: "show me data"
   Try: "show me the users table with name and email columns"
   ```

2. **Let it see the schema first:**
   ```
   First:  "list all tables"
   Then:   "describe the users table"
   Finally: "now show me users from last week"
   ```

3. **Correct and retry:**
   ```
   "that's wrong, I want users.created_at not orders.created_at"
   ```

### Problem: "Too Many Errors from LLM (3 attempts)"

**What's happening:**

pgcli has an **automatic retry mechanism** that gives the LLM 3 chances to fix SQL errors:

1. **Attempt 1:** LLM generates SQL → Error occurs
2. **Attempt 2:** Error is added to history → LLM tries to fix it
3. **Attempt 3:** If still failing, LLM gets one final try

**You'll see output like:**
```
> SELECT * FROM userz;  -- LLM's first attempt
Error (attempt 1/3): relation "userz" does not exist

> SELECT * FROM users;  -- LLM auto-corrects
+----------+
| name     |
|----------|
| Alice    |
+----------+
```

**If all 3 attempts fail:**
```
Error (attempt 3/3): ...
Too Many Errors from LLM (3 attempts)
```

**Solution:**
```bash
# Exit LLM mode
[Press F6]

# Run the query manually to see the real error
SELECT ...

# Re-enable LLM mode and be more specific
[Press F6]
"show me X with Y where Z"
```

**Common causes of retry exhaustion:**
- LLM responding with explanatory text instead of pure SQL
- Complex schema issues requiring more context
- Ambiguous prompt leading to wrong assumptions

### Problem: Token count very high / Context limit reached

**Cause:** Long conversation or large query results

**Solution:**
```bash
# Start fresh
exit
pgcli

# Or clear history (not implemented yet - restart is the solution)
```

**Workaround:**
```bash
# Limit result sizes
"show me first 10 rows from users"

# Be more specific to avoid exploratory queries
```

### Problem: OpenAI not working

**Cause:** Forgot to set provider

**Solution:**
```bash
# Must set BOTH variables
export PGCLI_LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# Verify
pgcli
# Should see OpenAI models being used
```

### Problem: Ollama not responding

**Cause:** Ollama server not running

**Solution:**
```bash
# Start Ollama server in separate terminal
ollama serve

# Keep it running while using pgcli
```

### Enable Debug Logging

To see what's happening internally:

**1. Edit config file:**
```bash
vi ~/.config/pgcli/config
```

**2. Add:**
```ini
[main]
log_file = /tmp/pgcli.log
```

**3. Check logs:**
```bash
tail -f /tmp/pgcli.log

# You'll see:
# DEBUG:pgcli.llm.agent:SQL result added to history: 1234 tokens
# DEBUG:pgcli.llm.agent:History trimmed: 30 msgs → 22 msgs
```

---

## FAQ

### Q: Does the LLM execute SQL automatically?

**A:** Yes. The LLM generates SQL and pgcli executes it immediately. There's no confirmation step.

**Safety:** Review what the LLM generates before using it for destructive operations (UPDATE, DELETE, DROP).

### Q: Can I see the SQL before it runs?

**A:** Yes, it's printed with `> ` prefix:
```
> SELECT * FROM users WHERE created_at > now() - interval '7 days';
```

### Q: Does it remember previous queries?

**A:** Yes, conversation history includes:
- Your prompts
- Generated SQL
- Query results

This allows follow-up questions like "show me just the name for that user".

### Q: What happens when context is full?

**A:** Automatic trimming:
- Old messages removed
- Last 10-20 exchanges kept
- System messages preserved

You won't lose critical context.

### Q: Can I use it offline?

**A:** Only with Ollama (local LLM). Anthropic and OpenAI require internet.

### Q: Is my data sent to the LLM provider?

**A:** Yes, the LLM sees:
- Your database schema (table/column names)
- Your natural language prompts
- Query results

**If privacy is critical:** Use Ollama (runs locally, nothing sent to cloud).

### Q: How much does it cost?

**Anthropic (Claude):**
- ~$0.01 - $0.05 per complex query
- Most queries: < $0.01

**OpenAI (GPT-4o-mini):**
- ~$0.001 - $0.01 per query
- Cheapest option

**Ollama:**
- Free (local)

### Q: Can it create tables or modify data?

**A:** Yes, but with **mandatory confirmation prompts** for safety.

When the LLM generates a destructive query (DELETE, UPDATE, DROP, etc.), pgcli will:
1. Show you the generated SQL
2. **Prompt you for confirmation**: "You're about to run a destructive command. Do you want to proceed? (y/n)"
3. Only execute if you type `y`

This protection prevents accidental data loss from LLM-generated queries.

**Example:**
```
postgres@localhost:mydb> delete old records
> DELETE FROM logs WHERE created_at < NOW() - INTERVAL '90 days';

You're about to run a destructive command.
Do you want to proceed? (y/n): n
Wise choice! Query cancelled.
```

**Recommendation:** LLM mode is best for SELECT queries and exploration. For planned destructive operations, use normal SQL mode (F6 OFF) for more control.

### Q: Why is it slow?

**A:** LLM mode involves:
1. Send prompt to LLM (network)
2. LLM calls database functions (network + DB queries)
3. LLM generates SQL (thinking time)
4. Execute SQL (DB query)

Total: 2-5 seconds vs <0.1s for normal SQL.

### Q: Can I use it in scripts?

**A:** No, LLM mode is interactive only. For scripts:
```bash
# Use normal pgcli or psql
pgcli -c "SELECT * FROM users"
```

### Q: What if it generates bad SQL?

**A:** pgcli has **automatic retry with error feedback**:

1. **LLM generates bad SQL** → Error occurs
2. **Error is automatically sent to LLM** → It tries to fix it
3. **Up to 3 attempts** before giving up

**Example:**
```
> SELECT * FROM userz;  -- Typo in table name
Error (attempt 1/3): relation "userz" does not exist

> SELECT * FROM users;  -- LLM auto-corrects!
+----------+
| name     |
|----------|
| Alice    |
+----------+
```

**If it still fails after 3 tries:**
1. Correct your prompt: "I meant 'name' not 'naem'"
2. Or switch to normal mode (F6) and fix manually

### Q: Can I customize the system prompt?

**A:** Not yet. The system prompt is optimized for SQL generation.

Feature request: https://github.com/dbcli/pgcli/issues

---

## Next Steps

- **Setup Guide:** [LLM_SETUP.md](LLM_SETUP.md) - Configure providers
- **Technical Details:** [PHASE3_FINDINGS.md](../PHASE3_FINDINGS.md) - Implementation
- **Report Issues:** https://github.com/dbcli/pgcli/issues

---

## Tips for Success

1. **Start broad, narrow down:**
   ```
   "list tables" → "describe users" → "show active users"
   ```

2. **Let the LLM see the schema first:**
   The first time you mention a table, let it call describe_tables.

3. **Be patient with function calls:**
   ```
   ...Listing tables...
   ...describing tables 'users' in public...
   ```
   These are necessary for quality SQL.

4. **Use limits for exploration:**
   ```
   "show me 5 examples from users"
   ```

5. **Reference previous results:**
   ```
   "now filter that to only active users"
   "use the same table but show different columns"
   ```

6. **Monitor your token usage:**
   Watch the status bar. If it's getting high (>100k), consider restarting.

Happy querying! 🎉
