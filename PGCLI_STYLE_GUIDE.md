# pgcli Code Style Guide

This guide documents the code style, conventions, and patterns used in the pgcli codebase. Follow these guidelines to write code that looks like it was written by the same team.

---

## Table of Contents
1. [File Organization](#file-organization)
2. [Import Conventions](#import-conventions)
3. [Code Style](#code-style)
4. [Python Patterns](#python-patterns)
5. [Database Interaction](#database-interaction)
6. [Testing Patterns](#testing-patterns)
7. [Error Handling](#error-handling)
8. [Logging](#logging)

---

## File Organization

### Module Structure
- Main application logic: `/pgcli/main.py`
- Database execution: `/pgcli/pgexecute.py`
- Completion logic: `/pgcli/pgcompleter.py`
- Configuration: `/pgcli/config.py`
- Packages/utilities: `/pgcli/packages/`
- Tests: `/tests/` (mirrors main structure)

### Naming Conventions

**Files and Directories:**
- Use lowercase with underscores: `pgexecute.py`, `key_bindings.py`
- Package directories use lowercase: `packages/`, `parseutils/`

**Classes:**
- Use PascalCase: `PGCli`, `PGExecute`, `PGCompleter`
- Exception classes end with "Error": `PgCliQuitError`, `InvalidMapFile`

**Functions and Methods:**
- Use snake_case: `execute_command()`, `connect_dsn()`, `get_tables()`
- Private methods start with underscore: `_evaluate_command()`, `_has_limit()`
- Helper functions within classes follow same pattern

**Variables:**
- Use snake_case: `user_input`, `table_names`, `log_file`
- Constants use UPPER_SNAKE_CASE: `DEFAULT_MAX_FIELD_WIDTH`, `SSH_TUNNEL_SUPPORT`
- Module-level loggers: `_logger` (with leading underscore)

**Named Tuples:**
- Use PascalCase for the type: `MetaQuery`, `ViewDef`, `OutputSettings`
- Fields use snake_case: `query`, `successful`, `total_time`

---

## Import Conventions

### Import Order (strictly followed)
1. **Third-party imports with try/except** (optional dependencies)
2. **Standard library imports** (alphabetically grouped)
3. **Third-party imports** (major libraries)
4. **Prompt toolkit imports** (grouped together)
5. **pgspecial imports**
6. **Relative imports** (local package imports)

### Example from `main.py`:
```python
from configobj import ConfigObj, ParseError
from pgspecial.namedqueries import NamedQueries
from .config import skip_initial_comment
from .llm_api import generate_sql, llm_append_output

import atexit
import os
import re
import sys
import traceback
import logging
import threading
import shutil
import functools
import datetime as dt
import itertools
import pathlib
import platform
from time import time, sleep
from typing import Optional

from cli_helpers.tabular_output import TabularOutputFormatter
from cli_helpers.tabular_output.preprocessors import align_decimals, format_numbers
from cli_helpers.utils import strip_ansi
from .explain_output_formatter import ExplainOutputFormatter
import click

try:
    import setproctitle
except ImportError:
    setproctitle = None

from prompt_toolkit.completion import DynamicCompleter, ThreadedCompleter
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
# ... more prompt_toolkit imports

from pgspecial.main import PGSpecial, NO_QUERY, PAGER_OFF, PAGER_LONG_OUTPUT
import pgspecial as special

from . import auth
from .pgcompleter import PGCompleter
from .pgtoolbar import create_toolbar_tokens_func
```

### Import Style Rules
- **Always use absolute imports for third-party**: `import click`, `from cli_helpers.tabular_output import ...`
- **Use relative imports for local modules**: `from .config import ...`, `from . import auth`
- **Import specific items when few are needed**: `from time import time, sleep`
- **Import module when many items needed**: `import pgspecial as special`
- **Handle optional dependencies with try/except at top of file**:
  ```python
  try:
      import setproctitle
  except ImportError:
      setproctitle = None
  ```

---

## Code Style

### Docstrings

**Style: Plain docstrings** (NOT Google or NumPy style)

**Module/Class docstrings:**
```python
def register_typecasters(connection):
    """Casts date and timestamp values to string, resolves issues with out-of-range
    dates (e.g. BC) which psycopg can't handle"""
    # implementation
```

**Method docstrings with parameters:**
```python
def run(
    self,
    statement,
    pgspecial=None,
    exception_formatter=None,
    on_error_resume=False,
    explain_mode=False,
):
    """Execute the sql in the database and return the results.

    :param statement: A string containing one or more sql statements
    :param pgspecial: PGSpecial object
    :param exception_formatter: A callable that accepts an Exception and
           returns a formatted (title, rows, headers, status) tuple that can
           act as a query result. If an exception_formatter is not supplied,
           psycopg2 exceptions are always raised.
    :param on_error_resume: Bool. If true, queries following an exception
           (assuming exception_formatter has been supplied) continue to
           execute.

    :return: Generator yielding tuples containing
             (title, rows, headers, status, query, success, is_special)
    """
```

**Complex function docstrings:**
```python
def last_word(text, include="alphanum_underscore"):
    r"""
    Find the last word in a sentence.

    >>> last_word('abc')
    'abc'
    >>> last_word(' abc')
    'abc'
    >>> last_word('')
    ''
    >>> last_word('abc def')
    'def'
    """
```

### Comments

**Inline comments:**
```python
# Remove spaces and EOL
statement = statement.strip()

# sql parse doesn't split on a comment first + special
# so we're going to do it
cleaned_command, removed_comments = remove_beginning_comments(statement)

# Query tuples are used for maintaining history
MetaQuery = namedtuple("Query", [...])
```

**TODO/FIXME comments:**
```python
# Ref: https://stackoverflow.com/questions/30425105/...
COLOR_CODE_REGEX = re.compile(r"\x1b(\[.*?[@-~]|\].*?(\x07|\x1b\\))")

# pg3: I don't know what is this
class ProtocolSafeCursor(psycopg.Cursor):
```

### String Quotes

**Preference: Double quotes for strings**
- Double quotes are standard: `"hello world"`
- Single quotes used for format strings within f-strings: `f"Table format {pattern} not recognized"`
- Triple double quotes for docstrings and multi-line strings: `"""SELECT * FROM ..."""`

```python
# Good
message = "File output disabled"
click.secho("Reconnecting...", fg="green")
query = """
    SELECT *
    FROM table
"""

# Also acceptable (for consistency within f-strings)
f'Default pager found in config file: "{configured_pager}"'
```

### Line Length
- **No strict limit observed**, but generally reasonable (under 100 chars when possible)
- Long strings (SQL queries, messages) are broken naturally
- Function signatures use multiple lines when long:

```python
def __init__(
    self,
    force_passwd_prompt=False,
    never_passwd_prompt=False,
    pgexecute=None,
    pgclirc_file=None,
    row_limit=None,
    application_name="pgcli",
    single_connection=False,
    less_chatty=None,
    prompt=None,
    prompt_dsn=None,
    auto_vertical_output=False,
    warn=None,
    ssh_tunnel_url: Optional[str] = None,
    log_file: Optional[str] = None,
    llm_log_dir: Optional[str] = None,
):
```

### Function/Class Definitions

**Class attributes defined at class level:**
```python
class PGCli:
    default_prompt = "\\u@\\h:\\d> "
    max_len_prompt = 30
```

**Class methods organized by functionality** (connection, execution, formatting, etc.)

**Default parameters in __init__:**
```python
def __init__(
    self,
    force_passwd_prompt=False,  # Boolean defaults
    never_passwd_prompt=False,
    pgexecute=None,             # Optional objects default to None
    application_name="pgcli",   # String defaults
):
```

---

## Python Patterns

### Type Hints

**Limited use - only where truly helpful:**
```python
from typing import Optional

def __init__(
    self,
    ssh_tunnel_url: Optional[str] = None,
    log_file: Optional[str] = None,
):
```

**Function signatures in `llm_api.py`:**
```python
from typing import Callable
from functools import wraps

def with_default_args(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(pgcli, *args, **kwargs):
        # implementation
```

**Type hints not heavily used** - code relies on duck typing and clear naming.

### Decorators

**Function wrapping:**
```python
from functools import wraps

def with_default_args(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(pgcli, *args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(pgcli, *args, **kwargs)
        bound_args.apply_defaults()
        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper

@with_default_args
def get_tables(pgcli):
    # implementation
```

**Test decorators:**
```python
@dbtest
def test_conn(executor):
    # test implementation

@pytest.mark.skipif(not CAN_CONNECT_TO_DB, reason="...")
def test_something():
    # test implementation
```

### Named Tuples

**Pattern: Define at module level, use for data structures**

```python
from collections import namedtuple

# Define the named tuple
MetaQuery = namedtuple(
    "Query",
    [
        "query",
        "successful",
        "total_time",
        "execution_time",
        "meta_changed",
        "db_changed",
        "path_changed",
        "mutated",
        "is_special",
    ],
)

# Set defaults using __new__.__defaults__
MetaQuery.__new__.__defaults__ = ("", False, 0, 0, False, False, False, False)

# Use it
query = MetaQuery(query=text, successful=False)
```

**Factory functions for complex named tuples:**
```python
_SchemaObject = namedtuple("SchemaObject", "name schema meta")

def SchemaObject(name, schema=None, meta=None):
    return _SchemaObject(name, schema, meta)
```

### Properties

**Use @property decorator for computed attributes:**
```python
@property
def short_host(self):
    try:
        ipaddress.ip_address(self.host)
        return self.host
    except ValueError:
        pass
    if "," in self.host:
        host, _, _ = self.host.partition(",")
    else:
        host = self.host
    short_host, _, _ = host.partition(".")
    return short_host
```

### Context Managers and Locks

**Threading locks:**
```python
self._completer_lock = threading.Lock()

# Usage
with self._completer_lock:
    self.completer.extend_query_history(text)
```

**Database cursors:**
```python
with self.conn.cursor() as cur:
    cur.execute(query)
    return [x[0] for x in cur.fetchall()]
```

### Generators and Iterators

**Prefer generators for large datasets:**
```python
def _relations(self, kinds=("r", "p", "f", "v", "m")):
    """Get table or view name metadata"""
    with self.conn.cursor() as cur:
        cur.execute(self.tables_query, [kinds])
        yield from cur

def tables(self):
    """Yields (schema_name, table_name) tuples"""
    yield from self._relations(kinds=["r", "p", "f"])
```

**Using itertools:**
```python
import itertools

output = itertools.chain(output, formatted)
new_cur = itertools.islice(cur, limit)
```

### Lambda Functions

**Use sparingly, mainly for simple callbacks:**
```python
refresh_callback = lambda: self.refresh_completions(persist_priorities="all")

case_function = lambda x: x

normalize_ref = lambda ref: ref if ref[0] == '"' else '"' + ref.lower() + '"'
```

---

## Database Interaction

### Query Execution Pattern

**Using pgexecute.run():**
```python
# In main.py
on_error_resume = self.on_error == "RESUME"
res = self.pgexecute.run(
    text,
    self.pgspecial,
    lambda x: exception_formatter(x, self.verbose_errors),
    on_error_resume,
    explain_mode=self.explain_mode,
)

for title, cur, headers, status, sql, success, is_special in res:
    # Process results
```

**Direct cursor execution:**
```python
# In pgexecute.py
with self.conn.cursor() as cur:
    cur.execute(query)
    return [x[0] for x in cur.fetchall()]
```

### SQL Query Formatting

**Multi-line SQL queries use triple quotes:**
```python
search_path_query = """
    SELECT * FROM unnest(current_schemas(true))"""

tables_query = """
    SELECT  n.nspname schema_name,
            c.relname table_name
    FROM    pg_catalog.pg_class c
            LEFT JOIN pg_catalog.pg_namespace n
                ON n.oid = c.relnamespace
    WHERE   c.relkind = ANY(%s)
    ORDER BY 1,2;"""
```

**Indentation within SQL:**
- Main keywords (SELECT, FROM, WHERE) align left
- Subqueries and clauses indented with spaces
- No tabs in SQL strings

### Parameter Passing

**Use psycopg3 parameter passing (not string formatting):**
```python
# Good - parameterized
cur.execute(self.tables_query, [kinds])
cur.execute(query, (spec,))

# Avoid - string formatting (only for dynamic SQL structure)
# This is used in llm_api.py but should be avoided elsewhere
f"""SELECT ... WHERE table_name = '{table_name}'"""
```

**Dynamic query construction (when necessary):**
```python
# Using psycopg.sql for identifiers
from psycopg import sql

return (
    sql.SQL(template)
    .format(
        name=sql.Identifier(result.nspname, result.relname),
        stmt=sql.SQL(result.viewdef),
    )
    .as_string(self.conn)
)
```

### Result Processing

**Iterate over cursor directly:**
```python
for row in cur:
    yield ForeignKey(*row)
```

**Extract headers from cursor.description:**
```python
if cur.description:
    headers = [x[0] for x in cur.description]
    return title, cur, headers, cur.statusmessage
```

**List comprehension for simple transformations:**
```python
return [x[0] for x in cur.fetchall()]
```

---

## Testing Patterns

### Test File Organization

**Mirror source structure:**
- `/tests/test_main.py` tests `/pgcli/main.py`
- `/tests/test_pgexecute.py` tests `/pgcli/pgexecute.py`
- `/tests/parseutils/` tests `/pgcli/packages/parseutils/`

### Test Utilities (`tests/utils.py`)

**Use shared fixtures and helpers:**
```python
from utils import dbtest, run

@dbtest
def test_conn(executor):
    run(executor, """create table test(a text)""")
    run(executor, """insert into test values('abc')""")
    assert run(executor, """select * from test""", join=True) == expected
```

### Test Naming

**Pattern: `test_<what_is_being_tested>`**
```python
def test_conn(executor):
def test_copy(executor):
def test_bools_are_treated_as_strings(executor):
def test_expanded_slash_G(executor, pgspecial):
def test_schemata_table_views_and_columns_query(executor):
```

### Fixtures and Markers

**Database tests use decorator:**
```python
@dbtest
def test_something(executor):
    # test implementation
```

**Skip conditions:**
```python
@pytest.mark.skipif(
    not CAN_CONNECT_TO_DB,
    reason="Need a postgres instance at localhost"
)
def test_something():
    pass

@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Not applicable in windows"
)
def test_something():
    pass
```

### Mocking

**Use unittest.mock:**
```python
from unittest.mock import patch, MagicMock

@patch('pgcli.main.PGExecute')
def test_something(mock_execute):
    # test implementation
```

### Assertions

**Use standard pytest assertions:**
```python
assert list(results) == expected
assert title == expected_title
assert pgspecial.expanded_output == False
```

**Use dedent for multi-line string comparisons:**
```python
from textwrap import dedent

assert run(executor, """select * from test""", join=True) == dedent(
    """\
    +-----+
    | a   |
    |-----|
    | abc |
    +-----+
    SELECT 1"""
)
```

---

## Error Handling

### Exception Patterns

**Catch specific exceptions:**
```python
try:
    pgexecute = PGExecute(...)
except (OperationalError, InterfaceError) as e:
    if should_ask_for_password(e):
        # handle password prompt
    else:
        raise e
```

**Custom exceptions inherit from built-in:**
```python
class PgCliQuitError(Exception):
    pass

class InvalidMapFile(ValueError):
    pass
```

### Error Messages

**User-facing errors use click.secho:**
```python
click.secho(str(e), err=True, fg="red")
click.secho("Reconnecting...", fg="green")
click.echo("Previous connection kept")
```

**Return error tuples for special commands:**
```python
if not pattern:
    message = "\\i: missing required argument"
    return [(None, None, None, message, "", False, True)]

try:
    with open(os.path.expanduser(pattern), encoding="utf-8") as f:
        query = f.read()
except OSError as e:
    return [(None, None, None, str(e), "", False, True)]
```

### Traceback Logging

**Log exceptions with traceback:**
```python
except Exception as e:
    logger.error("sql: %r, error: %r", text, e)
    logger.error("traceback: %r", traceback.format_exc())
    click.secho(str(e), err=True, fg="red")
```

### Exit Handling

**Use sys.exit for fatal errors:**
```python
except Exception as e:
    click.secho(str(e), err=True, fg="red")
    sys.exit(1)
```

**Raise custom exceptions for control flow:**
```python
def quit(self):
    raise PgCliQuitError

try:
    self.handle_watch_command(text)
except PgCliQuitError:
    if not self._check_ongoing_transaction_and_allow_quitting():
        continue
    raise
```

---

## Logging

### Logger Setup

**Module-level logger with underscore prefix:**
```python
import logging

_logger = logging.getLogger(__name__)
```

**Instance logger (for classes):**
```python
class PGCli:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
```

### Log Levels

**DEBUG - for SQL and detailed execution info:**
```python
_logger.debug("Regular sql statement. sql: %r", split_sql)
_logger.debug("Search path query. sql: %r", self.search_path_query)
self.logger.debug("Database connection failed: %r.", e)
```

**INFO - for configuration and setup:**
```python
self.logger.info(
    'Default pager found in config file: "%s"', configured_pager
)
```

**ERROR - for exceptions and failures:**
```python
_logger.error("sql: %r, error: %r", sql, e)
_logger.error("traceback: %r", traceback.format_exc())
self.logger.error("traceback: %r", traceback.format_exc())
```

### Log Message Format

**Use %r for repr() of objects:**
```python
_logger.debug("sql: %r", text)
logger.error("sql: %r, error: %r", text, e)
```

**Use %s for string interpolation:**
```python
self.logger.info('Default pager found: "%s"', configured_pager)
```

**Complex log messages:**
```python
self.logger.debug(
    "Launch Params: \n"
    "\tdatabase: %r"
    "\tuser: %r"
    "\thost: %r"
    "\tport: %r",
    database,
    user,
    host,
    port,
)
```

---

## Additional Conventions

### Configuration

**Use ConfigObj for configuration files:**
```python
from configobj import ConfigObj

cfg = ConfigObj(expanduser(usr_cfg), interpolation=False, encoding="utf-8")
```

**Access config values:**
```python
self.multi_line = c["main"].as_bool("multi_line")
self.row_limit = c["main"].as_int("row_limit")
self.table_format = c["main"]["table_format"]
```

### Click Command-Line Interface

**Use Click decorators:**
```python
@click.command()
@click.option("-h", "--host", default="", envvar="PGHOST", help="...")
@click.option("-p", "--port", default=5432, type=click.INT, help="...")
@click.argument("dbname", default=lambda: None, envvar="PGDATABASE", nargs=1)
def cli(dbname, host, port, ...):
    # implementation
```

### Environment Variables

**Read with os.environ or os.getenv:**
```python
passwd = os.environ.get("PGPASSWORD", "")
service = os.getenv("PGSERVICE")
```

### Path Handling

**Use pathlib for modern path operations:**
```python
import pathlib

log_dir_path = Path(log_dir)
log_dir_path.mkdir(parents=True, exist_ok=True)
log_file = log_dir_path / f"llm_interaction_{timestamp}.json"
```

**Use os.path for simpler operations:**
```python
filename = os.path.abspath(os.path.expanduser(pattern))
if not os.path.isfile(filename):
    # handle
```

---

## Do's and Don'ts

### DO:

✅ Use double quotes for strings consistently
✅ Follow the import order: third-party → stdlib → prompt_toolkit → pgspecial → relative
✅ Use snake_case for functions/methods/variables
✅ Use PascalCase for classes
✅ Write docstrings for public functions/methods
✅ Use f-strings for string formatting (Python 3.6+)
✅ Use context managers (with statements) for resources
✅ Catch specific exceptions, not bare `except:`
✅ Log errors with traceback
✅ Use click.secho for colored terminal output
✅ Use generators (yield) for large datasets
✅ Follow existing patterns in similar code

### DON'T:

❌ Mix single and double quotes arbitrarily
❌ Use wildcard imports (`from module import *`)
❌ Use bare `except:` without specifying exception type
❌ Use string concatenation in loops (use join or f-strings)
❌ Use global variables (use class attributes or function parameters)
❌ Write overly long lines (break naturally at 80-100 chars)
❌ Use `print()` for output (use click.echo or click.secho)
❌ Ignore exception context (always log or re-raise)
❌ Use tabs for indentation (always use 4 spaces)
❌ Create new patterns when existing ones work

---

## Quick Reference: The "pgcli Way"

### Function Definition
```python
def execute_command(self, text, handle_closed_connection=True):
    """Execute a command and return results.

    :param text: The command text to execute
    :param handle_closed_connection: Whether to handle connection errors
    :return: Query result object
    """
    logger = self.logger

    try:
        output, query = self._evaluate_command(text)
    except KeyboardInterrupt:
        logger.debug("cancelled query, sql: %r", text)
        click.secho("cancelled query", err=True, fg="red")
    except OperationalError as e:
        logger.error("sql: %r, error: %r", text, e)
        logger.error("traceback: %r", traceback.format_exc())
        click.secho(str(e), err=True, fg="red")
        if handle_closed_connection:
            self._handle_server_closed_connection(text)
    else:
        # Success path
        if output:
            self.echo_via_pager("\n".join(output))

    return query
```

### Database Query
```python
def get_tables(self):
    """Yields (schema_name, table_name) tuples"""
    with self.conn.cursor() as cur:
        cur.execute(self.tables_query, [kinds])
        yield from cur
```

### Test Function
```python
@dbtest
def test_query_execution(executor):
    """Test that queries execute correctly"""
    run(executor, "CREATE TABLE test(id int, name text)")
    run(executor, "INSERT INTO test VALUES (1, 'alice')")

    result = run(executor, "SELECT * FROM test", join=True)

    expected = dedent("""\
        +----+-------+
        | id | name  |
        |----+-------|
        | 1  | alice |
        +----+-------+
        SELECT 1""")

    assert result == expected
```

---

## Summary

The pgcli codebase values:
- **Consistency** over cleverness
- **Clarity** over brevity
- **Explicit** over implicit
- **Tested** code over untested
- **Logged** errors over silent failures

When in doubt, look at existing code in the same module or similar functionality elsewhere in the codebase. The patterns are consistent and well-established.
