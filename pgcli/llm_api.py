import os
import json
from openai import OpenAI
import openai
from datetime import datetime
import pathlib
from pathlib import Path
import inspect
from typing import Callable
from functools import wraps

def with_default_args(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(pgcli, *args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(pgcli, *args, **kwargs)
        bound_args.apply_defaults()
        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper

# Set up the OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set up the log directory
LOG_DIR = None

messages = [
    {
        "role": "system",
        "content": """You are an assistant that generates SQL queries based on user requests.
You have access to some functions like `get_tables` and `describe_tables` that you can call to gather vital and necessary information to create valid and working SQL queries like being able to retrieve the list of tables in the database.
Use these functions ALL THE TIME to help guide you to provide accurate SQL queries, specially when working with a table you have not seen before.
Make sure the SQL returns data in a human readable format and augmented with anything a user would benbefit from seeing. This include but is not limited to: timezone, extracting data from json/jsonb fields, concatening fields, etc...
You only return SQL queries and nothing else.
No matter what, you need to return valid SQL, including comments (-- ...).
If we ask the for the current time, return `select NOW()`.
Do not return any blockquote. Do not return markdown. Just SQL.
Expected output for SQL (write comments before SQL, to allow you to think)
-- explanation
SELECT -- let's format ...
  LOWER(code) code
FROM mytable
WHERE
  -- explanation/thoughts
  condition
  -- more reasonning
  AND condition2
"""
    },
    {
        "role": "user",
        "content": "Tell me a joke"
    },
    {
        "role": "assistant",
        "content": "-- Sorry, I can only assist you with SQL."
    },
    {
        "role": "user",
        "content": "Can you find the most recent group class for a specific instructor (InstructorID 38365)?"
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_53K7WRhyoJLvUOoADllWdfyn",
                "function": {
                    "arguments": "{}",
                    "name": "get_tables"
                },
                "type": "function"
            }
        ],
    },
    {
        "role": "tool",
        "content": """
schema,table,comment
public,group_classes,Contains group x classes that are available. Use the clubs.timezone with start_time and the trainer's name is in instructions->0->>'FirstName'
...
""",
        "tool_call_id": "call_53K7WRhyoJLvUOoADllWdfyn"
    },
    {
        "role": "assistant",
        "content": """
-- I'll query the group_classes and join with clubs to show proper timezone of start time.
-- Let's show the most important fields and format them properly.
-- I'll Extract the Coach's name from the JSON field and I'll concatenate club name and number.
SELECT
    -- the id field seems obvious
    gx.id,
    -- The schedule_id is helpful to be able to reference this with 24hf.
    gx.schedule_id,
    -- The name of the class so we know which class this is about
    gx.name,
    -- when the class starts, in the timezone of the club
    gx.start_time AT TIME ZONE c.timezone AS start_time,
    -- The name of the class coach
    gx.instructors->0->>'FirstName' AS coach,
    -- Which room the class occurs in
    gx.room,
    -- Which club this class takes place
    gx.club_number || ' - ' || c.name AS club,
    -- What type of class this is
    gx.training_type,
    -- When this was created for begugging purposes
    gx.created_at,
    -- If this class is deleted or not.
    gx.deleted_at IS NOT NULL AS deleted
FROM
    group_classes gx
JOIN -- Used to find the timezone for the start_time
    clubs c ON c.id = gx.club_id
WHERE -- Let's find the instructor that was requested
    gx.instructors->0->>'InstructorID' = '38365'
ORDER BY -- Since this table uses auto increment int id, the most recent is the higest id
        -- It will be faster this way than using the gx.created_at field since id is idx.
    gx.id DESC
LIMIT 1;"""
    },
    {
        "role": "user",
        "content": "how many 24go user have an active member and checked in a gym at least 5 times?",
    },
    {
        "role": "assistant",
        "content": """
-- We will query the users table and join with members and checkins
-- Formating the output properly is important for humans.
-- Because we need a count with a subcount of checkins, it's best to use a subquery
SELECT
    to_char(COUNT(*)::numeric, 'FM999,999,999,999') AS active_users -- Format to human readable
FROM (
    SELECT
        u.id
    FROM
        users u
    JOIN -- Ensure the user is an active member
        members m ON m.id = u.member_id
    JOIN -- Checkin records
        checkins c ON c.member_id = m.id
    WHERE
        -- Ensure the user is an active member
        m.is_active = TRUE
        -- Ensure the member is not deleted
        AND m.deleted_at IS NULL
        -- Ensure checkins are not deleted
        AND c.deleted_at IS NULL
    GROUP BY -- counting unique users
        u.id
    HAVING
        -- The user has checked in at least 5 times
        COUNT(c.id) >= 5
) subquery;"""
    }
]

def llm_append_output(role, output):
    messages.append({
        "role": role,
        "content": output
    })

def log_llm_interaction(request, response, log_dir=None):
    """
    Logs the LLM request and response to a file.

    Parameters:
        request (dict): The request sent to the LLM.
        response (object): The response received from the LLM.
        log_dir (str): The directory to save the log files. If None, logging is disabled.
    """
    if log_dir is None or log_dir == 'disable':
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    log_file = log_dir_path / f"llm_interaction_{timestamp}.json"
    
    # Convert request and response to serializable format
    serializable_request = json.loads(json.dumps(request, default=str))
    serializable_response = json.loads(json.dumps(response.model_dump(), default=str))
    
    log_data = {
        "timestamp": timestamp,
        "request": serializable_request,
        "response": serializable_response
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

def generate_sql(pgcli, user_input):
    """
    Generates an SQL query based on the user's input using OpenAI's GPT-4o-mini model.

    Parameters:
        user_input (str): The user's prompt or question.

    Returns:
        str: The generated SQL query.
    """
    # Define the model to use
    MODEL = "gpt-4o-mini"

    if not hasattr(generate_sql, 'initialized'):
        generate_sql.client = OpenAI()
        generate_sql.initialized = True

    messages.append({
        "role": "user",
        "content": user_input
    })

    # Avoid filling up the context over the limit.
    # This is a crude attempt. A better approach would be to count the
    # amount of tokens instead.
    if len(messages) > 110:
        print("trimming messages...")
        messages[:] = messages[:9] + messages[-91:]

    # Define the function schema
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_tables",
                "description": "Retrieve the list of tables in the current database.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "describe_tables",
                "description": "Describe one or more tables in the current database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_names": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "The name of one of the table to describe"
                        }
                    },
                    "required": [ "table_names" ],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "describe_enum",
                "description": "Describe an enum in the current database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enum_name": {
                            "type": "string",
                            "description": "The name of the enum to describe"
                        },
                        "schema": {
                            "type": "string",
                            "description": "The schema of the enum (default is 'public')",
                            "default": "public"
                        }
                    },
                    "required": [ "enum_name" ],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_table_statistics",
                "description": "Get statistics for a table in the current database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "The name of the table to get statistics for"
                        },
                        "schema": {
                            "type": "string",
                            "description": "The schema of the table (default is 'public')",
                            "default": "public"
                        }
                    },
                    "required": [ "table_name" ],
                    "additionalProperties": False
                }
            }
        }
    ]

    loop_counter = 0
    # Enter a loop to handle multiple function calls
    while True:
        try:
            loop_counter += 1
            if loop_counter > 10:
                raise Exception("Infinite Loop detected")

            # Prepare the request
            request = {
                "model": MODEL,
                "messages": messages,
                "tools": functions,
            }

            # Call the OpenAI Chat Completions API
            response = generate_sql.client.chat.completions.create(**request)

            # Log the interaction
            log_llm_interaction(request, response, pgcli.llm_log_dir)

            # Extract the assistant's message
            message = response.choices[0].message
            messages.append(message)

            # Handle edge cases
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise Exception("The conversation was too long. Please try a shorter query.")
            elif finish_reason == "content_filter":
                raise Exception("The content was filtered. Please rephrase your query.")
            elif finish_reason == "tool_calls" or (finish_reason == "stop" and message.tool_calls):
                # Handle tool calls
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Execute the function dynamically
                    function_to_call = globals().get(function_name)
                    if function_to_call:
                        # Prepare arguments
                        args = [pgcli]
                        args.extend(function_args.values())
                        function_response = function_to_call(*args)
                    else:
                        function_response = json.dumps({"error": f"Function {function_name} not found"})

                    # Append the function result to the messages
                    messages.append({
                        "role": "tool",
                        "content": function_response,
                        "tool_call_id": tool_call.id
                    })
            elif finish_reason == "stop":
                # Assistant has provided the final answer
                sql_query = message.content.strip()
                return sql_query
            else:
                raise Exception(f"Unexpected response from the model. Finish reason: {finish_reason}")

        except openai.OpenAIError as e:
            # Handle any API errors
            raise Exception(f"An issue occurred with the OpenAI API. {str(e)}")

@with_default_args
def get_tables(pgcli):
    print("...Listing tables...")
    output, query = pgcli._evaluate_command(
        f"""SELECT
    n.nspname AS schema_name,
    c.relname AS table_name,
    d.description AS table_description
FROM
    pg_class c
JOIN
    pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN
    pg_description d ON d.objoid = c.oid AND d.objsubid = 0
WHERE
    c.relkind = 'r'  -- 'r' is for ordinary tables
    AND n.nspname NOT IN ('pg_catalog', 'information_schema')  -- Exclude system schemas
ORDER BY
    n.nspname, c.relname;"""
    )
    return "\n".join(output)

@with_default_args
def describe_tables(pgcli, table_names, schema="public"):
    table_names_joined = "'" + "', '".join(table_names)  + "'"
    print(f"...describing tables {table_names_joined} in {schema}...")
    output, query = pgcli._evaluate_command(
        f"""SELECT
     cols.column_name,
     cols.data_type,
     pgd.description
FROM
     information_schema.columns AS cols
LEFT JOIN
     pg_catalog.pg_statio_all_tables AS st
     ON cols.table_schema = st.schemaname AND cols.table_name = st.relname
LEFT JOIN
     pg_catalog.pg_description AS pgd
     ON pgd.objoid = st.relid AND pgd.objsubid = cols.ordinal_position
WHERE
     cols.table_schema = '{schema}'
     AND cols.table_name in ({table_names_joined});"""
    )
    return "\n".join(output)

@with_default_args
def describe_enum(pgcli, enum_name, schema="public"):
    print(f"...describing enum {enum_name} in {schema}...")
    output, query = pgcli._evaluate_command(
        f"""SELECT
     e.enumlabel AS enum_value,
     e.enumsortorder AS sort_order
FROM
     pg_type t
JOIN
     pg_enum e ON t.oid = e.enumtypid
JOIN
     pg_catalog.pg_namespace n ON n.oid = t.typnamespace
WHERE
     t.typtype = 'e'
     AND n.nspname = '{schema}'
     AND t.typname = '{enum_name}'
ORDER BY
     e.enumsortorder;"""
    )
    return "\n".join(output)

@with_default_args
def get_table_statistics(pgcli, table_name, schema="public"):
    print(f"...getting statistics for table {table_name} in {schema}...")
    output, query = pgcli._evaluate_command(
        f"""SELECT
     pg_stat_user_tables.relname AS table_name,
     pg_stat_user_tables.n_live_tup AS row_count,
     pg_size_pretty(pg_total_relation_size('"' || schemaname || '"."' || relname || '"')) AS total_size,
     pg_stat_user_tables.seq_scan AS sequential_scans,
     pg_stat_user_tables.idx_scan AS index_scans,
     pg_stat_user_tables.n_tup_ins AS inserts,
     pg_stat_user_tables.n_tup_upd AS updates,
     pg_stat_user_tables.n_tup_del AS deletes,
     pg_stat_user_tables.last_vacuum,
     pg_stat_user_tables.last_analyze
FROM
     pg_stat_user_tables
WHERE
     pg_stat_user_tables.schemaname = '{schema}'
     AND pg_stat_user_tables.relname = '{table_name}';"""
    )
    return "\n".join(output)
