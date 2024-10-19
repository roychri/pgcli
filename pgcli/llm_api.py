import os
import json
from openai import OpenAI
import openai

# Set up the OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

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
Do not return any blockquote. Do not return markdown. Just SQL."""
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
        "content": """SELECT
    gx.id,
    gx.schedule_id,
    gx.name,
    gx.start_time AT TIME ZONE c.timezone AS start_time,
    gx.instructors->0->>'FirstName' AS coach,
    gx.room,
    gx.club_number || ' - ' || c.name AS club,
    gx.training_type,
    gx.created_at,
    gx.deleted_at IS NOT NULL AS deleted
FROM
    group_classes gx
JOIN
    clubs c ON c.id = gx.club_id
WHERE
    gx.instructors->0->>'InstructorID' = '38365'
ORDER BY
    gx.id DESC
LIMIT 1;"""
    },
    {
        "role": "user",
        "content": "how many 24go user have an active member and checked in a gym at least 5 times?",
    },
    {
        "role": "assistant",
        "content": """SELECT
    to_char(COUNT(*)::numeric, 'FM999,999,999,999') AS active_users -- Format to human readable
FROM (
    SELECT
        u.id
    FROM
        users u
    JOIN
        members m ON m.id = u.member_id -- Ensure the user is an active member
    JOIN
        checkins c ON c.member_id = m.id -- Checkin records
    WHERE
        m.is_active = TRUE -- Ensure the user is an active member
        AND m.deleted_at IS NULL -- Ensure the member is not deleted
        AND c.deleted_at IS NULL -- Ensure checkins are not deleted
    GROUP BY
        u.id
    HAVING
        COUNT(c.id) >= 5 -- The user has checked in at least 5 times
) subquery;"""
    }
]

def llm_append_output(role, output):
    messages.append({
        "role": role,
        "content": output
    })

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
        }
    ]

    loop_counter = 0
    # Enter a loop to handle multiple function calls
    while True:
        try:
            loop_counter += 1
            if loop_counter > 10:
                raise Exception("Infinite Loop detected")

            # Call the OpenAI Chat Completions API
            response = generate_sql.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=functions,
            )

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

                    # Execute the function
                    if function_name == "get_tables":
                        function_response = get_tables(pgcli)
                    elif function_name == "describe_tables":
                        schema = function_args.get('schema', 'public')
                        table_names = function_args["table_names"]
                        function_response = describe_tables(pgcli, table_names, schema)
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
