"""Database introspection tools for the Pydantic AI agent."""
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from .models import PgcliDeps
import sys


def _show_tool_status(message: str):
    """Show tool execution status that overwrites the previous line."""
    # Clear line and show new status (without newline)
    sys.stdout.write("\r\033[K")  # \r = return to start, \033[K = clear to end
    sys.stdout.write(message)
    sys.stdout.flush()


class DescribeTablesInput(BaseModel):
    """Input parameters for describe_tables tool."""
    table_names: List[str] = Field(..., description="List of table names to describe")
    schema_name: str = Field("public", description="Schema name (default: public)")


class DescribeEnumInput(BaseModel):
    """Input parameters for describe_enum tool."""
    enum_name: str = Field(..., description="Name of the enum type to describe")
    schema_name: str = Field("public", description="Schema name (default: public)")


class GetTableStatsInput(BaseModel):
    """Input parameters for get_table_statistics tool."""
    table_name: str = Field(..., description="Name of the table to get statistics for")
    schema_name: str = Field("public", description="Schema name (default: public)")


class GetIndexesInput(BaseModel):
    """Input parameters for get_indexes tool."""
    table_name: str = Field(..., description="Name of the table to get indexes for")
    schema_name: str = Field("public", description="Schema name (default: public)")


class GetConstraintsInput(BaseModel):
    """Input parameters for get_constraints tool."""
    table_name: str = Field(..., description="Name of the table to get constraints for")
    schema_name: str = Field("public", description="Schema name (default: public)")


class GetForeignKeysInput(BaseModel):
    """Input parameters for get_foreign_keys tool."""
    table_name: str = Field(..., description="Name of the table to get foreign keys for")
    schema_name: str = Field("public", description="Schema name (default: public)")


def get_tables(ctx: RunContext[PgcliDeps]) -> str:
    """List all tables in the database.

    Returns table names with their schemas and descriptions.
    This helps the LLM understand what tables are available.
    """
    _show_tool_status("...Listing tables...")
    pgcli = ctx.deps.pgcli
    output, query = pgcli._evaluate_command(
        """SELECT
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
    c.relkind = 'r'
    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY
    n.nspname, c.relname;"""
    )
    return "\n".join(output)


def describe_tables(ctx: RunContext[PgcliDeps], params: DescribeTablesInput) -> str:
    """Describe the columns of specified tables.

    Returns column names, data types, and descriptions for the requested tables.
    This helps the LLM understand table structure for query generation.
    """
    table_names_joined = "'" + "', '".join(params.table_names) + "'"
    _show_tool_status(f"...describing tables {table_names_joined} in {params.schema_name}...")

    pgcli = ctx.deps.pgcli
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
     cols.table_schema = '{params.schema_name}'
     AND cols.table_name in ({table_names_joined});"""
    )
    return "\n".join(output)


def describe_enum(ctx: RunContext[PgcliDeps], params: DescribeEnumInput) -> str:
    """Describe an enum type and its values.

    :param ctx: Run context with pgcli instance
    :param params: Enum name and schema
    :return: Enum values with sort order
    """
    _show_tool_status(f"...describing enum {params.enum_name} in {params.schema_name}...")
    pgcli = ctx.deps.pgcli
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
     AND n.nspname = '{params.schema_name}'
     AND t.typname = '{params.enum_name}'
ORDER BY
     e.enumsortorder;"""
    )
    return "\n".join(output)


def get_table_statistics(ctx: RunContext[PgcliDeps], params: GetTableStatsInput) -> str:
    """Get table statistics including row count, size, and access patterns.

    :param ctx: Run context with pgcli instance
    :param params: Table name and schema
    :return: Table statistics data
    """
    _show_tool_status(f"...getting statistics for table {params.table_name} in {params.schema_name}...")
    pgcli = ctx.deps.pgcli
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
     pg_stat_user_tables.schemaname = '{params.schema_name}'
     AND pg_stat_user_tables.relname = '{params.table_name}';"""
    )
    return "\n".join(output)


def get_indexes(ctx: RunContext[PgcliDeps], params: GetIndexesInput) -> str:
    """Get all indexes for a table.

    :param ctx: Run context with pgcli instance
    :param params: Table name and schema
    :return: Index information including columns, uniqueness, and definitions
    """
    _show_tool_status(f"...getting indexes for table {params.table_name} in {params.schema_name}...")
    pgcli = ctx.deps.pgcli
    output, query = pgcli._evaluate_command(
        f"""SELECT
     i.relname AS index_name,
     a.attname AS column_name,
     ix.indisunique AS is_unique,
     ix.indisprimary AS is_primary,
     am.amname AS index_type,
     pg_get_indexdef(i.oid) AS index_definition
FROM
     pg_class t,
     pg_class i,
     pg_index ix,
     pg_attribute a,
     pg_am am
WHERE
     t.oid = ix.indrelid
     AND i.oid = ix.indexrelid
     AND a.attrelid = t.oid
     AND a.attnum = ANY(ix.indkey)
     AND t.relkind = 'r'
     AND t.relname = '{params.table_name}'
     AND t.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = '{params.schema_name}')
     AND i.relam = am.oid
ORDER BY
     i.relname;"""
    )
    return "\n".join(output)


def get_constraints(ctx: RunContext[PgcliDeps], params: GetConstraintsInput) -> str:
    """Get all constraints for a table.

    :param ctx: Run context with pgcli instance
    :param params: Table name and schema
    :return: Constraint information including type and definition
    """
    _show_tool_status(f"...getting constraints for table {params.table_name} in {params.schema_name}...")
    pgcli = ctx.deps.pgcli
    output, query = pgcli._evaluate_command(
        f"""SELECT
     con.conname AS constraint_name,
     con.contype AS constraint_type,
     pg_get_constraintdef(con.oid) AS constraint_definition
FROM
     pg_constraint con
JOIN
     pg_class rel ON rel.oid = con.conrelid
JOIN
     pg_namespace nsp ON nsp.oid = rel.relnamespace
WHERE
     rel.relname = '{params.table_name}'
     AND nsp.nspname = '{params.schema_name}';"""
    )
    return "\n".join(output)


def get_foreign_keys(ctx: RunContext[PgcliDeps], params: GetForeignKeysInput) -> str:
    """Get all foreign key constraints for a table.

    :param ctx: Run context with pgcli instance
    :param params: Table name and schema
    :return: Foreign key constraint information
    """
    _show_tool_status(f"...getting foreign keys for table {params.table_name} in {params.schema_name}...")
    pgcli = ctx.deps.pgcli
    output, query = pgcli._evaluate_command(
        f"""SELECT
     con.conname AS constraint_name,
     pg_get_constraintdef(con.oid) AS constraint_definition
FROM
     pg_constraint con
JOIN
     pg_class rel ON rel.oid = con.conrelid
JOIN
     pg_namespace nsp ON nsp.oid = rel.relnamespace
WHERE
     con.contype = 'f'
     AND rel.relname = '{params.table_name}'
     AND nsp.nspname = '{params.schema_name}';"""
    )
    return "\n".join(output)
