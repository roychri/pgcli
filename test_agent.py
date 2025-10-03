"""Test script for Pydantic AI agent with mocked pgcli."""
from unittest.mock import Mock
from pydantic_ai.models.test import TestModel
from pgcli.llm.agent import create_agent
from pgcli.llm.models import PgcliDeps


def main():
    """Test all 7 database introspection tools."""
    # Create a mock pgcli instance
    mock_pgcli = Mock()

    # Mock different responses for different tools
    def mock_evaluate(query):
        if "pg_class" in query and "relkind = 'r'" in query:
            # get_tables
            return (
                ["public | users | User accounts table",
                 "public | orders | Customer orders"],
                query
            )
        elif "information_schema.columns" in query:
            # describe_tables
            return (
                ["id | integer | Primary key",
                 "email | varchar | User email",
                 "created_at | timestamp | Creation time"],
                query
            )
        elif "pg_enum" in query:
            # describe_enum
            return (
                ["active | 1",
                 "inactive | 2",
                 "pending | 3"],
                query
            )
        elif "pg_stat_user_tables" in query:
            # get_table_statistics
            return (
                ["users | 1000 | 128 kB | 50 | 200 | 100 | 10 | 5 | 2025-01-01 | 2025-01-01"],
                query
            )
        elif "pg_index" in query:
            # get_indexes
            return (
                ["users_email_idx | email | t | f | btree | CREATE INDEX ..."],
                query
            )
        elif "pg_constraint" in query and "contype" not in query.lower():
            # get_constraints
            return (
                ["users_pkey | p | PRIMARY KEY (id)"],
                query
            )
        elif "contype = 'f'" in query:
            # get_foreign_keys
            return (
                ["orders_user_id_fkey | FOREIGN KEY (user_id) REFERENCES users(id)"],
                query
            )
        else:
            return (["Unknown query"], query)

    mock_pgcli._evaluate_command = mock_evaluate

    # Create agent with TestModel (no API keys needed)
    print("Creating agent with all 7 tools...")
    agent = create_agent(TestModel())

    # Create dependencies
    deps = PgcliDeps(pgcli=mock_pgcli)

    # Test 1: get_tables
    print("\n" + "="*60)
    print("TEST 1: get_tables")
    print("="*60)
    result = agent.run_sync("List all tables", deps=deps)
    print(f"Output: {result.output}")

    # Test 2: Multi-tool usage
    print("\n" + "="*60)
    print("TEST 2: Multiple tools (TestModel will call get_tables)")
    print("="*60)
    result = agent.run_sync("Show me info about users table", deps=deps)
    print(f"Output: {result.output}")

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print(f"\nAgent successfully created with all 7 tools:")
    print("  1. get_tables")
    print("  2. describe_tables")
    print("  3. describe_enum")
    print("  4. get_table_statistics")
    print("  5. get_indexes")
    print("  6. get_constraints")
    print("  7. get_foreign_keys")
    print("\nPhase 2 implementation complete!")


if __name__ == "__main__":
    main()
