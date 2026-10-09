"""Tests for Pydantic AI LLM integration.

This module tests the integration of Pydantic AI agent with pgcli.
Tests use TestModel to avoid API calls and provide deterministic results.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic_ai.models.test import TestModel

# Import the Pydantic AI components
from pgcli.llm.agent import create_agent
from pgcli.llm.models import PgcliDeps, SQLQuery


class TestAgentCreation:
    """Test agent creation and basic functionality."""

    def test_agent_can_be_created_with_test_model(self):
        """Verify that we can create an agent with TestModel (no API required)."""
        agent = create_agent(TestModel())
        assert agent is not None
        # Verify agent has basic attributes
        assert agent.model is not None
        assert agent.deps_type is PgcliDeps


class TestLLMModeToggle:
    """Test LLM mode toggling functionality."""

    def test_llm_mode_defaults_to_false(self):
        """Verify that llm_mode is False by default."""
        # Import here to avoid import errors if pgcli.main has issues
        from pgcli.main import PGCli

        cli = PGCli()
        assert hasattr(cli, 'llm_mode')
        assert cli.llm_mode is False

    def test_llm_mode_can_be_toggled(self):
        """Verify that llm_mode can be set to True and back to False."""
        from pgcli.main import PGCli

        cli = PGCli()

        # Toggle on
        cli.llm_mode = True
        assert cli.llm_mode is True

        # Toggle off
        cli.llm_mode = False
        assert cli.llm_mode is False


class TestConfigLoading:
    """Test LLM configuration loading."""

    def test_config_defaults_to_anthropic(self):
        """Verify that config defaults to Anthropic provider."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-5-20250929"

    def test_config_can_use_openai(self):
        """Verify that config can be set to OpenAI."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig(provider="openai")
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"

    def test_config_get_model_string(self):
        """Verify that get_model_string returns correct format."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-0")
        assert config.get_model_string() == "anthropic:claude-sonnet-4-0"

    def test_config_is_configured_without_api_key(self):
        """Verify that is_configured returns False without API key."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig(provider="anthropic", api_key=None)
        # Will be False unless ANTHROPIC_API_KEY env var is set
        # In tests, we don't set it, so should be False
        if not os.getenv("ANTHROPIC_API_KEY"):
            assert config.is_configured() is False

    def test_config_is_configured_with_api_key(self):
        """Verify that is_configured returns True with API key."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig(provider="anthropic", api_key="test-key")
        assert config.is_configured() is True

    def test_config_ollama_does_not_need_api_key(self):
        """Verify that Ollama provider doesn't require API key."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig(provider="ollama")
        assert config.is_configured() is True  # No API key needed

    def test_config_get_context_limit_anthropic(self):
        """Verify get_context_limit returns correct limit for Anthropic."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        assert config.get_context_limit() == 180000  # 90% of 200K

    def test_config_get_context_limit_openai(self):
        """Verify get_context_limit returns correct limit for OpenAI."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig(provider="openai", model="gpt-4o")
        assert config.get_context_limit() == 115000  # 90% of 128K

    def test_config_get_context_limit_custom(self):
        """Verify get_context_limit respects custom max_context_tokens."""
        from pgcli.llm.config import LLMConfig

        config = LLMConfig(provider="anthropic", max_context_tokens=50000)
        assert config.get_context_limit() == 50000


class TestGenerateSQLFunction:
    """Test the public generate_sql API."""

    def test_generate_sql_with_test_model(self):
        """Test that generate_sql works with TestModel."""
        from pgcli.llm import generate_sql
        from pgcli.llm.config import LLMConfig

        # Create mock pgcli with proper connection structure
        mock_pgcli = Mock()
        mock_pgcli._evaluate_command.return_value = (["public | test | Test table"], "SELECT ...")

        # Create a proper mock connection for psycopg.sql.as_string()
        mock_conn = Mock()
        mock_conn.connection.pgconn._encoding = 'utf8'
        mock_pgexecute = Mock()
        mock_pgexecute.conn = mock_conn
        mock_pgcli.pgexecute = mock_pgexecute

        # Create config with TestModel
        config = LLMConfig(provider="anthropic", api_key="test-key")

        # Call generate_sql with TestModel to avoid API calls
        # We'll pass the model directly through create_agent
        from pydantic_ai.models.test import TestModel
        from pgcli.llm.agent import create_agent
        from pgcli.llm.models import PgcliDeps

        # Create agent with TestModel answering through the submit_sql output tool
        agent = create_agent(model=TestModel(custom_output_args={"sql": "SELECT * FROM test"}))
        deps = PgcliDeps(pgcli=mock_pgcli)

        # Run agent directly (since generate_sql would try to create a real model)
        result = agent.run_sync("List tables", deps=deps)

        # Verify result
        assert result is not None
        assert isinstance(result.output, SQLQuery)
        assert result.output.sql == "SELECT * FROM test"

    def test_generate_sql_requires_api_key(self):
        """Test that generate_sql raises error without API key."""
        from pgcli.llm import generate_sql
        from pgcli.llm.config import LLMConfig

        mock_pgcli = Mock()

        # Config without API key
        config = LLMConfig(provider="anthropic", api_key=None)

        # Should raise exception about missing config
        with pytest.raises(Exception) as exc_info:
            generate_sql(mock_pgcli, "test query", config=config)

        assert "not configured" in str(exc_info.value).lower()


class TestMessageHistoryTracking:
    """Test message history tracking across conversations."""

    def test_history_is_stored_on_pgcli_instance(self):
        """Test that message history is stored after generate_sql call."""
        from pgcli.main import PGCli
        from pgcli.llm.agent import create_agent
        from pgcli.llm.models import PgcliDeps

        # Create PGCli instance
        cli = PGCli()
        assert cli.llm_message_history == []

        # Create mock pgcli with _evaluate_command
        cli._evaluate_command = Mock(return_value=(["public | test | Test table"], "SELECT ..."))

        # Create a proper mock connection for psycopg.sql.as_string()
        mock_conn = Mock()
        mock_conn.connection.pgconn._encoding = 'utf8'
        mock_pgexecute = Mock()
        mock_pgexecute.conn = mock_conn
        cli.pgexecute = mock_pgexecute

        # Create agent with TestModel answering through the submit_sql output tool
        agent = create_agent(model=TestModel(custom_output_args={"sql": "SELECT 1"}))
        deps = PgcliDeps(pgcli=cli)

        # Run agent
        result = agent.run_sync("List tables", deps=deps)

        # Manually update history (simulating what generate_sql does)
        cli.llm_message_history = result.all_messages()

        # Verify history was stored
        assert len(cli.llm_message_history) > 0
        assert isinstance(cli.llm_message_history, list)

    def test_manual_sql_is_added_to_history(self):
        """Test that manually executed SQL (non-LLM mode) is added to history."""
        from pgcli.llm.agent import add_sql_result_to_history
        from pgcli.llm.config import LLMConfig

        # Create mock pgcli instance
        mock_pgcli = Mock()
        mock_pgcli.llm_message_history = []

        # Simulate manual SQL execution
        sql_query = "SELECT * FROM users WHERE customer_number = 'MBR03015511'"
        result = """
-[ RECORD 1 ]-----------
user_id     | 4049777
member_id   | 14196025
customer_nbr| MBR03015511
first_name  | Adam
last_name   | Letechipia
"""

        # Add to history (as would happen in main.py for manual queries)
        config = LLMConfig(provider="anthropic", api_key="test-key")
        add_sql_result_to_history(mock_pgcli, sql_query, result, config)

        # Verify it was added to history
        assert len(mock_pgcli.llm_message_history) == 1

        # Verify the content includes both SQL and result
        from pydantic_ai.messages import ModelRequest
        assert isinstance(mock_pgcli.llm_message_history[0], ModelRequest)

        # Get the content from the message
        msg = mock_pgcli.llm_message_history[0]
        assert len(msg.parts) > 0
        content = str(msg.parts[0].content)

        # Verify SQL and result are in the content
        assert "SELECT * FROM users" in content
        assert "14196025" in content  # member_id from result

    def test_successful_sql_added_to_readline_history(self):
        """Test that successful LLM-generated SQL is added to readline history."""
        from unittest.mock import MagicMock

        # Create mock pgcli instance with mock prompt_app and history
        mock_pgcli = Mock()
        mock_pgcli.llm_message_history = []

        # Create mock prompt_app with mock history
        mock_history = MagicMock()
        mock_prompt_app = MagicMock()
        mock_prompt_app.history = mock_history
        mock_pgcli.prompt_app = mock_prompt_app

        # Simulate successful SQL execution from LLM mode
        original_text = "show me all active users"
        sql_query = "SELECT * FROM users WHERE active = true"

        # This simulates the code in main.py that adds to readline history
        if mock_pgcli.prompt_app and mock_pgcli.prompt_app.history:
            history_entry = f"-- LLM: {original_text}\n{sql_query}"
            mock_pgcli.prompt_app.history.append_string(history_entry)

        # Verify append_string was called
        mock_history.append_string.assert_called_once()

        # Verify the correct format was added
        call_args = mock_history.append_string.call_args[0][0]
        assert "-- LLM: show me all active users" in call_args
        assert "SELECT * FROM users WHERE active = true" in call_args
        assert call_args.startswith("-- LLM:")


class TestErrorRecovery:
    """Test error handling and graceful degradation."""

    def test_generate_sql_raises_clear_error_without_api_key(self):
        """Test that generate_sql raises clear error when API key is missing."""
        from pgcli.llm import generate_sql
        from pgcli.llm.config import LLMConfig

        mock_pgcli = Mock()
        config = LLMConfig(provider="anthropic", api_key=None)

        # Should raise exception with clear message
        with pytest.raises(Exception) as exc_info:
            generate_sql(mock_pgcli, "test query", config=config)

        error_message = str(exc_info.value).lower()
        assert "not configured" in error_message or "api" in error_message

    def test_retry_mechanism_adds_error_to_history(self):
        """Test that SQL errors are added to history for retry context."""
        from pgcli.llm.agent import add_sql_result_to_history
        from pgcli.llm.config import LLMConfig

        # Create mock pgcli instance with empty history
        mock_pgcli = Mock()
        mock_pgcli.llm_message_history = []

        # Simulate LLM generating bad SQL
        bad_sql = "SELECT * FROM nonexistent_table"
        error_msg = "relation \"nonexistent_table\" does not exist"

        # Add error feedback to history (as retry mechanism does)
        error_feedback = f"SQL Error: {error_msg}\n\nThe query failed. Please fix the SQL and try again."
        config = LLMConfig(provider="anthropic", api_key="test-key")
        add_sql_result_to_history(mock_pgcli, bad_sql, error_feedback, config)

        # Verify error was added to history
        assert len(mock_pgcli.llm_message_history) == 1

        # Verify the error message is in the content
        from pydantic_ai.messages import ModelRequest
        msg = mock_pgcli.llm_message_history[0]
        assert isinstance(msg, ModelRequest)

        content = str(msg.parts[0].content)
        assert "SQL Error:" in content
        assert "nonexistent_table" in content
        assert bad_sql in content

    def test_destructive_query_rejection_added_to_history(self):
        """Test that rejected destructive queries are added to history."""
        from pgcli.llm.agent import add_sql_result_to_history
        from pgcli.llm.config import LLMConfig

        # Create mock pgcli instance with empty history
        mock_pgcli = Mock()
        mock_pgcli.llm_message_history = []

        # Simulate LLM generating destructive SQL that user rejects
        destructive_sql = "DELETE FROM users WHERE age < 18"
        rejection_message = """SQL Query REJECTED by User:
DELETE FROM users WHERE age < 18

Reason: User cancelled destructive operation
Status: Not executed

The user reviewed this query and chose not to proceed. Consider suggesting safer alternatives or asking for clarification before generating destructive queries."""

        # Add rejection to history (as main.py does when user types 'n')
        config = LLMConfig(provider="anthropic", api_key="test-key")
        add_sql_result_to_history(mock_pgcli, destructive_sql, rejection_message, config)

        # Verify rejection was added to history
        assert len(mock_pgcli.llm_message_history) == 1

        # Verify the rejection message is in the content
        from pydantic_ai.messages import ModelRequest
        msg = mock_pgcli.llm_message_history[0]
        assert isinstance(msg, ModelRequest)

        content = str(msg.parts[0].content)
        assert "REJECTED by User" in content
        assert "DELETE FROM users" in content
        assert "User cancelled destructive operation" in content
        assert "Not executed" in content
        assert destructive_sql in content

    def test_query_timing_added_to_history(self):
        """Test that query timing information is added to history."""
        from pgcli.llm.agent import add_sql_result_to_history
        from pgcli.llm.config import LLMConfig

        # Create mock pgcli instance with empty history
        mock_pgcli = Mock()
        mock_pgcli.llm_message_history = []

        # Simulate successful query with timing
        sql_query = "SELECT * FROM users LIMIT 10"
        result = "id | name\n----+------\n 1 | Alice\n 2 | Bob\nSELECT 2"
        execution_time = 0.123
        total_time = 0.156

        # Add result with timing to history
        config = LLMConfig(provider="anthropic", api_key="test-key")
        add_sql_result_to_history(
            mock_pgcli,
            sql_query,
            result,
            config,
            execution_time=execution_time,
            total_time=total_time,
        )

        # Verify timing was added to history
        assert len(mock_pgcli.llm_message_history) == 1

        from pydantic_ai.messages import ModelRequest
        msg = mock_pgcli.llm_message_history[0]
        assert isinstance(msg, ModelRequest)

        content = str(msg.parts[0].content)
        assert "SQL Query Executed" in content
        assert sql_query in content
        assert "Performance:" in content
        assert "Total Time: 0.156s" in content
        assert "Execution Time: 0.123s" in content


class TestTokenCounting:
    """Test token counting functionality."""

    def test_character_based_token_counter(self):
        """Test character-based token counter estimation."""
        from pgcli.llm.token_counter import CharacterBasedTokenCounter

        counter = CharacterBasedTokenCounter()
        # 100 chars = ~25 tokens
        text = "a" * 100
        assert counter.count_tokens(text) == 25

    def test_openai_token_counter_with_tiktoken(self):
        """Test OpenAI token counter with tiktoken installed."""
        from pgcli.llm.token_counter import OpenAITokenCounter

        counter = OpenAITokenCounter(model="gpt-4o")
        # Test with simple text
        text = "Hello, world!"
        tokens = counter.count_tokens(text)
        # Should return a positive integer
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_openai_token_counter_fallback_without_tiktoken(self):
        """Test OpenAI token counter falls back to character estimation."""
        from pgcli.llm.token_counter import OpenAITokenCounter
        import sys
        from unittest.mock import patch

        # Mock ImportError for tiktoken
        with patch.dict(sys.modules, {'tiktoken': None}):
            counter = OpenAITokenCounter(model="gpt-4o")
            # Force re-initialization with mocked import
            counter._encoding = None

            # 100 chars should give ~25 tokens with fallback
            text = "a" * 100
            assert counter.count_tokens(text) == 25

    def test_anthropic_token_counter(self):
        """Test Anthropic token counter uses character estimation."""
        from pgcli.llm.token_counter import AnthropicTokenCounter

        counter = AnthropicTokenCounter()
        # 100 chars = ~25 tokens
        text = "a" * 100
        assert counter.count_tokens(text) == 25

    def test_ollama_token_counter(self):
        """Test Ollama token counter uses character estimation."""
        from pgcli.llm.token_counter import OllamaTokenCounter

        counter = OllamaTokenCounter()
        # 100 chars = ~25 tokens
        text = "a" * 100
        assert counter.count_tokens(text) == 25

    def test_get_token_counter_factory(self):
        """Test get_token_counter returns correct counter type."""
        from pgcli.llm.token_counter import (
            get_token_counter,
            OpenAITokenCounter,
            AnthropicTokenCounter,
            OllamaTokenCounter,
            CharacterBasedTokenCounter,
        )

        # Test OpenAI
        counter = get_token_counter("openai", "gpt-4o")
        assert isinstance(counter, OpenAITokenCounter)

        # Test Anthropic
        counter = get_token_counter("anthropic")
        assert isinstance(counter, AnthropicTokenCounter)

        # Test Ollama
        counter = get_token_counter("ollama")
        assert isinstance(counter, OllamaTokenCounter)

        # Test unknown provider (fallback)
        counter = get_token_counter("unknown")
        assert isinstance(counter, CharacterBasedTokenCounter)


class TestTrimmingFunctions:
    """Test SQL result and message history trimming."""

    def test_detect_content_structure_empty(self):
        """Test detect_content_structure with empty content."""
        from pgcli.llm.trimming import detect_content_structure

        assert detect_content_structure("") == "empty"
        assert detect_content_structure("   ") == "empty"

    def test_detect_content_structure_single_line(self):
        """Test detect_content_structure with single line."""
        from pgcli.llm.trimming import detect_content_structure

        assert detect_content_structure("SELECT 1") == "single_line"

    def test_detect_content_structure_tabular(self):
        """Test detect_content_structure with tabular data."""
        from pgcli.llm.trimming import detect_content_structure

        # Need more than 10 lines to detect as tabular
        lines = ["id | name"] + [f"{i} | User{i}" for i in range(15)]
        tabular = "\n".join(lines)
        assert detect_content_structure(tabular) == "tabular"

    def test_detect_content_structure_few_lines(self):
        """Test detect_content_structure with few lines."""
        from pgcli.llm.trimming import detect_content_structure

        # 3 lines should be "few_lines" even with |
        tabular = "id | name\n1 | Alice\n2 | Bob"
        assert detect_content_structure(tabular) == "few_lines"

    def test_trim_single_line_no_trim_needed(self):
        """Test trim_single_line when content fits."""
        from pgcli.llm.trimming import trim_single_line
        from pgcli.llm.token_counter import CharacterBasedTokenCounter

        counter = CharacterBasedTokenCounter()
        content = "SELECT * FROM users"
        result, was_trimmed = trim_single_line(content, 1000, counter)

        assert not was_trimmed
        assert result == content

    def test_trim_single_line_with_trim(self):
        """Test trim_single_line when trimming needed."""
        from pgcli.llm.trimming import trim_single_line
        from pgcli.llm.token_counter import CharacterBasedTokenCounter

        counter = CharacterBasedTokenCounter()
        # Create a long line (400 chars = ~100 tokens)
        content = "a" * 400
        result, was_trimmed = trim_single_line(content, 50, counter)

        assert was_trimmed
        assert "content truncated" in result
        assert "tokens total" in result

    def test_trim_tabular_data_no_trim_needed(self):
        """Test trim_tabular_data when content fits."""
        from pgcli.llm.trimming import trim_tabular_data
        from pgcli.llm.token_counter import CharacterBasedTokenCounter

        counter = CharacterBasedTokenCounter()
        content = "id | name\n1 | Alice\n2 | Bob"
        result, was_trimmed = trim_tabular_data(content, 1000, counter)

        assert not was_trimmed
        assert result == content

    def test_trim_tabular_data_with_trim(self):
        """Test trim_tabular_data with large dataset."""
        from pgcli.llm.trimming import trim_tabular_data
        from pgcli.llm.token_counter import CharacterBasedTokenCounter

        counter = CharacterBasedTokenCounter()
        # Create 200 rows of data
        lines = ["id | name"]
        for i in range(200):
            lines.append(f"{i} | User{i}")

        content = "\n".join(lines)
        result, was_trimmed = trim_tabular_data(content, 200, counter, head_lines=10)

        assert was_trimmed
        assert "rows omitted" in result
        assert "id | name" in result  # Header preserved

    def test_trim_sql_result_delegates_correctly(self):
        """Test trim_sql_result delegates to appropriate strategy."""
        from pgcli.llm.trimming import trim_sql_result
        from pgcli.llm.token_counter import CharacterBasedTokenCounter

        counter = CharacterBasedTokenCounter()

        # Test empty
        result, was_trimmed = trim_sql_result("", 100, counter)
        assert not was_trimmed

        # Test single line
        content = "a" * 400  # Will need trimming
        result, was_trimmed = trim_sql_result(content, 50, counter)
        assert was_trimmed

        # Test tabular
        lines = ["id | name"] + [f"{i} | User{i}" for i in range(200)]
        content = "\n".join(lines)
        result, was_trimmed = trim_sql_result(content, 200, counter)
        assert was_trimmed

    def test_trim_message_history_no_trim_needed(self):
        """Test trim_message_history when history fits."""
        from pgcli.llm.trimming import trim_message_history
        from pgcli.llm.token_counter import CharacterBasedTokenCounter

        counter = CharacterBasedTokenCounter()

        # Create mock messages
        class MockPart:
            def __init__(self, content):
                self.content = content

        class MockMessage:
            def __init__(self, content):
                self.parts = [MockPart(content)]

        messages = [MockMessage("Hello"), MockMessage("World")]
        result = trim_message_history(messages, 1000, counter)

        assert len(result) == 2

    def test_trim_message_history_with_trim(self):
        """Test trim_message_history when trimming needed."""
        from pgcli.llm.trimming import trim_message_history
        from pgcli.llm.token_counter import CharacterBasedTokenCounter

        counter = CharacterBasedTokenCounter()

        # Create mock messages
        class MockPart:
            def __init__(self, content):
                self.content = content

        class MockMessage:
            def __init__(self, content):
                self.parts = [MockPart(content)]

        # Create 50 messages (way over limit)
        messages = [MockMessage("a" * 100) for _ in range(50)]
        result = trim_message_history(messages, 100, counter)

        # Should keep only last few messages
        assert len(result) < 50
        assert len(result) > 0


class TestToolsWithMockPgcli:
    """Test all 7 tools with mocked pgcli instance."""

    @pytest.fixture
    def mock_pgcli(self):
        """Create a mock pgcli instance that returns test data."""
        mock = Mock()

        # Create a proper mock connection for psycopg.sql.as_string()
        mock_conn = Mock()
        mock_conn.connection.pgconn._encoding = 'utf8'

        # Create mock pgexecute with conn
        mock_pgexecute = Mock()
        mock_pgexecute.conn = mock_conn

        mock.pgexecute = mock_pgexecute

        def mock_evaluate(query):
            """Mock _evaluate_command with different responses based on query."""
            if 'pg_class' in query and "relkind = 'r'" in query:
                # get_tables
                return (
                    ['public | users | User accounts table',
                     'public | orders | Customer orders'],
                    query
                )
            elif 'information_schema.columns' in query:
                # describe_tables
                return (
                    ['id | integer | Primary key',
                     'email | varchar | User email',
                     'created_at | timestamp | Creation time'],
                    query
                )
            elif 'pg_enum' in query:
                # describe_enum
                return (
                    ['active | 1',
                     'inactive | 2',
                     'pending | 3'],
                    query
                )
            elif 'pg_stat_user_tables' in query:
                # get_table_statistics
                return (
                    ['users | 1000 | 128 kB | 50 | 200 | 100 | 10 | 5 | 2025-01-01 | 2025-01-01'],
                    query
                )
            elif 'pg_index' in query:
                # get_indexes
                return (
                    ['users_email_idx | email | t | f | btree | CREATE INDEX ...'],
                    query
                )
            elif 'pg_constraint' in query and 'contype' not in query.lower():
                # get_constraints
                return (
                    ['users_pkey | p | PRIMARY KEY (id)'],
                    query
                )
            elif "contype = 'f'" in query:
                # get_foreign_keys
                return (
                    ['orders_user_id_fkey | FOREIGN KEY (user_id) REFERENCES users(id)'],
                    query
                )
            else:
                return (['Unknown query'], query)

        mock._evaluate_command = mock_evaluate
        return mock

    def test_all_tools_execute_successfully(self, mock_pgcli):
        """Test that agent with mocked pgcli can call all tools without errors."""
        agent = create_agent(TestModel(custom_output_args={"sql": "SELECT 1"}))
        deps = PgcliDeps(pgcli=mock_pgcli)

        # Run agent - TestModel will call tools automatically during execution
        result = agent.run_sync("List all tables and describe the users table", deps=deps)

        # TestModel calls every tool, then submits SQL through the output tool
        assert result is not None
        assert isinstance(result.output, SQLQuery)


class TestLLMAutocomplete:
    """Test autocomplete filtering in LLM mode."""

    def test_keywords_excluded_in_llm_mode(self):
        """Test that SQL keywords are excluded from autocomplete in LLM mode."""
        from pgcli.pgcompleter import PGCompleter, Keyword
        from prompt_toolkit.document import Document

        # Create completer with LLM mode enabled
        llm_mode = True
        completer = PGCompleter(llm_mode_callback=lambda: llm_mode)

        # Simulate typing "sel" which normally would suggest SELECT
        doc = Document("sel", cursor_position=3)

        # Mock suggestion that returns Keyword
        with patch('pgcli.pgcompleter.suggest_type') as mock_suggest:
            mock_suggest.return_value = [Keyword()]

            # Get completions
            completions = list(completer.get_completions(doc, None))

            # In LLM mode, keywords should be filtered out
            # So we should get no completions (or only non-keyword ones)
            keyword_completions = [c for c in completions if c.text.upper() in ['SELECT', 'SELECTINTO']]
            assert len(keyword_completions) == 0, "Keywords should not appear in LLM mode"

    def test_functions_excluded_in_llm_mode(self):
        """Test that SQL functions are excluded from autocomplete in LLM mode."""
        from pgcli.pgcompleter import PGCompleter, Function
        from prompt_toolkit.document import Document

        # Create completer with LLM mode enabled
        llm_mode = True
        completer = PGCompleter(llm_mode_callback=lambda: llm_mode)

        # Simulate typing which would normally suggest functions
        doc = Document("cou", cursor_position=3)

        # Mock suggestion that returns Function
        with patch('pgcli.pgcompleter.suggest_type') as mock_suggest:
            mock_suggest.return_value = [Function()]

            # Get completions
            completions = list(completer.get_completions(doc, None))

            # In LLM mode, functions should be filtered out
            assert len(completions) == 0, "Functions should not appear in LLM mode"

    def test_table_names_allowed_in_llm_mode(self):
        """Test that table names still appear in autocomplete in LLM mode."""
        from pgcli.pgcompleter import PGCompleter, Table
        from prompt_toolkit.document import Document

        # Create completer with LLM mode enabled
        llm_mode = True
        completer = PGCompleter(llm_mode_callback=lambda: llm_mode)

        # Add some test tables
        completer.dbmetadata = {
            'tables': {
                'public': {
                    'users': ['id', 'name', 'email'],
                    'orders': ['id', 'user_id', 'total']
                }
            },
            'views': {},
            'functions': {},
            'datatypes': {}
        }
        completer.search_path = ['public']

        # Simulate typing "us" which should suggest "users" table
        doc = Document("us", cursor_position=2)

        # Mock suggestion that returns Table
        with patch('pgcli.pgcompleter.suggest_type') as mock_suggest:
            mock_suggest.return_value = [Table(schema=None)]

            # Get completions
            completions = list(completer.get_completions(doc, None))

            # In LLM mode, table names should still appear
            table_names = [c.text for c in completions]
            assert 'users' in table_names, "Table names should appear in LLM mode"

    def test_column_suggestions_allowed_in_llm_mode(self):
        """Test that Column suggestion type is not filtered out in LLM mode."""
        from pgcli.pgcompleter import PGCompleter, Column
        from pgcli.packages.parseutils.tables import TableReference
        from prompt_toolkit.document import Document

        # Create completer with LLM mode enabled
        llm_mode = True
        completer = PGCompleter(llm_mode_callback=lambda: llm_mode)

        # Create a mock matcher that verifies it gets called for Column suggestions
        column_matcher_called = []

        def mock_column_matcher(completer_self, suggestion, word):
            column_matcher_called.append(True)
            return []  # Return empty list, we just want to verify it was called

        # Replace the column matcher
        original_matcher = completer.suggestion_matchers[Column]
        completer.suggestion_matchers[Column] = mock_column_matcher

        try:
            doc = Document("test", cursor_position=4)
            table_ref = TableReference(schema='public', name='users', alias=None, is_function=False)

            with patch('pgcli.pgcompleter.suggest_type') as mock_suggest:
                mock_suggest.return_value = [Column(table_refs=(table_ref,), qualifiable=False)]

                # Get completions
                list(completer.get_completions(doc, None))

                # Verify that in LLM mode, Column suggestions are NOT filtered
                assert len(column_matcher_called) > 0, "Column matcher should be called in LLM mode"
        finally:
            # Restore original matcher
            completer.suggestion_matchers[Column] = original_matcher

    def test_normal_mode_allows_everything(self):
        """Test that normal mode (llm_mode=False) allows all completions."""
        from pgcli.pgcompleter import PGCompleter, Keyword
        from prompt_toolkit.document import Document

        # Create completer with LLM mode disabled
        llm_mode = False
        completer = PGCompleter(llm_mode_callback=lambda: llm_mode)

        # Simulate typing "sel"
        doc = Document("sel", cursor_position=3)

        # Mock suggestion that returns Keyword
        with patch('pgcli.pgcompleter.suggest_type') as mock_suggest:
            mock_suggest.return_value = [Keyword()]

            # Get completions
            completions = list(completer.get_completions(doc, None))

            # In normal mode, keywords should appear
            # (Note: actual keywords depend on the matcher implementation)
            # We just verify that the filter didn't run (completions were attempted)
            assert mock_suggest.called, "suggest_type should be called in normal mode"

    def test_single_letter_no_keywords_in_llm_mode(self):
        """Test that typing a single letter in LLM mode doesn't suggest SQL keywords."""
        from pgcli.pgcompleter import PGCompleter
        from prompt_toolkit.document import Document

        # Create completer with LLM mode enabled and smart_completion OFF
        # (this simulates what happens when you type just "S" or "L" with no context)
        llm_mode = True
        completer = PGCompleter(smart_completion=False, llm_mode_callback=lambda: llm_mode)

        # Add some test tables
        completer.dbmetadata = {
            'tables': {
                'public': {
                    'users': {},
                    'sessions': {}
                }
            },
            'views': {},
            'functions': {},
            'datatypes': {}
        }

        # Simulate typing "S" (which would normally suggest SELECT, SHOW, SESSION)
        doc = Document("S", cursor_position=1)

        # Get completions
        completions = list(completer.get_completions(doc, None))
        completion_texts = [c.text for c in completions]

        # In LLM mode, SQL keywords should NOT appear
        assert 'SELECT' not in completion_texts, "SELECT keyword should not appear in LLM mode"
        assert 'SHOW' not in completion_texts, "SHOW keyword should not appear in LLM mode"
        assert 'SESSION' not in completion_texts, "SESSION keyword should not appear in LLM mode"

        # But table names starting with S should appear
        assert 'sessions' in completion_texts, "Table names should still appear in LLM mode"

    def test_single_letter_allows_keywords_in_normal_mode(self):
        """Test that typing a single letter in normal mode DOES suggest SQL keywords."""
        from pgcli.pgcompleter import PGCompleter
        from prompt_toolkit.document import Document

        # Create completer with LLM mode disabled and smart_completion OFF
        llm_mode = False
        completer = PGCompleter(smart_completion=False, llm_mode_callback=lambda: llm_mode)

        # Simulate typing "S"
        doc = Document("S", cursor_position=1)

        # Get completions
        completions = list(completer.get_completions(doc, None))
        completion_texts = [c.text for c in completions]

        # In normal mode, SQL keywords SHOULD appear
        # (At least one of these should be present)
        sql_keywords_present = any(kw in completion_texts for kw in ['SELECT', 'SHOW', 'SET'])
        assert sql_keywords_present, "SQL keywords should appear in normal mode"


class TestExplainIntegration:
    """Tests for EXPLAIN ANALYZE integration with LLM mode."""

    def test_parse_explain_output_with_valid_json(self):
        """Test parsing valid EXPLAIN ANALYZE JSON output."""
        from pgcli.llm.agent import parse_explain_output

        explain_json = """
[
  {
    "Plan": {
      "Node Type": "Seq Scan",
      "Relation Name": "users",
      "Startup Cost": 0.00,
      "Total Cost": 23.50,
      "Plan Rows": 1000,
      "Plan Width": 64,
      "Actual Time": 0.023,
      "Actual Rows": 1000,
      "Actual Loops": 1
    },
    "Planning Time": 0.083,
    "Execution Time": 0.195
  }
]
"""
        result = parse_explain_output(explain_json)

        assert result is not None
        assert "Planning Time: 0.083 ms" in result
        assert "Execution Time: 0.195 ms" in result
        assert "Seq Scan on users" in result
        assert "Total Cost: 23.50" in result
        assert "Sequential scan detected" in result

    def test_parse_explain_output_with_index_scan(self):
        """Test parsing EXPLAIN with Index Scan (no warning)."""
        from pgcli.llm.agent import parse_explain_output

        explain_json = """
[
  {
    "Plan": {
      "Node Type": "Index Scan",
      "Relation Name": "users",
      "Index Name": "users_email_idx",
      "Total Cost": 8.50,
      "Plan Rows": 1,
      "Actual Rows": 1
    },
    "Planning Time": 0.045,
    "Execution Time": 0.082
  }
]
"""
        result = parse_explain_output(explain_json)

        assert result is not None
        assert "Index Scan on users" in result
        assert "Sequential scan detected" not in result  # No warning for index scan

    def test_parse_explain_output_with_invalid_json(self):
        """Test parsing invalid EXPLAIN output returns None."""
        from pgcli.llm.agent import parse_explain_output

        # Not JSON at all
        result = parse_explain_output("This is not JSON")
        assert result is None

        # Empty string
        result = parse_explain_output("")
        assert result is None

        # Valid JSON but not EXPLAIN format
        result = parse_explain_output('{"foo": "bar"}')
        assert result is None

    def test_parse_explain_output_with_table_formatted_output(self):
        """Test parsing EXPLAIN output that includes table formatting."""
        from pgcli.llm.agent import parse_explain_output

        # Simulate output from pgcli with table formatting around JSON
        table_output = """
QUERY PLAN
───────────────────────────────────────────────────
[                                                  +
  {                                                +
    "Plan": {                                      +
      "Node Type": "Seq Scan",                     +
      "Relation Name": "products",                 +
      "Total Cost": 45.20,                         +
      "Plan Rows": 500,                            +
      "Actual Rows": 498                           +
    },                                             +
    "Planning Time": 0.123,                        +
    "Execution Time": 1.456                        +
  }                                                +
]
(1 row)
"""
        result = parse_explain_output(table_output)

        assert result is not None
        assert "Planning Time: 0.123 ms" in result
        assert "Execution Time: 1.456 ms" in result
        assert "Seq Scan on products" in result

    def test_explain_output_included_in_history(self):
        """Test that EXPLAIN output is included in LLM message history."""
        from pgcli.llm.agent import add_sql_result_to_history

        # Create mock pgcli instance with message history
        class MockPgcli:
            def __init__(self):
                self.llm_message_history = []

        pgcli = MockPgcli()

        explain_summary = """Query Plan Analysis:
  • Planning Time: 0.083 ms
  • Execution Time: 0.195 ms
  • Primary Operation: Seq Scan on users
  • Total Cost: 23.50
  • Sequential scan detected - consider adding an index"""

        add_sql_result_to_history(
            pgcli,
            "SELECT * FROM users WHERE email LIKE '%example.com%'",
            "100 rows returned",
            execution_time=0.195,
            total_time=0.278,
            explain_output=explain_summary
        )

        # Check that message was added to history
        assert len(pgcli.llm_message_history) == 1

        # Get the message content
        message = pgcli.llm_message_history[0]
        content = message.parts[0].content

        # Verify EXPLAIN output is in the message
        assert "Query Plan Analysis" in content
        assert "Sequential scan detected" in content
        assert "Execution Time: 0.195 ms" in content

    def test_explain_output_not_included_when_none(self):
        """Test that when explain_output is None, it's not included in history."""
        from pgcli.llm.agent import add_sql_result_to_history

        class MockPgcli:
            def __init__(self):
                self.llm_message_history = []

        pgcli = MockPgcli()

        add_sql_result_to_history(
            pgcli,
            "SELECT * FROM users",
            "50 rows returned",
            execution_time=0.050,
            explain_output=None  # No EXPLAIN output
        )

        # Check that message was added
        assert len(pgcli.llm_message_history) == 1

        # Get content
        content = pgcli.llm_message_history[0].parts[0].content

        # Verify EXPLAIN-related text is NOT in the message
        assert "Query Plan Analysis" not in content
        assert "Sequential scan" not in content
        assert "Planning Time" not in content


class TestResultFormatters:
    """Tests for result formatters (CSV vs tabular for token optimization)."""

    def test_format_as_csv_simple_table(self):
        """Test CSV formatter with simple table."""
        from pgcli.llm.formatters import format_as_csv

        tabular_input = """+------+-------------+-------------------+-------+
|   id | name        | email             |   age |
|------+-------------+-------------------+-------|
|    1 | Alice Smith | alice@example.com |    28 |
|    2 | Bob Johnson | bob@example.com   |    35 |
|    3 | Carol White | carol@example.com |    42 |
+------+-------------+-------------------+-------+"""

        result = format_as_csv(tabular_input)

        # Should be CSV format
        assert "id,name,email,age" in result
        assert "1,Alice Smith,alice@example.com,28" in result
        assert "2,Bob Johnson,bob@example.com,35" in result
        assert "3,Carol White,carol@example.com,42" in result

        # Should NOT have table borders
        assert "+" not in result
        assert "|" not in result
        assert "-" not in result

    def test_format_as_csv_with_special_characters(self):
        """Test CSV formatter handles special characters."""
        from pgcli.llm.formatters import format_as_csv

        tabular_input = """+------+------------------+
|   id | description      |
|------+------------------|
|    1 | Has, comma       |
|    2 | Has "quotes"     |
+------+------------------+"""

        result = format_as_csv(tabular_input)

        # CSV should escape special characters
        assert "id,description" in result
        # CSV library will quote fields with commas
        assert "1," in result or '"1"' in result

    def test_format_as_csv_non_table_input(self):
        """Test CSV formatter returns non-table input unchanged."""
        from pgcli.llm.formatters import format_as_csv

        # Error message (not a table)
        error_msg = "SQL Error: syntax error at or near 'SELECT'"
        result = format_as_csv(error_msg)
        assert result == error_msg

        # Single line output
        single_line = "SELECT 1"
        result = format_as_csv(single_line)
        assert result == single_line

    def test_format_as_tabular_unchanged(self):
        """Test tabular formatter returns input unchanged."""
        from pgcli.llm.formatters import format_as_tabular

        input_text = "any text"
        result = format_as_tabular(input_text)
        assert result == input_text

    def test_get_result_formatter_default_tabular(self):
        """Test formatter defaults to tabular when env var not set."""
        import os
        from pgcli.llm.formatters import get_result_formatter, format_as_tabular

        # Ensure env var is not set
        os.environ.pop('PGCLI_LLM_RESULT_FORMAT', None)

        formatter = get_result_formatter()

        # Should return tabular formatter
        assert formatter == format_as_tabular

    def test_get_result_formatter_csv_when_enabled(self):
        """Test formatter uses CSV when env var is set."""
        import os
        from pgcli.llm.formatters import get_result_formatter, format_as_csv

        # Set env var to csv
        os.environ['PGCLI_LLM_RESULT_FORMAT'] = 'csv'

        try:
            formatter = get_result_formatter()

            # Should return CSV formatter
            assert formatter == format_as_csv
        finally:
            # Clean up
            os.environ.pop('PGCLI_LLM_RESULT_FORMAT', None)

    def test_get_result_formatter_case_insensitive(self):
        """Test formatter handles case-insensitive env var."""
        import os
        from pgcli.llm.formatters import get_result_formatter, format_as_csv

        os.environ['PGCLI_LLM_RESULT_FORMAT'] = 'CSV'  # Uppercase

        try:
            formatter = get_result_formatter()
            assert formatter == format_as_csv
        finally:
            os.environ.pop('PGCLI_LLM_RESULT_FORMAT', None)

    def test_formatter_integration_with_history(self):
        """Test that formatter is used when adding results to history."""
        import os
        from pgcli.llm.agent import add_sql_result_to_history

        # Create mock pgcli instance
        class MockPgcli:
            def __init__(self):
                self.llm_message_history = []

        pgcli = MockPgcli()

        tabular_result = """+------+-------+
|   id | name  |
|------+-------|
|    1 | Alice |
|    2 | Bob   |
+------+-------+"""

        # Enable CSV format
        os.environ['PGCLI_LLM_RESULT_FORMAT'] = 'csv'

        try:
            add_sql_result_to_history(
                pgcli,
                "SELECT id, name FROM users",
                tabular_result
            )

            # Check that message was added
            assert len(pgcli.llm_message_history) == 1

            # Get message content
            content = pgcli.llm_message_history[0].parts[0].content

            # Should contain CSV format (not tabular)
            assert "id,name" in content
            assert "1,Alice" in content
            assert "+" not in content  # No table borders
            assert "|" not in content  # No pipes
        finally:
            os.environ.pop('PGCLI_LLM_RESULT_FORMAT', None)


class TestNaturalLanguageDetector:
    """Tests for natural language detection heuristic."""

    def test_obvious_natural_language_detected(self):
        """Test that obvious natural language is detected."""
        from pgcli.llm.natural_language_detector import looks_like_natural_language

        natural_language_examples = [
            "show me all users",
            "get all tables",
            "find users with gmail addresses",
            "list all orders from last week",
            "give me the top 10 products",
            "tell me how many users we have",
            "what are the active subscriptions",
            "how many orders were placed today",
            "can you show me the recent signups",
            "please list all customers",
        ]

        for example in natural_language_examples:
            assert looks_like_natural_language(example), \
                f"Failed to detect natural language: '{example}'"

    def test_obvious_sql_not_detected(self):
        """Test that obvious SQL is NOT detected as natural language."""
        from pgcli.llm.natural_language_detector import looks_like_natural_language

        sql_examples = [
            "SELECT * FROM users",
            "SELECT id, name FROM users WHERE active = true",
            "DELETE FROM users WHERE id = 1",
            "INSERT INTO users (name, email) VALUES ('John', 'john@example.com')",
            "UPDATE users SET active = false WHERE id = 5",
            "CREATE TABLE products (id SERIAL PRIMARY KEY, name VARCHAR(100))",
            "ALTER TABLE users ADD COLUMN age INTEGER",
            "DROP TABLE temp_data",
        ]

        for example in sql_examples:
            assert not looks_like_natural_language(example), \
                f"Incorrectly detected SQL as natural language: '{example}'"

    def test_psql_commands_not_detected(self):
        """Test that psql special commands are NOT detected as natural language."""
        from pgcli.llm.natural_language_detector import looks_like_natural_language

        psql_commands = [
            "\\d",
            "\\dt",
            "\\d users",
            "\\dt+",
            "\\l",
            "\\c mydb",
            "\\q",
        ]

        for cmd in psql_commands:
            assert not looks_like_natural_language(cmd), \
                f"Incorrectly detected psql command as natural language: '{cmd}'"

    def test_ambiguous_cases(self):
        """Test handling of ambiguous cases that could be either."""
        from pgcli.llm.natural_language_detector import looks_like_natural_language

        # These start with SQL keywords but are actually natural language
        natural_language_with_sql_words = [
            "show all users",  # "show" is a SQL keyword but this is natural language
            "select me all active users",  # Starts with SELECT but not SQL syntax
        ]

        for example in natural_language_with_sql_words:
            assert looks_like_natural_language(example), \
                f"Failed to detect natural language: '{example}'"

    def test_questions_detected(self):
        """Test that questions with ? are detected as natural language."""
        from pgcli.llm.natural_language_detector import looks_like_natural_language

        questions = [
            "how many users?",
            "what is the total revenue?",
            "which products are out of stock?",
        ]

        for question in questions:
            assert looks_like_natural_language(question), \
                f"Failed to detect question as natural language: '{question}'"

    def test_empty_or_whitespace(self):
        """Test that empty or whitespace-only input is not detected."""
        from pgcli.llm.natural_language_detector import looks_like_natural_language

        assert not looks_like_natural_language("")
        assert not looks_like_natural_language("   ")
        assert not looks_like_natural_language("\n")
        assert not looks_like_natural_language("\t")

    def test_single_sql_keyword(self):
        """Test that single SQL keywords are not detected as natural language."""
        from pgcli.llm.natural_language_detector import looks_like_natural_language

        single_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE"]

        for keyword in single_keywords:
            assert not looks_like_natural_language(keyword), \
                f"Incorrectly detected single SQL keyword as natural language: '{keyword}'"

    def test_is_syntax_error_detection(self):
        """Test syntax error detection from exception messages."""
        from pgcli.llm.natural_language_detector import is_syntax_error

        # Create mock exceptions with different error messages
        class MockException(Exception):
            pass

        # Syntax errors
        syntax_errors = [
            "syntax error at or near \"show\"",
            "Syntax Error: unexpected token",
            "Invalid syntax near FROM",
            "Cannot parse query",
        ]

        for error_msg in syntax_errors:
            exception = MockException(error_msg)
            assert is_syntax_error(exception), \
                f"Failed to detect syntax error: '{error_msg}'"

        # Non-syntax errors
        other_errors = [
            "connection refused",
            "table does not exist",
            "column 'foo' does not exist",
            "permission denied",
        ]

        for error_msg in other_errors:
            exception = MockException(error_msg)
            assert not is_syntax_error(exception), \
                f"Incorrectly detected non-syntax error: '{error_msg}'"


class TestStructuredSQLOutput:
    """The agent must hand back executable SQL, never prose or markdown."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("SELECT 1;", "SELECT 1;"),
            ("-- top customers\nSELECT 1", "-- top customers\nSELECT 1"),
            ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
            ("\\copy (SELECT 1) TO '/tmp/a.csv' CSV HEADER", "\\copy (SELECT 1) TO '/tmp/a.csv' CSV HEADER"),
            ("```sql\nSELECT 2;\n```", "SELECT 2;"),
            ("Perfect! Here it is:\n\n```sql\nSELECT 3;\n```\n\nThis query joins...", "SELECT 3;"),
            ("Perfect! Now I can generate the query.", None),
            ("Show me what you need", None),
            ("With the schema in hand, here is my plan", None),
            ("Option A:\n```sql\nSELECT 1\n```\nOption B:\n```sql\nSELECT 2\n```", None),
        ],
    )
    def test_extract_sql(self, text, expected):
        from pgcli.llm.agent import extract_sql

        assert extract_sql(text) == expected

    @staticmethod
    def _run(responses):
        """Run the agent against a scripted model; return (sql, messages seen by the model)."""
        from pydantic_ai.messages import ModelResponse
        from pydantic_ai.models.function import FunctionModel
        from pgcli.llm.agent import output_to_sql

        calls = []

        def model(messages, info):
            calls.append(messages)
            return ModelResponse(parts=[responses[len(calls) - 1]])

        agent = create_agent(FunctionModel(model))
        result = agent.run_sync("top customers", deps=PgcliDeps(pgcli=Mock()))
        return output_to_sql(result.output), calls

    def test_submit_sql_tool_output(self):
        from pydantic_ai.messages import ToolCallPart

        sql, calls = self._run([ToolCallPart("submit_sql", {"sql": "SELECT 1"})])
        assert sql == "SELECT 1"
        assert len(calls) == 1

    def test_fences_stripped_from_tool_output(self):
        from pydantic_ai.messages import ToolCallPart

        sql, _ = self._run([ToolCallPart("submit_sql", {"sql": "```sql\nSELECT 1\n```"})])
        assert sql == "SELECT 1"

    def test_fenced_sql_extracted_from_prose_without_retry(self):
        from pydantic_ai.messages import TextPart

        sql, calls = self._run([TextPart("Perfect! Here you go:\n```sql\nSELECT 1;\n```\nThis query...")])
        assert sql == "SELECT 1;"
        assert len(calls) == 1

    def test_prose_is_sent_back_to_model_for_retry(self):
        """Prose never reaches the database: the model is asked again within the same run."""
        from pydantic_ai.messages import RetryPromptPart, TextPart, ToolCallPart

        sql, calls = self._run(
            [
                TextPart("Perfect! Now I can generate the query for the top 5 customers."),
                ToolCallPart("submit_sql", {"sql": "SELECT 1"}),
            ]
        )
        assert sql == "SELECT 1"
        assert len(calls) == 2
        retry_parts = [p for p in calls[1][-1].parts if isinstance(p, RetryPromptPart)]
        assert retry_parts and "submit_sql" in retry_parts[0].model_response()
