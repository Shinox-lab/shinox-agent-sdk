"""
Tests for shinox_agent.tools.history — History pattern detection and formatting.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from shinox_agent.tools.history import (
    should_auto_fetch_history,
    format_history_as_context,
    format_results_as_context,
    get_session_history,
    get_session_results,
    HistoryMessage,
    SessionHistory,
)


class TestShouldAutoFetchHistory:
    """Test heuristic pattern matching for auto-history injection."""

    # --- Summarization triggers ---
    def test_summarize(self):
        should, patterns = should_auto_fetch_history("Summarize the conversion result")
        assert should is True
        assert len(patterns) > 0

    def test_combine(self):
        should, patterns = should_auto_fetch_history("Combine all the agent results")
        assert should is True

    def test_consolidate(self):
        should, patterns = should_auto_fetch_history("Consolidate findings from all agents")
        assert should is True

    def test_aggregate(self):
        should, patterns = should_auto_fetch_history("Aggregate the data from previous stages")
        assert should is True

    def test_merge(self):
        should, patterns = should_auto_fetch_history("Merge the research into a report")
        assert should is True

    # --- Reference triggers ---
    def test_based_on(self):
        should, patterns = should_auto_fetch_history("Based on the analysis, write a report")
        assert should is True

    def test_previously(self):
        should, patterns = should_auto_fetch_history("Previously discussed options should be evaluated")
        assert should is True

    def test_earlier(self):
        should, patterns = should_auto_fetch_history("As mentioned earlier, the rate is 3.95")
        assert should is True

    def test_what_did_say(self):
        should, patterns = should_auto_fetch_history("What did the accounting agent report?")
        assert should is True

    # --- Dependency triggers ---
    def test_using_the_data(self):
        should, patterns = should_auto_fetch_history("Using the data from stage 1, calculate totals")
        assert should is True

    def test_with_the_information(self):
        should, patterns = should_auto_fetch_history("With the information provided, draft a response")
        assert should is True

    # --- No match (direct tasks) ---
    def test_direct_task(self):
        should, patterns = should_auto_fetch_history("Convert 7 USD to MYR")
        assert should is False
        assert patterns == []

    def test_simple_question(self):
        should, patterns = should_auto_fetch_history("What is the capital of France?")
        assert should is False

    def test_code_task(self):
        should, patterns = should_auto_fetch_history("Write a Python function to sort a list")
        assert should is False

    # --- Multiple patterns ---
    def test_multiple_patterns_matched(self):
        should, patterns = should_auto_fetch_history(
            "Summarize the results based on previous findings"
        )
        assert should is True
        assert len(patterns) >= 2


class TestFormatHistoryAsContext:
    def test_empty_history(self):
        history = SessionHistory(
            session_id="s1", total_count=0, messages=[], has_more=False
        )
        result = format_history_as_context(history)
        assert result == ""

    def test_single_message(self):
        history = SessionHistory(
            session_id="s1",
            total_count=1,
            messages=[
                HistoryMessage(
                    message_id="m1",
                    source_agent_id="accounting-agent",
                    target_agent_id=None,
                    interaction_type="TASK_RESULT",
                    content="7 USD = 27.67 MYR",
                    created_at="2025-01-15T10:30:00Z",
                )
            ],
            has_more=False,
        )
        result = format_history_as_context(history)
        assert "accounting-agent" in result
        assert "TASK_RESULT" in result
        assert "7 USD = 27.67 MYR" in result

    def test_truncation(self):
        long_content = "x" * 3000
        history = SessionHistory(
            session_id="s1",
            total_count=1,
            messages=[
                HistoryMessage(
                    message_id="m1",
                    source_agent_id="agent-1",
                    target_agent_id=None,
                    interaction_type="TASK_RESULT",
                    content=long_content,
                    created_at="2025-01-15T10:30:00Z",
                )
            ],
            has_more=False,
        )
        result = format_history_as_context(history, max_chars_per_message=100)
        assert "..." in result
        # Original 3000 chars should be truncated
        assert len(result) < 3000

    def test_type_filtering(self):
        history = SessionHistory(
            session_id="s1",
            total_count=2,
            messages=[
                HistoryMessage(
                    message_id="m1", source_agent_id="a1",
                    target_agent_id=None, interaction_type="TASK_RESULT",
                    content="result", created_at="2025-01-01",
                ),
                HistoryMessage(
                    message_id="m2", source_agent_id="a2",
                    target_agent_id=None, interaction_type="INFO_UPDATE",
                    content="update", created_at="2025-01-01",
                ),
            ],
            has_more=False,
        )
        result = format_history_as_context(history, include_types=["TASK_RESULT"])
        assert "result" in result
        assert "update" not in result

    def test_target_agent_shown(self):
        history = SessionHistory(
            session_id="s1",
            total_count=1,
            messages=[
                HistoryMessage(
                    message_id="m1", source_agent_id="a1",
                    target_agent_id="a2", interaction_type="TASK_ASSIGNMENT",
                    content="Do this", created_at="2025-01-01",
                )
            ],
            has_more=False,
        )
        result = format_history_as_context(history)
        assert "To: a2" in result


class TestFormatResultsAsContext:
    def test_empty_results(self):
        result = format_results_as_context([])
        assert result == ""

    def test_single_result(self):
        results = [{"agent_id": "accounting-agent", "content": "7 USD = 27.67 MYR"}]
        result = format_results_as_context(results)
        assert "accounting-agent" in result
        assert "7 USD = 27.67 MYR" in result
        assert "Previous Results" in result

    def test_truncation(self):
        results = [{"agent_id": "a1", "content": "x" * 5000}]
        result = format_results_as_context(results, max_chars_per_result=100)
        assert "..." in result

    def test_multiple_results(self):
        results = [
            {"agent_id": "agent-1", "content": "Result 1"},
            {"agent_id": "agent-2", "content": "Result 2"},
        ]
        result = format_results_as_context(results)
        assert "agent-1" in result
        assert "agent-2" in result
        assert "Result 1" in result
        assert "Result 2" in result


class TestGetSessionHistory:
    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "session_id": "session.test",
            "total_count": 1,
            "messages": [
                {
                    "message_id": "m1",
                    "source_agent_id": "a1",
                    "target_agent_id": None,
                    "interaction_type": "TASK_RESULT",
                    "content": "result content",
                    "created_at": "2025-01-15T10:30:00Z",
                    "metadata": {},
                }
            ],
            "has_more": False,
        }

        with patch("shinox_agent.tools.history.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            history = await get_session_history("session.test", "http://localhost:9000")

            assert history.session_id == "session.test"
            assert history.total_count == 1
            assert len(history.messages) == 1
            assert history.messages[0].content == "result content"

    @pytest.mark.asyncio
    async def test_fetch_with_filters(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "session_id": "s1",
            "total_count": 0,
            "messages": [],
            "has_more": False,
        }

        with patch("shinox_agent.tools.history.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await get_session_history(
                "s1", "http://localhost:9000",
                filter_agent="agent-1",
                filter_type="TASK_RESULT",
            )

            # Verify the call was made with filter params
            call_args = mock_client.get.call_args
            params = call_args.kwargs.get("params", {})
            assert params["filter_agent"] == "agent-1"
            assert params["filter_type"] == "TASK_RESULT"


class TestGetSessionResults:
    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "session_id": "s1",
            "count": 1,
            "results": [
                {"agent_id": "a1", "content": "result", "created_at": "2025-01-01"},
            ],
        }

        with patch("shinox_agent.tools.history.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await get_session_results("s1", "http://localhost:9000")

            assert len(results) == 1
            assert results[0]["agent_id"] == "a1"
