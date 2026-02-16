"""
Tests for ShinoxWorkerAgent asynchronous task processing.

Tests the fire-and-forget protocol where agents immediately acknowledge
task receipt, execute brain invocation in background, and publish
TASK_RESULT when complete.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shinox_agent.worker import ShinoxWorkerAgent, BackgroundTask
from shinox_agent.schemas import AgentMessage, A2AHeaders
from tests.conftest import MockAgentCard, make_agent_message


def _make_async_worker(
    agent_name="Async Worker",
    async_mode=True,
    max_concurrent_tasks=10,
    triggers=None,
):
    """Create a ShinoxWorkerAgent with async mode and mocked infrastructure."""
    card = MockAgentCard(name=agent_name)
    brain = AsyncMock()
    brain.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Brain response")]
    })

    with patch("shinox_agent.base.KafkaBroker") as mock_broker_cls, \
         patch("shinox_agent.base.FastStream"), \
         patch("shinox_agent.base.setup_json_logging"):
        mock_broker = MagicMock()
        mock_broker.subscriber = MagicMock(return_value=lambda fn: fn)
        mock_broker.publish = AsyncMock()
        mock_broker_cls.return_value = mock_broker

        agent = ShinoxWorkerAgent(
            agent_card=card,
            brain=brain,
            triggers=triggers,
            enable_semantic_wake=False,
            async_mode=async_mode,
            max_concurrent_tasks=max_concurrent_tasks,
        )
        agent.broker = mock_broker

    return agent


class TestAsyncAcknowledgment:
    """INFO_UPDATE ack published immediately; handler returns before brain completes."""

    @pytest.mark.asyncio
    async def test_ack_published_immediately(self):
        """Handler should publish INFO_UPDATE ack and return before brain finishes."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        # Make brain slow so we can verify handler returns quickly
        brain_started = asyncio.Event()
        brain_done = asyncio.Event()

        async def slow_brain(*args, **kwargs):
            brain_started.set()
            await brain_done.wait()
            return {"messages": [MagicMock(content="Slow brain response")]}

        agent.brain.ainvoke = slow_brain

        msg = make_agent_message(
            content="Do this task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        # Handler should return quickly (async dispatch)
        await agent._default_session_handler(msg, agent)

        # Verify INFO_UPDATE ack was published
        publish_calls = agent.broker.publish.call_args_list
        assert len(publish_calls) >= 1
        ack_call = publish_calls[0]
        ack_msg = ack_call.args[0]
        assert isinstance(ack_msg, AgentMessage)
        assert ack_msg.headers.interaction_type == "INFO_UPDATE"
        assert ack_msg.metadata["status"] == "async_processing"
        assert "task_id" in ack_msg.metadata
        assert ack_msg.metadata["original_message_id"] == msg.id

        # Brain should have started but not completed yet (or be about to start)
        # Let the brain finish for cleanup
        brain_done.set()
        # Give background task time to complete
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_handler_returns_before_brain_completes(self):
        """Handler must exit before brain invocation finishes."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        brain_called = asyncio.Event()

        async def slow_brain(*args, **kwargs):
            brain_called.set()
            await asyncio.sleep(10)  # Very slow
            return {"messages": [MagicMock(content="response")]}

        agent.brain.ainvoke = slow_brain

        msg = make_agent_message(
            content="Long task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        # Handler should return immediately
        await asyncio.wait_for(
            agent._default_session_handler(msg, agent),
            timeout=1.0,
        )

        # Clean up: cancel background tasks
        for bt in agent._background_tasks.values():
            if bt.asyncio_task and not bt.asyncio_task.done():
                bt.asyncio_task.cancel()
                try:
                    await bt.asyncio_task
                except (asyncio.CancelledError, Exception):
                    pass


class TestBackgroundTaskExecution:
    """Background task publishes TASK_RESULT with task_id correlation."""

    @pytest.mark.asyncio
    async def test_task_result_published_with_task_id(self):
        """Background task should publish TASK_RESULT with task_id in metadata."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Do this task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Should have INFO_UPDATE ack + TASK_RESULT
        publish_calls = agent.broker.publish.call_args_list
        assert len(publish_calls) >= 2

        # Second publish should be TASK_RESULT
        result_msg = publish_calls[1].args[0]
        assert isinstance(result_msg, AgentMessage)
        assert result_msg.headers.interaction_type == "TASK_RESULT"
        assert "task_id" in result_msg.metadata

        # task_id should match the one from the ack
        ack_task_id = publish_calls[0].args[0].metadata["task_id"]
        assert result_msg.metadata["task_id"] == ack_task_id

    @pytest.mark.asyncio
    async def test_task_marked_completed(self):
        """Background task should be marked as completed after success."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Do this task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        await asyncio.sleep(0.1)

        # Task should be tracked and completed
        tasks = agent.get_background_tasks()
        assert len(tasks) == 1
        task_status = list(tasks.values())[0]
        assert task_status["status"] == "completed"


class TestBackgroundTaskErrors:
    """Brain failures publish error TASK_RESULT; task tracked as 'failed'."""

    @pytest.mark.asyncio
    async def test_error_publishes_error_task_result(self):
        """Brain error should publish TASK_RESULT with error metadata."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        agent.brain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        msg = make_agent_message(
            content="Do this task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        await asyncio.sleep(0.1)

        # Should have INFO_UPDATE ack + error TASK_RESULT
        publish_calls = agent.broker.publish.call_args_list
        assert len(publish_calls) >= 2

        result_msg = publish_calls[1].args[0]
        assert result_msg.headers.interaction_type == "TASK_RESULT"
        assert result_msg.metadata["status"] == "error"
        assert "LLM timeout" in result_msg.metadata["error"]
        assert "task_id" in result_msg.metadata

    @pytest.mark.asyncio
    async def test_error_marks_task_failed(self):
        """Brain error should mark the background task as failed."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        agent.brain.ainvoke = AsyncMock(side_effect=ValueError("Bad input"))

        msg = make_agent_message(
            content="Do this task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        await asyncio.sleep(0.1)

        tasks = agent.get_background_tasks()
        assert len(tasks) == 1
        task_status = list(tasks.values())[0]
        assert task_status["status"] == "failed"


class TestConcurrencyLimit:
    """Exceeding max_concurrent falls back to synchronous processing."""

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_when_limit_reached(self):
        """When max_concurrent_tasks is reached, handler should process synchronously."""
        agent = _make_async_worker(max_concurrent_tasks=1)
        agent.active_sessions.add("session.test")

        # Fill up with a running background task
        bg_task = BackgroundTask(
            task_id="existing-task",
            conversation_id="session.test",
            source_agent_id="other",
            original_message_id="msg-000",
            interaction_type="TASK_ASSIGNMENT",
            status="running",
        )
        agent._background_tasks["existing-task"] = bg_task

        msg = make_agent_message(
            content="Another task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)

        # Brain should have been invoked synchronously (directly)
        agent.brain.ainvoke.assert_called_once()

        # No new INFO_UPDATE ack should have been published (sync path
        # publishes TASK_RESULT directly, not INFO_UPDATE)
        publish_calls = agent.broker.publish.call_args_list
        ack_published = any(
            call.args[0].headers.interaction_type == "INFO_UPDATE"
            and call.args[0].metadata.get("status") == "async_processing"
            for call in publish_calls
        )
        assert not ack_published


class TestTaskLevelAsync:
    """metadata.async=True overrides sync agent to go async."""

    @pytest.mark.asyncio
    async def test_metadata_async_overrides_sync_mode(self):
        """Message with metadata.async=True should go async even if agent is sync."""
        agent = _make_async_worker(async_mode=False)  # Sync agent
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Do this async",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
            metadata={"async": True},
        )

        # Make brain slow to verify async dispatch
        brain_done = asyncio.Event()

        async def slow_brain(*args, **kwargs):
            await brain_done.wait()
            return {"messages": [MagicMock(content="response")]}

        agent.brain.ainvoke = slow_brain

        # Handler should return quickly (async)
        await asyncio.wait_for(
            agent._default_session_handler(msg, agent),
            timeout=1.0,
        )

        # INFO_UPDATE ack should be published
        publish_calls = agent.broker.publish.call_args_list
        assert len(publish_calls) >= 1
        ack_msg = publish_calls[0].args[0]
        assert ack_msg.headers.interaction_type == "INFO_UPDATE"
        assert ack_msg.metadata["status"] == "async_processing"

        # Cleanup
        brain_done.set()
        await asyncio.sleep(0.1)
        for bt in agent._background_tasks.values():
            if bt.asyncio_task and not bt.asyncio_task.done():
                bt.asyncio_task.cancel()
                try:
                    await bt.asyncio_task
                except (asyncio.CancelledError, Exception):
                    pass


class TestSyncModeUnchanged:
    """Default async_mode=False preserves existing behavior exactly."""

    @pytest.mark.asyncio
    async def test_sync_mode_invokes_brain_directly(self):
        """With async_mode=False, brain should be invoked synchronously."""
        agent = _make_async_worker(async_mode=False)
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Do this task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)

        # Brain invoked directly (synchronously)
        agent.brain.ainvoke.assert_called_once()

        # TASK_RESULT published (not INFO_UPDATE ack)
        publish_calls = agent.broker.publish.call_args_list
        assert len(publish_calls) >= 1
        result_msg = publish_calls[0].args[0]
        assert result_msg.headers.interaction_type == "TASK_RESULT"

        # No background tasks tracked
        assert len(agent._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_sync_mode_no_ack_published(self):
        """Sync mode should not publish any INFO_UPDATE async ack."""
        agent = _make_async_worker(async_mode=False)
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Sync task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)

        publish_calls = agent.broker.publish.call_args_list
        ack_published = any(
            call.args[0].metadata.get("status") == "async_processing"
            for call in publish_calls
        )
        assert not ack_published

    @pytest.mark.asyncio
    async def test_group_query_never_async(self):
        """GROUP_QUERY should never use async dispatch (only ALWAYS_WAKE_TYPES)."""
        agent = _make_async_worker(async_mode=True, triggers=["currency"])
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="What is the currency rate?",
            source_agent_id="user-agent",
            interaction_type="GROUP_QUERY",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)

        # Brain invoked synchronously (keyword trigger, not async dispatched)
        agent.brain.ainvoke.assert_called_once()

        # No async ack
        publish_calls = agent.broker.publish.call_args_list
        ack_published = any(
            call.args[0].metadata.get("status") == "async_processing"
            for call in publish_calls
        )
        assert not ack_published


class TestGracefulShutdown:
    """Shutdown cancels running background tasks."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_running_tasks(self):
        """_shutdown_background_tasks should cancel all running background tasks."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        # Create a slow brain
        async def slow_brain(*args, **kwargs):
            await asyncio.sleep(100)
            return {"messages": [MagicMock(content="response")]}

        agent.brain.ainvoke = slow_brain

        msg = make_agent_message(
            content="Long task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)

        # Should have a running background task
        assert len(agent._background_tasks) == 1
        task_id = list(agent._background_tasks.keys())[0]
        bt = agent._background_tasks[task_id]
        assert bt.asyncio_task is not None
        assert not bt.asyncio_task.done()

        # Shutdown should cancel it
        await agent._shutdown_background_tasks()

        assert bt.asyncio_task.done()
        assert bt.status == "failed"


class TestBackgroundTaskIntrospection:
    """get_background_tasks() returns correct status."""

    @pytest.mark.asyncio
    async def test_get_background_tasks_returns_status(self):
        """get_background_tasks should return task info with correct fields."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Do task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        await asyncio.sleep(0.1)  # Let background task complete

        tasks = agent.get_background_tasks()
        assert len(tasks) == 1

        task_info = list(tasks.values())[0]
        assert "status" in task_info
        assert "conversation_id" in task_info
        assert "source_agent_id" in task_info
        assert "started_at" in task_info
        assert "elapsed_seconds" in task_info
        assert task_info["conversation_id"] == "session.test"
        assert task_info["source_agent_id"] == "squad-lead"

    def test_get_background_tasks_empty_initially(self):
        """get_background_tasks should return empty dict initially."""
        agent = _make_async_worker()
        assert agent.get_background_tasks() == {}

    @pytest.mark.asyncio
    async def test_completed_task_shows_completed(self):
        """Completed background tasks should show status=completed."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Quick task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        await asyncio.sleep(0.1)

        tasks = agent.get_background_tasks()
        task_info = list(tasks.values())[0]
        assert task_info["status"] == "completed"

    @pytest.mark.asyncio
    async def test_failed_task_shows_failed(self):
        """Failed background tasks should show status=failed."""
        agent = _make_async_worker()
        agent.active_sessions.add("session.test")
        agent.brain.ainvoke = AsyncMock(side_effect=RuntimeError("Boom"))

        msg = make_agent_message(
            content="Fail task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        await asyncio.sleep(0.1)

        tasks = agent.get_background_tasks()
        task_info = list(tasks.values())[0]
        assert task_info["status"] == "failed"
