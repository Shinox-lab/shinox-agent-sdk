"""
Tests for ShinoxWorkerAgent wake-up decision logic.

This tests the _default_session_handler method which is the core decision tree
for when an agent should wake up (invoke LLM) vs. observe vs. ignore.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shinox_agent.schemas import AgentMessage, A2AHeaders
from tests.conftest import MockAgentCard, MockSkill, make_agent_message


def _make_worker(
    agent_name="Test Worker",
    triggers=None,
    enable_semantic=False,
    enable_peer_collaboration=False,
):
    """Create a ShinoxWorkerAgent with mocked infrastructure."""
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

        from shinox_agent.worker import ShinoxWorkerAgent
        agent = ShinoxWorkerAgent(
            agent_card=card,
            brain=brain,
            triggers=triggers,
            enable_semantic_wake=enable_semantic,
            enable_peer_collaboration=enable_peer_collaboration,
        )
        agent.broker = mock_broker

    return agent


class TestSelfMessageFiltering:
    @pytest.mark.asyncio
    async def test_ignores_own_messages(self):
        """Agent should never process its own messages."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            source_agent_id=agent.agent_id,  # Message from self
            interaction_type="TASK_RESULT",
            conversation_id="session.test",
        )

        agent.brain.ainvoke = AsyncMock()
        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()


class TestExplicitTargeting:
    @pytest.mark.asyncio
    async def test_wakes_on_direct_target(self):
        """Agent should wake when target_agent_id matches its ID."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Do this task",
            source_agent_id="squad-lead",
            target_agent_id=agent.agent_id,
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_wakes_on_mention(self):
        """Agent should wake when @mentioned in content."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content=f"Hey @{agent.agent_id} can you help?",
            source_agent_id="other-agent",
            interaction_type="GROUP_QUERY",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()


class TestAlwaysWakeTypes:
    @pytest.mark.asyncio
    async def test_wakes_on_task_assignment(self):
        """TASK_ASSIGNMENT should always wake the agent."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Convert 7 USD to MYR",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_wakes_on_direct_command(self):
        """DIRECT_COMMAND should always wake the agent."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Execute deployment",
            source_agent_id="squad-lead",
            interaction_type="DIRECT_COMMAND",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_assignment_bypasses_session_check(self):
        """TASK_ASSIGNMENT should wake even if session not in active_sessions."""
        agent = _make_worker()
        # Don't add session to active_sessions

        msg = make_agent_message(
            content="Do work",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.unknown",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()


class TestObserveOnlyTypes:
    @pytest.mark.asyncio
    async def test_info_update_observed_not_woken(self):
        """INFO_UPDATE should update context but not invoke brain."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="System status: healthy",
            source_agent_id="monitoring-agent",
            interaction_type="INFO_UPDATE",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()

        # Context should be updated
        context = agent.get_conversation_context("session.test")
        assert context is not None
        assert len(context.recent_messages) == 1

    @pytest.mark.asyncio
    async def test_task_result_observed_not_woken(self):
        """TASK_RESULT from another agent should be observed, not woken."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Result: 27.67 MYR",
            source_agent_id="accounting-agent",
            interaction_type="TASK_RESULT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_joined_observed_not_woken(self):
        """AGENT_JOINED should be observed, not woken."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Agent coder-agent joined",
            source_agent_id="coder-agent",
            interaction_type="AGENT_JOINED",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()


class TestSessionFiltering:
    @pytest.mark.asyncio
    async def test_ignores_unknown_session_for_non_wake_types(self):
        """Messages from unknown sessions should be ignored (except ALWAYS_WAKE)."""
        agent = _make_worker()
        # Don't add any sessions

        msg = make_agent_message(
            content="Some group query",
            source_agent_id="other-agent",
            interaction_type="GROUP_QUERY",
            conversation_id="session.unknown",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()


class TestGroupVisibleWithKeywords:
    """Test GROUP_VISIBLE_TYPES with keyword fallback (no semantic matcher)."""

    @pytest.mark.asyncio
    async def test_keyword_trigger_wakes(self):
        """Keyword match in GROUP_QUERY should wake the agent."""
        agent = _make_worker(triggers=["currency", "exchange"])
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="What is the currency exchange rate?",
            source_agent_id="user-agent",
            interaction_type="GROUP_QUERY",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_keyword_match_observes(self):
        """No keyword match in GROUP_QUERY should observe, not wake."""
        agent = _make_worker(triggers=["database", "sql"])
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="What is the weather today?",
            source_agent_id="user-agent",
            interaction_type="GROUP_QUERY",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()

        # Should still observe
        context = agent.get_conversation_context("session.test")
        assert context is not None
        assert len(context.recent_messages) == 1

    @pytest.mark.asyncio
    async def test_peer_request_keyword_trigger(self):
        """PEER_REQUEST with matching keyword should wake."""
        agent = _make_worker(triggers=["schema"])
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Help me design a database schema",
            source_agent_id="coder-agent",
            interaction_type="PEER_REQUEST",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()


class TestGroupVisibleWithSemanticMatcher:
    @pytest.mark.asyncio
    async def test_semantic_match_above_threshold_wakes(self):
        """Semantic match above threshold should wake."""
        agent = _make_worker(enable_semantic=True)
        agent.active_sessions.add("session.test")

        # Mock semantic matcher to return high score
        mock_matcher = MagicMock()
        mock_matcher.initialized = True
        mock_matcher.should_wake.return_value = (True, 0.85, "skill:currency")
        agent._semantic_matcher = mock_matcher

        msg = make_agent_message(
            content="What is the exchange rate?",
            source_agent_id="user-agent",
            interaction_type="GROUP_QUERY",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_match_below_threshold_observes(self):
        """Semantic match below threshold should observe, not wake."""
        agent = _make_worker(enable_semantic=True)
        agent.active_sessions.add("session.test")

        mock_matcher = MagicMock()
        mock_matcher.initialized = True
        mock_matcher.should_wake.return_value = (False, 0.3, "none")
        agent._semantic_matcher = mock_matcher

        msg = make_agent_message(
            content="How is the weather?",
            source_agent_id="user-agent",
            interaction_type="GROUP_QUERY",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()

        # Should still observe
        context = agent.get_conversation_context("session.test")
        assert len(context.recent_messages) == 1

    @pytest.mark.asyncio
    async def test_preferred_agent_always_wakes(self):
        """Agent listed as preferred in PEER_REQUEST should wake regardless of semantic score."""
        agent = _make_worker(enable_semantic=True)
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Help with database design",
            source_agent_id="coder-agent",
            interaction_type="PEER_REQUEST",
            conversation_id="session.test",
            metadata={"preferred_agents": [agent.agent_id]},
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_called_once()


class TestCollaborationResponseTypes:
    @pytest.mark.asyncio
    async def test_peer_response_without_collaboration_observes(self):
        """PEER_RESPONSE with collaboration disabled should just observe."""
        agent = _make_worker(enable_peer_collaboration=False)
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Here's the help you needed",
            source_agent_id="helper-agent",
            interaction_type="PEER_RESPONSE",
            conversation_id="session.test",
            metadata={"collaboration_correlation_id": "corr-123"},
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_peer_response_with_matching_correlation_wakes(self):
        """PEER_RESPONSE with matching correlation_id should wake."""
        agent = _make_worker(enable_peer_collaboration=True)
        agent.active_sessions.add("session.test")

        # Set up a pending collaboration
        from shinox_agent.worker import PendingCollaboration
        context = agent._get_or_create_context("session.test")
        context.pending_collaborations["corr-123"] = PendingCollaboration(
            correlation_id="corr-123",
            original_task_content="original task",
            my_attempt_content="my attempt",
            original_source_agent_id="squad-lead",
            original_conversation_id="session.test",
        )

        msg = make_agent_message(
            content="Here's the help you needed",
            source_agent_id="helper-agent",
            interaction_type="PEER_RESPONSE",
            conversation_id="session.test",
            metadata={"collaboration_correlation_id": "corr-123"},
        )

        # Mock _handle_peer_response since we're testing wake logic, not peer handling
        agent._handle_peer_response = AsyncMock()

        await agent._default_session_handler(msg, agent)
        agent._handle_peer_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_peer_response_with_unknown_correlation_observes(self):
        """PEER_RESPONSE with unmatched correlation_id should observe."""
        agent = _make_worker(enable_peer_collaboration=True)
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Help for someone else",
            source_agent_id="helper-agent",
            interaction_type="PEER_RESPONSE",
            conversation_id="session.test",
            metadata={"collaboration_correlation_id": "unknown-corr"},
        )

        await agent._default_session_handler(msg, agent)
        agent.brain.ainvoke.assert_not_called()


class TestContextTracking:
    @pytest.mark.asyncio
    async def test_wake_updates_interaction_tracking(self):
        """Waking up should track the interaction in context."""
        agent = _make_worker()
        agent.active_sessions.add("session.test")

        msg = make_agent_message(
            content="Do this task",
            source_agent_id="squad-lead",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.test",
        )

        await agent._default_session_handler(msg, agent)

        context = agent.get_conversation_context("session.test")
        assert context is not None
        assert context.has_contributed is True
        assert "squad-lead" in context.interacted_with

    def test_get_recent_messages(self):
        """get_recent_messages should return most recent observed messages."""
        agent = _make_worker()
        context = agent._get_or_create_context("session.test")

        # Add some messages
        for i in range(5):
            context.recent_messages.append({
                "id": f"msg-{i}", "source": f"agent-{i}",
                "type": "INFO_UPDATE", "content_preview": f"msg {i}",
            })

        recent = agent.get_recent_messages("session.test", limit=3)
        assert len(recent) == 3
        assert recent[-1]["id"] == "msg-4"  # Most recent

    def test_get_recent_messages_unknown_session(self):
        agent = _make_worker()
        result = agent.get_recent_messages("unknown-session")
        assert result == []
