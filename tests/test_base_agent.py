"""
Tests for shinox_agent.base — ShinoxAgent helper methods.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shinox_agent.schemas import AgentMessage, A2AHeaders
from tests.conftest import MockAgentCard, MockSkill


class TestGenerateAgentId:
    """Test _generate_agent_id derives correct ID from card name."""

    def _make_agent(self, name: str):
        """Helper to create agent with patched broker and avoid Kafka connection."""
        card = MockAgentCard(name=name)
        with patch("shinox_agent.base.KafkaBroker"), \
             patch("shinox_agent.base.FastStream"), \
             patch("shinox_agent.base.setup_json_logging"):
            from shinox_agent.base import ShinoxAgent
            agent = ShinoxAgent(agent_card=card)
        return agent

    def test_simple_name(self):
        agent = self._make_agent("Test Agent")
        assert agent.agent_id == "test-agent"

    def test_multi_word_name(self):
        agent = self._make_agent("Accounting Specialist Agent")
        assert agent.agent_id == "accounting-specialist-agent"

    def test_single_word(self):
        agent = self._make_agent("Coder")
        assert agent.agent_id == "coder"

    def test_already_lowercase(self):
        agent = self._make_agent("squad-lead")
        assert agent.agent_id == "squad-lead"


class TestGenerateIntroduction:
    def _make_agent(self, card):
        with patch("shinox_agent.base.KafkaBroker"), \
             patch("shinox_agent.base.FastStream"), \
             patch("shinox_agent.base.setup_json_logging"):
            from shinox_agent.base import ShinoxAgent
            return ShinoxAgent(agent_card=card)

    def test_contains_agent_id(self):
        card = MockAgentCard(name="Test Agent")
        agent = self._make_agent(card)
        assert "test-agent" in agent.self_introduction

    def test_contains_description(self):
        card = MockAgentCard(name="Test Agent", description="I test things")
        agent = self._make_agent(card)
        assert "I test things" in agent.self_introduction

    def test_contains_capabilities_from_tags(self):
        card = MockAgentCard(
            name="Test Agent",
            skills=[MockSkill(tags=["finance", "math"])],
        )
        agent = self._make_agent(card)
        intro = agent.self_introduction
        assert "finance" in intro or "math" in intro


class TestParseMessage:
    def _make_agent(self):
        card = MockAgentCard(name="Parser Agent")
        with patch("shinox_agent.base.KafkaBroker"), \
             patch("shinox_agent.base.FastStream"), \
             patch("shinox_agent.base.setup_json_logging"):
            from shinox_agent.base import ShinoxAgent
            return ShinoxAgent(agent_card=card)

    def test_dict_passthrough(self):
        agent = self._make_agent()
        data = {"content": "hello", "id": "123"}
        result = agent._parse_message(data)
        assert result == data

    def test_json_string(self):
        agent = self._make_agent()
        data = {"content": "hello", "id": "123"}
        result = agent._parse_message(json.dumps(data))
        assert result == data

    def test_bytes(self):
        agent = self._make_agent()
        data = {"content": "hello", "id": "123"}
        result = agent._parse_message(json.dumps(data).encode("utf-8"))
        assert result == data

    def test_invalid_json_returns_none(self):
        agent = self._make_agent()
        result = agent._parse_message("not valid json {{{")
        assert result is None

    def test_pydantic_model(self):
        agent = self._make_agent()
        msg = AgentMessage(
            content="test",
            headers=A2AHeaders(
                source_agent_id="a1",
                interaction_type="TASK_RESULT",
                conversation_id="s1",
            ),
        )
        result = agent._parse_message(msg)
        assert isinstance(result, dict)
        assert result["content"] == "test"

    def test_none_type_returns_none(self):
        agent = self._make_agent()
        result = agent._parse_message(42)  # Unexpected type
        assert result is None


class TestHandleSystemCommand:
    @pytest.mark.asyncio
    async def test_join_session_adds_to_active(self):
        card = MockAgentCard(name="Test Agent")
        with patch("shinox_agent.base.KafkaBroker") as mock_broker_cls, \
             patch("shinox_agent.base.FastStream"), \
             patch("shinox_agent.base.setup_json_logging"):
            mock_broker = MagicMock()
            mock_broker.subscriber = MagicMock(return_value=lambda fn: fn)
            mock_broker.publish = AsyncMock()
            mock_broker_cls.return_value = mock_broker

            from shinox_agent.base import ShinoxAgent
            from shinox_agent.schemas import SystemCommand, SystemCommandMetadata

            agent = ShinoxAgent(agent_card=card)
            agent.broker = mock_broker

            cmd = SystemCommand(
                id="cmd-1",
                type="SYSTEM_COMMAND",
                source="director",
                timestamp="2025-01-01T00:00:00Z",
                metadata=SystemCommandMetadata(
                    command="JOIN_SESSION",
                    session_id="session.test-squad",
                ),
            )

            # Patch _subscribe_to_session and _send_join_acknowledgment
            agent._subscribe_to_session = AsyncMock()
            agent._send_join_acknowledgment = AsyncMock()

            await agent._handle_system_command(cmd)

            assert "session.test-squad" in agent.active_sessions
            agent._subscribe_to_session.assert_called_once_with("session.test-squad")
            agent._send_join_acknowledgment.assert_called_once_with("session.test-squad")

    @pytest.mark.asyncio
    async def test_duplicate_join_ignored(self):
        card = MockAgentCard(name="Test Agent")
        with patch("shinox_agent.base.KafkaBroker") as mock_broker_cls, \
             patch("shinox_agent.base.FastStream"), \
             patch("shinox_agent.base.setup_json_logging"):
            mock_broker = MagicMock()
            mock_broker.subscriber = MagicMock(return_value=lambda fn: fn)
            mock_broker_cls.return_value = mock_broker

            from shinox_agent.base import ShinoxAgent
            from shinox_agent.schemas import SystemCommand, SystemCommandMetadata

            agent = ShinoxAgent(agent_card=card)
            agent.active_sessions.add("session.test-squad")
            agent._subscribe_to_session = AsyncMock()
            agent._send_join_acknowledgment = AsyncMock()

            cmd = SystemCommand(
                id="cmd-1",
                type="SYSTEM_COMMAND",
                source="director",
                timestamp="2025-01-01T00:00:00Z",
                metadata=SystemCommandMetadata(
                    command="JOIN_SESSION",
                    session_id="session.test-squad",
                ),
            )

            await agent._handle_system_command(cmd)

            # Should not subscribe again
            agent._subscribe_to_session.assert_not_called()
