"""
Tests for shinox_agent.schemas — Pydantic message models and type sets.
"""

import pytest
from datetime import datetime, timezone

from shinox_agent.schemas import (
    AgentMessage,
    A2AHeaders,
    SystemCommand,
    SystemCommandMetadata,
    GROUP_VISIBLE_TYPES,
    OBSERVE_ONLY_TYPES,
    ALWAYS_WAKE_TYPES,
    COLLABORATION_RESPONSE_TYPES,
)


class TestA2AHeaders:
    def test_required_fields(self):
        h = A2AHeaders(
            source_agent_id="agent-1",
            interaction_type="TASK_ASSIGNMENT",
            conversation_id="session.1",
        )
        assert h.source_agent_id == "agent-1"
        assert h.target_agent_id is None
        assert h.governance_status == "VERIFIED"

    def test_all_fields(self):
        h = A2AHeaders(
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            interaction_type="DIRECT_COMMAND",
            conversation_id="session.1",
            correlation_id="corr-123",
            governance_status="PENDING",
        )
        assert h.target_agent_id == "agent-2"
        assert h.correlation_id == "corr-123"
        assert h.governance_status == "PENDING"

    def test_invalid_interaction_type_rejected(self):
        with pytest.raises(Exception):
            A2AHeaders(
                source_agent_id="agent-1",
                interaction_type="INVALID_TYPE",
                conversation_id="session.1",
            )

    def test_all_valid_interaction_types(self):
        valid_types = [
            "DIRECT_COMMAND", "TASK_ASSIGNMENT", "TASK_RESULT", "INFO_UPDATE",
            "SESSION_BRIEFING", "AGENT_JOINED", "AGENT_STATE_SNAPSHOT",
            "SQUAD_COMPLETION", "GROUP_QUERY", "PEER_REQUEST", "PEER_RESPONSE",
            "EXPERTISE_OFFER", "BROADCAST_QUERY",
        ]
        for itype in valid_types:
            h = A2AHeaders(
                source_agent_id="a", interaction_type=itype, conversation_id="s"
            )
            assert h.interaction_type == itype


class TestAgentMessage:
    def test_defaults(self):
        msg = AgentMessage(
            content="Hello",
            headers=A2AHeaders(
                source_agent_id="a1",
                interaction_type="TASK_RESULT",
                conversation_id="s1",
            ),
        )
        assert msg.content == "Hello"
        assert msg.id  # Auto-generated UUID
        assert msg.timestamp  # Auto-generated timestamp
        assert msg.tool_calls is None
        assert msg.metadata == {}

    def test_round_trip_serialization(self):
        msg = AgentMessage(
            id="test-id",
            content="Test content",
            metadata={"key": "value"},
            headers=A2AHeaders(
                source_agent_id="a1",
                target_agent_id="a2",
                interaction_type="TASK_ASSIGNMENT",
                conversation_id="s1",
            ),
        )

        data = msg.model_dump()
        restored = AgentMessage(**data)

        assert restored.id == "test-id"
        assert restored.content == "Test content"
        assert restored.metadata == {"key": "value"}
        assert restored.headers.source_agent_id == "a1"
        assert restored.headers.target_agent_id == "a2"

    def test_with_tool_calls(self):
        msg = AgentMessage(
            content="Using a tool",
            tool_calls=[{"name": "search", "args": {"query": "test"}}],
            headers=A2AHeaders(
                source_agent_id="a1",
                interaction_type="TASK_RESULT",
                conversation_id="s1",
            ),
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "search"


class TestSystemCommand:
    def test_creation(self):
        cmd = SystemCommand(
            id="cmd-1",
            type="SYSTEM_COMMAND",
            source="director",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=SystemCommandMetadata(
                command="JOIN_SESSION",
                session_id="session.test-1",
            ),
        )
        assert cmd.metadata.command == "JOIN_SESSION"
        assert cmd.metadata.session_id == "session.test-1"
        assert cmd.metadata.priority == "NORMAL"

    def test_extra_fields_allowed_in_metadata(self):
        cmd = SystemCommand(
            id="cmd-1",
            type="SYSTEM_COMMAND",
            source="director",
            timestamp="2025-01-01T00:00:00Z",
            metadata=SystemCommandMetadata(
                command="JOIN_SESSION",
                session_id="session.test-1",
                custom_field="extra",
            ),
        )
        assert cmd.metadata.custom_field == "extra"


class TestInteractionTypeSets:
    def test_sets_are_disjoint(self):
        """All wake-up sets should be mutually exclusive."""
        all_sets = [
            GROUP_VISIBLE_TYPES,
            OBSERVE_ONLY_TYPES,
            ALWAYS_WAKE_TYPES,
            COLLABORATION_RESPONSE_TYPES,
        ]

        for i, set_a in enumerate(all_sets):
            for j, set_b in enumerate(all_sets):
                if i != j:
                    overlap = set_a & set_b
                    assert not overlap, f"Sets {i} and {j} overlap: {overlap}"

    def test_always_wake_types(self):
        assert "TASK_ASSIGNMENT" in ALWAYS_WAKE_TYPES
        assert "DIRECT_COMMAND" in ALWAYS_WAKE_TYPES

    def test_group_visible_types(self):
        assert "GROUP_QUERY" in GROUP_VISIBLE_TYPES
        assert "PEER_REQUEST" in GROUP_VISIBLE_TYPES
        assert "EXPERTISE_OFFER" in GROUP_VISIBLE_TYPES
        assert "BROADCAST_QUERY" in GROUP_VISIBLE_TYPES

    def test_observe_only_types(self):
        assert "INFO_UPDATE" in OBSERVE_ONLY_TYPES
        assert "TASK_RESULT" in OBSERVE_ONLY_TYPES
        assert "AGENT_JOINED" in OBSERVE_ONLY_TYPES

    def test_collaboration_response_types(self):
        assert "PEER_RESPONSE" in COLLABORATION_RESPONSE_TYPES
