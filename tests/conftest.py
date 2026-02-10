"""
Shared test fixtures for Shinox Agent SDK tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional

from shinox_agent.schemas import AgentMessage, A2AHeaders


# --- Mock Agent Card ---

@dataclass
class MockSkill:
    id: str = "skill-1"
    name: str = "test_skill"
    description: str = "A test skill"
    tags: list = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["testing", "mock"]


@dataclass
class MockAgentCard:
    name: str = "Test Agent"
    description: str = "A test agent for unit tests"
    skills: list = None
    url: str = "http://localhost:8001"
    version: str = "1.0.0"

    def __post_init__(self):
        if self.skills is None:
            self.skills = [MockSkill()]

    def model_dump(self, mode=None):
        return {
            "name": self.name,
            "description": self.description,
            "skills": [
                {"id": s.id, "name": s.name, "description": s.description, "tags": s.tags}
                for s in self.skills
            ],
            "url": self.url,
            "version": self.version,
        }


@pytest.fixture
def mock_agent_card():
    """A basic mock agent card."""
    return MockAgentCard()


@pytest.fixture
def mock_agent_card_with_skills():
    """An agent card with multiple skills."""
    return MockAgentCard(
        name="Accounting Specialist",
        description="Handles currency conversion and financial calculations",
        skills=[
            MockSkill(id="s1", name="currency_conversion", description="Convert between currencies", tags=["finance", "currency"]),
            MockSkill(id="s2", name="invoice_processing", description="Process and validate invoices", tags=["finance", "invoices"]),
        ],
    )


@pytest.fixture
def mock_brain():
    """A mock LangGraph brain."""
    brain = AsyncMock()
    brain.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Brain response content")]
    })
    return brain


# --- Message Factories ---

def make_agent_message(
    content: str = "Hello world",
    source_agent_id: str = "other-agent",
    target_agent_id: Optional[str] = None,
    interaction_type: str = "TASK_ASSIGNMENT",
    conversation_id: str = "session.test-123",
    governance_status: str = "VERIFIED",
    msg_id: str = "msg-001",
    metadata: dict = None,
    correlation_id: Optional[str] = None,
) -> AgentMessage:
    """Create a test AgentMessage."""
    return AgentMessage(
        id=msg_id,
        content=content,
        metadata=metadata or {},
        correlation_id=correlation_id,
        headers=A2AHeaders(
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            interaction_type=interaction_type,
            conversation_id=conversation_id,
            governance_status=governance_status,
        ),
    )


@pytest.fixture
def task_assignment_msg():
    """A TASK_ASSIGNMENT message."""
    return make_agent_message(
        content="Convert 7 USD to MYR",
        interaction_type="TASK_ASSIGNMENT",
        source_agent_id="squad-lead-agent",
        target_agent_id="test-agent",
    )


@pytest.fixture
def info_update_msg():
    """An INFO_UPDATE message (observe only)."""
    return make_agent_message(
        content="System status: all agents healthy",
        interaction_type="INFO_UPDATE",
        source_agent_id="monitoring-agent",
    )


@pytest.fixture
def group_query_msg():
    """A GROUP_QUERY message (semantic matching)."""
    return make_agent_message(
        content="What is the exchange rate for USD to MYR?",
        interaction_type="GROUP_QUERY",
        source_agent_id="user-agent",
    )
