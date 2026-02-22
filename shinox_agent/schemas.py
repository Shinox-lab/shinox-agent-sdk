"""
Shinox Agent SDK - Message Schemas

Pydantic models for A2A protocol messages used in the Shinox mesh.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union, Set
from uuid import uuid4
from datetime import datetime, timezone


# Type definitions based on A2A Protocol
# Extended with Multi-Agent Coordination types for group chat semantics
InteractionType = Literal[
    # --- Orchestrated (Squad Lead directed) ---
    "DIRECT_COMMAND",       # Explicit command to specific agent
    "TASK_ASSIGNMENT",      # Task from orchestrator → always wake
    "TASK_RESULT",          # Result from worker → Squad Lead processes
    "INFO_UPDATE",          # Status broadcast → observe only (no LLM)
    "SESSION_BRIEFING",     # Initial briefing → Squad Lead handles
    "AGENT_JOINED",         # Agent announcement → passive
    "AGENT_STATE_SNAPSHOT", # State snapshot → passive
    "SQUAD_COMPLETION",     # Squad finished → Squad Lead handles

    # --- Decentralized (Group Chat / Peer-to-Peer) ---
    "GROUP_QUERY",          # Question to the group → semantic wake-up
    "PEER_REQUEST",         # Agent requests help → semantic wake-up
    "PEER_RESPONSE",        # Response to a PEER_REQUEST → correlation-based wake-up
    "EXPERTISE_OFFER",      # Agent offers expertise → semantic wake-up
    "BROADCAST_QUERY",      # Legacy: general broadcast → semantic wake-up
]

GovernanceStatus = Literal["PENDING", "VERIFIED", "BLOCKED"]


# Group-visible message types that trigger semantic wake-up matching
# These are messages visible to all agents where semantic relevance determines who responds
GROUP_VISIBLE_TYPES: Set[str] = {
    "GROUP_QUERY",      # User/agent asks group a question
    "PEER_REQUEST",     # Agent requests help from capable peers
    "EXPERTISE_OFFER",  # Agent proactively offers expertise
    "BROADCAST_QUERY",  # Legacy broadcast (first capable responds)
}

# Message types that agents should observe (update context) but not wake LLM
OBSERVE_ONLY_TYPES: Set[str] = {
    "INFO_UPDATE",      # Status updates
    "TASK_RESULT",      # Other agents' results (context building)
    "AGENT_JOINED",     # Announcements
    "AGENT_STATE_SNAPSHOT",
}

# Message types that always wake the target agent
ALWAYS_WAKE_TYPES: Set[str] = {
    "DIRECT_COMMAND",
    "TASK_ASSIGNMENT",
}

# Collaboration response types — targeted wake-up via correlation_id matching
# NOT in GROUP_VISIBLE_TYPES (no semantic matching — targeted by correlation_id)
# NOT in OBSERVE_ONLY_TYPES (workers need to wake on matching responses)
# NOT in ALWAYS_WAKE_TYPES (only the original requester should wake)
COLLABORATION_RESPONSE_TYPES: Set[str] = {
    "PEER_RESPONSE",
}


class A2AHeaders(BaseModel):
    """Headers for agent-to-agent messages following A2A protocol."""

    source_agent_id: str
    target_agent_id: Optional[str] = None
    interaction_type: InteractionType
    conversation_id: str
    correlation_id: Optional[str] = None
    governance_status: GovernanceStatus = "VERIFIED"


class AgentMessage(BaseModel):
    """Standard message format for agent communication."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    headers: A2AHeaders


class SystemCommandMetadata(BaseModel):
    """Metadata for system commands from Director."""

    command: str
    session_id: str
    priority: str = "NORMAL"
    session_title: Optional[str] = None
    session_briefing: Optional[str] = None
    agent_context: Optional[str] = None

    class Config:
        extra = "allow"


class SystemCommand(BaseModel):
    """System command message (e.g., JOIN_SESSION from Director)."""

    id: str
    type: Literal["SYSTEM_COMMAND"]
    source: str
    timestamp: Union[int, str, datetime]
    content: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: SystemCommandMetadata

    class Config:
        extra = "allow"
