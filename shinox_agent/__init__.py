"""
Shinox Agent SDK

A Python library for building agents in the Shinox decentralized mesh network.

Quick Start (Worker Agent):
    from shinox_agent import ShinoxWorkerAgent

    agent = ShinoxWorkerAgent(
        agent_card=my_card,
        brain=my_brain,
        enable_semantic_wake=True,  # Enable semantic matching
        wake_threshold=0.65,        # Similarity threshold
    )
    app = agent.app

Quick Start (Custom Agent):
    from shinox_agent import ShinoxAgent

    async def handler(msg, agent):
        await agent.publish_message(...)

    agent = ShinoxAgent(
        agent_card=my_card,
        session_handler=handler,
    )
    app = agent.app

Semantic Wake-up:
    Install with: pip install shinox-agent-sdk[semantic]

    The agent will use vector similarity matching to decide
    when to respond to broadcast messages based on its capabilities.

Agent History Tool:
    from shinox_agent.tools import AgentHistoryTool, get_session_history

    # Fetch session history for context
    history = await get_session_history(session_id, registry_url)

    # Use as LangChain tool in agent brains
    tool = AgentHistoryTool(registry_url=url, session_id=session_id)
"""

from .base import ShinoxAgent
from .logging import setup_json_logging
from .worker import ShinoxWorkerAgent, BackgroundTask
from .schemas import (
    AgentMessage,
    A2AHeaders,
    SystemCommand,
    SystemCommandMetadata,
    InteractionType,
    GovernanceStatus,
    GROUP_VISIBLE_TYPES,
    OBSERVE_ONLY_TYPES,
    ALWAYS_WAKE_TYPES,
    COLLABORATION_RESPONSE_TYPES,
)

# Optional: Semantic matching (requires sentence-transformers)
try:
    from .embeddings import SemanticMatcher, SemanticMatcherFactory
except ImportError:
    SemanticMatcher = None
    SemanticMatcherFactory = None

# Optional: Agent History Tool
try:
    from .tools import (
        AgentHistoryTool,
        get_session_history,
        get_session_results,
        should_auto_fetch_history,
    )
except ImportError:
    AgentHistoryTool = None
    get_session_history = None
    get_session_results = None
    should_auto_fetch_history = None

__version__ = "0.1.2"

__all__ = [
    # Agent classes
    "ShinoxAgent",
    "ShinoxWorkerAgent",
    "BackgroundTask",
    # Logging
    "setup_json_logging",
    # Message schemas
    "AgentMessage",
    "A2AHeaders",
    "SystemCommand",
    "SystemCommandMetadata",
    # Type aliases
    "InteractionType",
    "GovernanceStatus",
    # Interaction type sets (for wake-up logic)
    "GROUP_VISIBLE_TYPES",
    "OBSERVE_ONLY_TYPES",
    "ALWAYS_WAKE_TYPES",
    "COLLABORATION_RESPONSE_TYPES",
    # Semantic matching (optional)
    "SemanticMatcher",
    "SemanticMatcherFactory",
    # History Tool (optional)
    "AgentHistoryTool",
    "get_session_history",
    "get_session_results",
    "should_auto_fetch_history",
]
