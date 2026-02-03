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
"""

from .base import ShinoxAgent
from .worker import ShinoxWorkerAgent
from .schemas import (
    AgentMessage,
    A2AHeaders,
    SystemCommand,
    SystemCommandMetadata,
    InteractionType,
    GovernanceStatus,
)

# Optional: Semantic matching (requires sentence-transformers)
try:
    from .embeddings import SemanticMatcher, SemanticMatcherFactory
except ImportError:
    SemanticMatcher = None
    SemanticMatcherFactory = None

__version__ = "0.1.0"

__all__ = [
    # Agent classes
    "ShinoxAgent",
    "ShinoxWorkerAgent",
    # Message schemas
    "AgentMessage",
    "A2AHeaders",
    "SystemCommand",
    "SystemCommandMetadata",
    # Type aliases
    "InteractionType",
    "GovernanceStatus",
    # Semantic matching (optional)
    "SemanticMatcher",
    "SemanticMatcherFactory",
]
