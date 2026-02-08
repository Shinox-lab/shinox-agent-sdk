"""
Shinox Agent SDK - Agent History Tool

Provides agents with the ability to query session history on-demand.
Works together with semantic wake-up for intelligent context retrieval.

Usage:
    from shinox_agent.tools import AgentHistoryTool, get_session_history

    # As a standalone function
    history = await get_session_history(session_id, registry_url)

    # As a LangChain tool for agent brains
    tool = AgentHistoryTool(registry_url=registry_url)
    # Add to your agent's tools list
"""

import os
import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Default registry URL
DEFAULT_REGISTRY_URL = os.getenv("AGENT_REGISTRY_URL", "http://localhost:9000")

# Patterns that indicate a task might need history context
HISTORY_TRIGGER_PATTERNS = [
    # Summarization patterns
    r"\bsummar(y|ize|izing|isation)\b",
    r"\bcombine\b",
    r"\bconsolidate\b",
    r"\baggregate\b",
    r"\bmerge\b",
    # Reference patterns
    r"\bbased on\b",
    r"\bprevious(ly)?\b",
    r"\bearlier\b",
    r"\babove\b",
    r"\bwhat did .+ (say|report|find|result)\b",
    r"\bwhat (was|were) the result\b",
    r"\baccording to\b",
    r"\bfrom the\b.*\bresult\b",
    # Dependency patterns
    r"\busing the\b.*\b(data|result|output|response)\b",
    r"\bwith the\b.*\b(information|context)\b",
]

# Compiled regex patterns for efficiency
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in HISTORY_TRIGGER_PATTERNS]


@dataclass
class HistoryMessage:
    """A message from session history."""
    message_id: str
    source_agent_id: str
    target_agent_id: Optional[str]
    interaction_type: str
    content: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SessionHistory:
    """Session history response."""
    session_id: str
    total_count: int
    messages: List[HistoryMessage]
    has_more: bool


def should_auto_fetch_history(task_instruction: str) -> Tuple[bool, List[str]]:
    """
    Determine if a task instruction likely needs history context.

    Uses heuristic pattern matching to detect tasks that reference
    or depend on previous results.

    Args:
        task_instruction: The task instruction text

    Returns:
        Tuple of (should_fetch, matched_patterns)
    """
    matched = []
    for i, pattern in enumerate(_COMPILED_PATTERNS):
        if pattern.search(task_instruction):
            matched.append(HISTORY_TRIGGER_PATTERNS[i])

    return len(matched) > 0, matched


async def get_session_history(
    session_id: str,
    registry_url: str = DEFAULT_REGISTRY_URL,
    filter_agent: Optional[str] = None,
    filter_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> SessionHistory:
    """
    Fetch session history from the Agent Registry.

    Args:
        session_id: The session/conversation ID
        registry_url: URL of the Agent Registry
        filter_agent: Optional - only get messages from this agent
        filter_type: Optional - filter by interaction_type (e.g., "TASK_RESULT")
        limit: Max number of messages (default: 50)
        offset: Number of messages to skip

    Returns:
        SessionHistory with messages

    Raises:
        httpx.HTTPError: If the request fails
    """
    params = {
        "limit": limit,
        "offset": offset,
        "include_content": True,
    }

    if filter_agent:
        params["filter_agent"] = filter_agent
    if filter_type:
        params["filter_type"] = filter_type

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{registry_url}/session/{session_id}/history",
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

    messages = [
        HistoryMessage(
            message_id=msg["message_id"],
            source_agent_id=msg["source_agent_id"],
            target_agent_id=msg.get("target_agent_id"),
            interaction_type=msg["interaction_type"],
            content=msg["content"],
            created_at=msg["created_at"],
            metadata=msg.get("metadata"),
        )
        for msg in data["messages"]
    ]

    return SessionHistory(
        session_id=data["session_id"],
        total_count=data["total_count"],
        messages=messages,
        has_more=data["has_more"],
    )


async def get_session_results(
    session_id: str,
    registry_url: str = DEFAULT_REGISTRY_URL,
    limit: int = 20,
) -> List[Dict[str, str]]:
    """
    Fetch TASK_RESULT messages from a session (convenience function).

    Args:
        session_id: The session/conversation ID
        registry_url: URL of the Agent Registry
        limit: Max number of results (default: 20)

    Returns:
        List of dicts with agent_id, content, created_at
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{registry_url}/session/{session_id}/results",
            params={"limit": limit},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

    return data["results"]


def format_history_as_context(
    history: SessionHistory,
    max_chars_per_message: int = 2000,
    include_types: Optional[List[str]] = None,
) -> str:
    """
    Format session history as a context block for injection into prompts.

    Args:
        history: The SessionHistory object
        max_chars_per_message: Max characters per message content
        include_types: Optional list of interaction types to include (default: all)

    Returns:
        Formatted string suitable for prompt injection
    """
    if not history.messages:
        return ""

    lines = ["## Session History (Previous Messages)\n"]

    for msg in history.messages:
        # Filter by type if specified
        if include_types and msg.interaction_type not in include_types:
            continue

        # Truncate long content
        content = msg.content
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message] + "..."

        # Format the message
        lines.append(f"### [{msg.interaction_type}] {msg.source_agent_id}")
        if msg.target_agent_id:
            lines.append(f"To: {msg.target_agent_id}")
        lines.append(f"Time: {msg.created_at}")
        lines.append(f"\n{content}\n")
        lines.append("---\n")

    return "\n".join(lines)


def format_results_as_context(
    results: List[Dict[str, str]],
    max_chars_per_result: int = 2000,
) -> str:
    """
    Format task results as a context block for injection into prompts.

    Args:
        results: List of result dicts from get_session_results()
        max_chars_per_result: Max characters per result content

    Returns:
        Formatted string suitable for prompt injection
    """
    if not results:
        return ""

    lines = ["## Previous Results from Squad Members\n"]

    for result in results:
        agent_id = result["agent_id"]
        content = result["content"]

        # Truncate long content
        if len(content) > max_chars_per_result:
            content = content[:max_chars_per_result] + "..."

        lines.append(f"### {agent_id}")
        lines.append(f"{content}\n")
        lines.append("---\n")

    return "\n".join(lines)


# --- LangChain Tool Implementation ---

try:
    from langchain_core.tools import BaseTool
    from pydantic import Field as PydanticField

    class AgentHistoryTool(BaseTool):
        """
        LangChain tool for querying session history.

        Allows agents to fetch previous messages and results from their session.

        Example:
            tool = AgentHistoryTool(
                registry_url="http://localhost:9000",
                session_id="session.my_squad_123"
            )
            result = await tool.ainvoke({"filter_type": "TASK_RESULT"})
        """

        name: str = "read_session_history"
        description: str = """Query the message history for the current session.
Use this tool when you need to:
- See what other agents have reported or produced
- Get context from previous task results
- Summarize or combine information from multiple agents
- Reference earlier messages in the conversation

Input should be a JSON object with optional filters:
- filter_agent: Only get messages from a specific agent ID
- filter_type: Filter by message type (e.g., "TASK_RESULT", "TASK_ASSIGNMENT")
- limit: Max number of messages to return (default: 20)

Returns formatted history with agent IDs, message types, and content."""

        registry_url: str = PydanticField(
            default=DEFAULT_REGISTRY_URL,
            description="URL of the Agent Registry"
        )
        session_id: str = PydanticField(
            default="",
            description="Current session ID (set at runtime)"
        )
        max_results: int = PydanticField(
            default=20,
            description="Default max results"
        )

        def _run(
            self,
            filter_agent: Optional[str] = None,
            filter_type: Optional[str] = None,
            limit: Optional[int] = None,
        ) -> str:
            """Sync version - wraps async."""
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Can't run async in sync context with running loop
                return "Error: Use ainvoke() for async execution"
            else:
                return asyncio.run(
                    self._arun(filter_agent, filter_type, limit)
                )

        async def _arun(
            self,
            filter_agent: Optional[str] = None,
            filter_type: Optional[str] = None,
            limit: Optional[int] = None,
        ) -> str:
            """Async execution."""
            if not self.session_id:
                return "Error: session_id not set. Cannot query history."

            try:
                effective_limit = limit or self.max_results

                # If filter_type is specified, use the general history endpoint
                if filter_type or filter_agent:
                    history = await get_session_history(
                        session_id=self.session_id,
                        registry_url=self.registry_url,
                        filter_agent=filter_agent,
                        filter_type=filter_type,
                        limit=effective_limit,
                    )
                    return format_history_as_context(history)
                else:
                    # Default to getting task results (most common use case)
                    results = await get_session_results(
                        session_id=self.session_id,
                        registry_url=self.registry_url,
                        limit=effective_limit,
                    )
                    return format_results_as_context(results)

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching history: {e}")
                return f"Error fetching history: HTTP {e.response.status_code}"
            except Exception as e:
                logger.error(f"Error fetching history: {e}")
                return f"Error fetching history: {str(e)}"

except ImportError:
    # LangChain not available - provide a stub
    class AgentHistoryTool:  # type: ignore
        """Stub when LangChain is not installed."""

        def __init__(self, **kwargs):
            raise ImportError(
                "langchain_core is required for AgentHistoryTool. "
                "Install with: pip install langchain-core"
            )
