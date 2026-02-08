"""
Shinox Agent SDK - Tools

This module provides LangChain tools that agents can use during execution.
"""

from .history import (
    AgentHistoryTool,
    get_session_history,
    get_session_results,
    should_auto_fetch_history,
    HISTORY_TRIGGER_PATTERNS,
)

__all__ = [
    "AgentHistoryTool",
    "get_session_history",
    "get_session_results",
    "should_auto_fetch_history",
    "HISTORY_TRIGGER_PATTERNS",
]
