"""
Shinox Agent SDK - Worker Agent Class

The ShinoxWorkerAgent provides a simple way to create task-executing agents
with default wake-up logic and brain invocation.

Supports semantic wake-up using vector similarity for intelligent message filtering.
"""

import logging
import os
from typing import Optional, Any

from .base import ShinoxAgent
from .schemas import AgentMessage

# Try to import AgentCard from a2a.types, fall back to Any
try:
    from a2a.types import AgentCard
except ImportError:
    AgentCard = Any  # type: ignore

logger = logging.getLogger(__name__)

# Default semantic wake-up threshold
DEFAULT_WAKE_THRESHOLD = float(os.getenv("SEMANTIC_WAKE_THRESHOLD", "0.65"))


class ShinoxWorkerAgent(ShinoxAgent):
    """
    Worker agent with default wake-up logic and simple brain invocation.

    This is the simplest way to create a task-executing agent:

        from shinox_agent import ShinoxWorkerAgent
        from brain import brain
        from agent import agent_card

        agent = ShinoxWorkerAgent(
            agent_card=agent_card,
            brain=brain,
            enable_semantic_wake=True,  # Enable semantic matching
        )
        app = agent.app

    The worker will:
    1. Join sessions when invited by Director
    2. Listen for messages where it's @mentioned or targeted
    3. Use semantic similarity for broadcast messages (if enabled)
    4. Invoke the brain and publish the result

    Wake-up triggers (priority order):
    1. Explicit targeting (target_agent_id, @mention)
    2. TASK_ASSIGNMENT interaction type
    3. Semantic similarity > threshold (for BROADCAST_QUERY)
    4. Fallback to keyword triggers (if semantic disabled/unavailable)
    """

    def __init__(
        self,
        agent_card: AgentCard,
        brain: Any,
        agent_url: Optional[str] = None,
        triggers: Optional[list[str]] = None,
        enable_semantic_wake: bool = True,
        wake_threshold: float = DEFAULT_WAKE_THRESHOLD,
    ):
        """
        Initialize a Worker Agent.

        Args:
            agent_card: A2A AgentCard defining the agent's identity and capabilities.
            brain: LangGraph/LangChain brain with ainvoke(state, config) method.
                   Expected signature: brain.ainvoke({"messages": [...]}, config={"configurable": {"thread_id": ...}})
            agent_url: URL where agent is accessible (for registry).
            triggers: Additional keywords that wake up the agent (case-insensitive).
                      Used as fallback when semantic matching is unavailable.
            enable_semantic_wake: Enable semantic similarity matching for wake-up decisions.
            wake_threshold: Minimum similarity score to wake (0-1). Default: 0.65.
        """
        self.brain = brain
        self._custom_triggers = triggers or []
        self._enable_semantic_wake = enable_semantic_wake
        self._wake_threshold = wake_threshold
        self._semantic_matcher = None

        # Initialize base with our default handler
        super().__init__(
            agent_card=agent_card,
            session_handler=self._default_session_handler,
            triggers=triggers,
            agent_url=agent_url,
        )

        # Add semantic matcher initialization to startup
        original_startup = self.app.on_startup
        self.app.on_startup(self._init_semantic_matcher)

    async def _init_semantic_matcher(self):
        """Initialize semantic matcher after agent is registered."""
        if not self._enable_semantic_wake:
            logger.info(f"[{self.agent_id}] Semantic wake-up disabled")
            return

        try:
            from .embeddings import SemanticMatcher
            self._semantic_matcher = SemanticMatcher(
                agent_id=self.agent_id,
                registry_url=self.registry_url,
                wake_threshold=self._wake_threshold,
            )
            await self._semantic_matcher.initialize()
            logger.info(
                f"[{self.agent_id}] Semantic matcher initialized "
                f"(threshold: {self._wake_threshold})"
            )
        except ImportError:
            logger.warning(
                f"[{self.agent_id}] sentence-transformers not available. "
                "Falling back to keyword matching. "
                "Install with: pip install shinox-agent-sdk[semantic]"
            )
        except Exception as e:
            logger.warning(
                f"[{self.agent_id}] Failed to initialize semantic matcher: {e}. "
                "Falling back to keyword matching."
            )

    async def _default_session_handler(self, msg: AgentMessage, agent: 'ShinoxWorkerAgent'):
        """Default handler with wake-up logic and brain invocation."""
        headers = msg.headers
        my_id = agent.agent_id
        interaction_type = headers.interaction_type

        # --- Filtering ---

        # Ignore messages from self (loop prevention)
        if headers.source_agent_id == my_id:
            return

        # Session filtering (only process if we're in the session)
        if headers.conversation_id not in agent.active_sessions:
            # But allow direct inbox messages (TASK_ASSIGNMENT)
            if interaction_type != "TASK_ASSIGNMENT":
                return

        # --- Wake-up Logic ---
        # Priority order:
        # 1. Explicit targeting (target_agent_id, @mention) - ALWAYS wake
        # 2. TASK_ASSIGNMENT - ALWAYS wake
        # 3. BROADCAST_QUERY - Use semantic matching
        # 4. Fallback to keyword triggers (if semantic unavailable)
        should_wake = False
        wake_reason = None

        # 1. ALWAYS wake if directly targeted by agent ID
        if headers.target_agent_id == my_id:
            should_wake = True
            wake_reason = "direct_target"

        # 2. ALWAYS wake if @mentioned in content
        elif f"@{my_id}" in msg.content:
            should_wake = True
            wake_reason = "mention"

        # 3. Wake on TASK_ASSIGNMENT (explicit work from orchestrator)
        elif interaction_type == "TASK_ASSIGNMENT":
            should_wake = True
            wake_reason = "task_assignment"

        # 4. Semantic matching for BROADCAST_QUERY (first capable agent responds)
        elif interaction_type == "BROADCAST_QUERY":
            if agent._semantic_matcher and agent._semantic_matcher.initialized:
                wake, score, component = agent._semantic_matcher.should_wake(msg.content)
                if wake:
                    should_wake = True
                    wake_reason = f"semantic:{component}(score={score:.2f})"
            else:
                # Fallback to keyword triggers for BROADCAST_QUERY
                content_lower = msg.content.lower()
                for trigger in agent._custom_triggers:
                    if trigger.lower() in content_lower:
                        should_wake = True
                        wake_reason = f"keyword:{trigger}"
                        break

        # NOTE: Workers do NOT wake on:
        # - SESSION_BRIEFING: Squad-lead handles briefings and creates assignments
        # - INFO_UPDATE: Broadcast status updates, not direct tasks
        # - TASK_RESULT: Other agents' results, for passive observation
        # - AGENT_JOINED: Announcements only

        if not should_wake:
            logger.debug(f"[{my_id}] Ignoring {interaction_type} from {headers.source_agent_id}")
            return

        print(f"[{my_id}] Processing message from {headers.source_agent_id} (reason: {wake_reason})")

        # --- Brain Invocation ---
        try:
            from langchain_core.messages import HumanMessage

            # Prepare state
            lc_msg = HumanMessage(content=msg.content, name=headers.source_agent_id)
            state = {"messages": [lc_msg]}
            config = {"configurable": {"thread_id": headers.conversation_id}}

            # Invoke brain
            result = await self.brain.ainvoke(state, config=config)

            # Extract response
            response_content = result["messages"][-1].content

            # Publish result
            await agent.publish_task_result(
                content=str(response_content),
                conversation_id=headers.conversation_id,
                target_agent_id=headers.source_agent_id,
            )

        except Exception as e:
            logger.error(f"[{my_id}] Brain invocation error: {e}")
            await agent.publish_task_result(
                content=f"Error processing request: {str(e)}",
                conversation_id=headers.conversation_id,
                target_agent_id=headers.source_agent_id,
            )

    async def publish_task_result(
        self,
        content: str,
        conversation_id: str,
        target_agent_id: Optional[str] = None,
    ):
        """
        Publish a task result back to the session.

        Args:
            content: The result content
            conversation_id: The session ID
            target_agent_id: Optional agent to target (usually the requester)
        """
        await self.publish_message(
            content=content,
            conversation_id=conversation_id,
            interaction_type="TASK_RESULT",
            target_agent_id=target_agent_id,
            route_through_governance=True,
        )
