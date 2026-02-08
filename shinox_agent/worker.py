"""
Shinox Agent SDK - Worker Agent Class

The ShinoxWorkerAgent provides a simple way to create task-executing agents
with default wake-up logic and brain invocation.

Supports:
- Semantic wake-up using vector similarity for intelligent message filtering
- Auto-history injection for tasks that need context from previous results
- Group chat semantics with GROUP_QUERY, PEER_REQUEST, EXPERTISE_OFFER
- Peer-to-peer communication with broadcast_query() and request_peer_help()
"""

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Deque

from .base import ShinoxAgent
from .schemas import (
    AgentMessage,
    GROUP_VISIBLE_TYPES,
    OBSERVE_ONLY_TYPES,
    ALWAYS_WAKE_TYPES,
    COLLABORATION_RESPONSE_TYPES,
)

# Try to import AgentCard from a2a.types, fall back to Any
try:
    from a2a.types import AgentCard
except ImportError:
    AgentCard = Any  # type: ignore

logger = logging.getLogger(__name__)

# Default semantic wake-up threshold
DEFAULT_WAKE_THRESHOLD = float(os.getenv("SEMANTIC_WAKE_THRESHOLD", "0.65"))

# Auto-history injection enabled by default
DEFAULT_AUTO_HISTORY = os.getenv("AUTO_HISTORY_INJECTION", "true").lower() == "true"

# Context window size for observed messages (per session)
DEFAULT_CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "50"))


@dataclass
class PendingCollaboration:
    """Tracks an active peer collaboration request."""
    correlation_id: str
    original_task_content: str           # What was asked of us
    my_attempt_content: str              # Our low-confidence attempt
    original_source_agent_id: str        # Who assigned us the task (e.g., squad lead)
    original_conversation_id: str
    round_number: int = 1
    max_rounds: int = 3
    created_at: float = field(default_factory=time.time)
    timeout_seconds: float = 60.0
    timeout_task: Optional[asyncio.Task] = field(default=None, repr=False)
    accepted_response_id: Optional[str] = None  # First-response latch


@dataclass
class ConversationContext:
    """Tracks conversation context for intelligent participation."""
    session_id: str
    # Recent messages observed (for context awareness)
    recent_messages: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_CONTEXT_WINDOW_SIZE)
    )
    # Agents I've interacted with in this session
    interacted_with: set = field(default_factory=set)
    # Whether I've contributed to this conversation
    has_contributed: bool = False
    # Last message I responded to (for threading)
    last_response_to: Optional[str] = None
    # Active peer collaborations: correlation_id -> PendingCollaboration
    pending_collaborations: Dict[str, 'PendingCollaboration'] = field(default_factory=dict)


class ShinoxWorkerAgent(ShinoxAgent):
    """
    Worker agent with default wake-up logic and simple brain invocation.

    This is the simplest way to create a task-executing agent:

        from shinox_agent import ShinoxWorkerAgent
        from brain import brain
        from agent import agent_card

        agent = ShinoxWorkerAgent(
            agent_card=agent_card,
            brain=my_brain,
            enable_semantic_wake=True,  # Enable semantic matching
            enable_auto_history=True,   # Auto-inject context when needed
        )
        app = agent.app

    The worker will:
    1. Join sessions when invited by Director
    2. Listen for messages where it's @mentioned or targeted
    3. Use semantic similarity for group messages (GROUP_QUERY, PEER_REQUEST, etc.)
    4. Auto-fetch history when task suggests it needs context (if enabled)
    5. Observe messages for context building (even when not waking)
    6. Invoke the brain and publish the result

    Wake-up triggers (priority order):
    1. Explicit targeting (target_agent_id, @mention) - ALWAYS wake
    2. TASK_ASSIGNMENT or DIRECT_COMMAND - ALWAYS wake
    3. GROUP_VISIBLE messages (GROUP_QUERY, PEER_REQUEST, EXPERTISE_OFFER, BROADCAST_QUERY)
       - Use semantic matching against capabilities
       - If score > threshold → wake and respond
    4. Fallback to keyword triggers (if semantic unavailable)

    Observation (no LLM wake):
    - INFO_UPDATE, TASK_RESULT, AGENT_JOINED → update context, don't invoke brain
    """

    def __init__(
        self,
        agent_card: AgentCard,
        brain: Any,
        agent_url: Optional[str] = None,
        triggers: Optional[list[str]] = None,
        enable_semantic_wake: bool = True,
        wake_threshold: float = DEFAULT_WAKE_THRESHOLD,
        enable_auto_history: bool = DEFAULT_AUTO_HISTORY,
        context_window_size: int = DEFAULT_CONTEXT_WINDOW_SIZE,
        enable_peer_collaboration: bool = False,
        collaboration_timeout: float = 60.0,
        collaboration_max_rounds: int = 3,
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
            enable_auto_history: Enable automatic history injection for tasks that need context.
                                 Default: True. Detects patterns like "summarize", "combine", "based on".
            context_window_size: Max messages to keep in local context per session. Default: 50.
            enable_peer_collaboration: Enable peer-to-peer collaboration flow. When True,
                                       workers defer publishing TASK_RESULT when stuck and
                                       instead wait for peer responses. Default: False.
            collaboration_timeout: Seconds to wait for a peer response before falling back. Default: 60.
            collaboration_max_rounds: Max back-and-forth rounds with peers. Default: 3.
        """
        self.brain = brain
        self._custom_triggers = triggers or []
        self._enable_semantic_wake = enable_semantic_wake
        self._wake_threshold = wake_threshold
        self._enable_auto_history = enable_auto_history
        self._context_window_size = context_window_size
        self._semantic_matcher = None

        # Peer collaboration settings (opt-in)
        self._enable_peer_collaboration = (
            enable_peer_collaboration
            or os.getenv("ENABLE_PEER_COLLABORATION", "false").lower() == "true"
        )
        self._collaboration_timeout = collaboration_timeout
        self._collaboration_max_rounds = collaboration_max_rounds

        # Conversation context tracking per session
        self._conversation_contexts: Dict[str, ConversationContext] = {}

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

    def _get_or_create_context(self, session_id: str) -> ConversationContext:
        """Get or create conversation context for a session."""
        if session_id not in self._conversation_contexts:
            self._conversation_contexts[session_id] = ConversationContext(
                session_id=session_id,
                recent_messages=deque(maxlen=self._context_window_size),
            )
        return self._conversation_contexts[session_id]

    def _observe_message(self, msg: AgentMessage):
        """
        Observe a message without waking the LLM.
        Updates local context for future reference.
        """
        context = self._get_or_create_context(msg.headers.conversation_id)
        context.recent_messages.append({
            "id": msg.id,
            "source": msg.headers.source_agent_id,
            "target": msg.headers.target_agent_id,
            "type": msg.headers.interaction_type,
            "content_preview": msg.content[:200] if msg.content else "",
            "timestamp": str(msg.timestamp),
        })
        logger.debug(
            f"[{self.agent_id}] Observed {msg.headers.interaction_type} "
            f"from {msg.headers.source_agent_id}"
        )

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
            # But allow direct inbox messages (TASK_ASSIGNMENT, DIRECT_COMMAND)
            if interaction_type not in ALWAYS_WAKE_TYPES:
                return

        # --- Wake-up Logic ---
        # Priority order:
        # 1. Explicit targeting (target_agent_id, @mention) - ALWAYS wake
        # 2. ALWAYS_WAKE_TYPES (TASK_ASSIGNMENT, DIRECT_COMMAND) - ALWAYS wake
        # 3. GROUP_VISIBLE_TYPES - Use semantic matching
        # 4. OBSERVE_ONLY_TYPES - Update context, don't wake LLM
        # 5. Fallback to keyword triggers (if semantic unavailable)
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

        # 3. Wake on ALWAYS_WAKE_TYPES (explicit work from orchestrator)
        elif interaction_type in ALWAYS_WAKE_TYPES:
            should_wake = True
            wake_reason = f"{interaction_type.lower()}"

        # 4. Semantic matching for GROUP_VISIBLE_TYPES (group chat semantics)
        # These are messages visible to all where semantic relevance determines response
        elif interaction_type in GROUP_VISIBLE_TYPES:
            # Check if we're a preferred agent (for PEER_REQUEST with preferences)
            preferred_agents = msg.metadata.get("preferred_agents", [])
            if my_id in preferred_agents:
                should_wake = True
                wake_reason = "preferred_peer"

            # Use semantic matching to determine relevance
            elif agent._semantic_matcher and agent._semantic_matcher.initialized:
                wake, score, component = agent._semantic_matcher.should_wake(msg.content)
                if wake:
                    should_wake = True
                    wake_reason = f"semantic:{component}(score={score:.2f})"
                else:
                    # Didn't wake, but still observe for context
                    agent._observe_message(msg)

            # Fallback to keyword triggers
            else:
                content_lower = msg.content.lower()
                for trigger in agent._custom_triggers:
                    if trigger.lower() in content_lower:
                        should_wake = True
                        wake_reason = f"keyword:{trigger}"
                        break

                if not should_wake:
                    # Still observe even if not waking
                    agent._observe_message(msg)

        # 4b. PEER_RESPONSE — Wake only if correlation_id matches a pending collaboration
        elif interaction_type in COLLABORATION_RESPONSE_TYPES:
            if agent._enable_peer_collaboration:
                corr_id = msg.metadata.get("collaboration_correlation_id")
                context = agent._get_or_create_context(headers.conversation_id)
                if corr_id and corr_id in context.pending_collaborations:
                    pending = context.pending_collaborations[corr_id]
                    if pending.accepted_response_id is None:  # First-response latch
                        should_wake = True
                        wake_reason = f"peer_response:correlation={corr_id[:8]}"
                    else:
                        agent._observe_message(msg)
                        return
                else:
                    agent._observe_message(msg)
                    return
            else:
                agent._observe_message(msg)
                return

        # 5. OBSERVE_ONLY_TYPES - Update context, don't invoke LLM
        elif interaction_type in OBSERVE_ONLY_TYPES:
            agent._observe_message(msg)
            logger.debug(
                f"[{my_id}] Observing {interaction_type} from {headers.source_agent_id} "
                "(context update only)"
            )
            return  # Don't wake, just observe

        # NOTE: Other types (SESSION_BRIEFING, SQUAD_COMPLETION) are handled by Squad Lead

        if not should_wake:
            logger.debug(f"[{my_id}] Ignoring {interaction_type} from {headers.source_agent_id}")
            return

        # Track that we're responding to this agent
        context = agent._get_or_create_context(headers.conversation_id)
        context.interacted_with.add(headers.source_agent_id)
        context.has_contributed = True
        context.last_response_to = msg.id

        print(f"[{my_id}] Processing message from {headers.source_agent_id} (reason: {wake_reason})")

        # --- Auto-History Injection ---
        # Check if task needs history context and inject if needed
        enhanced_content = msg.content

        if agent._enable_auto_history:
            try:
                from .tools.history import (
                    should_auto_fetch_history,
                    get_session_results,
                    format_results_as_context,
                )

                needs_history, matched_patterns = should_auto_fetch_history(msg.content)

                if needs_history:
                    logger.info(
                        f"[{my_id}] Auto-history triggered by patterns: {matched_patterns}"
                    )

                    # Fetch task results from session
                    results = await get_session_results(
                        session_id=headers.conversation_id,
                        registry_url=agent.registry_url,
                        limit=20,
                    )

                    if results:
                        context_block = format_results_as_context(results)
                        enhanced_content = f"{context_block}\n---\n\n**Your Task:** {msg.content}"
                        logger.info(
                            f"[{my_id}] Injected {len(results)} previous results as context"
                        )

            except ImportError:
                logger.debug(f"[{my_id}] Auto-history tools not available")
            except Exception as e:
                logger.warning(f"[{my_id}] Auto-history injection failed: {e}")

        # --- Collaboration Re-Invocation (peer_response wake) ---
        if wake_reason and wake_reason.startswith("peer_response:"):
            await agent._handle_peer_response(msg, headers, context)
            return

        # --- Brain Invocation ---
        try:
            from langchain_core.messages import HumanMessage

            # Determine if we woke on a PEER_REQUEST (acting as helper)
            is_helping_peer = interaction_type == "PEER_REQUEST"
            incoming_corr_id = msg.metadata.get("collaboration_correlation_id") if is_helping_peer else None

            # Prepare state (with potentially enhanced content)
            lc_msg = HumanMessage(content=enhanced_content, name=headers.source_agent_id)
            state = {"messages": [lc_msg]}
            config = {"configurable": {"thread_id": headers.conversation_id}}

            # Invoke brain
            result = await self.brain.ainvoke(state, config=config)

            # Extract response
            response_content = result["messages"][-1].content

            # Self-awareness: assess response quality
            response_confidence = None
            response_needs_help = False
            try:
                from .stuck_detection import assess_response
                assessment = assess_response(str(response_content))
                response_confidence = assessment.confidence
                response_needs_help = assessment.needs_help
                if assessment.is_stuck:
                    logger.info(
                        f"[{my_id}] Self-assessment: confidence={assessment.confidence}, "
                        f"patterns={assessment.matched_patterns}"
                    )
            except ImportError:
                logger.debug(f"[{my_id}] stuck_detection not available, skipping self-assessment")

            # --- Result Dispatch ---

            # Case 1: We're a helper responding to a PEER_REQUEST → publish PEER_RESPONSE
            if is_helping_peer:
                await agent.publish_peer_response(
                    content=str(response_content),
                    conversation_id=headers.conversation_id,
                    target_agent_id=headers.source_agent_id,
                    correlation_id=incoming_corr_id,
                    confidence=response_confidence,
                )
                return

            # Case 2: We're stuck AND peer collaboration is enabled → defer TASK_RESULT
            if response_needs_help and agent._enable_peer_collaboration:
                corr_id = str(uuid.uuid4())
                pending = PendingCollaboration(
                    correlation_id=corr_id,
                    original_task_content=msg.content[:2000],
                    my_attempt_content=str(response_content)[:2000],
                    original_source_agent_id=headers.source_agent_id,
                    original_conversation_id=headers.conversation_id,
                    max_rounds=agent._collaboration_max_rounds,
                    timeout_seconds=agent._collaboration_timeout,
                )
                context.pending_collaborations[corr_id] = pending

                # Start timeout handler
                pending.timeout_task = asyncio.create_task(
                    agent._collaboration_timeout_handler(corr_id, headers.conversation_id)
                )

                # Request peer help with correlation_id
                help_request = (
                    f"I'm having difficulty with a task and need help.\n\n"
                    f"Original task: {msg.content[:500]}\n\n"
                    f"My attempt: {str(response_content)[:500]}\n\n"
                    f"Confidence: {response_confidence}"
                )
                await agent.request_peer_help(
                    request=help_request,
                    conversation_id=headers.conversation_id,
                    correlation_id=corr_id,
                )
                logger.info(
                    f"[{my_id}] Deferred result, requesting peer collaboration "
                    f"(correlation={corr_id[:8]}, confidence={response_confidence})"
                )
                return

            # Case 3: Normal flow — publish TASK_RESULT
            await agent.publish_task_result(
                content=str(response_content),
                conversation_id=headers.conversation_id,
                target_agent_id=headers.source_agent_id,
                confidence=response_confidence,
            )

            # Case 4: Stuck but collaboration disabled → legacy supplementary PEER_REQUEST
            if response_needs_help:
                try:
                    help_request = (
                        f"I'm having difficulty with a task and need help.\n\n"
                        f"Original task: {msg.content[:500]}\n\n"
                        f"My attempt: {str(response_content)[:500]}\n\n"
                        f"Confidence: {response_confidence}"
                    )
                    await agent.request_peer_help(
                        request=help_request,
                        conversation_id=headers.conversation_id,
                    )
                    logger.info(f"[{my_id}] Requested peer help (confidence: {response_confidence})")
                except Exception as help_err:
                    logger.warning(f"[{my_id}] Failed to request peer help: {help_err}")

        except Exception as e:
            logger.error(f"[{my_id}] Brain invocation error: {e}")
            await agent.publish_task_result(
                content=f"Error processing request: {str(e)}",
                conversation_id=headers.conversation_id,
                target_agent_id=headers.source_agent_id,
            )

    # --- Peer Collaboration Handlers ---

    async def _handle_peer_response(
        self,
        msg: AgentMessage,
        headers,
        context: ConversationContext,
    ):
        """Handle a PEER_RESPONSE that matched a pending collaboration."""
        my_id = self.agent_id
        corr_id = msg.metadata.get("collaboration_correlation_id")

        if not corr_id or corr_id not in context.pending_collaborations:
            logger.warning(f"[{my_id}] Peer response with unknown correlation_id: {corr_id}")
            return

        pending = context.pending_collaborations[corr_id]

        # First-response latch
        pending.accepted_response_id = msg.id

        # Cancel the timeout task
        if pending.timeout_task and not pending.timeout_task.done():
            pending.timeout_task.cancel()

        logger.info(
            f"[{my_id}] Received peer response from {headers.source_agent_id} "
            f"(correlation={corr_id[:8]}, round={pending.round_number})"
        )

        # Build enriched prompt for re-invocation
        enriched_content = (
            f"You previously attempted this task but had low confidence.\n\n"
            f"Original task: {pending.original_task_content}\n\n"
            f"Your previous attempt: {pending.my_attempt_content}\n\n"
            f"A peer agent ({headers.source_agent_id}) has provided this help:\n"
            f"{msg.content}\n\n"
            f"Please incorporate this information and provide an improved response."
        )

        try:
            from langchain_core.messages import HumanMessage

            lc_msg = HumanMessage(content=enriched_content, name=headers.source_agent_id)
            state = {"messages": [lc_msg]}
            config = {"configurable": {"thread_id": pending.original_conversation_id}}

            # Re-invoke brain with enriched context
            result = await self.brain.ainvoke(state, config=config)
            response_content = result["messages"][-1].content

            # Assess the improved response
            response_confidence = None
            response_needs_help = False
            try:
                from .stuck_detection import assess_response
                assessment = assess_response(str(response_content))
                response_confidence = assessment.confidence
                response_needs_help = assessment.needs_help
            except ImportError:
                pass

            # Improved — publish final TASK_RESULT
            if not response_needs_help:
                await self.publish_task_result(
                    content=str(response_content),
                    conversation_id=pending.original_conversation_id,
                    target_agent_id=pending.original_source_agent_id,
                    confidence=response_confidence,
                )
                # Clean up
                del context.pending_collaborations[corr_id]
                logger.info(
                    f"[{my_id}] Collaboration successful after round {pending.round_number} "
                    f"(confidence={response_confidence})"
                )
                return

            # Still stuck — try another round if allowed
            if pending.round_number < pending.max_rounds:
                pending.round_number += 1
                pending.my_attempt_content = str(response_content)[:2000]
                pending.accepted_response_id = None  # Reset latch for next round

                # Start new timeout
                pending.timeout_task = asyncio.create_task(
                    self._collaboration_timeout_handler(corr_id, pending.original_conversation_id)
                )

                # Send another PEER_REQUEST
                help_request = (
                    f"I'm still having difficulty with a task after peer input (round {pending.round_number}).\n\n"
                    f"Original task: {pending.original_task_content[:500]}\n\n"
                    f"My latest attempt: {str(response_content)[:500]}\n\n"
                    f"Confidence: {response_confidence}"
                )
                await self.request_peer_help(
                    request=help_request,
                    conversation_id=pending.original_conversation_id,
                    correlation_id=corr_id,
                )
                logger.info(
                    f"[{my_id}] Collaboration round {pending.round_number}/{pending.max_rounds} "
                    f"(still stuck, confidence={response_confidence})"
                )
                return

            # Max rounds exhausted — publish best attempt
            await self.publish_task_result(
                content=str(response_content),
                conversation_id=pending.original_conversation_id,
                target_agent_id=pending.original_source_agent_id,
                confidence=response_confidence,
            )
            del context.pending_collaborations[corr_id]
            logger.info(
                f"[{my_id}] Collaboration max rounds reached ({pending.max_rounds}), "
                f"publishing best attempt (confidence={response_confidence})"
            )

        except Exception as e:
            logger.error(f"[{my_id}] Collaboration re-invocation error: {e}")
            # Fallback: publish original attempt
            await self.publish_task_result(
                content=pending.my_attempt_content,
                conversation_id=pending.original_conversation_id,
                target_agent_id=pending.original_source_agent_id,
            )
            context.pending_collaborations.pop(corr_id, None)

    async def _collaboration_timeout_handler(self, correlation_id: str, conversation_id: str):
        """
        Background task that fires if no peer responds within the timeout window.
        Publishes the worker's original low-confidence attempt as a TASK_RESULT fallback.
        """
        my_id = self.agent_id
        context = self._get_or_create_context(conversation_id)
        pending = context.pending_collaborations.get(correlation_id)

        if not pending:
            return

        try:
            await asyncio.sleep(pending.timeout_seconds)

            # Check if still pending (may have been resolved by peer response)
            if correlation_id not in context.pending_collaborations:
                return

            pending = context.pending_collaborations[correlation_id]
            logger.info(
                f"[{my_id}] Collaboration timed out after {pending.timeout_seconds}s "
                f"(correlation={correlation_id[:8]}, round={pending.round_number})"
            )

            # Publish low-confidence result with timeout metadata
            await self.publish_task_result(
                content=pending.my_attempt_content,
                conversation_id=pending.original_conversation_id,
                target_agent_id=pending.original_source_agent_id,
                confidence=0.3,
            )
            # Add timeout metadata by publishing an info update
            logger.info(
                f"[{my_id}] Published timeout fallback TASK_RESULT "
                f"(correlation={correlation_id[:8]})"
            )

            # Clean up
            del context.pending_collaborations[correlation_id]

        except asyncio.CancelledError:
            logger.debug(
                f"[{my_id}] Collaboration timeout cancelled "
                f"(correlation={correlation_id[:8]})"
            )
            raise
        except Exception as e:
            logger.error(f"[{my_id}] Collaboration timeout handler error: {e}")
            context.pending_collaborations.pop(correlation_id, None)

    async def publish_task_result(
        self,
        content: str,
        conversation_id: str,
        target_agent_id: Optional[str] = None,
        confidence: Optional[float] = None,
    ):
        """
        Publish a task result back to the session.

        Args:
            content: The result content
            conversation_id: The session ID
            target_agent_id: Optional agent to target (usually the requester)
            confidence: Optional confidence score (0-1) for result quality
        """
        metadata = {}
        if confidence is not None:
            metadata["confidence"] = confidence

        await self.publish_message(
            content=content,
            conversation_id=conversation_id,
            interaction_type="TASK_RESULT",
            target_agent_id=target_agent_id,
            route_through_governance=True,
            metadata=metadata if metadata else None,
        )

    async def publish_peer_response(
        self,
        content: str,
        conversation_id: str,
        target_agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        confidence: Optional[float] = None,
    ):
        """
        Publish a response to a PEER_REQUEST.

        Unlike TASK_RESULT (which goes to the Squad Lead for processing),
        PEER_RESPONSE is targeted at the requesting worker via correlation_id
        matching, enabling direct peer-to-peer collaboration.

        Args:
            content: The response content
            conversation_id: The session ID
            target_agent_id: Optional agent to target (the requester)
            correlation_id: Correlation ID from the original PEER_REQUEST
            confidence: Optional confidence score (0-1) for response quality
        """
        metadata = {}
        if correlation_id:
            metadata["collaboration_correlation_id"] = correlation_id
        if confidence is not None:
            metadata["confidence"] = confidence

        await self.publish_message(
            content=content,
            conversation_id=conversation_id,
            interaction_type="PEER_RESPONSE",
            target_agent_id=target_agent_id,
            route_through_governance=True,
            metadata=metadata if metadata else None,
        )
        logger.info(
            f"[{self.agent_id}] Published peer response "
            f"(correlation={correlation_id[:8] if correlation_id else 'none'})"
        )

    # --- Peer Communication Methods (Multi-Agent Coordination) ---

    async def broadcast_query(
        self,
        question: str,
        conversation_id: str,
    ):
        """
        Ask a question to the group. Any agent with relevant capabilities may respond.

        This enables group chat semantics where agents with semantic relevance
        to the question will wake up and respond.

        Args:
            question: The question to ask the group
            conversation_id: The session ID

        Example:
            await agent.broadcast_query(
                "What's the current exchange rate for USD to MYR?",
                conversation_id=session_id,
            )
            # accounting-agent with semantic match will wake and respond
        """
        await self.publish_message(
            content=question,
            conversation_id=conversation_id,
            interaction_type="GROUP_QUERY",
            target_agent_id=None,  # Broadcast to all
            route_through_governance=True,
        )
        logger.info(f"[{self.agent_id}] Broadcast query to group: {question[:50]}...")

    async def request_peer_help(
        self,
        request: str,
        conversation_id: str,
        preferred_agents: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Request help from peer agents. Relevant agents will wake based on semantic matching.

        If preferred_agents is specified, those agents get direct notification,
        but other semantically relevant agents can still contribute.

        Args:
            request: What help you need
            conversation_id: The session ID
            preferred_agents: Optional list of agent IDs to prefer
            correlation_id: Optional correlation ID for peer collaboration tracking.
                            When provided, the responding peer includes it in their
                            PEER_RESPONSE so the requester can match the response.

        Example:
            await agent.request_peer_help(
                "I need help designing a PostgreSQL schema for user auth",
                conversation_id=session_id,
                preferred_agents=["database-agent"],
            )
            # database-agent wakes (preferred), security-agent may also wake (semantic match)
        """
        metadata = {}
        if preferred_agents:
            metadata["preferred_agents"] = preferred_agents
        if correlation_id:
            metadata["collaboration_correlation_id"] = correlation_id

        await self.publish_message(
            content=request,
            conversation_id=conversation_id,
            interaction_type="PEER_REQUEST",
            target_agent_id=None,  # Broadcast, but with preferences
            route_through_governance=True,
            metadata=metadata if metadata else None,
        )
        logger.info(
            f"[{self.agent_id}] Requested peer help: {request[:50]}... "
            f"(preferred: {preferred_agents}, correlation={correlation_id[:8] if correlation_id else 'none'})"
        )

    async def offer_expertise(
        self,
        offer: str,
        conversation_id: str,
    ):
        """
        Proactively offer expertise to the group.

        Use this when you detect a conversation where your skills might be helpful,
        even if not explicitly requested.

        Args:
            offer: What expertise you're offering
            conversation_id: The session ID

        Example:
            # After observing a conversation about authentication
            await agent.offer_expertise(
                "I notice you're discussing auth. I can help with OAuth2 implementation.",
                conversation_id=session_id,
            )
        """
        await self.publish_message(
            content=offer,
            conversation_id=conversation_id,
            interaction_type="EXPERTISE_OFFER",
            target_agent_id=None,  # Visible to all
            route_through_governance=True,
        )
        logger.info(f"[{self.agent_id}] Offered expertise: {offer[:50]}...")

    def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """
        Get the conversation context for a session.

        Returns observed messages, interaction history, and participation status.

        Args:
            conversation_id: The session ID

        Returns:
            ConversationContext or None if not tracking this session
        """
        return self._conversation_contexts.get(conversation_id)

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent observed messages from a session.

        Args:
            conversation_id: The session ID
            limit: Max messages to return (from most recent)

        Returns:
            List of message summaries
        """
        context = self._conversation_contexts.get(conversation_id)
        if not context:
            return []

        messages = list(context.recent_messages)
        return messages[-limit:] if limit < len(messages) else messages
