"""
Shinox Agent SDK - Semantic Embeddings Module

Provides semantic wake-up capabilities using vector similarity matching.
Agents can use this to intelligently decide when to respond to messages
based on semantic relevance rather than just keyword matching.

Usage:
    from shinox_agent.embeddings import SemanticMatcher

    matcher = SemanticMatcher(
        agent_id="my-agent",
        registry_url="http://localhost:9000",
        wake_threshold=0.65
    )
    await matcher.initialize()

    # Check if a message should wake the agent
    should_wake, score = matcher.should_wake("convert 7 USD to MYR")
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_WAKE_THRESHOLD = 0.65
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Lazy-loaded embedding model
_embedding_model = None


def _get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            logger.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install shinox-agent-sdk[semantic]"
            )
            return None
    return _embedding_model


def compute_embedding(text: str) -> Optional[List[float]]:
    """Compute embedding for a text string."""
    model = _get_embedding_model()
    if model is None:
        return None
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    try:
        import numpy as np
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except ImportError:
        # Fallback without numpy
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0


class SemanticMatcher:
    """
    Semantic matching for agent wake-up decisions.

    Uses multi-vector similarity with max pooling:
    1. Compares message to each skill embedding
    2. Compares message to description embedding
    3. Takes the maximum similarity score
    4. Wakes up if score exceeds threshold
    """

    def __init__(
        self,
        agent_id: str,
        registry_url: str = "http://localhost:9000",
        wake_threshold: float = DEFAULT_WAKE_THRESHOLD,
        description_boost: float = 0.0,
        skill_boost: float = 0.1,
    ):
        """
        Initialize the semantic matcher.

        Args:
            agent_id: The agent's ID
            registry_url: URL of the agent registry
            wake_threshold: Minimum similarity score to wake (0-1)
            description_boost: Boost factor for description matches
            skill_boost: Boost factor for skill matches (default 10%)
        """
        self.agent_id = agent_id
        self.registry_url = registry_url
        self.wake_threshold = wake_threshold
        self.description_boost = description_boost
        self.skill_boost = skill_boost

        # Cached embeddings
        self.description_embedding: Optional[List[float]] = None
        self.skill_embeddings: Dict[str, List[float]] = {}
        self.initialized = False

        # Fallback mode (when embeddings unavailable)
        self.fallback_mode = False

    async def initialize(self) -> bool:
        """
        Fetch embeddings from registry.

        Returns True if successful, False if fallback mode activated.
        """
        url = f"{self.registry_url}/agent/{self.agent_id}/embeddings"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10.0)

                if resp.status_code == 200:
                    data = resp.json()
                    self.description_embedding = data.get("description_embedding")

                    # Extract skill embeddings
                    skill_data = data.get("skill_embeddings", {})
                    for skill_name, skill_info in skill_data.items():
                        if skill_info.get("embedding"):
                            self.skill_embeddings[skill_name] = skill_info["embedding"]

                    self.initialized = True
                    logger.info(
                        f"[{self.agent_id}] Loaded embeddings: "
                        f"description={'yes' if self.description_embedding else 'no'}, "
                        f"skills={len(self.skill_embeddings)}"
                    )
                    return True

                elif resp.status_code == 404:
                    logger.warning(
                        f"[{self.agent_id}] No embeddings found in registry. "
                        "Agent may not have been registered yet. Using fallback mode."
                    )
                    self.fallback_mode = True
                    return False

                else:
                    logger.warning(
                        f"[{self.agent_id}] Failed to fetch embeddings: {resp.status_code}. "
                        "Using fallback mode."
                    )
                    self.fallback_mode = True
                    return False

        except Exception as e:
            logger.warning(
                f"[{self.agent_id}] Error fetching embeddings: {e}. Using fallback mode."
            )
            self.fallback_mode = True
            return False

    def compute_similarity(self, message: str) -> Tuple[float, str]:
        """
        Compute similarity between message and agent capabilities.

        Uses multi-vector matching with max pooling:
        - Compares message to each skill embedding individually
        - Compares message to description embedding
        - Returns the maximum similarity score

        Args:
            message: The message content to check

        Returns:
            Tuple of (similarity_score, matched_component)
            where matched_component is "skill:{name}" or "description"
        """
        if not self.initialized:
            return 0.0, "not_initialized"

        # Compute message embedding
        msg_embedding = compute_embedding(message)
        if msg_embedding is None:
            return 0.0, "embedding_failed"

        max_score = 0.0
        matched_component = "none"

        # Compare against each skill embedding (max pooling)
        for skill_name, skill_emb in self.skill_embeddings.items():
            score = cosine_similarity(msg_embedding, skill_emb)
            # Apply skill boost
            boosted_score = score * (1 + self.skill_boost)
            if boosted_score > max_score:
                max_score = boosted_score
                matched_component = f"skill:{skill_name}"

        # Compare against description embedding
        if self.description_embedding:
            score = cosine_similarity(msg_embedding, self.description_embedding)
            # Apply description boost
            boosted_score = score * (1 + self.description_boost)
            if boosted_score > max_score:
                max_score = boosted_score
                matched_component = "description"

        return max_score, matched_component

    def should_wake(self, message: str) -> Tuple[bool, float, str]:
        """
        Determine if the agent should wake up for this message.

        Args:
            message: The message content

        Returns:
            Tuple of (should_wake, score, matched_component)
        """
        if self.fallback_mode:
            # In fallback mode, return False to let keyword matching handle it
            return False, 0.0, "fallback_mode"

        score, component = self.compute_similarity(message)
        wake = score >= self.wake_threshold

        if wake:
            logger.debug(
                f"[{self.agent_id}] Semantic wake: score={score:.3f} "
                f"(threshold={self.wake_threshold}), matched={component}"
            )
        else:
            logger.debug(
                f"[{self.agent_id}] Below threshold: score={score:.3f} "
                f"(threshold={self.wake_threshold})"
            )

        return wake, score, component


class SemanticMatcherFactory:
    """Factory for creating and caching SemanticMatcher instances."""

    _instances: Dict[str, SemanticMatcher] = {}

    @classmethod
    def get_matcher(
        cls,
        agent_id: str,
        registry_url: str = "http://localhost:9000",
        wake_threshold: float = DEFAULT_WAKE_THRESHOLD,
    ) -> SemanticMatcher:
        """Get or create a SemanticMatcher for the given agent."""
        if agent_id not in cls._instances:
            cls._instances[agent_id] = SemanticMatcher(
                agent_id=agent_id,
                registry_url=registry_url,
                wake_threshold=wake_threshold,
            )
        return cls._instances[agent_id]

    @classmethod
    def clear_cache(cls):
        """Clear all cached matchers."""
        cls._instances.clear()
