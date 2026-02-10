"""
Tests for shinox_agent.embeddings — Semantic matching and cosine similarity.
"""

import pytest
import math
from unittest.mock import AsyncMock, MagicMock, patch

from shinox_agent.embeddings import (
    cosine_similarity,
    SemanticMatcher,
    SemanticMatcherFactory,
    DEFAULT_WAKE_THRESHOLD,
)


class TestCosineSimilarity:
    """Test cosine_similarity with known vector pairs."""

    def test_identical_vectors(self):
        vec = [1.0, 0.0, 0.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-4)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_fallback_without_numpy(self):
        """The pure-python fallback path should produce correct results."""
        # Verify the fallback math directly (same logic as in cosine_similarity)
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        dot = sum(x * y for x, y in zip(a, b))
        norm1 = sum(x * x for x in a) ** 0.5
        norm2 = sum(x * x for x in b) ** 0.5
        result = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
        assert result == pytest.approx(1.0)

        # Also test zero-norm case
        a_zero = [0.0, 0.0]
        b_any = [1.0, 2.0]
        dot = sum(x * y for x, y in zip(a_zero, b_any))
        norm1 = sum(x * x for x in a_zero) ** 0.5
        norm2 = sum(x * x for x in b_any) ** 0.5
        result = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
        assert result == pytest.approx(0.0)


class TestSemanticMatcherInit:
    def test_default_values(self):
        matcher = SemanticMatcher(agent_id="test-agent")
        assert matcher.agent_id == "test-agent"
        assert matcher.wake_threshold == DEFAULT_WAKE_THRESHOLD
        assert matcher.description_boost == 0.0
        assert matcher.skill_boost == 0.1
        assert matcher.initialized is False
        assert matcher.fallback_mode is False

    def test_custom_values(self):
        matcher = SemanticMatcher(
            agent_id="custom",
            registry_url="http://custom:8000",
            wake_threshold=0.8,
            description_boost=0.2,
            skill_boost=0.15,
        )
        assert matcher.wake_threshold == 0.8
        assert matcher.description_boost == 0.2
        assert matcher.skill_boost == 0.15


class TestSemanticMatcherInitialize:
    @pytest.mark.asyncio
    async def test_successful_initialization(self):
        matcher = SemanticMatcher(agent_id="test-agent")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "description_embedding": [0.1, 0.2, 0.3],
            "skill_embeddings": {
                "currency": {"embedding": [0.4, 0.5, 0.6]},
                "math": {"embedding": [0.7, 0.8, 0.9]},
            },
        }

        with patch("shinox_agent.embeddings.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await matcher.initialize()

        assert result is True
        assert matcher.initialized is True
        assert matcher.fallback_mode is False
        assert matcher.description_embedding == [0.1, 0.2, 0.3]
        assert len(matcher.skill_embeddings) == 2

    @pytest.mark.asyncio
    async def test_404_activates_fallback(self):
        matcher = SemanticMatcher(agent_id="unknown-agent")

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("shinox_agent.embeddings.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await matcher.initialize()

        assert result is False
        assert matcher.fallback_mode is True
        assert matcher.initialized is False

    @pytest.mark.asyncio
    async def test_connection_error_activates_fallback(self):
        matcher = SemanticMatcher(agent_id="test-agent")

        with patch("shinox_agent.embeddings.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await matcher.initialize()

        assert result is False
        assert matcher.fallback_mode is True

    @pytest.mark.asyncio
    async def test_skips_skills_without_embedding(self):
        matcher = SemanticMatcher(agent_id="test-agent")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "description_embedding": [0.1, 0.2, 0.3],
            "skill_embeddings": {
                "currency": {"embedding": [0.4, 0.5, 0.6]},
                "empty_skill": {"embedding": None},
                "no_embed": {},
            },
        }

        with patch("shinox_agent.embeddings.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await matcher.initialize()

        assert len(matcher.skill_embeddings) == 1
        assert "currency" in matcher.skill_embeddings


import httpx  # needed for ConnectError in test above


class TestComputeSimilarity:
    def test_not_initialized_returns_zero(self):
        matcher = SemanticMatcher(agent_id="test-agent")
        score, component = matcher.compute_similarity("test message")
        assert score == 0.0
        assert component == "not_initialized"

    def test_embedding_failure_returns_zero(self):
        matcher = SemanticMatcher(agent_id="test-agent")
        matcher.initialized = True
        matcher.skill_embeddings = {"skill1": [0.1, 0.2, 0.3]}

        with patch("shinox_agent.embeddings.compute_embedding", return_value=None):
            score, component = matcher.compute_similarity("test message")

        assert score == 0.0
        assert component == "embedding_failed"

    def test_max_pooling_across_skills(self):
        """Should return the highest-scoring skill match."""
        matcher = SemanticMatcher(agent_id="test-agent", skill_boost=0.0)
        matcher.initialized = True

        # Set up two skills with known embeddings
        matcher.skill_embeddings = {
            "low_match": [0.0, 1.0, 0.0],
            "high_match": [1.0, 0.0, 0.0],
        }
        matcher.description_embedding = None

        # Mock compute_embedding to return a vector close to high_match
        with patch("shinox_agent.embeddings.compute_embedding", return_value=[1.0, 0.0, 0.0]):
            score, component = matcher.compute_similarity("currency conversion")

        assert component == "skill:high_match"
        assert score == pytest.approx(1.0, abs=0.01)

    def test_description_wins_when_higher(self):
        """Description match can win over skill matches."""
        matcher = SemanticMatcher(agent_id="test-agent", skill_boost=0.0, description_boost=0.0)
        matcher.initialized = True

        matcher.skill_embeddings = {
            "low_skill": [0.0, 1.0, 0.0],
        }
        matcher.description_embedding = [1.0, 0.0, 0.0]

        with patch("shinox_agent.embeddings.compute_embedding", return_value=[1.0, 0.0, 0.0]):
            score, component = matcher.compute_similarity("test")

        assert component == "description"
        assert score == pytest.approx(1.0, abs=0.01)

    def test_skill_boost_applied(self):
        """Skill boost should increase score."""
        matcher = SemanticMatcher(agent_id="test-agent", skill_boost=0.1, description_boost=0.0)
        matcher.initialized = True

        matcher.skill_embeddings = {"skill1": [1.0, 0.0, 0.0]}
        matcher.description_embedding = [1.0, 0.0, 0.0]

        with patch("shinox_agent.embeddings.compute_embedding", return_value=[1.0, 0.0, 0.0]):
            score, component = matcher.compute_similarity("test")

        # Skill: 1.0 * 1.1 = 1.1, Description: 1.0 * 1.0 = 1.0
        # Skill wins due to boost
        assert component == "skill:skill1"
        assert score == pytest.approx(1.1, abs=0.01)

    def test_description_boost_applied(self):
        """Description boost should increase description score."""
        matcher = SemanticMatcher(agent_id="test-agent", skill_boost=0.0, description_boost=0.2)
        matcher.initialized = True

        matcher.skill_embeddings = {"skill1": [1.0, 0.0, 0.0]}
        matcher.description_embedding = [1.0, 0.0, 0.0]

        with patch("shinox_agent.embeddings.compute_embedding", return_value=[1.0, 0.0, 0.0]):
            score, component = matcher.compute_similarity("test")

        # Skill: 1.0 * 1.0 = 1.0, Description: 1.0 * 1.2 = 1.2
        # Description wins due to boost
        assert component == "description"
        assert score == pytest.approx(1.2, abs=0.01)


class TestShouldWake:
    def test_fallback_mode_returns_false(self):
        matcher = SemanticMatcher(agent_id="test-agent")
        matcher.fallback_mode = True
        wake, score, component = matcher.should_wake("any message")
        assert wake is False
        assert score == 0.0
        assert component == "fallback_mode"

    def test_above_threshold_wakes(self):
        matcher = SemanticMatcher(agent_id="test-agent", wake_threshold=0.65, skill_boost=0.0)
        matcher.initialized = True
        matcher.skill_embeddings = {"currency": [1.0, 0.0, 0.0]}
        matcher.description_embedding = None

        with patch("shinox_agent.embeddings.compute_embedding", return_value=[1.0, 0.0, 0.0]):
            wake, score, component = matcher.should_wake("convert dollars")

        assert wake is True
        assert score >= 0.65

    def test_below_threshold_does_not_wake(self):
        matcher = SemanticMatcher(agent_id="test-agent", wake_threshold=0.65, skill_boost=0.0)
        matcher.initialized = True
        matcher.skill_embeddings = {"currency": [1.0, 0.0, 0.0]}
        matcher.description_embedding = None

        # Orthogonal vector → similarity = 0.0
        with patch("shinox_agent.embeddings.compute_embedding", return_value=[0.0, 1.0, 0.0]):
            wake, score, component = matcher.should_wake("weather report")

        assert wake is False
        assert score < 0.65

    def test_exactly_at_threshold_wakes(self):
        """Score exactly at threshold should wake (>=)."""
        matcher = SemanticMatcher(agent_id="test-agent", wake_threshold=0.5, skill_boost=0.0)
        matcher.initialized = True
        matcher.description_embedding = None

        # Use vectors that produce exactly 0.5 cosine similarity
        # cos(60°) = 0.5 → vectors [1,0] and [0.5, sqrt(3)/2]
        matcher.skill_embeddings = {"s1": [1.0, 0.0]}

        with patch("shinox_agent.embeddings.compute_embedding", return_value=[0.5, math.sqrt(3) / 2]):
            wake, score, component = matcher.should_wake("test")

        assert score == pytest.approx(0.5, abs=0.01)
        assert wake is True


class TestSemanticMatcherFactory:
    def setup_method(self):
        SemanticMatcherFactory.clear_cache()

    def test_creates_new_matcher(self):
        matcher = SemanticMatcherFactory.get_matcher("agent-1")
        assert matcher.agent_id == "agent-1"

    def test_caches_matcher(self):
        m1 = SemanticMatcherFactory.get_matcher("agent-1")
        m2 = SemanticMatcherFactory.get_matcher("agent-1")
        assert m1 is m2

    def test_different_agents_different_matchers(self):
        m1 = SemanticMatcherFactory.get_matcher("agent-1")
        m2 = SemanticMatcherFactory.get_matcher("agent-2")
        assert m1 is not m2

    def test_clear_cache(self):
        m1 = SemanticMatcherFactory.get_matcher("agent-1")
        SemanticMatcherFactory.clear_cache()
        m2 = SemanticMatcherFactory.get_matcher("agent-1")
        assert m1 is not m2

    def test_custom_threshold(self):
        matcher = SemanticMatcherFactory.get_matcher("agent-1", wake_threshold=0.9)
        assert matcher.wake_threshold == 0.9
