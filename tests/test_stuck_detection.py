"""
Tests for shinox_agent.stuck_detection — Response assessment and pattern detection.
"""

import pytest
from shinox_agent.stuck_detection import assess_response, ResponseAssessment


class TestInabilityPatterns:
    def test_i_dont_know(self):
        result = assess_response("I don't know the answer to that question.")
        assert result.is_stuck is True
        assert result.confidence == 0.3
        assert any("inability" in p for p in result.matched_patterns)

    def test_i_cannot(self):
        result = assess_response("I cannot help with that request.")
        assert result.is_stuck is True
        assert result.confidence == 0.3

    def test_im_unable(self):
        result = assess_response("I'm unable to process this information.")
        assert result.is_stuck is True
        assert result.confidence == 0.3

    def test_outside_my_capabilities(self):
        result = assess_response("That is outside my capabilities.")
        assert result.is_stuck is True
        assert result.confidence == 0.3

    def test_beyond_my_scope(self):
        result = assess_response("This task is beyond my scope.")
        assert result.is_stuck is True
        assert result.confidence == 0.3

    def test_not_equipped(self):
        result = assess_response("I'm not equipped to handle database queries.")
        assert result.is_stuck is True
        assert result.confidence == 0.3


class TestClarificationPatterns:
    def test_need_more_information(self):
        result = assess_response("I need more information to complete this task.")
        assert result.is_stuck is True
        assert result.confidence == 0.7
        assert any("clarification" in p for p in result.matched_patterns)

    def test_could_you_clarify(self):
        result = assess_response("Could you clarify what you mean by 'optimize'?")
        assert result.is_stuck is True
        assert result.confidence == 0.7

    def test_request_is_unclear(self):
        result = assess_response("The request is unclear, please provide more details.")
        assert result.is_stuck is True
        assert result.confidence == 0.7


class TestHedgingPatterns:
    def test_not_certain(self):
        result = assess_response("I'm not certain about this, but the answer might be 42.")
        assert result.is_stuck is True
        assert result.confidence == 0.5
        assert any("hedging" in p for p in result.matched_patterns)

    def test_might_not_be_accurate(self):
        result = assess_response("This might not be accurate, but here's my best guess.")
        assert result.is_stuck is True
        assert result.confidence == 0.5

    def test_think_maybe(self):
        result = assess_response("I think maybe the conversion rate is 3.95.")
        assert result.is_stuck is True
        assert result.confidence == 0.5


class TestCleanResponses:
    def test_clean_response(self):
        result = assess_response("The conversion result is 7 USD = 27.67 MYR.")
        assert result.is_stuck is False
        assert result.confidence == 1.0
        assert result.matched_patterns == []

    def test_empty_string(self):
        result = assess_response("")
        assert result.is_stuck is False
        assert result.confidence == 1.0

    def test_long_clean_response(self):
        text = "Here is a detailed analysis. " * 100
        result = assess_response(text)
        assert result.is_stuck is False
        assert result.confidence == 1.0


class TestMultiplePatterns:
    def test_inability_and_hedging(self):
        result = assess_response(
            "I'm not certain about this. I don't know the exact rate."
        )
        assert result.is_stuck is True
        # Should take the lowest confidence (inability = 0.3)
        assert result.confidence == 0.3
        assert len(result.matched_patterns) >= 2

    def test_all_three_categories(self):
        result = assess_response(
            "I cannot do this. I need more information. I'm not certain."
        )
        assert result.is_stuck is True
        assert result.confidence == 0.3  # Lowest from inability
        assert len(result.matched_patterns) >= 3


class TestNeedsHelp:
    def test_inability_needs_help(self):
        """Inability confidence (0.3) < default threshold (0.4) → needs_help=True."""
        result = assess_response("I cannot help with that.")
        assert result.needs_help is True

    def test_hedging_no_help_needed(self):
        """Hedging confidence (0.5) > default threshold (0.4) → needs_help=False."""
        result = assess_response("I'm not certain about this answer.")
        assert result.needs_help is False

    def test_clarification_no_help_needed(self):
        """Clarification confidence (0.7) > default threshold (0.4) → needs_help=False."""
        result = assess_response("I need more information about the requirements.")
        assert result.needs_help is False

    def test_clean_no_help_needed(self):
        result = assess_response("The answer is 42.")
        assert result.needs_help is False


class TestCaseInsensitivity:
    def test_uppercase(self):
        result = assess_response("I DON'T KNOW THE ANSWER")
        assert result.is_stuck is True

    def test_mixed_case(self):
        result = assess_response("I'm Not Certain about this")
        assert result.is_stuck is True
