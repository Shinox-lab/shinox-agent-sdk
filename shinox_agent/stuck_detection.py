"""
Shinox Agent SDK - Stuck Detection Utility

Detects when an agent's response indicates confusion, inability, or low confidence.
Used by workers to self-assess and seek help when stuck.
"""

import os
import re
from dataclasses import dataclass
from typing import List

# Configurable confidence threshold for triggering help-seeking
STUCK_CONFIDENCE_THRESHOLD = float(os.getenv("STUCK_CONFIDENCE_THRESHOLD", "0.4"))

# --- Pattern Categories ---
# Each tuple: (compiled_regex, category, confidence_score)
# Lower confidence = more likely stuck

_INABILITY_PATTERNS = [
    r"I don'?t know",
    r"I cannot",
    r"I'm unable",
    r"outside my capabilities",
    r"beyond my scope",
    r"not equipped to",
    r"I don'?t have access",
]

_CLARIFICATION_PATTERNS = [
    r"I need more information",
    r"could you clarify",
    r"the request is unclear",
]

_HEDGING_PATTERNS = [
    r"I'm not certain",
    r"this might not be accurate",
    r"I think maybe",
]

STUCK_PATTERNS = []
for pattern in _INABILITY_PATTERNS:
    STUCK_PATTERNS.append((re.compile(pattern, re.IGNORECASE), "inability", 0.3))
for pattern in _CLARIFICATION_PATTERNS:
    STUCK_PATTERNS.append((re.compile(pattern, re.IGNORECASE), "clarification_needed", 0.7))
for pattern in _HEDGING_PATTERNS:
    STUCK_PATTERNS.append((re.compile(pattern, re.IGNORECASE), "hedging", 0.5))


@dataclass
class ResponseAssessment:
    """Assessment of an agent's response quality."""
    is_stuck: bool
    confidence: float
    matched_patterns: List[str]
    needs_help: bool


def assess_response(text: str) -> ResponseAssessment:
    """
    Scan response text for stuck patterns and return an assessment.

    Confidence scoring:
    - 0.3 for inability phrases ("I don't know", "I cannot", etc.)
    - 0.5 for hedging phrases ("I'm not certain", "this might not be accurate")
    - 0.7 for clarification-needed phrases ("I need more information")
    - 1.0 for clean responses (no stuck patterns detected)

    Args:
        text: The agent's response text to assess.

    Returns:
        ResponseAssessment with is_stuck, confidence, matched_patterns, needs_help.
    """
    matched = []
    lowest_confidence = 1.0

    for regex, category, confidence in STUCK_PATTERNS:
        if regex.search(text):
            matched.append(f"{category}:{regex.pattern}")
            if confidence < lowest_confidence:
                lowest_confidence = confidence

    is_stuck = len(matched) > 0
    needs_help = lowest_confidence < STUCK_CONFIDENCE_THRESHOLD

    return ResponseAssessment(
        is_stuck=is_stuck,
        confidence=lowest_confidence,
        matched_patterns=matched,
        needs_help=needs_help,
    )
