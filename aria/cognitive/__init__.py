"""
aria.cognitive — Intent parsing and hybrid routing for ARIA.

The cognitive layer is ARIA's decision-making core. It translates
raw user input into precise capability selections with parameters.

Components:
    - **IntentParser**: Two-stage intent extraction (rules + LLM).
    - **Router**: Hybrid capability selector (rules → semantic → LLM).
    - **RoutingRules**: Deterministic pattern-matching engine.
    - **PromptBuilder**: Context-rich LLM prompt construction.

Usage:
    >>> from aria.cognitive import IntentParser, Router
    >>> intent = await parser.parse("read /tmp/hello.txt")
    >>> result = await router.route(intent)
    >>> result.capability_name
    'read_file'
"""

from aria.cognitive.intent import Intent, IntentParser
from aria.cognitive.prompt_builder import PromptBuilder
from aria.cognitive.router import Router, RouterResult
from aria.cognitive.rules import RoutingRule, RoutingRules

__all__ = [
    "IntentParser",
    "Intent",
    "Router",
    "RouterResult",
    "RoutingRules",
    "RoutingRule",
    "PromptBuilder",
]
