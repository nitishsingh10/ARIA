"""
Base Generator
===============

Abstract base class for all training-example generators.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from aria_brain.schema.examples import TrainingExample
from aria_brain.schema.router_output import RouterOutput, IntentOutput, RoutingOutput


class BaseGenerator(ABC):
    """Base class every generator inherits from."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    @abstractmethod
    def generate_all(self) -> list[TrainingExample]:
        """Generate all examples this generator is responsible for."""
        ...

    # ── helpers ──────────────────────────────────────────────────────

    def _build_router_output(
        self,
        capability_name: str | None,
        parameters: dict,
        action: str,
        complexity: str = "simple",
        requires_planning: bool = False,
        intent_type: str = "capability",
        confidence: float = 0.95,
        entities: list[str] | None = None,
        reasoning: str = "",
    ) -> RouterOutput:
        """Construct a fully-validated RouterOutput."""
        return RouterOutput(
            intent=IntentOutput(
                action=action,
                complexity=complexity,
                requires_planning=requires_planning,
                intent_type=intent_type,
                confidence=confidence,
            ),
            routing=RoutingOutput(
                capability_name=capability_name,
                parameters=parameters,
                fallback="conversation",
            ),
            entities=entities or [],
            reasoning=reasoning[:100] if reasoning else "Matched user intent to capability.",
        )

    def _make_example(
        self,
        user_input: str,
        router_output: RouterOutput,
        capability: str,
        difficulty: str = "easy",
        source: str = "template",
    ) -> TrainingExample:
        """Build a TrainingExample from an input and a RouterOutput."""
        return TrainingExample(
            input=user_input,
            output=router_output.to_training_output(),
            capability=capability,
            difficulty=difficulty,
            language="en",
            source=source,
        )
