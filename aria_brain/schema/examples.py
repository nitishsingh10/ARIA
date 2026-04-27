"""
ARIA Training Example & Dataset Models
=======================================

TrainingExample: a single input→output pair for fine-tuning.
Dataset: a validated, stats-aware container of examples.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TrainingExample(BaseModel):
    """One training row: user input → router JSON output."""

    input: str = Field(
        ...,
        min_length=1,
        description="Raw user message.",
    )
    output: str = Field(
        ...,
        min_length=2,
        description="Compact JSON string (RouterOutput).",
    )
    capability: str = Field(
        ...,
        description="Ground-truth capability name or 'conversation'/'memory'.",
    )
    difficulty: str = Field(
        default="easy",
        description="easy | medium | hard",
    )
    language: str = Field(
        default="en",
        description="Language code (v1 is English-only).",
    )
    source: str = Field(
        default="template",
        description="template | llm_generated | augmented",
    )


class Dataset(BaseModel):
    """
    A validated, stats-aware container of TrainingExamples.

    Stats are auto-computed on initialization.
    """

    examples: list[TrainingExample] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="1.0")
    stats: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_stats(self) -> "Dataset":
        """Auto-compute dataset statistics after construction."""
        examples = self.examples
        total = len(examples)

        if total == 0:
            self.stats = {
                "total": 0,
                "by_capability": {},
                "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
                "by_source": {"template": 0, "llm_generated": 0, "augmented": 0},
                "conversation_pct": 0.0,
                "memory_pct": 0.0,
            }
            return self

        cap_counter = Counter(ex.capability for ex in examples)
        diff_counter = Counter(ex.difficulty for ex in examples)
        src_counter = Counter(ex.source for ex in examples)

        conv_count = cap_counter.get("conversation", 0)
        mem_count = cap_counter.get("memory", 0)

        self.stats = {
            "total": total,
            "by_capability": dict(sorted(cap_counter.items())),
            "by_difficulty": {
                "easy": diff_counter.get("easy", 0),
                "medium": diff_counter.get("medium", 0),
                "hard": diff_counter.get("hard", 0),
            },
            "by_source": {
                "template": src_counter.get("template", 0),
                "llm_generated": src_counter.get("llm_generated", 0),
                "augmented": src_counter.get("augmented", 0),
            },
            "conversation_pct": round(conv_count / total, 4) if total else 0.0,
            "memory_pct": round(mem_count / total, 4) if total else 0.0,
        }
        return self
