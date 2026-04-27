"""
Dataset Balancer
=================
Ensures class balance across capabilities, conversation, and memory
examples via oversampling (augmentation) and undersampling.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

from aria_brain.schema.examples import TrainingExample
from aria_brain.schema.router_output import KNOWN_CAPABILITIES

# ── Augmentation helpers ────────────────────────────────────────────
_PREFIXES = ["can you ", "please ", "hey aria ", "could you ", "I need you to ", "go ahead and "]
_SUFFIXES = [" for me", " now", " quickly", " please", " right now", " asap"]
_SYNONYM_MAP: dict[str, list[str]] = {
    "read": ["open", "show", "view", "display"],
    "open": ["read", "show", "view"],
    "show": ["display", "read", "view"],
    "delete": ["remove", "erase", "trash"],
    "remove": ["delete", "erase", "trash"],
    "create": ["make", "set up", "initialize"],
    "make": ["create", "set up", "build"],
    "run": ["execute", "launch", "start"],
    "execute": ["run", "launch", "perform"],
    "fetch": ["grab", "download", "get"],
    "get": ["fetch", "grab", "retrieve"],
    "list": ["show", "display", "enumerate"],
    "copy": ["duplicate", "clone", "replicate"],
    "move": ["relocate", "transfer", "shift"],
    "search": ["find", "look for", "locate"],
    "find": ["search", "look for", "locate"],
    "kill": ["stop", "terminate", "end"],
    "stop": ["kill", "terminate", "end"],
}
_ALT_PATHS = [
    "/tmp/alt.txt", "~/other/file.md", "./data/sample.csv",
    "/home/user/test.py", "~/notes/draft.txt", "backup.json",
]


class DatasetBalancer:
    """Balance a dataset by over/under-sampling and augmentation."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def balance(
        self,
        examples: list[TrainingExample],
        target_per_capability: int = 120,
        min_conversation_pct: float = 0.15,
        min_memory_pct: float = 0.08,
    ) -> list[TrainingExample]:
        """Return a balanced copy of the example list."""
        cap_counter = Counter(ex.capability for ex in examples)
        grouped: dict[str, list[TrainingExample]] = {}
        for ex in examples:
            grouped.setdefault(ex.capability, []).append(ex)

        balanced: list[TrainingExample] = []

        # Balance each capability to the target
        for cap_name in KNOWN_CAPABILITIES:
            bucket = grouped.get(cap_name, [])
            if len(bucket) >= target_per_capability:
                # Undersample
                balanced.extend(self.rng.sample(bucket, target_per_capability))
            elif bucket:
                # Keep all originals and oversample to fill
                balanced.extend(bucket)
                needed = target_per_capability - len(bucket)
                for _ in range(needed):
                    source = self.rng.choice(bucket)
                    balanced.append(self.augment(source))
            # If bucket is empty, skip (LLM gen might not have run)

        # Handle multi_step
        multi_step = grouped.get("multi_step", [])
        balanced.extend(multi_step)

        # Calculate total so far to compute conversation / memory needs
        current_total = len(balanced)
        estimated_total = max(current_total, 2000)

        # ── Conversation ────────────────────────────────────────────
        conv_bucket = grouped.get("conversation", [])
        conv_target = max(int(estimated_total * min_conversation_pct), len(conv_bucket))
        if len(conv_bucket) >= conv_target:
            balanced.extend(self.rng.sample(conv_bucket, conv_target))
        elif conv_bucket:
            balanced.extend(conv_bucket)
            for _ in range(conv_target - len(conv_bucket)):
                balanced.append(self.augment(self.rng.choice(conv_bucket)))

        # ── Memory ──────────────────────────────────────────────────
        mem_bucket = grouped.get("memory", [])
        mem_target = max(int(estimated_total * min_memory_pct), len(mem_bucket))
        if len(mem_bucket) >= mem_target:
            balanced.extend(self.rng.sample(mem_bucket, mem_target))
        elif mem_bucket:
            balanced.extend(mem_bucket)
            for _ in range(mem_target - len(mem_bucket)):
                balanced.append(self.augment(self.rng.choice(mem_bucket)))

        self.rng.shuffle(balanced)
        return balanced

    def augment(self, ex: TrainingExample) -> TrainingExample:
        """Create a slight variation of an example."""
        new = copy.deepcopy(ex)
        new.source = "augmented"

        transforms = [
            self._prepend_prefix,
            self._append_suffix,
            self._swap_synonym,
            self._change_casing,
            self._swap_path,
        ]
        transform = self.rng.choice(transforms)
        new.input = transform(new.input)

        # Ensure input stays non-empty
        if not new.input or len(new.input) < 3:
            new.input = ex.input
        return new

    # ── transforms ──────────────────────────────────────────────────

    def _prepend_prefix(self, text: str) -> str:
        return self.rng.choice(_PREFIXES) + text

    def _append_suffix(self, text: str) -> str:
        return text + self.rng.choice(_SUFFIXES)

    def _swap_synonym(self, text: str) -> str:
        lower = text.lower()
        for word, synonyms in _SYNONYM_MAP.items():
            if f" {word} " in f" {lower} ":
                replacement = self.rng.choice(synonyms)
                # Preserve original casing of first char
                idx = lower.find(word)
                if idx != -1:
                    return text[:idx] + replacement + text[idx + len(word):]
        return text

    def _change_casing(self, text: str) -> str:
        choice = self.rng.choice(["upper", "lower", "title"])
        if choice == "upper":
            return text.upper()
        elif choice == "lower":
            return text.lower()
        else:
            return text.title()

    def _swap_path(self, text: str) -> str:
        for path in _ALT_PATHS:
            if "/" in text:
                parts = text.split()
                for i, part in enumerate(parts):
                    if "/" in part or "~" in part:
                        parts[i] = self.rng.choice(_ALT_PATHS)
                        return " ".join(parts)
        return text
