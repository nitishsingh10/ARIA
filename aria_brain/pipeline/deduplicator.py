"""
Dataset Deduplicator
=====================
Removes near-duplicate training examples using character-level
n-gram Jaccard similarity.
"""
from __future__ import annotations

from collections import defaultdict
from aria_brain.schema.examples import TrainingExample


def _ngrams(text: str, n: int = 3) -> set[str]:
    """Produce character n-grams for a string."""
    text = text.lower().strip()
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 1))}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class DatasetDeduplicator:
    """Remove near-duplicate examples from a list."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.threshold = similarity_threshold

    def deduplicate(self, examples: list[TrainingExample]) -> list[TrainingExample]:
        """Return examples with near-duplicates removed."""
        # Group by capability for efficiency
        groups: dict[str, list[tuple[int, TrainingExample]]] = defaultdict(list)
        for i, ex in enumerate(examples):
            groups[ex.capability].append((i, ex))

        keep_indices: set[int] = set()

        for cap, group in groups.items():
            # Precompute n-grams
            grams = [(idx, ex, _ngrams(ex.input)) for idx, ex in group]
            seen: list[tuple[int, set[str]]] = []

            for idx, ex, ng in grams:
                is_dup = False
                for seen_idx, seen_ng in seen:
                    if _jaccard(ng, seen_ng) >= self.threshold:
                        is_dup = True
                        break
                if not is_dup:
                    seen.append((idx, ng))
                    keep_indices.add(idx)

        # Preserve original order
        return [ex for i, ex in enumerate(examples) if i in keep_indices]
