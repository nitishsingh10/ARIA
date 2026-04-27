"""
Dataset Exporter
=================
Exports training data to JSONL (Alpaca format), handles
train/val/test splitting, and writes statistics reports.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from aria_brain.schema.examples import TrainingExample, Dataset

# ── System instruction for the Alpaca format ────────────────────────
_SYSTEM_INSTRUCTION = (
    "You are ARIA's router. Analyze the user input and output a "
    "JSON routing decision."
)


class DatasetExporter:
    """Export a Dataset to disk in training-ready formats."""

    def __init__(self, seed: int | None = 42) -> None:
        self.rng = random.Random(seed)

    # ── JSONL export ────────────────────────────────────────────────

    def export_jsonl(
        self,
        dataset: Dataset,
        output_path: str,
        format: str = "alpaca",
    ) -> None:
        """Write all examples to a single JSONL file in Alpaca format."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        examples = list(dataset.examples)
        self.rng.shuffle(examples)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                row = self._to_alpaca(ex) if format == "alpaca" else self._to_raw(ex)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Train / Val / Test split ────────────────────────────────────

    def export_train_test_split(
        self,
        dataset: Dataset,
        output_dir: str,
        test_size: float = 0.1,
        val_size: float = 0.1,
    ) -> dict:
        """Stratified split into train/val/test JSONL files."""
        os.makedirs(output_dir, exist_ok=True)

        # Group by capability for stratification
        groups: dict[str, list[TrainingExample]] = {}
        for ex in dataset.examples:
            groups.setdefault(ex.capability, []).append(ex)

        train_examples: list[TrainingExample] = []
        val_examples: list[TrainingExample] = []
        test_examples: list[TrainingExample] = []

        for cap, exs in groups.items():
            self.rng.shuffle(exs)
            n = len(exs)
            n_test = max(1, int(n * test_size))
            n_val = max(1, int(n * val_size))
            n_train = n - n_test - n_val

            if n_train < 1:
                # Too few examples — put everything in train
                train_examples.extend(exs)
                continue

            test_examples.extend(exs[:n_test])
            val_examples.extend(exs[n_test : n_test + n_val])
            train_examples.extend(exs[n_test + n_val :])

        # Shuffle each split
        self.rng.shuffle(train_examples)
        self.rng.shuffle(val_examples)
        self.rng.shuffle(test_examples)

        # Write files
        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")
        test_path = os.path.join(output_dir, "test.jsonl")

        self._write_jsonl(train_examples, train_path)
        self._write_jsonl(val_examples, val_path)
        self._write_jsonl(test_examples, test_path)

        return {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "test_size": len(test_examples),
        }

    # ── Statistics report ───────────────────────────────────────────

    def export_stats(self, dataset: Dataset, output_path: str) -> None:
        """Write a human-readable statistics report."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        stats = dataset.stats
        lines: list[str] = [
            "=" * 50,
            "ARIA Router Training Dataset — Statistics",
            "=" * 50,
            "",
            f"Total examples:       {stats.get('total', 0)}",
            f"Conversation %:       {stats.get('conversation_pct', 0):.1%}",
            f"Memory %:             {stats.get('memory_pct', 0):.1%}",
            "",
            "─── By Difficulty ───",
        ]
        for diff, count in stats.get("by_difficulty", {}).items():
            lines.append(f"  {diff:12s}: {count}")

        lines.append("")
        lines.append("─── By Source ───")
        for src, count in stats.get("by_source", {}).items():
            lines.append(f"  {src:15s}: {count}")

        lines.append("")
        lines.append("─── By Capability ───")
        for cap, count in sorted(stats.get("by_capability", {}).items()):
            lines.append(f"  {cap:30s}: {count}")

        lines.append("")
        lines.append(f"Version:              {dataset.version}")
        lines.append(f"Created:              {dataset.created_at.isoformat()}")
        lines.append("=" * 50)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ── private helpers ─────────────────────────────────────────────

    def _to_alpaca(self, ex: TrainingExample) -> dict:
        return {
            "instruction": _SYSTEM_INSTRUCTION,
            "input": ex.input,
            "output": ex.output,
        }

    def _to_raw(self, ex: TrainingExample) -> dict:
        return ex.model_dump()

    def _write_jsonl(self, examples: list[TrainingExample], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                row = self._to_alpaca(ex)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
