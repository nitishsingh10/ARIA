#!/usr/bin/env python3
"""
ARIA Dataset Validation Script
================================
Validates an existing JSONL dataset against the RouterOutput schema.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rich.console import Console
from aria_brain.schema.examples import TrainingExample, Dataset
from aria_brain.pipeline.validator import DatasetValidator

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ARIA training dataset")
    parser.add_argument("input", help="Path to JSONL file to validate")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show individual errors")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        console.print(f"[red]File not found: {args.input}[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]Validating:[/bold cyan] {args.input}")
    console.print("=" * 40)

    examples: list[TrainingExample] = []
    parse_errors = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ex = TrainingExample(
                    input=row.get("input", ""),
                    output=row.get("output", ""),
                    capability=row.get("capability", "unknown"),
                    difficulty=row.get("difficulty", "easy"),
                    language=row.get("language", "en"),
                    source=row.get("source", "template"),
                )
                examples.append(ex)
            except Exception as e:
                parse_errors += 1
                if args.verbose:
                    console.print(f"  [red]Line {line_num}: Parse error — {e}[/red]")

    console.print(f"  Parsed: {len(examples)} examples ({parse_errors} parse errors)")

    dataset = Dataset(examples=examples, version="1.0")
    validator = DatasetValidator()
    report = validator.validate_dataset(dataset)

    console.print(f"\n  Total:   {report.total}")
    console.print(f"  Valid:   [green]{report.valid}[/green]")
    console.print(f"  Invalid: [red]{report.invalid}[/red]")
    console.print(f"  Quality: [cyan]{report.quality_score:.1%}[/cyan]")

    if args.verbose and report.errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in report.errors[:20]:
            console.print(f"  #{err['example_id']}: {err['error_message']}")

    if report.warnings:
        console.print(f"\n  Warnings: {len(report.warnings)}")

    console.print("")


if __name__ == "__main__":
    main()
