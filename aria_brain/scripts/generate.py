#!/usr/bin/env python3
"""
ARIA Dataset Generator — Main Script
======================================
Orchestrates template generation, edge cases, multi-step,
optional LLM generation, validation, balancing, and export.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

# Allow running from aria_brain/scripts/ or project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rich.console import Console
from rich.table import Table

from aria_brain.generators.capability_generator import CapabilityGenerator
from aria_brain.generators.edge_case_generator import EdgeCaseGenerator
from aria_brain.generators.multi_step_generator import MultiStepGenerator
from aria_brain.pipeline.validator import DatasetValidator
from aria_brain.pipeline.deduplicator import DatasetDeduplicator
from aria_brain.pipeline.balancer import DatasetBalancer
from aria_brain.pipeline.exporter import DatasetExporter
from aria_brain.schema.examples import Dataset

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA Router Training Dataset Generator",
    )
    parser.add_argument("--n-templates", default=60, type=int,
                        help="Examples per capability from templates (default: 60)")
    parser.add_argument("--n-llm", default=50, type=int,
                        help="Examples per capability from LLM (default: 50)")
    parser.add_argument("--n-edge-cases", default=200, type=int,
                        help="Target edge-case examples (default: 200)")
    parser.add_argument("--output-dir", default="./data",
                        help="Output directory (default: ./data)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Gemma4/Ollama generation (faster)")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    console.print("\n[bold cyan]ARIA Dataset Generator[/bold cyan]")
    console.print("=" * 40)

    all_examples = []

    # ── Step 1: Template-based examples ─────────────────────────────
    console.print("\n[yellow]Step 1:[/yellow] Generating template examples...")
    gen = CapabilityGenerator(seed=args.seed)
    examples = gen.generate_all(args.n_templates)
    all_examples.extend(examples)
    console.print(f"  ✓ Generated [green]{len(examples)}[/green] template examples")

    # ── Step 2: Edge cases ──────────────────────────────────────────
    console.print("[yellow]Step 2:[/yellow] Generating edge cases...")
    edge_gen = EdgeCaseGenerator(seed=args.seed)
    edge_examples = edge_gen.generate_all()
    all_examples.extend(edge_examples)
    console.print(f"  ✓ Generated [green]{len(edge_examples)}[/green] edge cases")

    # ── Step 3: Multi-step examples ─────────────────────────────────
    console.print("[yellow]Step 3:[/yellow] Generating multi-step examples...")
    multi_gen = MultiStepGenerator(seed=args.seed)
    multi_examples = multi_gen.generate_all()
    all_examples.extend(multi_examples)
    console.print(f"  ✓ Generated [green]{len(multi_examples)}[/green] multi-step examples")

    # ── Step 4: LLM-generated examples (optional) ──────────────────
    if not args.skip_llm:
        console.print("[yellow]Step 4:[/yellow] Generating LLM examples (Gemma4)...")
        try:
            from aria_brain.generators.ollama_generator import OllamaGenerator
            llm_gen = OllamaGenerator(seed=args.seed)
            llm_examples = llm_gen.generate_full_dataset(
                n_per_capability=args.n_llm,
            )
            all_examples.extend(llm_examples)
            console.print(f"  ✓ Generated [green]{len(llm_examples)}[/green] LLM examples")
        except Exception as e:
            console.print(f"  [red]✗ LLM generation failed: {e}[/red]")
    else:
        console.print("[yellow]Step 4:[/yellow] [dim]Skipping LLM generation (--skip-llm)[/dim]")

    # ── Step 5: Deduplicate ─────────────────────────────────────────
    console.print("[yellow]Step 5:[/yellow] Deduplicating...")
    dedup = DatasetDeduplicator(similarity_threshold=0.85)
    before_dedup = len(all_examples)
    all_examples = dedup.deduplicate(all_examples)
    removed = before_dedup - len(all_examples)
    console.print(f"  ✓ Removed [red]{removed}[/red] near-duplicates → {len(all_examples)} remaining")

    # ── Step 6: Validate ────────────────────────────────────────────
    console.print("[yellow]Step 6:[/yellow] Validating dataset...")
    validator = DatasetValidator()
    valid = []
    invalid_count = 0
    for ex in all_examples:
        vr = validator.validate_example(ex)
        if vr.is_valid:
            valid.append(ex)
        else:
            invalid_count += 1
    console.print(f"  ✓ Valid: [green]{len(valid)}[/green] / {len(all_examples)}  "
                   f"([red]{invalid_count}[/red] removed)")

    # ── Step 7: Balance ─────────────────────────────────────────────
    console.print("[yellow]Step 7:[/yellow] Balancing dataset...")
    balancer = DatasetBalancer(seed=args.seed)
    balanced = balancer.balance(valid)
    console.print(f"  ✓ Balanced to [green]{len(balanced)}[/green] examples")

    # ── Step 8: Build dataset and export ────────────────────────────
    console.print("[yellow]Step 8:[/yellow] Exporting...")
    dataset = Dataset(
        examples=balanced,
        version="1.0",
        created_at=datetime.now(timezone.utc),
    )

    exporter = DatasetExporter(seed=args.seed)
    paths = exporter.export_train_test_split(dataset, args.output_dir)
    stats_path = os.path.join(args.output_dir, "stats.txt")
    exporter.export_stats(dataset, stats_path)

    # ── Summary ─────────────────────────────────────────────────────
    console.print("\n[bold green]Dataset complete![/bold green]")
    console.print(f"  Train: [cyan]{paths['train_size']}[/cyan] examples")
    console.print(f"  Val:   [cyan]{paths['val_size']}[/cyan] examples")
    console.print(f"  Test:  [cyan]{paths['test_size']}[/cyan] examples")
    console.print(f"  Files: [dim]{args.output_dir}/[/dim]")

    # ── Capability table ────────────────────────────────────────────
    console.print("\n[bold]Dataset Statistics:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Capability", style="cyan", min_width=25)
    table.add_column("Count", justify="right", style="green")
    for cap, count in sorted(dataset.stats.get("by_capability", {}).items()):
        table.add_row(cap, str(count))
    console.print(table)

    # Difficulty breakdown
    console.print("\n[bold]Difficulty:[/bold]")
    for diff, count in dataset.stats.get("by_difficulty", {}).items():
        console.print(f"  {diff:10s}: {count}")

    console.print(f"\n  Stats saved to: [dim]{stats_path}[/dim]\n")


if __name__ == "__main__":
    main()
