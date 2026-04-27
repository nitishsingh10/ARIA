#!/usr/bin/env python3
"""
ARIA Dataset Statistics Script
================================
Shows detailed statistics for an existing JSONL dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA dataset statistics")
    parser.add_argument("input", nargs="?", default="./data/train.jsonl",
                        help="Path to JSONL file (default: ./data/train.jsonl)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        console.print(f"[red]File not found: {args.input}[/red]")
        sys.exit(1)

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    console.print(f"\n[bold cyan]ARIA Dataset Statistics[/bold cyan]")
    console.print(f"File: {args.input}")
    console.print(f"Total examples: [green]{len(rows)}[/green]")
    console.print("=" * 50)

    # Parse outputs to get routing info
    capabilities: list[str] = []
    actions: list[str] = []
    complexities: list[str] = []
    confidences: list[float] = []

    for row in rows:
        output_str = row.get("output", "{}")
        try:
            parsed = json.loads(output_str)
            cap = parsed.get("routing", {}).get("capability_name") or "null"
            capabilities.append(cap)
            actions.append(parsed.get("intent", {}).get("action", "unknown"))
            complexities.append(parsed.get("intent", {}).get("complexity", "unknown"))
            conf = parsed.get("intent", {}).get("confidence", 0.0)
            confidences.append(conf)
        except (json.JSONDecodeError, TypeError):
            capabilities.append("parse_error")

    # Capability distribution
    cap_counter = Counter(capabilities)
    table = Table(title="Capability Distribution", show_header=True,
                  header_style="bold magenta")
    table.add_column("Capability", style="cyan", min_width=25)
    table.add_column("Count", justify="right", style="green")
    table.add_column("Pct", justify="right", style="yellow")
    for cap, count in sorted(cap_counter.items(), key=lambda x: -x[1]):
        pct = f"{count / len(rows):.1%}"
        table.add_row(cap, str(count), pct)
    console.print(table)

    # Action distribution
    action_counter = Counter(actions)
    console.print("\n[bold]Action Distribution:[/bold]")
    for action, count in sorted(action_counter.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 5, 40)
        console.print(f"  {action:12s}: {count:4d} {bar}")

    # Complexity distribution
    comp_counter = Counter(complexities)
    console.print("\n[bold]Complexity Distribution:[/bold]")
    for comp, count in sorted(comp_counter.items(), key=lambda x: -x[1]):
        console.print(f"  {comp:12s}: {count}")

    # Confidence stats
    if confidences:
        avg = sum(confidences) / len(confidences)
        lo = min(confidences)
        hi = max(confidences)
        console.print(f"\n[bold]Confidence:[/bold]  avg={avg:.3f}  min={lo:.3f}  max={hi:.3f}")

    # Input length stats
    input_lens = [len(row.get("input", "")) for row in rows]
    if input_lens:
        avg_len = sum(input_lens) / len(input_lens)
        console.print(f"[bold]Input Length:[/bold] avg={avg_len:.0f}  "
                       f"min={min(input_lens)}  max={max(input_lens)}")

    console.print("")


if __name__ == "__main__":
    main()
