"""
aria/main.py — CLI entrypoint for ARIA.

Provides the following commands via Typer:
    aria run      → start the interactive REPL
    aria check    → health-check Ollama connectivity and config validity
    aria version  → print version information

Usage:
    $ aria run
    $ aria check
    $ aria version
"""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aria import __version__
from aria.config import get_settings, load_settings
from aria.core.logger import get_logger, setup_logging
from aria.llm.client import OllamaClient
from aria.llm.prompts import ARIA_CHAT_PROMPT

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------
app = typer.Typer(
    name="aria",
    help="ARIA — Adaptive Runtime Intelligence Architecture",
    add_completion=False,
)

console = Console()
log = get_logger("main")

# ---------------------------------------------------------------------------
# ASCII banner
# ---------------------------------------------------------------------------
_BANNER = r"""
    ╔═══════════════════════════════════════════╗
    ║     █████╗ ██████╗ ██╗ █████╗            ║
    ║    ██╔══██╗██╔══██╗██║██╔══██╗           ║
    ║    ███████║██████╔╝██║███████║           ║
    ║    ██╔══██║██╔══██╗██║██╔══██║           ║
    ║    ██║  ██║██║  ██║██║██║  ██║           ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝           ║
    ║                                           ║
    ║   Adaptive Runtime Intelligence Architecture ║
    ╚═══════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run() -> None:
    """Start the ARIA interactive REPL."""
    settings = _init()

    client = OllamaClient(settings.llm)

    # Pre-flight health check
    if not client.health_check():
        console.print(
            "[bold red]⚠  Ollama is not running![/bold red] "
            f"Expected at [cyan]{settings.llm.base_url}[/cyan]\n"
            "Start it with: [bold]ollama serve[/bold]",
        )
        raise typer.Exit(code=1)

    # Initialise memory system
    from aria.memory import MemoryManager

    memory = MemoryManager(settings.aria, client)

    # Show banner
    console.print(_BANNER, style="bold cyan")
    console.print(
        f"  [bold green]v{settings.aria.version}[/bold green]  •  "
        f"model: [bold]{settings.llm.model}[/bold]  •  "
        f"provider: [bold]{settings.llm.provider}[/bold]\n"
    )
    console.print(
        '  Type your message and press Enter. '
        'Type [bold]"quit"[/bold] or [bold]"exit"[/bold] to leave.\n'
        '  Special commands: [bold]stats[/bold]  [bold]new session[/bold]\n',
    )

    # Run the async REPL
    try:
        asyncio.run(_repl(client, memory))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")


@app.command()
def check() -> None:
    """Run a system health check (Ollama, config, model availability)."""
    settings = _init()

    console.print(Panel("[bold]ARIA Health Check[/bold]", style="cyan"))

    # --- Config validity ---
    try:
        load_settings()
        console.print("  [green]✔[/green]  Configuration loaded and valid")
    except Exception as exc:
        console.print(f"  [red]✘[/red]  Configuration error: {exc}")
        raise typer.Exit(code=1)

    # --- Ollama connectivity ---
    client = OllamaClient(settings.llm)
    if client.health_check():
        console.print(
            f"  [green]✔[/green]  Ollama is reachable at "
            f"[cyan]{settings.llm.base_url}[/cyan]"
        )
    else:
        console.print(
            f"  [red]✘[/red]  Ollama is NOT reachable at "
            f"[cyan]{settings.llm.base_url}[/cyan]"
        )
        console.print("     → Make sure Ollama is running: [bold]ollama serve[/bold]")
        raise typer.Exit(code=1)

    # --- Model availability ---
    console.print(
        f"  [green]✔[/green]  Configured model: "
        f"[bold]{settings.llm.model}[/bold]"
    )

    console.print()
    console.print("[bold green]All checks passed![/bold green] 🎉")


@app.command()
def version() -> None:
    """Print ARIA version information."""
    settings = get_settings()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]ARIA[/bold cyan]  v{settings.aria.version}\n"
                f"Python  {sys.version.split()[0]}\n"
                f"Model   {settings.llm.model} @ {settings.llm.base_url}"
            ),
            title="Version Info",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


async def _repl(client: OllamaClient, memory: "MemoryManager") -> None:
    """Run the interactive read-eval-print loop with memory integration.

    Accepts user input, retrieves relevant memory, sends the context
    to Ollama, prints the response, and stores the conversation turn.
    Exits on ``quit``, ``exit``, or ``Ctrl+C``.

    Args:
        client: An initialised OllamaClient instance.
        memory: An initialised MemoryManager instance.
    """
    from aria.memory import MemoryManager  # noqa: F811

    try:
        while True:
            try:
                user_input = console.input("[bold blue]You ❯[/bold blue] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in {"quit", "exit"}:
                console.print("[bold yellow]Goodbye! 👋[/bold yellow]")
                break

            # --- Handle special REPL commands ---
            if stripped.lower() == "stats":
                _print_stats(memory)
                continue
            if stripped.lower() == "new session":
                new_id = memory.new_session()
                console.print(
                    f"[bold green]New session started:[/bold green] {new_id}\n"
                )
                continue

            # --- Memory-augmented chat ---
            with console.status("[bold cyan]ARIA thinking…[/bold cyan]", spinner="dots"):
                try:
                    # Retrieve relevant memory for this query
                    memory_context = await memory.recall_for_prompt(stripped)

                    # Build system prompt with memory
                    system_prompt = ARIA_CHAT_PROMPT
                    if memory_context:
                        system_prompt = f"{ARIA_CHAT_PROMPT}\n\n{memory_context}"

                    # Get conversation history from context
                    messages = memory.get_context_for_llm(last_n=10)
                    messages.append({"role": "user", "content": stripped})

                    reply = await client.chat(messages, system=system_prompt)
                except Exception as exc:
                    log.error("Chat request failed", data={"error": str(exc)})
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    continue

            console.print(f"\n[bold green]ARIA ❯[/bold green] {reply}\n")

            # Store the conversation turn in memory
            try:
                await memory.store_conversation_turn(stripped, reply)
            except Exception as exc:
                log.warning("Failed to store conversation", data={"error": str(exc)})

    finally:
        # Save knowledge graph on exit
        try:
            memory.knowledge.save()
        except Exception:
            pass
        await client.close()


def _print_stats(memory: "MemoryManager") -> None:
    """Print memory system statistics as a Rich table.

    Args:
        memory: The MemoryManager instance.
    """
    from aria.memory import MemoryManager  # noqa: F811

    stats = memory.stats()

    table = Table(title="ARIA Memory Stats", border_style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    # Vector store
    vs = stats.get("vector_store", {})
    for coll, count in vs.items():
        table.add_row("Vector Store", f"{coll}", str(count))

    # Knowledge graph
    kg = stats.get("knowledge_graph", {})
    table.add_row("Knowledge Graph", "entities", str(kg.get("entities", 0)))
    table.add_row("Knowledge Graph", "relations", str(kg.get("relations", 0)))

    # Context
    ctx = stats.get("context", {})
    table.add_row("Session", "messages", str(ctx.get("message_count", 0)))
    table.add_row("Session", "duration (s)", str(ctx.get("duration_seconds", 0)))
    table.add_row("Session", "session_id", ctx.get("session_id", "")[:12] + "…")

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init() -> "aria.config.Settings":  # noqa: F821
    """Load settings and configure logging.

    Returns:
        The validated Settings object.
    """
    from aria.config import Settings  # noqa: F811 — used for type only

    settings = get_settings()
    setup_logging(
        level=settings.aria.log_level,
        fmt=settings.aria.log_format,
    )
    return settings


# ---------------------------------------------------------------------------
# Allow direct execution: python -m aria.main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
