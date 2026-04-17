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

    # Show banner
    console.print(_BANNER, style="bold cyan")
    console.print(
        f"  [bold green]v{settings.aria.version}[/bold green]  •  "
        f"model: [bold]{settings.llm.model}[/bold]  •  "
        f"provider: [bold]{settings.llm.provider}[/bold]\n"
    )
    console.print(
        '  Type your message and press Enter. '
        'Type [bold]"quit"[/bold] or [bold]"exit"[/bold] to leave.\n',
    )

    # Run the async REPL
    try:
        asyncio.run(_repl(client))
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


async def _repl(client: OllamaClient) -> None:
    """Run the interactive read-eval-print loop.

    Accepts user input, sends it to Ollama, and prints the response.
    Exits on ``quit``, ``exit``, or ``Ctrl+C``.

    Args:
        client: An initialised OllamaClient instance.
    """
    messages: list[dict[str, str]] = []

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

            messages.append({"role": "user", "content": stripped})

            # Show a thinking indicator
            with console.status("[bold cyan]ARIA thinking…[/bold cyan]", spinner="dots"):
                try:
                    reply = await client.chat(messages, system=ARIA_CHAT_PROMPT)
                except Exception as exc:
                    log.error("Chat request failed", data={"error": str(exc)})
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    messages.pop()  # remove the failed user message
                    continue

            messages.append({"role": "assistant", "content": reply})
            console.print(f"\n[bold green]ARIA ❯[/bold green] {reply}\n")
    finally:
        await client.close()


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
