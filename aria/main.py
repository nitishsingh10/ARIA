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
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from aria.config import AriaConfig
    from aria.cognitive import IntentParser, PromptBuilder, Router
    from aria.memory import MemoryManager
    from aria.planner import Planner, PlanExecutor

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

    memory = MemoryManager(settings, client)

    # Initialise cognitive layer
    from aria.capabilities import GLOBAL_REGISTRY
    from aria.cognitive import IntentParser, PromptBuilder, Router
    from aria.planner import Planner, PlanExecutor

    intent_parser = IntentParser(settings, client, memory)
    router = Router(settings, client, GLOBAL_REGISTRY, memory)
    prompt_builder = PromptBuilder(settings, memory, GLOBAL_REGISTRY)
    planner = Planner(settings, client, GLOBAL_REGISTRY, memory)
    executor = PlanExecutor(settings, client, GLOBAL_REGISTRY, memory)

    # Show banner
    console.print(_BANNER, style="bold cyan")
    console.print(
        f"  [bold green]v{settings.version}[/bold green]  •  "
        f"model: [bold]{settings.llm.model}[/bold]  •  "
        f"provider: [bold]{settings.llm.provider}[/bold]  •  "
        f"capabilities: [bold]{len(GLOBAL_REGISTRY)}[/bold]\n"
    )
    console.print(
        '  Type your message and press Enter. '
        'Type [bold]"quit"[/bold] or [bold]"exit"[/bold] to leave.\n'
        '  Special commands: [bold]stats[/bold]  [bold]new session[/bold]\n',
    )

    # Run the async REPL
    try:
        asyncio.run(_repl(client, memory, intent_parser, router, prompt_builder, planner, executor))
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

    # --- Capabilities ---
    from aria.capabilities import GLOBAL_REGISTRY

    console.print(
        f"  [green]✔[/green]  Capabilities loaded: "
        f"[bold]{len(GLOBAL_REGISTRY)}[/bold]"
    )

    # --- Cognitive layer ---
    console.print("  [green]✔[/green]  Cognitive layer (IntentParser + Router)")

    console.print()
    console.print("[bold green]All checks passed![/bold green] 🎉")


@app.command()
def version() -> None:
    """Print ARIA version information."""
    settings = get_settings()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]ARIA[/bold cyan]  v{settings.version}\n"
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


async def _repl(
    client: OllamaClient,
    memory: "MemoryManager",
    intent_parser: "IntentParser",
    router: "Router",
    prompt_builder: "PromptBuilder",
    planner: "Planner",
    executor: "PlanExecutor",
) -> None:
    """Run the interactive REPL with cognitive routing and planning.

    For each user input:
        1. Parse intent (fast rules → LLM fallback).
        2. Route to capability (rules → semantic → LLM).
        3. Plan execution (if multi-step required).
        4. Execute capability/plan or handle as conversation.
        5. Store the conversation turn in memory.

    Args:
        client: An initialised OllamaClient instance.
        memory: An initialised MemoryManager instance.
        intent_parser: The IntentParser.
        router: The Router.
        prompt_builder: The PromptBuilder.
        planner: The Planner.
        executor: The PlanExecutor.
    """
    from aria.capabilities import GLOBAL_REGISTRY

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
            if stripped.lower() == "help":
                from aria.capabilities import GLOBAL_REGISTRY
                console.print("\n[bold cyan]Available Capabilities:[/bold cyan]")
                for cap in GLOBAL_REGISTRY.search(""):
                    console.print(f"  • [bold]{cap.name}[/bold]: {cap.description}")
                console.print()
                continue

            # --- Cognitive pipeline ---
            reply = ""
            with console.status("[bold cyan]ARIA thinking…[/bold cyan]", spinner="dots"):
                try:
                    # Step 1: Parse intent
                    intent = await intent_parser.parse(stripped)

                    # Step 2: Route to capability
                    result = await router.route(intent)

                    # Log the decision
                    console.print(
                        f"  [dim][Intent]  action={intent.action} "
                        f"type={intent.intent_type} "
                        f"conf={intent.confidence:.2f}[/dim]"
                    )
                    console.print(
                        f"  [dim][Router]  capability={result.capability_name} "
                        f"method={result.routing_method} "
                        f"conf={result.confidence:.2f}[/dim]"
                    )

                except Exception as exc:
                    log.error("Cognitive pipeline failed", data={"error": str(exc)})
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    continue

            # --- Step 3: Execute or chat ---
            try:
                if intent.requires_planning or result.requires_planning:
                    # Multi-step Plan Execution
                    plan = await planner.create_plan(intent, result)
                    console.print(f"\n[bold magenta]Plan generated[/bold magenta]\n{planner.explain_plan(plan)}\n")
                    
                    if plan.requires_confirmation:
                        confirm = console.input("[bold yellow]This plan contains sensitive actions. Proceed? (y/n): [/bold yellow]")
                        if confirm.lower() != 'y':
                            reply = "Plan cancelled."
                        else:
                            with console.status("[bold cyan]Executing plan…[/bold cyan]", spinner="dots"):
                                exec_r = await executor.execute(plan)
                            reply = exec_r.summary()
                    else:
                        with console.status("[bold cyan]Executing plan…[/bold cyan]", spinner="dots"):
                            exec_r = await executor.execute(plan)
                        reply = exec_r.summary()

                elif result.capability_name:
                    # Single capability execution
                    with console.status(f"[bold cyan]Running {result.capability_name}…[/bold cyan]", spinner="dots"):
                        cap_out = await GLOBAL_REGISTRY.execute(result.capability_name, result.parameters)

                    if cap_out.success:
                        data = cap_out.data
                        def format_output(d, indent=0):
                            lines = []
                            if isinstance(d, dict):
                                for key, val in d.items():
                                    if isinstance(val, (dict, list)):
                                        lines.append(f"{' '*indent}▪ {key}:")
                                        lines.extend(format_output(val, indent+2))
                                    else:
                                        lines.append(f"{' '*indent}▪ {key}: {val}")
                            elif isinstance(d, list):
                                for item in d:
                                    if isinstance(item, (dict, list)):
                                        lines.extend(format_output(item, indent+2))
                                    else:
                                        lines.append(f"{' '*indent}• {item}")
                            else:
                                lines.append(f"{' '*indent}{d}")
                            return lines
                            
                        if data is None:
                            reply = "Done."
                        else:
                            reply = "\n".join(format_output(data))
                    else:
                        reply = f"Error: {cap_out.error}"

                elif result.routing_method == "memory":
                    # Handle memory operations
                    if intent.action == "remember":
                        await memory.remember(stripped)
                        reply = "Got it, I'll remember that."
                    else:
                        recall = await memory.recall_for_prompt(stripped)  # Will use memory.recall() wrapper later.
                        # Wait, memory.recall gives MemoryResult, recall_for_prompt gives str
                        recall_res = await memory.recall(stripped)
                        reply = recall_res.formatted_summary or "No relevant memories found."

                else:
                    # Conversation fallback
                    with console.status("[bold cyan]ARIA thinking…[/bold cyan]", spinner="dots"):
                        sys_prompt = await prompt_builder.build_chat_prompt(stripped)
                        messages = memory.get_context_for_llm(last_n=10)
                        messages.append({"role": "user", "content": stripped})
                        response = await client.chat(messages=messages, system=sys_prompt)
                        reply = response

            except Exception as exc:
                log.error("Execution failed", data={"error": str(exc)})
                reply = f"Error: {exc}"

            console.print(f"\n[bold green]ARIA ❯[/bold green] {reply}\n")

            # --- Step 4: Store conversation turn ---
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


# ---------------------------------------------------------------------------
# Stats display
# ---------------------------------------------------------------------------


def _print_stats(memory: "MemoryManager") -> None:
    """Print memory system statistics as a Rich table.

    Args:
        memory: The MemoryManager instance.
    """
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

    # Capabilities
    from aria.capabilities import GLOBAL_REGISTRY
    table.add_row("Registry", "capabilities", str(len(GLOBAL_REGISTRY.search(""))))

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init() -> "AriaConfig":
    """Load settings and configure logging.

    Returns:
        The validated AriaConfig object.
    """
    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        fmt=settings.log_format,
    )
    return settings


# ---------------------------------------------------------------------------
# Allow direct execution: python -m aria.main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
