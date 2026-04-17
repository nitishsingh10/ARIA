# ARIA — Adaptive Runtime Intelligence Architecture

> A local-first, capability-driven AI runtime system powered by Ollama.

## Overview

ARIA is an intelligent local AI agent runtime that leverages Ollama for LLM inference. It provides a capability-driven architecture where modular tools can be composed, routed, and executed by the AI agent — all running on your local machine with no cloud dependency.

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running locally
- The `llama3` model pulled: `ollama pull llama3`

### Installation

```bash
# Clone and install in editable mode
cd aria
pip install -e .

# Copy env template
cp .env.example .env
```

### Usage

```bash
# Check system health (Ollama running? config valid?)
aria check

# Start the interactive REPL
aria run

# Print version info
aria version
```

### Configuration

ARIA reads configuration from `aria.yaml` in the current directory, falling back to `~/.aria/aria.yaml`. See the included `aria.yaml` for all available options.

## Architecture

```
aria/
├── config.py          # Pydantic config loader
├── main.py            # CLI entrypoint (typer)
├── llm/
│   ├── client.py      # Ollama client wrapper
│   └── prompts.py     # Base system prompts
├── core/
│   └── logger.py      # Structured logger (loguru)
├── capabilities/      # Tool/capability modules (Task 2+)
├── memory/            # Persistent memory (Task 4+)
└── interfaces/        # CLI, API, voice, desktop (Task 11+)
```

## License

MIT
