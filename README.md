# ARIA — Adaptive Runtime Intelligence Architecture

> A local-first, capability-driven AI runtime system powered by Ollama.
> No cloud. No API keys. Everything runs on your machine.

---

## Overview

ARIA is an intelligent local AI agent that can **read files, execute code, browse the web, manage processes, remember conversations, and learn about you** — all through natural language. It uses a hybrid cognitive engine that routes your requests through deterministic rules before falling back to LLM reasoning, making it both **fast and reliable**.

### Key Features

- 🧠 **Cognitive Routing** — Hybrid intent parser + tool router (rules → semantic → LLM)
- ⚡ **22 Capabilities** — Deterministic, real system actions (never hallucinated)
- 💾 **Persistent Memory** — ChromaDB vector store + knowledge graph + session context
- 🔒 **Safety First** — Destructive actions require explicit confirmation
- 🏠 **100% Local** — Runs on Ollama, no cloud dependency
- 📝 **Learns About You** — Extracts preferences, tools, projects from conversation

---

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally
- The `llama3` model pulled: `ollama pull llama3`

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ARIA.git
cd ARIA

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Copy env template
cp .env.example .env
```

### Start Ollama

```bash
ollama serve
```

### Run ARIA

```bash
# Check system health
aria check

# Start the interactive REPL
aria run

# Print version info
aria version
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `aria run` | Start the interactive REPL with cognitive routing |
| `aria check` | Health-check Ollama, config, capabilities, and cognitive layer |
| `aria version` | Print ARIA version, Python version, and model info |

### REPL Commands

Once inside `aria run`, these special commands are available:

| Command | Description |
|---------|-------------|
| `stats` | Show memory statistics (vector store, knowledge graph, session) |
| `new session` | End current session and start a fresh one |
| `quit` / `exit` | Exit the REPL |

---

## Capabilities (22 Tools)

ARIA has 22 deterministic capabilities organized into four categories. **Every action is real** — ARIA never fabricates file contents, command outputs, or system states.

### 📁 System — File Operations

| Capability | Description | Safety |
|-----------|-------------|--------|
| `read_file` | Read the full contents of a file | ✅ Safe |
| `write_file` | Write text content to a file | ⚠️ Confirm |
| `append_file` | Append text to the end of a file | ✅ Safe |
| `copy_file` | Copy a file to a new location | ✅ Safe |
| `move_file` | Move or rename a file | ⚠️ Confirm |
| `delete_file` | Permanently delete a file | ⚠️ Confirm |
| `file_info` | Get detailed metadata (size, permissions, dates) | ✅ Safe |
| `search_files` | Search for files matching a glob pattern | ✅ Safe |

### 📂 System — Folder Operations

| Capability | Description | Safety |
|-----------|-------------|--------|
| `list_directory` | List directory contents with metadata | ✅ Safe |
| `create_directory` | Create a directory (with parents) | ✅ Safe |
| `delete_directory` | Delete an entire directory | ⚠️ Confirm |
| `folder_tree` | Render an ASCII tree of a directory | ✅ Safe |

### ⚙️ System — Process Operations

| Capability | Description | Safety |
|-----------|-------------|--------|
| `list_processes` | List running processes with CPU/memory usage | ✅ Safe |
| `get_process` | Get detailed info about a process by PID | ✅ Safe |
| `kill_process` | Kill a running process | ⚠️ Confirm |
| `run_command` | Execute a shell command (dangerous commands blocked) | ⚠️ Confirm |

### 🐍 Execution — Code Runners

| Capability | Description | Safety |
|-----------|-------------|--------|
| `run_python` | Execute Python in a sandboxed subprocess | ⚠️ Confirm 🔒 |
| `run_in_docker` | Run code in an isolated Docker container | ⚠️ Confirm 🔒 |

### 🌐 External — Information

| Capability | Description | Safety |
|-----------|-------------|--------|
| `fetch_url` | Fetch a URL and extract text/links | ✅ Safe |
| `get_system_info` | Get OS, CPU, memory, disk info | ✅ Safe |
| `get_current_time` | Get current date/time in multiple formats | ✅ Safe |
| `get_env_var` | Read environment variables (secrets redacted) | ✅ Safe |

> **⚠️ Confirm** = Requires `confirm=True` parameter for destructive actions  
> **🔒** = Runs in a sandboxed environment (subprocess or Docker)

---

## Cognitive Layer

ARIA uses a **hybrid routing system** that avoids LLM calls when possible, making most commands resolve in under 1ms.

### Intent Parser (Two-Stage)

```
User Input → fast_parse() → Intent (if conf ≥ 0.7, done!)
                          ↓ (if conf < 0.7)
                     llm_parse() → Intent
```

1. **Fast Parse** — Rule-based pattern matching, zero LLM calls, <1ms
2. **LLM Parse** — Called only when fast parse is uncertain (confidence < 0.7)

### Router (Four-Step Cascade)

```
Intent → Step 1: conversation/memory? → immediate return
       → Step 2: deterministic rules?  → sub-millisecond
       → Step 3: semantic search?      → fast tag matching
       → Step 4: LLM routing           → last resort (~2-4s)
```

### Routing Rules (9 Built-in)

| Pattern | Routes To | Example |
|---------|-----------|---------|
| `read/open/cat <path>` | `read_file` | `read /tmp/hello.txt` |
| `write/save <path>` | `write_file` | `write ~/notes.txt` |
| `ls/list/dir <path>` | `list_directory` | `ls ./src` |
| `` ```python ... ``` `` | `run_python` | Fenced code blocks |
| `run/exec/sh: <cmd>` | `run_command` | `run: ls -la` |
| `fetch/curl <url>` | `fetch_url` | `fetch https://example.com` |
| `system info/sysinfo` | `get_system_info` | `system info` |
| `what time/today` | `get_current_time` | `what time is it` |
| `tree/folder structure` | `folder_tree` | `tree ~/projects` |

---

## Memory System

ARIA has a three-tier persistent memory system:

### 🔍 Vector Memory (ChromaDB)

Semantic similarity search across three collections:
- **conversations** — Chat history and summaries
- **documents** — Files and content ARIA has read
- **facts** — Extracted facts about the user

### 🧩 Knowledge Graph (JSON)

Structured entity-relation storage with rule-based extraction:
- Automatically extracts **names, tools, preferences, and projects** from conversation
- Example: *"I use VS Code and prefer dark themes"* → `uses: VS Code`, `prefers: dark themes`
- Persists to `~/.aria/data/knowledge_graph.json`

### 💬 Session Context

Ephemeral in-memory conversation tracker:
- Rolling buffer of recent messages (last 50)
- Tool usage tracking
- LLM-formatted context for each request

---

## Architecture

```
aria/
├── __init__.py            # Package version
├── config.py              # Pydantic config loader (YAML + env)
├── main.py                # CLI entrypoint (Typer) + REPL loop
│
├── llm/
│   ├── client.py          # Ollama REST client (chat, embed, health)
│   └── prompts.py         # Base system prompts
│
├── core/
│   └── logger.py          # Structured JSON logger (loguru)
│
├── cognitive/             # 🧠 Intent parsing & routing
│   ├── intent.py          # Two-stage intent parser
│   ├── router.py          # Hybrid capability router
│   ├── rules.py           # 9 deterministic routing rules
│   └── prompt_builder.py  # Context-rich LLM prompts
│
├── capabilities/          # ⚡ 22 deterministic tools
│   ├── base.py            # Capability abstract base class
│   ├── registry.py        # Central capability registry
│   ├── system/            # File, folder, process ops
│   ├── execution/         # Python & Docker runners
│   └── external/          # Web fetch, system info, time
│
├── memory/                # 💾 Persistent memory
│   ├── vector_store.py    # ChromaDB semantic store
│   ├── knowledge_graph.py # JSON entity-relation graph
│   ├── context.py         # Session context tracker
│   └── memory_manager.py  # Unified memory API
│
└── interfaces/            # 🖥️ CLI, API, voice (future)
```

---

## Configuration

ARIA reads configuration from `aria.yaml` (project-local) or `~/.aria/aria.yaml` (global).

```yaml
llm:
  provider: ollama
  model: llama3               # Any Ollama model
  base_url: http://localhost:11434
  temperature: 0.2
  max_tokens: 4096
  timeout: 60

aria:
  name: ARIA
  version: "0.1.0"
  log_level: INFO             # TRACE | DEBUG | INFO | WARNING | ERROR
  log_format: json            # json | pretty
  data_dir: ~/.aria/data
  memory_dir: ~/.aria/memory
```

### Environment Overrides

| Variable | Overrides |
|----------|-----------|
| `OLLAMA_BASE_URL` | `llm.base_url` |
| `OLLAMA_MODEL` | `llm.model` |
| `ARIA_LOG_LEVEL` | `aria.log_level` |

---

## Example Session

```
You ❯ what time is it
  [Intent]  action=search type=capability conf=0.90
  [Router]  capability=get_current_time method=rule conf=0.95

ARIA ❯ [get_current_time] ✓ (2ms)
  iso: 2026-04-17T17:55:00+05:30
  unix: 1776402300
  formatted: Thursday, April 17, 2026, 05:55 PM

You ❯ read /tmp/hello.txt
  [Intent]  action=read type=capability conf=0.85
  [Router]  capability=read_file method=rule conf=0.95

ARIA ❯ [read_file] ✓ (1ms)
  Hello, World!

You ❯ remember my name is Nitish
  [Intent]  action=remember type=memory conf=0.90
  [Router]  capability=None method=memory conf=1.00

ARIA ❯ Got it, I'll remember that.

You ❯ what is a neural network
  [Intent]  action=explain type=conversation conf=0.75
  [Router]  capability=None method=conversation conf=1.00

ARIA ❯ A neural network is a computational model inspired by...
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| LLM Runtime | Ollama (local, via REST API) |
| Default Model | llama3 (configurable) |
| Vector Store | ChromaDB (MiniLM-L6-v2 embeddings) |
| Config | YAML + Pydantic validation |
| CLI | Typer + Rich |
| Logging | Structured JSON via loguru |
| Package Manager | pip with pyproject.toml |

---

## Development Status

- [x] **Task 1** — Scaffold, config, Ollama client, CLI
- [x] **Task 2** — Capability layer (22 tools)
- [x] **Task 3** — Memory system (vector + graph + context)
- [x] **Task 4** — Cognitive layer (intent parser + router)
- [ ] **Task 5** — Agent loop (multi-step planning + execution)
- [ ] **Task 6** — Interfaces (API, voice, desktop)

---

## License

MIT
