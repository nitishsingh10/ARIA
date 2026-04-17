"""
aria/cognitive/rules.py — Deterministic routing rules for ARIA.

Provides zero-LLM routing for common patterns. Rules are always
tried before the LLM router, guaranteeing sub-millisecond responses
for well-known input patterns.

Example:
    >>> rules = RoutingRules()
    >>> match = rules.match("read /tmp/hello.txt", intent)
    >>> match[0].capability_name
    'read_file'
    >>> match[1]
    {'path': '/tmp/hello.txt'}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from aria.cognitive.intent import Intent
from aria.core.logger import get_logger

log = get_logger("cognitive.rules")


# ---------------------------------------------------------------------------
# RoutingRule dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoutingRule:
    """A single deterministic routing rule.

    Attributes:
        name: Rule identifier (e.g. ``"read_file_direct"``).
        pattern: Regex pattern matched against user input.
        capability_name: Exact name in GLOBAL_REGISTRY.
        parameter_extractor: Function ``(match, user_input, intent) → dict``.
        priority: Higher values are checked first.
        description: Human-readable description.
    """

    name: str
    pattern: re.Pattern
    capability_name: str
    parameter_extractor: Callable
    priority: int = 50
    description: str = ""


# ---------------------------------------------------------------------------
# Parameter extractor helpers
# ---------------------------------------------------------------------------

def _extract_path(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract a file/directory path from regex group 2."""
    path = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else "."
    return {"path": path}


def _extract_path_or_dot(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract a path from group 2, defaulting to '.' if empty."""
    path = ""
    if match.lastindex and match.lastindex >= 2:
        path = match.group(2).strip()
    return {"path": path if path else "."}


def _extract_python_code(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract Python code from a fenced code block."""
    code = match.group(1).strip()
    return {"code": code}


def _extract_command(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract a shell command from the input."""
    command = match.group(2).strip()
    return {"command": command}


def _extract_url(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract a URL from the input."""
    url = match.group(2).strip()
    return {"url": url}


def _extract_nothing(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Return empty parameters (for capabilities that need none)."""
    return {}


def _extract_write_file(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract path for write_file, content left empty for user to fill."""
    path = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
    return {"path": path, "content": ""}


def _extract_create_dir(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract path for create_directory from flexible patterns.

    Handles:
        - 'mkdir /tmp/test' → group 2
        - 'create folder myproject' → group 3
        - 'create a folder called myproject' → group 3
    """
    # Try group 3 first (natural language: "create folder X")
    if match.lastindex and match.lastindex >= 3 and match.group(3):
        return {"path": match.group(3).strip()}
    # Then group 2 (direct: "mkdir /tmp/test")
    if match.lastindex and match.lastindex >= 2 and match.group(2):
        return {"path": match.group(2).strip()}
    return {"path": "."}


def _extract_search_files(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract pattern and optional directory for file search."""
    pattern = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else "*"
    # Check if group 3 has a directory
    directory = "."
    if match.lastindex and match.lastindex >= 3 and match.group(3):
        directory = match.group(3).strip()
    return {"pattern": pattern, "directory": directory}


def _extract_move_file(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract source and destination from move/rename commands."""
    source = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
    dest = match.group(3).strip() if match.lastindex and match.lastindex >= 3 else ""
    return {"source": source, "destination": dest, "confirm": True}


def _extract_copy_file(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract source and destination from copy commands."""
    source = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
    dest = match.group(3).strip() if match.lastindex and match.lastindex >= 3 else ""
    return {"source": source, "destination": dest}


def _extract_delete_path(match: re.Match, user_input: str, intent: Intent) -> dict:
    """Extract path for delete, with confirm=True."""
    path = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
    return {"path": path, "confirm": True}


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

BUILTIN_RULES: List[RoutingRule] = [
    # ── File operations ───────────────────────────────────────
    RoutingRule(
        name="read_file_direct",
        pattern=re.compile(
            r"^(read|open|show|cat|print)\s+([~/.][\S]+)",
            re.IGNORECASE,
        ),
        capability_name="read_file",
        parameter_extractor=_extract_path,
        priority=100,
        description="Direct file read: 'read /path/to/file'",
    ),
    RoutingRule(
        name="write_file_direct",
        pattern=re.compile(
            r"^(write|save|create file)\s+([~/.][\S]+)",
            re.IGNORECASE,
        ),
        capability_name="write_file",
        parameter_extractor=_extract_write_file,
        priority=100,
        description="Direct file write: 'write /path/to/file'",
    ),
    RoutingRule(
        name="delete_file_direct",
        pattern=re.compile(
            r"^(delete|remove|rm)\s+(?:file\s+)?([~/.][\S]+)",
            re.IGNORECASE,
        ),
        capability_name="delete_file",
        parameter_extractor=_extract_delete_path,
        priority=95,
        description="Delete a file: 'delete /path/to/file'",
    ),
    RoutingRule(
        name="move_file_direct",
        pattern=re.compile(
            r"^(move|mv|rename)\s+([~/.][\S]+)\s+(?:to\s+)?([~/.][\S]+)",
            re.IGNORECASE,
        ),
        capability_name="move_file",
        parameter_extractor=_extract_move_file,
        priority=95,
        description="Move/rename: 'move /a.txt to /b.txt'",
    ),
    RoutingRule(
        name="copy_file_direct",
        pattern=re.compile(
            r"^(copy|cp)\s+([~/.][\S]+)\s+(?:to\s+)?([~/.][\S]+)",
            re.IGNORECASE,
        ),
        capability_name="copy_file",
        parameter_extractor=_extract_copy_file,
        priority=95,
        description="Copy: 'copy /a.txt to /b.txt'",
    ),
    RoutingRule(
        name="file_info_direct",
        pattern=re.compile(
            r"^(file info|info|stat|file size)\s+([~/.][\S]+)",
            re.IGNORECASE,
        ),
        capability_name="file_info",
        parameter_extractor=_extract_path,
        priority=90,
        description="File metadata: 'file info /path/to/file'",
    ),
    RoutingRule(
        name="search_files_direct",
        pattern=re.compile(
            r"^(?:search|find|look for)\s+(?:files?\s+)?([*\w.]+)(?:\s+in\s+([~/.][\S]+))?",
            re.IGNORECASE,
        ),
        capability_name="search_files",
        parameter_extractor=lambda m, u, i: {
            "pattern": m.group(1).strip(),
            "directory": m.group(2).strip() if m.lastindex and m.lastindex >= 2 and m.group(2) else ".",
        },
        priority=90,
        description="Search files: 'search *.py in ./src'",
    ),

    # ── Folder operations ─────────────────────────────────────
    RoutingRule(
        name="list_directory_direct",
        pattern=re.compile(
            r"^(ls|list|dir)\s*([~/.][\S]*)?",
            re.IGNORECASE,
        ),
        capability_name="list_directory",
        parameter_extractor=_extract_path_or_dot,
        priority=100,
        description="Directory listing: 'ls /path' or 'ls'",
    ),
    RoutingRule(
        name="create_directory_direct",
        pattern=re.compile(
            r"^(mkdir|make dir|make directory)\s+([~/.]?[\S]+)",
            re.IGNORECASE,
        ),
        capability_name="create_directory",
        parameter_extractor=_extract_path,
        priority=100,
        description="Create directory (short): 'mkdir /tmp/test'",
    ),
    RoutingRule(
        name="create_folder_natural",
        pattern=re.compile(
            r"^(?:create|make|new)\s+(?:a\s+)?(?:folder|directory|dir)\s+(?:called\s+|named\s+|at\s+)?([~/.]?[\S]+)",
            re.IGNORECASE,
        ),
        capability_name="create_directory",
        parameter_extractor=lambda m, u, i: {"path": m.group(1).strip()},
        priority=98,
        description="Create directory (natural): 'create a folder called myproject'",
    ),
    RoutingRule(
        name="delete_directory_direct",
        pattern=re.compile(
            r"^(?:delete|remove|rm|rmdir)\s+(?:folder|directory|dir)\s+([~/.]?[\S]+)",
            re.IGNORECASE,
        ),
        capability_name="delete_directory",
        parameter_extractor=lambda m, u, i: {"path": m.group(1).strip(), "confirm": True},
        priority=95,
        description="Delete directory: 'delete folder /tmp/test'",
    ),
    RoutingRule(
        name="folder_tree_direct",
        pattern=re.compile(
            r"^(tree|show tree|folder structure)\s*([~/.][\S]*)?",
            re.IGNORECASE,
        ),
        capability_name="folder_tree",
        parameter_extractor=_extract_path_or_dot,
        priority=85,
        description="Show directory tree: 'tree /path'",
    ),

    # ── Execution ─────────────────────────────────────────────
    RoutingRule(
        name="run_python_direct",
        pattern=re.compile(
            r"```python\n(.*?)```",
            re.DOTALL,
        ),
        capability_name="run_python",
        parameter_extractor=_extract_python_code,
        priority=90,
        description="Execute fenced Python code block",
    ),
    RoutingRule(
        name="run_command_direct",
        pattern=re.compile(
            r"^(run|exec|shell|bash|sh):\s*(.+)",
            re.IGNORECASE,
        ),
        capability_name="run_command",
        parameter_extractor=_extract_command,
        priority=90,
        description="Direct shell command: 'run: ls -la'",
    ),

    # ── External ──────────────────────────────────────────────
    RoutingRule(
        name="fetch_url_direct",
        pattern=re.compile(
            r"^(fetch|get|scrape|curl)\s+(https?://\S+)",
            re.IGNORECASE,
        ),
        capability_name="fetch_url",
        parameter_extractor=_extract_url,
        priority=95,
        description="Fetch a URL: 'fetch https://example.com'",
    ),
    RoutingRule(
        name="system_info_direct",
        pattern=re.compile(
            r"(?:system info|sysinfo|hardware info|memory usage|disk usage|cpu info)",
            re.IGNORECASE,
        ),
        capability_name="get_system_info",
        parameter_extractor=_extract_nothing,
        priority=80,
        description="Get system information",
    ),
    RoutingRule(
        name="current_time_direct",
        pattern=re.compile(
            r"(?:what time|current time|what'?s the time|what'?s today|what day)",
            re.IGNORECASE,
        ),
        capability_name="get_current_time",
        parameter_extractor=_extract_nothing,
        priority=80,
        description="Get current time/date",
    ),
]


# ---------------------------------------------------------------------------
# RoutingRules engine
# ---------------------------------------------------------------------------

class RoutingRules:
    """Deterministic rule-matching engine.

    Loads built-in rules sorted by priority (descending). Returns
    the first matching rule and its extracted parameters.
    """

    def __init__(self) -> None:
        """Initialise with built-in rules, sorted by priority."""
        self._rules: List[RoutingRule] = sorted(
            BUILTIN_RULES, key=lambda r: r.priority, reverse=True
        )
        log.debug(
            "RoutingRules loaded",
            data={"count": len(self._rules)},
        )

    def match(
        self,
        user_input: str,
        intent: Intent,
    ) -> Optional[Tuple[RoutingRule, dict]]:
        """Find the first matching rule for the given input.

        Rules are checked in priority order (highest first). The
        first rule whose regex matches the input wins.

        Args:
            user_input: The raw user text.
            intent: The parsed Intent (passed to extractors).

        Returns:
            A tuple of ``(RoutingRule, extracted_params)`` or None.
        """
        for rule in self._rules:
            match = rule.pattern.search(user_input)
            if match:
                try:
                    params = rule.parameter_extractor(match, user_input, intent)
                    log.debug(
                        f"Rule matched: {rule.name}",
                        data={
                            "capability": rule.capability_name,
                            "params": params,
                        },
                    )
                    return (rule, params)
                except Exception as exc:
                    log.warning(
                        f"Rule extractor failed: {rule.name}",
                        data={"error": str(exc)},
                    )
                    continue
        return None

    def add_rule(self, rule: RoutingRule) -> None:
        """Register a new routing rule.

        The rule is inserted in the correct priority position.

        Args:
            rule: The routing rule to add.
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        log.debug(f"Added routing rule: {rule.name}")

    @property
    def rules(self) -> List[RoutingRule]:
        """Return the list of all rules (read-only).

        Returns:
            A list of ``RoutingRule`` instances.
        """
        return list(self._rules)
