"""
aria/capabilities/system/process_ops.py — Process and shell capabilities.

Provides shell command execution, process listing, inspection, and
process termination.

Safety:
    - ``run_command`` blocks dangerous patterns (see ``_BLOCKED_PATTERNS``).
    - ``kill_process`` never kills PID 1 or the current process.
    - stdout/stderr are capped at 100 KB.
    - All destructive operations require ``confirm=True``.

Example:
    >>> from aria.capabilities.registry import GLOBAL_REGISTRY
    >>> r = await GLOBAL_REGISTRY.execute("run_command", {"command": "ls -la"})
    >>> print(r.data["stdout"])
"""

from __future__ import annotations

import asyncio
import datetime
import os
import re
import time
from typing import Any, Dict, List, Optional

import psutil
from pydantic import Field

from aria.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityOutput,
)
from aria.core.logger import get_logger

log = get_logger("capabilities.process_ops")

# ─── Safety ───────────────────────────────────────────────────────────
_MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB

# Patterns that are unconditionally blocked.
_BLOCKED_PATTERNS: list[str] = [
    r"\brm\s+-rf\s+/\s*$",        # rm -rf /
    r"\bsudo\s+rm\b",             # sudo rm
    r"\bmkfs\b",                  # mkfs
    r"\bdd\s+if=",                # dd if=
    r"\bshutdown\b",              # shutdown
    r"\breboot\b",                # reboot
    r":\(\)\s*\{\s*:\|:\&\s*\};", # fork bomb
]

_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS), re.IGNORECASE)


def _is_command_safe(command: str) -> tuple[bool, str]:
    """Check whether a shell command passes the safety filter.

    Args:
        command: The raw command string.

    Returns:
        A tuple of (is_safe, reason).
    """
    if _BLOCKED_RE.search(command):
        return False, "Command matches a blocked dangerous pattern."
    return True, ""


def _truncate(text: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Truncate text to *max_bytes* UTF-8 bytes.

    Args:
        text: Input text.
        max_bytes: Maximum byte length.

    Returns:
        Possibly truncated string.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace") + "\n[TRUNCATED]"


# ═══════════════════════════════════════════════════════════════════════
# run_command
# ═══════════════════════════════════════════════════════════════════════


class RunCommandInput(CapabilityInput):
    """Input schema for the ``run_command`` capability.

    Attributes:
        command: The shell command to execute.
        cwd: Working directory (defaults to current dir).
        timeout: Max seconds before the command is killed.
        env_vars: Additional environment variables.
    """

    command: str = Field(description="Shell command to execute.")
    cwd: Optional[str] = Field(
        default=None, description="Working directory."
    )
    timeout: int = Field(default=30, description="Timeout in seconds.")
    env_vars: Dict[str, str] = Field(
        default_factory=dict, description="Additional env vars."
    )


class RunCommandCapability(Capability):
    """Execute a shell command and capture its output.

    Blocks known-dangerous commands (see module docstring).  Caps
    stdout/stderr at 100 KB each.

    Example input::

        {"command": "ls -la /tmp", "timeout": 10}

    Example output::

        {
            "stdout": "total 0\\n...",
            "stderr": "",
            "return_code": 0,
            "duration_ms": 42.1
        }
    """

    name = "run_command"
    description = "Execute a shell command and return stdout, stderr, and exit code."
    input_schema = RunCommandInput
    tags = ["system", "process", "execute"]
    requires_confirmation = True

    async def execute(self, input_data: RunCommandInput) -> CapabilityOutput:
        """Execute the command.

        Args:
            input_data: Validated ``RunCommandInput``.

        Returns:
            CapabilityOutput with stdout, stderr, return_code, duration_ms.

        Raises:
            CapabilityError: If the command matches a blocked pattern.
        """
        safe, reason = _is_command_safe(input_data.command)
        if not safe:
            raise CapabilityError(
                message=f"Blocked dangerous command: {reason}",
                capability_name=self.name,
                input_data=input_data.model_dump(),
            )

        env = os.environ.copy()
        env.update(input_data.env_vars)

        cwd = input_data.cwd
        if cwd:
            cwd = str(os.path.expanduser(cwd))

        start = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_shell(
                input_data.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            stdout_raw, stderr_raw = await asyncio.wait_for(
                proc.communicate(), timeout=input_data.timeout
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1_000
            proc.kill()  # type: ignore[union-attr]
            return CapabilityOutput(
                success=False,
                error=f"Command timed out after {input_data.timeout}s",
                data={"duration_ms": round(duration_ms, 2)},
            )

        duration_ms = (time.perf_counter() - start) * 1_000
        stdout = _truncate(stdout_raw.decode("utf-8", errors="replace"))
        stderr = _truncate(stderr_raw.decode("utf-8", errors="replace"))

        return CapabilityOutput(
            success=(proc.returncode == 0),
            data={
                "stdout": stdout,
                "stderr": stderr,
                "return_code": proc.returncode,
                "duration_ms": round(duration_ms, 2),
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# list_processes
# ═══════════════════════════════════════════════════════════════════════


class ListProcessesInput(CapabilityInput):
    """Input schema for the ``list_processes`` capability.

    Attributes:
        filter_name: Optional substring to filter process names.
    """

    filter_name: Optional[str] = Field(
        default=None, description="Substring filter for process names."
    )


class ListProcessesCapability(Capability):
    """List running processes, optionally filtered by name.

    Example input::

        {"filter_name": "python"}

    Example output::

        {
            "processes": [
                {"pid": 1234, "name": "python3", "cpu_percent": 1.2,
                 "memory_mb": 45.3, "status": "running"}
            ]
        }
    """

    name = "list_processes"
    description = "List running system processes with CPU and memory usage."
    input_schema = ListProcessesInput
    tags = ["system", "process", "read"]
    requires_confirmation = False

    async def execute(
        self, input_data: ListProcessesInput
    ) -> CapabilityOutput:
        """List processes.

        Args:
            input_data: Validated ``ListProcessesInput``.

        Returns:
            CapabilityOutput with a list of process info dicts.
        """
        procs: list[dict[str, Any]] = []
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_info", "status"]
        ):
            try:
                info = proc.info  # type: ignore[attr-defined]
                name = info.get("name", "")
                if (
                    input_data.filter_name
                    and input_data.filter_name.lower() not in name.lower()
                ):
                    continue

                mem_info = info.get("memory_info")
                mem_mb = round(mem_info.rss / (1024 * 1024), 1) if mem_info else 0

                procs.append(
                    {
                        "pid": info["pid"],
                        "name": name,
                        "cpu_percent": info.get("cpu_percent", 0.0),
                        "memory_mb": mem_mb,
                        "status": info.get("status", "unknown"),
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return CapabilityOutput(
            success=True,
            data={"processes": procs},
        )


# ═══════════════════════════════════════════════════════════════════════
# get_process
# ═══════════════════════════════════════════════════════════════════════


class GetProcessInput(CapabilityInput):
    """Input schema for the ``get_process`` capability.

    Attributes:
        pid: The process ID to inspect.
    """

    pid: int = Field(description="Process ID to inspect.")


class GetProcessCapability(Capability):
    """Get detailed information about a specific process.

    Example input::

        {"pid": 1234}

    Example output::

        {
            "pid": 1234,
            "name": "python3",
            "status": "running",
            "cpu_percent": 2.1,
            "memory_mb": 125.4,
            "cmdline": "python3 main.py",
            "created_at": "2025-06-01T10:00:00"
        }
    """

    name = "get_process"
    description = "Get detailed information about a running process by PID."
    input_schema = GetProcessInput
    tags = ["system", "process", "read"]
    requires_confirmation = False

    async def execute(self, input_data: GetProcessInput) -> CapabilityOutput:
        """Inspect a process.

        Args:
            input_data: Validated ``GetProcessInput``.

        Returns:
            CapabilityOutput with process details.
        """
        try:
            proc = psutil.Process(input_data.pid)
            mem = proc.memory_info()
            return CapabilityOutput(
                success=True,
                data={
                    "pid": proc.pid,
                    "name": proc.name(),
                    "status": proc.status(),
                    "cpu_percent": proc.cpu_percent(interval=0.1),
                    "memory_mb": round(mem.rss / (1024 * 1024), 1),
                    "cmdline": " ".join(proc.cmdline()),
                    "created_at": datetime.datetime.fromtimestamp(
                        proc.create_time()
                    ).isoformat(),
                },
            )
        except psutil.NoSuchProcess:
            return CapabilityOutput(
                success=False,
                error=f"Process {input_data.pid} not found.",
            )
        except psutil.AccessDenied:
            return CapabilityOutput(
                success=False,
                error=f"Access denied to process {input_data.pid}.",
            )


# ═══════════════════════════════════════════════════════════════════════
# kill_process
# ═══════════════════════════════════════════════════════════════════════


class KillProcessInput(CapabilityInput):
    """Input schema for the ``kill_process`` capability.

    Attributes:
        pid: The process ID to kill.
        confirm: Must be ``True`` to actually kill.
        signal: Signal name — ``"TERM"`` (graceful) or ``"KILL"`` (force).
    """

    pid: int = Field(description="Process ID to kill.")
    confirm: bool = Field(
        default=False, description="Must be True to confirm."
    )
    signal: str = Field(
        default="TERM",
        description="Signal: TERM (graceful) or KILL (force).",
    )


class KillProcessCapability(Capability):
    """Terminate a running process by PID.

    Refuses to kill PID 1, PID 0, or the current ARIA process.

    Example input::

        {"pid": 9999, "confirm": true, "signal": "TERM"}

    Example output::

        {"pid": 9999, "killed": true}
    """

    name = "kill_process"
    description = "Kill a running process by PID (requires confirm=True)."
    input_schema = KillProcessInput
    tags = ["system", "process", "execute"]
    requires_confirmation = True

    async def execute(self, input_data: KillProcessInput) -> CapabilityOutput:
        """Kill a process.

        Args:
            input_data: Validated ``KillProcessInput``.

        Returns:
            CapabilityOutput with pid and killed flag.
        """
        if not input_data.confirm:
            return CapabilityOutput(
                success=False,
                error="Kill refused: confirm must be True.",
            )

        if input_data.pid in (0, 1):
            return CapabilityOutput(
                success=False,
                error=f"Refused: cannot kill PID {input_data.pid}.",
            )
        if input_data.pid == os.getpid():
            return CapabilityOutput(
                success=False,
                error="Refused: cannot kill the current ARIA process.",
            )

        try:
            proc = psutil.Process(input_data.pid)
            sig_upper = input_data.signal.upper()
            if sig_upper == "KILL":
                proc.kill()
            else:
                proc.terminate()
            return CapabilityOutput(
                success=True,
                data={"pid": input_data.pid, "killed": True},
            )
        except psutil.NoSuchProcess:
            return CapabilityOutput(
                success=False,
                error=f"Process {input_data.pid} not found.",
            )
        except psutil.AccessDenied:
            return CapabilityOutput(
                success=False,
                error=f"Access denied to process {input_data.pid}.",
            )


# ═══════════════════════════════════════════════════════════════════════
# Capability instances for registration
# ═══════════════════════════════════════════════════════════════════════

run_command = RunCommandCapability()
list_processes = ListProcessesCapability()
get_process = GetProcessCapability()
kill_process = KillProcessCapability()

ALL_PROCESS_CAPABILITIES = [
    run_command,
    list_processes,
    get_process,
    kill_process,
]
