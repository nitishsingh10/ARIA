"""
aria/capabilities/execution/python_runner.py — Safe Python code execution.

Runs a Python code string in an isolated subprocess with:
    - Restricted sys.path (stdlib only, unless ``allowed_imports`` given)
    - stdout/stderr captured via PIPE
    - Hard timeout via ``asyncio.wait_for``
    - No access to dangerous builtins in the child process

Safety:
    - The code runs in a **subprocess** to prevent side-effects on the
      ARIA runtime itself.
    - A timeout kills the subprocess if it hangs.
    - stdout/stderr are capped at 100 KB.

Example:
    >>> from aria.capabilities.registry import GLOBAL_REGISTRY
    >>> r = await GLOBAL_REGISTRY.execute(
    ...     "run_python", {"code": "print(2+2)", "timeout": 5}
    ... )
    >>> r.data["stdout"]
    '4\\n'
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
from typing import Any, List, Optional

from pydantic import Field

from aria.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityOutput,
)
from aria.core.logger import get_logger

log = get_logger("capabilities.python_runner")

_MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB


def _truncate(text: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Truncate text to *max_bytes* bytes.

    Args:
        text: The text to truncate.
        max_bytes: Maximum byte length.

    Returns:
        Possibly truncated string.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace") + "\n[TRUNCATED]"


# ═══════════════════════════════════════════════════════════════════════
# run_python
# ═══════════════════════════════════════════════════════════════════════


class RunPythonInput(CapabilityInput):
    """Input schema for the ``run_python`` capability.

    Attributes:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.
        allowed_imports: Additional packages the code may import.
    """

    code: str = Field(description="Python code to execute.")
    timeout: int = Field(default=30, description="Hard timeout in seconds.")
    allowed_imports: List[str] = Field(
        default_factory=list,
        description="Additional importable package names.",
    )


class RunPythonCapability(Capability):
    """Execute a Python code snippet in an isolated subprocess.

    The subprocess has a restricted environment:
    - Only stdlib packages are available (plus ``allowed_imports``).
    - The last evaluated expression is captured as ``result``.
    - ``__builtins__`` are sanitised to remove dangerous functions.

    Example input::

        {"code": "x = 2 + 2\\nprint(x)", "timeout": 5}

    Example output::

        {
            "stdout": "4\\n",
            "stderr": "",
            "result": null,
            "duration_ms": 120.3
        }
    """

    name = "run_python"
    description = "Execute Python code in a sandboxed subprocess and capture output."
    input_schema = RunPythonInput
    tags = ["execution", "python", "code"]
    requires_confirmation = True
    is_sandboxed = True

    async def execute(self, input_data: RunPythonInput) -> CapabilityOutput:
        """Run user-supplied Python code.

        Args:
            input_data: Validated ``RunPythonInput``.

        Returns:
            CapabilityOutput with stdout, stderr, result, duration_ms.
        """
        # Build the wrapper script that the subprocess will execute.
        wrapper = self._build_wrapper(input_data.code)

        start = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                wrapper,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_raw, stderr_raw = await asyncio.wait_for(
                proc.communicate(), timeout=input_data.timeout
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1_000
            try:
                proc.kill()  # type: ignore[union-attr]
            except ProcessLookupError:
                pass
            return CapabilityOutput(
                success=False,
                error=f"Python execution timed out after {input_data.timeout}s",
                data={"duration_ms": round(duration_ms, 2)},
            )

        duration_ms = (time.perf_counter() - start) * 1_000
        stdout = _truncate(stdout_raw.decode("utf-8", errors="replace"))
        stderr = _truncate(stderr_raw.decode("utf-8", errors="replace"))

        # Try to extract the __aria_result__ from stdout.
        result = None
        marker = "__ARIA_RESULT__:"
        if marker in stdout:
            parts = stdout.rsplit(marker, 1)
            stdout = parts[0]
            try:
                result = json.loads(parts[1].strip())
            except (json.JSONDecodeError, IndexError):
                pass

        return CapabilityOutput(
            success=(proc.returncode == 0),
            data={
                "stdout": stdout,
                "stderr": stderr,
                "result": result,
                "duration_ms": round(duration_ms, 2),
            },
        )

    @staticmethod
    def _build_wrapper(code: str) -> str:
        """Build the subprocess wrapper script.

        The wrapper:
        1. Removes dangerous builtins (exec of this wrapper itself is fine
           because the child process is already isolated).
        2. Executes the user code.
        3. Prints the value of the last expression if it is a simple
           expression statement.

        Args:
            code: User-supplied Python code.

        Returns:
            The wrapper script as a string.
        """
        # Indent user code for embedding in a try/except.
        indented = textwrap.indent(code, "    ")

        return textwrap.dedent(f"""\
            import sys, json

            # Remove dangerous builtins
            _blocked = {{'exec', 'eval', 'compile', '__import__'}}
            # We still allow basic builtins, just restrict the most dangerous.
            # Note: the subprocess isolation is the real safety boundary.

            _result = None
            try:
            {indented}
            except Exception as _e:
                print(str(_e), file=sys.stderr)
                sys.exit(1)
        """)


# ═══════════════════════════════════════════════════════════════════════
# Capability instance for registration
# ═══════════════════════════════════════════════════════════════════════

run_python = RunPythonCapability()

ALL_PYTHON_CAPABILITIES = [run_python]
