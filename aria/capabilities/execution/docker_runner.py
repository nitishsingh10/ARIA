"""
aria/capabilities/execution/docker_runner.py — Docker-sandboxed code execution.

Runs arbitrary code inside a Docker container with:
    - Network isolation (``--network=none``)
    - Memory cap (256 MB) and CPU limit (0.5 cores)
    - Hard timeout
    - Automatic container cleanup

Supported languages:
    - python  → ``python:3.11-slim``
    - bash    → ``ubuntu:22.04``
    - node    → ``node:20-slim``

Example:
    >>> from aria.capabilities.registry import GLOBAL_REGISTRY
    >>> r = await GLOBAL_REGISTRY.execute("run_in_docker", {
    ...     "code": "print('hello from docker')",
    ...     "language": "python",
    ...     "timeout": 30,
    ... })
    >>> r.data["stdout"]
    'hello from docker\\n'
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

from pydantic import Field

from aria.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityOutput,
)
from aria.core.logger import get_logger

log = get_logger("capabilities.docker_runner")

_MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB

# Language → (default image, filename, run command template)
_LANGUAGE_MAP: dict[str, tuple[str, str, str]] = {
    "python": ("python:3.11-slim", "main.py", "python /workspace/{filename}"),
    "bash": ("ubuntu:22.04", "main.sh", "bash /workspace/{filename}"),
    "node": ("node:20-slim", "main.js", "node /workspace/{filename}"),
}


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


def _docker_available() -> bool:
    """Check whether the ``docker`` CLI is available on PATH.

    Returns:
        True if Docker is found.
    """
    return shutil.which("docker") is not None


# ═══════════════════════════════════════════════════════════════════════
# run_in_docker
# ═══════════════════════════════════════════════════════════════════════


class RunInDockerInput(CapabilityInput):
    """Input schema for the ``run_in_docker`` capability.

    Attributes:
        code: Source code to execute.
        language: Language key (``python``, ``bash``, ``node``).
        image: Override the Docker image (defaults per language).
        timeout: Max execution time in seconds.
        files: Extra files to mount (``{filename: content}``).
    """

    code: str = Field(description="Source code to execute.")
    language: str = Field(
        default="python", description="Language: python, bash, or node."
    )
    image: Optional[str] = Field(
        default=None,
        description="Override Docker image (defaults per language).",
    )
    timeout: int = Field(default=60, description="Timeout in seconds.")
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional files: {filename: content}.",
    )


class RunInDockerCapability(Capability):
    """Execute code inside a Docker container.

    The container runs with network disabled, memory limited to 256 MB,
    and CPU limited to 0.5 cores.  A temporary directory is mounted
    at ``/workspace`` and cleaned up afterwards.

    Any files written to ``/workspace/output/`` in the container are
    returned in ``output_files``.

    Example input::

        {"code": "print('hi')", "language": "python", "timeout": 30}

    Example output::

        {
            "stdout": "hi\\n",
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 1500.0,
            "output_files": {}
        }
    """

    name = "run_in_docker"
    description = (
        "Run code inside a Docker container with network isolation "
        "and resource limits."
    )
    input_schema = RunInDockerInput
    tags = ["execution", "docker", "sandbox", "code"]
    requires_confirmation = True
    is_sandboxed = True

    async def execute(self, input_data: RunInDockerInput) -> CapabilityOutput:
        """Run code in a Docker container.

        Args:
            input_data: Validated ``RunInDockerInput``.

        Returns:
            CapabilityOutput with stdout, stderr, exit_code, duration_ms,
            and output_files.

        Raises:
            CapabilityError: If Docker is not available or the language
                is unsupported.
        """
        if not _docker_available():
            raise CapabilityError(
                message=(
                    "Docker is not available. Install Docker and ensure "
                    "the 'docker' CLI is on PATH."
                ),
                capability_name=self.name,
            )

        lang = input_data.language.lower()
        if lang not in _LANGUAGE_MAP:
            supported = ", ".join(sorted(_LANGUAGE_MAP))
            raise CapabilityError(
                message=(
                    f"Unsupported language '{lang}'. "
                    f"Supported: {supported}"
                ),
                capability_name=self.name,
            )

        default_image, filename, run_template = _LANGUAGE_MAP[lang]
        image = input_data.image or default_image
        run_cmd = run_template.format(filename=filename)

        # Create temp workspace
        tmpdir = tempfile.mkdtemp(prefix="aria_docker_")
        try:
            # Write the main code file
            main_path = os.path.join(tmpdir, filename)
            with open(main_path, "w", encoding="utf-8") as f:
                f.write(input_data.code)

            # Write extra files
            for fname, content in input_data.files.items():
                fpath = os.path.join(tmpdir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)

            # Create output directory
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Build docker command
            docker_cmd = [
                "docker", "run", "--rm",
                "--network=none",
                "--memory=256m",
                "--cpus=0.5",
                "-v", f"{tmpdir}:/workspace:rw",
                "-w", "/workspace",
                image,
                "sh", "-c", run_cmd,
            ]

            start = time.perf_counter()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd,
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
                    error=f"Docker execution timed out after {input_data.timeout}s",
                    data={"duration_ms": round(duration_ms, 2)},
                )

            duration_ms = (time.perf_counter() - start) * 1_000

            stdout = _truncate(
                stdout_raw.decode("utf-8", errors="replace")
            )
            stderr = _truncate(
                stderr_raw.decode("utf-8", errors="replace")
            )

            # Collect output files
            output_files: dict[str, str] = {}
            if os.path.isdir(output_dir):
                for fname in os.listdir(output_dir):
                    fpath = os.path.join(output_dir, fname)
                    if os.path.isfile(fpath):
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                output_files[fname] = f.read()
                        except UnicodeDecodeError:
                            output_files[fname] = "<binary file>"

            return CapabilityOutput(
                success=(proc.returncode == 0),
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": proc.returncode,
                    "duration_ms": round(duration_ms, 2),
                    "output_files": output_files,
                },
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# Capability instance for registration
# ═══════════════════════════════════════════════════════════════════════

run_in_docker = RunInDockerCapability()

ALL_DOCKER_CAPABILITIES = [run_in_docker]
