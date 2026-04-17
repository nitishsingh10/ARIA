"""
aria.capabilities — Capability layer for ARIA.

Imports, instantiates, and registers all capabilities into the
``GLOBAL_REGISTRY``.  This is the single entry point for discovering
and executing ARIA's deterministic action library.

Usage:
    >>> from aria.capabilities import GLOBAL_REGISTRY
    >>> caps = GLOBAL_REGISTRY.list_all()
    >>> result = await GLOBAL_REGISTRY.execute("read_file", {"path": "/tmp/x.txt"})
"""

from __future__ import annotations

# -- Base contracts --
from aria.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityOutput,
)
from aria.capabilities.registry import GLOBAL_REGISTRY, CapabilityRegistry

# -- System capabilities --
from aria.capabilities.system.file_ops import (
    ALL_FILE_CAPABILITIES,
    AppendFileCapability,
    CopyFileCapability,
    DeleteFileCapability,
    FileInfoCapability,
    MoveFileCapability,
    ReadFileCapability,
    SearchFilesCapability,
    WriteFileCapability,
)
from aria.capabilities.system.folder_ops import (
    ALL_FOLDER_CAPABILITIES,
    CreateDirectoryCapability,
    DeleteDirectoryCapability,
    FolderTreeCapability,
    ListDirectoryCapability,
)
from aria.capabilities.system.process_ops import (
    ALL_PROCESS_CAPABILITIES,
    GetProcessCapability,
    KillProcessCapability,
    ListProcessesCapability,
    RunCommandCapability,
)

# -- Execution capabilities --
from aria.capabilities.execution.python_runner import (
    ALL_PYTHON_CAPABILITIES,
    RunPythonCapability,
)
from aria.capabilities.execution.docker_runner import (
    ALL_DOCKER_CAPABILITIES,
    RunInDockerCapability,
)

# -- External capabilities --
from aria.capabilities.external.web_fetch import (
    ALL_WEB_CAPABILITIES,
    FetchUrlCapability,
)
from aria.capabilities.external.system_info import (
    ALL_SYSTEM_INFO_CAPABILITIES,
    GetCurrentTimeCapability,
    GetEnvVarCapability,
    GetSystemInfoCapability,
)

from aria.core.logger import get_logger

_log = get_logger("capabilities")

# ---------------------------------------------------------------------------
# Register everything
# ---------------------------------------------------------------------------

_ALL_CAPABILITIES = (
    ALL_FILE_CAPABILITIES
    + ALL_FOLDER_CAPABILITIES
    + ALL_PROCESS_CAPABILITIES
    + ALL_PYTHON_CAPABILITIES
    + ALL_DOCKER_CAPABILITIES
    + ALL_WEB_CAPABILITIES
    + ALL_SYSTEM_INFO_CAPABILITIES
)

for _cap in _ALL_CAPABILITIES:
    GLOBAL_REGISTRY.register(_cap)

_log.info(
    f"Capability layer initialised: {len(GLOBAL_REGISTRY)} capabilities registered",
    data={"count": len(GLOBAL_REGISTRY)},
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Core
    "GLOBAL_REGISTRY",
    "CapabilityRegistry",
    "Capability",
    "CapabilityInput",
    "CapabilityOutput",
    "CapabilityError",
    # System
    "ReadFileCapability",
    "WriteFileCapability",
    "AppendFileCapability",
    "DeleteFileCapability",
    "CopyFileCapability",
    "MoveFileCapability",
    "SearchFilesCapability",
    "FileInfoCapability",
    "ListDirectoryCapability",
    "CreateDirectoryCapability",
    "DeleteDirectoryCapability",
    "FolderTreeCapability",
    "RunCommandCapability",
    "ListProcessesCapability",
    "GetProcessCapability",
    "KillProcessCapability",
    # Execution
    "RunPythonCapability",
    "RunInDockerCapability",
    # External
    "FetchUrlCapability",
    "GetSystemInfoCapability",
    "GetCurrentTimeCapability",
    "GetEnvVarCapability",
]
