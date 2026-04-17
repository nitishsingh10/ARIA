"""
aria/capabilities/system/folder_ops.py — Folder / directory capabilities.

Provides listing, creation, deletion, and tree-view of directories.
All operations log their actions and respect the safety contract.

Safety:
    - ``delete_directory`` with ``recursive=True`` refuses to operate on
      paths above the user's home directory.
    - Destructive operations require ``confirm=True``.

Example:
    >>> from aria.capabilities.registry import GLOBAL_REGISTRY
    >>> result = await GLOBAL_REGISTRY.execute(
    ...     "list_directory", {"path": "~/projects"}
    ... )
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Optional

from pydantic import Field

from aria.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityOutput,
)
from aria.core.logger import get_logger

log = get_logger("capabilities.folder_ops")


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _resolve_path(raw: str) -> Path:
    """Expand ~ and resolve a path to an absolute form.

    Args:
        raw: A user-provided path string.

    Returns:
        The fully resolved ``Path``.
    """
    return Path(raw).expanduser().resolve()


def _is_within_home(path: Path) -> bool:
    """Check whether *path* is within the current user's home directory.

    Args:
        path: A resolved absolute path.

    Returns:
        True if the path is inside ``Path.home()``.
    """
    try:
        path.relative_to(Path.home())
        return True
    except ValueError:
        return False


# ═══════════════════════════════════════════════════════════════════════
# list_directory
# ═══════════════════════════════════════════════════════════════════════


class ListDirectoryInput(CapabilityInput):
    """Input schema for the ``list_directory`` capability.

    Attributes:
        path: Directory path to list.
        show_hidden: Include dotfiles / hidden entries.
        include_sizes: Report file sizes and modification times.
    """

    path: str = Field(description="Directory to list.")
    show_hidden: bool = Field(
        default=False, description="Include hidden files."
    )
    include_sizes: bool = Field(
        default=True, description="Include size and modification time."
    )


class ListDirectoryCapability(Capability):
    """List the immediate contents of a directory.

    Returns items with name, type, size, and modification time.

    Example input::

        {"path": "~/projects", "show_hidden": false}

    Example output::

        {
            "items": [
                {"name": "main.py", "type": "file",
                 "size_bytes": 1024, "modified_at": "..."},
                {"name": "src", "type": "directory",
                 "size_bytes": null, "modified_at": "..."}
            ],
            "total_items": 2
        }
    """

    name = "list_directory"
    description = "List the contents of a directory with file metadata."
    input_schema = ListDirectoryInput
    tags = ["system", "folder", "read"]
    requires_confirmation = False

    async def execute(self, input_data: ListDirectoryInput) -> CapabilityOutput:
        """List directory contents.

        Args:
            input_data: Validated ``ListDirectoryInput``.

        Returns:
            CapabilityOutput with items list and total_items count.
        """
        root = _resolve_path(input_data.path)
        if not root.is_dir():
            return CapabilityOutput(
                success=False, error=f"Not a directory: {root}"
            )

        items = []
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not input_data.show_hidden and entry.name.startswith("."):
                continue

            item: dict = {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
            }

            if input_data.include_sizes:
                try:
                    st = entry.stat()
                    item["size_bytes"] = st.st_size if entry.is_file() else None
                    item["modified_at"] = datetime.datetime.fromtimestamp(
                        st.st_mtime
                    ).isoformat()
                except OSError:
                    item["size_bytes"] = None
                    item["modified_at"] = None

            items.append(item)

        return CapabilityOutput(
            success=True,
            data={"items": items, "total_items": len(items)},
        )


# ═══════════════════════════════════════════════════════════════════════
# create_directory
# ═══════════════════════════════════════════════════════════════════════


class CreateDirectoryInput(CapabilityInput):
    """Input schema for the ``create_directory`` capability.

    Attributes:
        path: Directory path to create.
        parents: Create intermediate parent directories.
    """

    path: str = Field(description="Directory path to create.")
    parents: bool = Field(
        default=True, description="Create parent directories as needed."
    )


class CreateDirectoryCapability(Capability):
    """Create a new directory (including parents if requested).

    Example input::

        {"path": "~/projects/new_project/src"}

    Example output::

        {"path": "/home/user/projects/new_project/src", "created": true}
    """

    name = "create_directory"
    description = "Create a directory, optionally including parent directories."
    input_schema = CreateDirectoryInput
    tags = ["system", "folder", "write"]
    requires_confirmation = False

    async def execute(
        self, input_data: CreateDirectoryInput
    ) -> CapabilityOutput:
        """Create a directory.

        Args:
            input_data: Validated ``CreateDirectoryInput``.

        Returns:
            CapabilityOutput with path and created flag.
        """
        path = _resolve_path(input_data.path)

        if path.exists():
            return CapabilityOutput(
                success=True,
                data={"path": str(path), "created": False},
            )

        path.mkdir(parents=input_data.parents, exist_ok=True)
        return CapabilityOutput(
            success=True,
            data={"path": str(path), "created": True},
        )


# ═══════════════════════════════════════════════════════════════════════
# delete_directory
# ═══════════════════════════════════════════════════════════════════════


class DeleteDirectoryInput(CapabilityInput):
    """Input schema for the ``delete_directory`` capability.

    Attributes:
        path: Directory to delete.
        recursive: Remove contents recursively.
        confirm: Must be ``True`` to actually delete.
    """

    path: str = Field(description="Directory to delete.")
    recursive: bool = Field(
        default=False, description="Delete contents recursively."
    )
    confirm: bool = Field(
        default=False, description="Must be True to confirm deletion."
    )


class DeleteDirectoryCapability(Capability):
    """Delete a directory from the filesystem.

    Refuses to operate unless ``confirm=True``.  When
    ``recursive=True``, also refuses if the path is above
    the user's home directory.

    Example input::

        {"path": "/tmp/scratch_dir", "recursive": true, "confirm": true}

    Example output::

        {"path": "/tmp/scratch_dir", "deleted": true}
    """

    name = "delete_directory"
    description = "Delete a directory (requires confirm=True)."
    input_schema = DeleteDirectoryInput
    tags = ["system", "folder", "delete"]
    requires_confirmation = True

    async def execute(
        self, input_data: DeleteDirectoryInput
    ) -> CapabilityOutput:
        """Delete a directory.

        Args:
            input_data: Validated ``DeleteDirectoryInput``.

        Returns:
            CapabilityOutput with path and deleted flag.
        """
        if not input_data.confirm:
            return CapabilityOutput(
                success=False,
                error="Deletion refused: confirm must be True.",
            )

        path = _resolve_path(input_data.path)

        if not path.exists():
            return CapabilityOutput(
                success=False, error=f"Directory not found: {path}"
            )
        if not path.is_dir():
            return CapabilityOutput(
                success=False,
                error=f"Not a directory (use delete_file): {path}",
            )

        # Safety: refuse recursive delete above home dir
        if input_data.recursive and not _is_within_home(path):
            return CapabilityOutput(
                success=False,
                error=(
                    "Refused: recursive delete of paths outside user home "
                    f"directory is not allowed. Path: {path}"
                ),
            )

        if input_data.recursive:
            shutil.rmtree(str(path))
        else:
            try:
                path.rmdir()  # only works on empty dirs
            except OSError as exc:
                return CapabilityOutput(
                    success=False,
                    error=(
                        f"Directory not empty. Use recursive=True to force. "
                        f"Error: {exc}"
                    ),
                )

        return CapabilityOutput(
            success=True,
            data={"path": str(path), "deleted": True},
        )


# ═══════════════════════════════════════════════════════════════════════
# folder_tree
# ═══════════════════════════════════════════════════════════════════════


class FolderTreeInput(CapabilityInput):
    """Input schema for the ``folder_tree`` capability.

    Attributes:
        path: Root directory to render.
        max_depth: Maximum recursion depth.
        show_hidden: Include hidden entries.
    """

    path: str = Field(description="Root directory for the tree.")
    max_depth: int = Field(default=3, description="Maximum depth to render.")
    show_hidden: bool = Field(
        default=False, description="Include hidden files."
    )


class FolderTreeCapability(Capability):
    """Render an ASCII tree view of a directory structure.

    Example input::

        {"path": "~/projects/aria", "max_depth": 2}

    Example output::

        {"tree": "aria/\\n├── main.py\\n├── config.py\\n└── llm/\\n    ├── ..."}
    """

    name = "folder_tree"
    description = "Render an ASCII tree of a directory structure."
    input_schema = FolderTreeInput
    tags = ["system", "folder", "read"]
    requires_confirmation = False

    async def execute(self, input_data: FolderTreeInput) -> CapabilityOutput:
        """Render a directory tree.

        Args:
            input_data: Validated ``FolderTreeInput``.

        Returns:
            CapabilityOutput with the formatted tree string.
        """
        root = _resolve_path(input_data.path)
        if not root.is_dir():
            return CapabilityOutput(
                success=False, error=f"Not a directory: {root}"
            )

        lines: list[str] = [root.name + "/"]
        self._walk(root, "", input_data.max_depth, input_data.show_hidden, lines)

        return CapabilityOutput(
            success=True,
            data={"tree": "\n".join(lines)},
        )

    @staticmethod
    def _walk(
        directory: Path,
        prefix: str,
        depth: int,
        show_hidden: bool,
        lines: list[str],
    ) -> None:
        """Recursively build the tree lines.

        Args:
            directory: Current directory.
            prefix: Indent prefix for this level.
            depth: Remaining depth levels.
            show_hidden: Whether to include dotfiles.
            lines: Accumulator list of output lines.
        """
        if depth <= 0:
            return

        entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")

            if entry.is_dir():
                child_prefix = prefix + ("    " if is_last else "│   ")
                FolderTreeCapability._walk(
                    entry, child_prefix, depth - 1, show_hidden, lines
                )


# ═══════════════════════════════════════════════════════════════════════
# Capability instances for registration
# ═══════════════════════════════════════════════════════════════════════

list_directory = ListDirectoryCapability()
create_directory = CreateDirectoryCapability()
delete_directory = DeleteDirectoryCapability()
folder_tree = FolderTreeCapability()

ALL_FOLDER_CAPABILITIES = [
    list_directory,
    create_directory,
    delete_directory,
    folder_tree,
]
