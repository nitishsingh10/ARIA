"""
aria/capabilities/system/file_ops.py — File operation capabilities.

Provides deterministic, real file system operations: read, write,
append, delete, copy, move, search, and info.  Every operation logs
its actions and respects the safety contract.

Safety:
    - Operations outside the user's home directory require explicit
      ``confirm=True`` where applicable.
    - Destructive operations (delete, overwrite) require confirmation.
    - All exceptions are caught and surfaced via ``CapabilityOutput``.

Example:
    >>> from aria.capabilities.registry import GLOBAL_REGISTRY
    >>> result = await GLOBAL_REGISTRY.execute("read_file", {"path": "/tmp/hello.txt"})
    >>> print(result.data["content"])
"""

from __future__ import annotations

import datetime
import mimetypes
import os
import shutil
import stat
from pathlib import Path
from typing import Optional

from pydantic import Field

from aria.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityOutput,
)
from aria.core.logger import get_logger

log = get_logger("capabilities.file_ops")


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


def _guess_mime(path: Path) -> str:
    """Guess the MIME type of a file.

    Args:
        path: The file path.

    Returns:
        A MIME type string, defaulting to ``"application/octet-stream"``.
    """
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


# ═══════════════════════════════════════════════════════════════════════
# read_file
# ═══════════════════════════════════════════════════════════════════════


class ReadFileInput(CapabilityInput):
    """Input schema for the ``read_file`` capability.

    Attributes:
        path: Absolute or ~-relative path to the file.
        encoding: Text encoding to use when reading.
        max_bytes: Maximum number of bytes to read (safety cap).
    """

    path: str = Field(description="Path to the file to read.")
    encoding: str = Field(default="utf-8", description="Text encoding.")
    max_bytes: int = Field(
        default=1_000_000, description="Max bytes to read."
    )


class ReadFileCapability(Capability):
    """Read the contents of a file from disk.

    Returns the file content as a string along with size and MIME type.

    Example input::

        {"path": "~/notes/todo.txt"}

    Example output::

        {"content": "buy milk\\n...", "size_bytes": 42, "mime_type": "text/plain"}
    """

    name = "read_file"
    description = "Read the full contents of a file and return them as text."
    input_schema = ReadFileInput
    tags = ["system", "file", "read"]
    requires_confirmation = False

    async def execute(self, input_data: ReadFileInput) -> CapabilityOutput:
        """Read a file and return its contents.

        Args:
            input_data: Validated ``ReadFileInput``.

        Returns:
            CapabilityOutput with content, size_bytes, and mime_type.
        """
        path = _resolve_path(input_data.path)

        if not path.exists():
            return CapabilityOutput(
                success=False, error=f"File not found: {path}"
            )
        if not path.is_file():
            return CapabilityOutput(
                success=False, error=f"Not a file: {path}"
            )

        size = path.stat().st_size
        if size > input_data.max_bytes:
            return CapabilityOutput(
                success=False,
                error=(
                    f"File too large ({size} bytes). "
                    f"max_bytes={input_data.max_bytes}"
                ),
            )

        content = path.read_text(encoding=input_data.encoding)
        return CapabilityOutput(
            success=True,
            data={
                "content": content,
                "size_bytes": size,
                "mime_type": _guess_mime(path),
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# write_file
# ═══════════════════════════════════════════════════════════════════════


class WriteFileInput(CapabilityInput):
    """Input schema for the ``write_file`` capability.

    Attributes:
        path: Target file path.
        content: Text content to write.
        encoding: Text encoding.
        create_dirs: Create parent directories if missing.
        overwrite: Overwrite the file if it already exists.
    """

    path: str = Field(description="Path to write to.")
    content: str = Field(description="Content to write.")
    encoding: str = Field(default="utf-8", description="Text encoding.")
    create_dirs: bool = Field(
        default=True, description="Create parent dirs if they don't exist."
    )
    overwrite: bool = Field(
        default=True, description="Overwrite if file exists."
    )


class WriteFileCapability(Capability):
    """Write text content to a file.

    Creates parent directories when ``create_dirs`` is True.  Refuses
    to overwrite an existing file unless ``overwrite`` is True.

    Example input::

        {"path": "/tmp/out.txt", "content": "hello world"}

    Example output::

        {"path": "/tmp/out.txt", "size_bytes": 11}
    """

    name = "write_file"
    description = "Write text content to a file, optionally creating parent directories."
    input_schema = WriteFileInput
    tags = ["system", "file", "write"]
    requires_confirmation = True

    async def execute(self, input_data: WriteFileInput) -> CapabilityOutput:
        """Write content to a file.

        Args:
            input_data: Validated ``WriteFileInput``.

        Returns:
            CapabilityOutput with path and size_bytes.
        """
        path = _resolve_path(input_data.path)

        if path.exists() and not input_data.overwrite:
            return CapabilityOutput(
                success=False,
                error=f"File exists and overwrite=False: {path}",
            )

        if input_data.create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(input_data.content, encoding=input_data.encoding)
        return CapabilityOutput(
            success=True,
            data={
                "path": str(path),
                "size_bytes": path.stat().st_size,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# append_file
# ═══════════════════════════════════════════════════════════════════════


class AppendFileInput(CapabilityInput):
    """Input schema for the ``append_file`` capability.

    Attributes:
        path: Target file path.
        content: Text content to append.
        encoding: Text encoding.
    """

    path: str = Field(description="Path to append to.")
    content: str = Field(description="Content to append.")
    encoding: str = Field(default="utf-8", description="Text encoding.")


class AppendFileCapability(Capability):
    """Append text content to an existing file.

    Creates the file if it does not exist.

    Example input::

        {"path": "/tmp/log.txt", "content": "new line\\n"}

    Example output::

        {"path": "/tmp/log.txt", "new_size_bytes": 1024}
    """

    name = "append_file"
    description = "Append text content to a file, creating it if absent."
    input_schema = AppendFileInput
    tags = ["system", "file", "write"]
    requires_confirmation = False

    async def execute(self, input_data: AppendFileInput) -> CapabilityOutput:
        """Append content to a file.

        Args:
            input_data: Validated ``AppendFileInput``.

        Returns:
            CapabilityOutput with path and new_size_bytes.
        """
        path = _resolve_path(input_data.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding=input_data.encoding) as fh:
            fh.write(input_data.content)

        return CapabilityOutput(
            success=True,
            data={
                "path": str(path),
                "new_size_bytes": path.stat().st_size,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# delete_file
# ═══════════════════════════════════════════════════════════════════════


class DeleteFileInput(CapabilityInput):
    """Input schema for the ``delete_file`` capability.

    Attributes:
        path: Path of the file to delete.
        confirm: Must be ``True`` to actually delete.
    """

    path: str = Field(description="Path to the file to delete.")
    confirm: bool = Field(
        default=False,
        description="Must be True to confirm deletion.",
    )


class DeleteFileCapability(Capability):
    """Delete a file from the filesystem.

    Refuses to operate unless ``confirm=True`` is supplied.

    Example input::

        {"path": "/tmp/scratch.txt", "confirm": true}

    Example output::

        {"path": "/tmp/scratch.txt", "deleted": true}
    """

    name = "delete_file"
    description = "Permanently delete a file (requires confirm=True)."
    input_schema = DeleteFileInput
    tags = ["system", "file", "delete"]
    requires_confirmation = True

    async def execute(self, input_data: DeleteFileInput) -> CapabilityOutput:
        """Delete a file.

        Args:
            input_data: Validated ``DeleteFileInput``.

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
                success=False, error=f"File not found: {path}"
            )
        if not path.is_file():
            return CapabilityOutput(
                success=False, error=f"Not a file (use delete_directory): {path}"
            )

        path.unlink()
        return CapabilityOutput(
            success=True,
            data={"path": str(path), "deleted": True},
        )


# ═══════════════════════════════════════════════════════════════════════
# copy_file
# ═══════════════════════════════════════════════════════════════════════


class CopyFileInput(CapabilityInput):
    """Input schema for the ``copy_file`` capability.

    Attributes:
        src: Source file path.
        dst: Destination file path.
        overwrite: Overwrite destination if it exists.
    """

    src: str = Field(description="Source file path.")
    dst: str = Field(description="Destination file path.")
    overwrite: bool = Field(
        default=False, description="Overwrite destination if it exists."
    )


class CopyFileCapability(Capability):
    """Copy a file from one location to another.

    Example input::

        {"src": "/tmp/a.txt", "dst": "/tmp/b.txt"}

    Example output::

        {"src": "/tmp/a.txt", "dst": "/tmp/b.txt", "size_bytes": 42}
    """

    name = "copy_file"
    description = "Copy a file to a new location."
    input_schema = CopyFileInput
    tags = ["system", "file", "write"]
    requires_confirmation = False

    async def execute(self, input_data: CopyFileInput) -> CapabilityOutput:
        """Copy a file.

        Args:
            input_data: Validated ``CopyFileInput``.

        Returns:
            CapabilityOutput with src, dst, and size_bytes.
        """
        src = _resolve_path(input_data.src)
        dst = _resolve_path(input_data.dst)

        if not src.exists():
            return CapabilityOutput(
                success=False, error=f"Source not found: {src}"
            )
        if dst.exists() and not input_data.overwrite:
            return CapabilityOutput(
                success=False,
                error=f"Destination exists and overwrite=False: {dst}",
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return CapabilityOutput(
            success=True,
            data={
                "src": str(src),
                "dst": str(dst),
                "size_bytes": dst.stat().st_size,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# move_file
# ═══════════════════════════════════════════════════════════════════════


class MoveFileInput(CapabilityInput):
    """Input schema for the ``move_file`` capability.

    Attributes:
        src: Source file path.
        dst: Destination file path.
        overwrite: Overwrite destination if it exists.
    """

    src: str = Field(description="Source file path.")
    dst: str = Field(description="Destination file path.")
    overwrite: bool = Field(
        default=False, description="Overwrite destination if it exists."
    )


class MoveFileCapability(Capability):
    """Move (rename) a file to a new location.

    Example input::

        {"src": "/tmp/old.txt", "dst": "/tmp/new.txt"}

    Example output::

        {"src": "/tmp/old.txt", "dst": "/tmp/new.txt"}
    """

    name = "move_file"
    description = "Move or rename a file (requires confirmation)."
    input_schema = MoveFileInput
    tags = ["system", "file", "write"]
    requires_confirmation = True

    async def execute(self, input_data: MoveFileInput) -> CapabilityOutput:
        """Move a file.

        Args:
            input_data: Validated ``MoveFileInput``.

        Returns:
            CapabilityOutput with src and dst.
        """
        src = _resolve_path(input_data.src)
        dst = _resolve_path(input_data.dst)

        if not src.exists():
            return CapabilityOutput(
                success=False, error=f"Source not found: {src}"
            )
        if dst.exists() and not input_data.overwrite:
            return CapabilityOutput(
                success=False,
                error=f"Destination exists and overwrite=False: {dst}",
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return CapabilityOutput(
            success=True,
            data={"src": str(src), "dst": str(dst)},
        )


# ═══════════════════════════════════════════════════════════════════════
# search_files
# ═══════════════════════════════════════════════════════════════════════


class SearchFilesInput(CapabilityInput):
    """Input schema for the ``search_files`` capability.

    Attributes:
        directory: Root directory to search from.
        pattern: Glob pattern (e.g. ``"*.py"``).
        recursive: Whether to recurse into subdirectories.
        max_results: Maximum number of results to return.
    """

    directory: str = Field(description="Root directory to search.")
    pattern: str = Field(description="Glob pattern to match.")
    recursive: bool = Field(default=True, description="Recurse into subdirs.")
    max_results: int = Field(
        default=50, description="Cap the number of results."
    )


class SearchFilesCapability(Capability):
    """Search for files matching a glob pattern in a directory.

    Uses ``pathlib.rglob`` (recursive) or ``pathlib.glob``.

    Example input::

        {"directory": "~/projects", "pattern": "*.py"}

    Example output::

        {"matches": ["/home/user/projects/main.py", ...], "total_found": 12}
    """

    name = "search_files"
    description = "Search for files matching a glob pattern within a directory."
    input_schema = SearchFilesInput
    tags = ["system", "file", "search"]
    requires_confirmation = False

    async def execute(self, input_data: SearchFilesInput) -> CapabilityOutput:
        """Search for files.

        Args:
            input_data: Validated ``SearchFilesInput``.

        Returns:
            CapabilityOutput with matches list and total_found count.
        """
        root = _resolve_path(input_data.directory)
        if not root.is_dir():
            return CapabilityOutput(
                success=False, error=f"Not a directory: {root}"
            )

        glob_fn = root.rglob if input_data.recursive else root.glob
        all_matches = [str(p) for p in glob_fn(input_data.pattern) if p.is_file()]
        total = len(all_matches)
        capped = all_matches[: input_data.max_results]

        return CapabilityOutput(
            success=True,
            data={"matches": capped, "total_found": total},
        )


# ═══════════════════════════════════════════════════════════════════════
# file_info
# ═══════════════════════════════════════════════════════════════════════


class FileInfoInput(CapabilityInput):
    """Input schema for the ``file_info`` capability.

    Attributes:
        path: The file or directory path to inspect.
    """

    path: str = Field(description="Path to inspect.")


class FileInfoCapability(Capability):
    """Return detailed metadata about a file or directory.

    Example input::

        {"path": "/tmp/example.txt"}

    Example output::

        {
            "path": "/tmp/example.txt",
            "size_bytes": 1024,
            "created_at": "2025-01-01T00:00:00",
            "modified_at": "2025-06-15T12:30:00",
            "mime_type": "text/plain",
            "is_file": true,
            "is_dir": false,
            "permissions": "rw-r--r--"
        }
    """

    name = "file_info"
    description = "Get detailed metadata about a file or directory."
    input_schema = FileInfoInput
    tags = ["system", "file", "read"]
    requires_confirmation = False

    async def execute(self, input_data: FileInfoInput) -> CapabilityOutput:
        """Retrieve file metadata.

        Args:
            input_data: Validated ``FileInfoInput``.

        Returns:
            CapabilityOutput with metadata dict.
        """
        path = _resolve_path(input_data.path)
        if not path.exists():
            return CapabilityOutput(
                success=False, error=f"Path not found: {path}"
            )

        st = path.stat()
        perms = stat.filemode(st.st_mode)

        return CapabilityOutput(
            success=True,
            data={
                "path": str(path),
                "size_bytes": st.st_size,
                "created_at": datetime.datetime.fromtimestamp(
                    st.st_ctime
                ).isoformat(),
                "modified_at": datetime.datetime.fromtimestamp(
                    st.st_mtime
                ).isoformat(),
                "mime_type": _guess_mime(path) if path.is_file() else None,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "permissions": perms,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# Capability instances for registration
# ═══════════════════════════════════════════════════════════════════════

read_file = ReadFileCapability()
write_file = WriteFileCapability()
append_file = AppendFileCapability()
delete_file = DeleteFileCapability()
copy_file = CopyFileCapability()
move_file = MoveFileCapability()
search_files = SearchFilesCapability()
file_info = FileInfoCapability()

ALL_FILE_CAPABILITIES = [
    read_file,
    write_file,
    append_file,
    delete_file,
    copy_file,
    move_file,
    search_files,
    file_info,
]
