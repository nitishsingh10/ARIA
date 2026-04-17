"""
aria/memory/context.py — Session context tracker for ARIA.

Tracks the current conversation session: messages exchanged, tools
used, and active task. This is ephemeral in-memory state — it is
NOT persisted between sessions. Persistence is the VectorStore's job.

Example:
    >>> ctx = SessionContext()
    >>> ctx.add_user_message("Hello ARIA")
    >>> ctx.add_assistant_message("Hi! How can I help?")
    >>> msgs = ctx.to_llm_format(last_n=10)
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ContextMessage:
    """A single message in the session context.

    Attributes:
        role: One of ``"user"``, ``"assistant"``, ``"system"``, ``"tool"``.
        content: The message text.
        timestamp: When the message was created (UTC).
        metadata: Extra data such as tool_name, capability_name, etc.
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SessionContext
# ---------------------------------------------------------------------------


class SessionContext:
    """Ephemeral in-memory tracker for a single conversation session.

    Maintains a bounded deque of ``ContextMessage`` objects plus
    session-level metadata (active task, working directory, etc.).

    Args:
        session_id: Unique session identifier (auto-generated if None).
        max_messages: Maximum messages to keep in the rolling buffer.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_messages: int = 50,
    ) -> None:
        """Initialise a new session context.

        Args:
            session_id: Explicit ID or auto-UUID.
            max_messages: Size limit of the message deque.
        """
        self.session_id: str = session_id or str(uuid.uuid4())
        self.messages: Deque[ContextMessage] = deque(maxlen=max_messages)
        self.created_at: datetime = datetime.now(timezone.utc)
        self.active_task: Optional[str] = None
        self.working_directory: str = "."
        self._tools_used: List[str] = []

    # ------------------------------------------------------------------
    # Add messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        """Add a generic message to the session.

        Args:
            role: Message role (user / assistant / system / tool).
            content: Message text.
            metadata: Optional extra metadata.

        Returns:
            The created ``ContextMessage``.
        """
        msg = ContextMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        return msg

    def add_user_message(self, content: str) -> ContextMessage:
        """Convenience: add a user message.

        Args:
            content: The user's text.

        Returns:
            The created ``ContextMessage``.
        """
        return self.add_message("user", content)

    def add_assistant_message(self, content: str) -> ContextMessage:
        """Convenience: add an assistant message.

        Args:
            content: ARIA's response text.

        Returns:
            The created ``ContextMessage``.
        """
        return self.add_message("assistant", content)

    def add_tool_result(
        self,
        tool_name: str,
        result: str,
        success: bool,
    ) -> ContextMessage:
        """Add a tool execution result to the session.

        Args:
            tool_name: The capability / tool that was invoked.
            result: The tool's output text.
            success: Whether the tool succeeded.

        Returns:
            The created ``ContextMessage``.
        """
        if tool_name not in self._tools_used:
            self._tools_used.append(tool_name)
        return self.add_message(
            "tool",
            result,
            metadata={
                "tool_name": tool_name,
                "success": success,
            },
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_messages(
        self, last_n: Optional[int] = None
    ) -> List[ContextMessage]:
        """Get session messages, optionally the most recent N.

        Args:
            last_n: If given, return only the last N messages.

        Returns:
            A list of ``ContextMessage`` objects.
        """
        msgs = list(self.messages)
        if last_n is not None:
            msgs = msgs[-last_n:]
        return msgs

    def to_llm_format(self, last_n: int = 10) -> List[Dict[str, str]]:
        """Convert recent messages to Ollama-compatible chat format.

        Tool messages are collapsed into assistant messages so the
        LLM doesn't see raw tool output.

        Args:
            last_n: Number of most recent messages to include.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts.
        """
        result: List[Dict[str, str]] = []
        for msg in self.get_messages(last_n):
            if msg.role == "tool":
                # Collapse tool results into assistant messages
                tool_name = msg.metadata.get("tool_name", "tool")
                success = msg.metadata.get("success", True)
                status = "✓" if success else "✗"
                result.append({
                    "role": "assistant",
                    "content": f"[Tool: {tool_name} {status}] {msg.content}",
                })
            else:
                result.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        return result

    # ------------------------------------------------------------------
    # Summary & metadata
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Get a summary of the current session.

        Returns:
            A dict with session_id, message_count, duration_seconds,
            tools_used, and active_task.
        """
        now = datetime.now(timezone.utc)
        duration = (now - self.created_at).total_seconds()
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "duration_seconds": round(duration, 1),
            "tools_used": list(self._tools_used),
            "active_task": self.active_task,
        }

    def set_active_task(self, task: str) -> None:
        """Set the description of the current active task.

        Args:
            task: A short description of what the user is working on.
        """
        self.active_task = task

    def set_working_directory(self, path: str) -> None:
        """Set the working directory for this session.

        Args:
            path: The directory path.
        """
        self.working_directory = path

    def clear(self) -> None:
        """Clear all messages and reset the session.

        Does NOT change the session_id — use ``MemoryManager.new_session()``
        for that.
        """
        self.messages.clear()
        self._tools_used.clear()
        self.active_task = None
