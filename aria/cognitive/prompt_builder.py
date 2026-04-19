"""
aria/cognitive/prompt_builder.py — Context-rich LLM prompt construction.

Builds system prompts for all LLM calls in the cognitive layer:
chat, intent parsing, and tool routing. Centralises prompt logic
so the rest of the system never constructs raw prompt strings.

Example:
    >>> pb = PromptBuilder(config.aria, memory, registry)
    >>> system = await pb.build_chat_prompt("hello")
    >>> response = await ollama.chat(messages, system=system)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from aria.capabilities.base import Capability
from aria.capabilities.registry import CapabilityRegistry
from aria.config import AriaConfig
from aria.core.logger import get_logger
from aria.memory.memory_manager import MemoryManager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aria.cognitive.intent import Intent

log = get_logger("cognitive.prompt_builder")


class PromptBuilder:
    """Constructs rich, context-aware system prompts for ARIA.

    Args:
        config: The ``AriaConfig`` section.
        memory_manager: For retrieving memory context.
        registry: For generating tool lists.
    """

    def __init__(
        self,
        config: AriaConfig,
        memory_manager: MemoryManager,
        registry: CapabilityRegistry,
    ) -> None:
        self._config = config
        self._memory = memory_manager
        self._registry = registry

    # ------------------------------------------------------------------
    # Chat prompt
    # ------------------------------------------------------------------

    async def build_chat_prompt(
        self,
        user_input: str,
        include_memory: bool = True,
        include_tools: bool = False,
    ) -> str:
        """Build the system prompt for general conversation.

        Includes:
            - ARIA identity block.
            - Memory context (from recall_for_prompt).
            - Current session summary.
            - Optional compact tool list.
            - Datetime + working directory.

        Args:
            user_input: The user's latest message (for memory recall).
            include_memory: Whether to include memory context.
            include_tools: Whether to include tool list.

        Returns:
            A complete system prompt string.
        """
        parts: List[str] = [self.build_system_identity()]

        # Memory context
        if include_memory:
            try:
                memory_ctx = await self._memory.recall_for_prompt(user_input)
                if memory_ctx:
                    parts.append(memory_ctx)
            except Exception as exc:
                log.warning("Memory recall failed", data={"error": str(exc)})

        # Session summary
        try:
            summary = self._memory.context.get_summary()
            if summary.get("message_count", 0) > 0:
                parts.append(
                    f"Session: {summary['message_count']} messages, "
                    f"{summary['duration_seconds']}s elapsed."
                )
        except Exception:
            pass

        # Tool list
        if include_tools:
            caps = self._registry.list_all()
            if caps:
                parts.append(self.format_tool_list(caps, compact=True))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Intent prompt
    # ------------------------------------------------------------------

    async def build_intent_prompt(self, user_input: str) -> str:
        """Build a compact prompt for intent parsing.

        Includes only: role directive, output format, and a few examples.

        Args:
            user_input: The user's input (not included in the system
                prompt — passed as the user message separately).

        Returns:
            A system prompt for the intent parser.
        """
        return (
            "You are an intent parser for ARIA, a local AI assistant.\n"
            "Extract the user's intent as structured JSON.\n\n"
            "Available actions: read, write, run, search, fetch, summarize, "
            "delete, list, create, move, explain, remember, recall, chat\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "action": "...",\n'
            '  "entities": [...],\n'
            '  "parameters": {...},\n'
            '  "complexity": "simple|multi_step|complex",\n'
            '  "requires_planning": false,\n'
            '  "intent_type": "capability|conversation|memory|system_command",\n'
            '  "confidence": 0.85\n'
            "}\n\n"
            "Examples:\n"
            '  "read /tmp/foo.py" → {"action":"read","entities":["/tmp/foo.py"],...}\n'
            '  "what is rust" → {"action":"explain","entities":["rust"],...}\n'
            '  "remember I like Python" → {"action":"remember","entities":[],...}'
        )

    # ------------------------------------------------------------------
    # Router prompt
    # ------------------------------------------------------------------

    async def build_router_prompt(
        self,
        intent: "Intent",
        candidates: List[Capability],
    ) -> str:
        """Build a routing prompt for capability selection.

        Includes only the intent summary and compact tool specs
        for candidate capabilities — never the full 22-tool list.

        Args:
            intent: The parsed Intent.
            candidates: Candidate capabilities from semantic search.

        Returns:
            A system prompt for the router LLM call.
        """
        # Avoid circular import
        from aria.cognitive.intent import Intent

        tool_list = self.format_tool_list(candidates, compact=True)

        return (
            "You are a tool router for ARIA. Pick the best tool for the "
            "user's intent. Return ONLY valid JSON, no explanation.\n\n"
            f"User intent: action={intent.action}, "
            f"entities={intent.entities}\n"
            f"User input: \"{intent.raw_text}\"\n\n"
            f"Available tools:\n{tool_list}\n\n"
            "Return this JSON:\n"
            "{\n"
            '  "capability_name": "exact_tool_name or null",\n'
            '  "parameters": {},\n'
            '  "confidence": 0.85,\n'
            '  "reasoning": "one sentence",\n'
            '  "alternatives": []\n'
            "}"
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def build_system_identity(self) -> str:
        """Build the static ARIA identity block.

        Returns:
            A string with ARIA's identity, date, and working directory.
        """
        now = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
        cwd = os.getcwd()

        return (
            f"You are {self._config.name} (Adaptive Runtime Intelligence "
            f"Architecture), a local AI runtime running v{self._config.version}. "
            "You help users accomplish real tasks by using tools. You are "
            "precise, honest, and never fabricate file paths, command "
            "outputs, or system states.\n"
            f"Today is {now}. Working directory: {cwd}."
        )

    # ------------------------------------------------------------------
    # Tool formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_tool_list(
        capabilities: List[Capability],
        compact: bool = True,
    ) -> str:
        """Format a list of capabilities as a readable string.

        Args:
            capabilities: The capabilities to format.
            compact: If True, one line per tool (name + description).
                If False, full JSON schema per tool.

        Returns:
            A formatted string listing the tools.
        """
        if not capabilities:
            return "No tools available."

        if compact:
            lines = ["Available tools:"]
            for cap in capabilities:
                tags_str = ", ".join(cap.tags[:3]) if cap.tags else ""
                lines.append(f"- {cap.name}: {cap.description} [{tags_str}]")
            return "\n".join(lines)
        else:
            import json
            specs = [cap.to_tool_spec() for cap in capabilities]
            return json.dumps(specs, indent=2)
