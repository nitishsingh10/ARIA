"""
aria/cognitive/intent.py — Intent parsing and classification for ARIA.

Takes raw user text and extracts structured intent: what the user
wants to do, what entities are involved, and the complexity level.

Uses a two-stage approach:
    1. **fast_parse**: Rule-based, zero LLM calls, runs in <1ms.
    2. **llm_parse**: Called only when fast_parse confidence < 0.7.

Example:
    >>> parser = IntentParser(config.aria, ollama, memory)
    >>> intent = await parser.parse("read /tmp/hello.txt")
    >>> intent.action
    'read'
    >>> intent.entities
    ['/tmp/hello.txt']
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aria.config import AriaConfig
from aria.core.logger import get_logger
from aria.llm.client import OllamaClient
from aria.memory.memory_manager import MemoryManager

log = get_logger("cognitive.intent")

# ---------------------------------------------------------------------------
# Valid actions
# ---------------------------------------------------------------------------
_VALID_ACTIONS = {
    "read", "write", "run", "search", "fetch", "summarize",
    "delete", "list", "create", "move", "explain", "remember",
    "recall", "chat",
}

_VALID_INTENT_TYPES = {"capability", "conversation", "memory", "system_command"}
_VALID_COMPLEXITIES = {"simple", "multi_step", "complex"}

# ---------------------------------------------------------------------------
# Entity extraction patterns (compiled once)
# ---------------------------------------------------------------------------
_RE_FILE_PATH = re.compile(r'(?:^|\s)([~/.][\w./\-*]+(?:\.\w+)?)')
_RE_URL = re.compile(r'https?://\S+')
_RE_QUOTED = re.compile(r'["\']([^"\']+)["\']')
_RE_CODE_BLOCK = re.compile(r'```(?:\w+)?\n?(.*?)```', re.DOTALL)

# ---------------------------------------------------------------------------
# Fast-parse keyword maps
# ---------------------------------------------------------------------------
_ACTION_KEYWORDS: List[tuple] = [
    # (keywords_tuple, action, intent_type, base_confidence)
    (("read ", "open ", "show me ", "cat ", "print "), "read", "capability", 0.85),
    (("write ", "save ", "create file "), "write", "capability", 0.85),
    (("run ", "execute ", "python "), "run", "capability", 0.80),
    (("search ", "find ", "look for ", "grep "), "search", "capability", 0.80),
    (("list ", "ls ", "dir "), "list", "capability", 0.85),
    (("delete ", "remove ", "rm "), "delete", "capability", 0.80),
    (("move ", "mv ", "rename "), "move", "capability", 0.80),
    (("remember ", "note that ", "save that "), "remember", "memory", 0.90),
    (("what is ", "explain ", "how does ", "how do ", "why "), "explain", "conversation", 0.75),
    (("summarize ", "summarise ", "tldr "), "summarize", "capability", 0.80),
    (("create ", "make ", "mkdir "), "create", "capability", 0.80),
]

# Patterns that need special detection (not prefix-based)
_URL_ACTION_KEYWORDS = ("fetch ", "get ", "scrape ", "curl ")
_TIME_PATTERNS = re.compile(
    r"(?:what time|current time|what'?s the time|what'?s today|what day)",
    re.IGNORECASE,
)
_SYSINFO_PATTERNS = re.compile(
    r"(?:system info|sysinfo|hardware info|memory usage|disk usage|cpu info)",
    re.IGNORECASE,
)

# Recall patterns — must be checked BEFORE remember keywords
# These catch questions about what ARIA knows/remembers.
_RECALL_PATTERNS = re.compile(
    r"(?:"
    r"what (?:do you |all do you |can you )?(?:know|remember|recall)"
    r"|do you (?:remember|recall|know)"
    r"|tell me (?:what you know|about me)"
    r"|what(?:'?s| is) my (?:name|project|preference)"
    r"|^recall\b"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Intent dataclass
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    """Structured intent extracted from user input.

    Attributes:
        raw_text: The original user input.
        action: Primary verb — one of the _VALID_ACTIONS.
        entities: Extracted paths, URLs, quoted strings, etc.
        parameters: Key-value pairs extracted from the input.
        complexity: ``"simple"``, ``"multi_step"``, or ``"complex"``.
        requires_planning: True if multi_step or complex.
        confidence: Parsing confidence (0.0–1.0).
        intent_type: ``"capability"``, ``"conversation"``,
            ``"memory"``, or ``"system_command"``.
        raw_llm_response: Full LLM JSON response for debugging.
    """

    raw_text: str = ""
    action: str = "chat"
    entities: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    complexity: str = "simple"
    requires_planning: bool = False
    confidence: float = 0.0
    intent_type: str = "conversation"
    raw_llm_response: str = ""


# ---------------------------------------------------------------------------
# IntentParser
# ---------------------------------------------------------------------------

class IntentParser:
    """Two-stage intent parser: fast rules first, LLM fallback second.

    Args:
        config: The ``AriaConfig`` section.
        ollama_client: An initialised ``OllamaClient``.
        memory_manager: The ``MemoryManager`` instance.
    """

    def __init__(
        self,
        config: AriaConfig,
        ollama_client: OllamaClient,
        memory_manager: MemoryManager,
    ) -> None:
        self._config = config
        self._ollama = ollama_client
        self._memory = memory_manager

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def parse(self, user_input: str) -> Intent:
        """Parse user input into a structured Intent.

        Strategy:
            1. Try fast_parse() first (no LLM).
            2. If confidence < 0.7, call llm_parse().
            3. Merge results and return.

        Args:
            user_input: The raw user text.

        Returns:
            A populated ``Intent`` dataclass.
        """
        fast = self.fast_parse(user_input)

        log.debug(
            "Fast parse result",
            data={
                "action": fast.action,
                "confidence": fast.confidence,
                "intent_type": fast.intent_type,
            },
        )

        if fast.confidence >= 0.7:
            return fast

        # LLM fallback
        try:
            llm_intent = await self.llm_parse(user_input, fast)
            # Merge: prefer LLM action if it has higher confidence,
            # but keep fast-parsed entities if LLM didn't find any
            if not llm_intent.entities and fast.entities:
                llm_intent.entities = fast.entities
            if not llm_intent.parameters and fast.parameters:
                llm_intent.parameters = fast.parameters
            return llm_intent
        except Exception as exc:
            log.warning(
                "LLM parse failed, using fast parse",
                data={"error": str(exc)},
            )
            fast.confidence = max(fast.confidence, 0.5)
            return fast

    # ------------------------------------------------------------------
    # Fast parse (rule-based, <1ms)
    # ------------------------------------------------------------------

    def fast_parse(self, user_input: str) -> Intent:
        """Rule-based intent parsing with zero LLM calls.

        Detects action verbs, extracts entities (paths, URLs,
        quoted strings, code blocks), and assigns confidence based
        on pattern strength.

        Args:
            user_input: Raw user text.

        Returns:
            An ``Intent`` with best-effort classification.
        """
        text = user_input.strip()
        text_lower = text.lower()

        intent = Intent(raw_text=text)

        # --- Extract entities first ---
        intent.entities = self._extract_entities(text)

        # --- Check for code blocks ---
        code_match = _RE_CODE_BLOCK.search(text)
        if code_match:
            intent.action = "run"
            intent.intent_type = "capability"
            intent.confidence = 0.90
            intent.parameters["code"] = code_match.group(1).strip()
            return intent

        # --- Check for URLs ---
        url_match = _RE_URL.search(text)
        if url_match:
            for kw in _URL_ACTION_KEYWORDS:
                if text_lower.startswith(kw):
                    intent.action = "fetch"
                    intent.intent_type = "capability"
                    intent.confidence = 0.90
                    intent.parameters["url"] = url_match.group(0)
                    return intent
            # URL present but no fetch keyword — still mark it
            if not any(text_lower.startswith(kw[0]) for kw in _ACTION_KEYWORDS):
                intent.action = "fetch"
                intent.intent_type = "capability"
                intent.confidence = 0.75
                intent.parameters["url"] = url_match.group(0)
                return intent

        # --- Recall patterns (before keywords to beat "remember" prefix) ---
        if _RECALL_PATTERNS.search(text):
            intent.action = "recall"
            intent.intent_type = "memory"
            intent.confidence = 0.90
            return intent

        # --- Time patterns ---
        if _TIME_PATTERNS.search(text):
            intent.action = "search"
            intent.intent_type = "capability"
            intent.confidence = 0.90
            intent.parameters["_special"] = "time"
            return intent

        # --- System info patterns ---
        if _SYSINFO_PATTERNS.search(text):
            intent.action = "search"
            intent.intent_type = "capability"
            intent.confidence = 0.90
            intent.parameters["_special"] = "sysinfo"
            return intent

        # --- Keyword-based action detection ---
        for keywords, action, itype, conf in _ACTION_KEYWORDS:
            for kw in keywords:
                if text_lower.startswith(kw) or text_lower == kw.strip():
                    intent.action = action
                    intent.intent_type = itype
                    intent.confidence = conf

                    # Extract path parameter for file operations
                    if action in ("read", "write", "list", "delete", "create"):
                        paths = [e for e in intent.entities
                                 if e.startswith(("/", "~", "."))]
                        if paths:
                            intent.parameters["path"] = paths[0]

                    return intent

        # --- Run command pattern: "run: <command>" ---
        run_match = re.match(
            r'^(?:run|exec|shell|bash|sh):\s*(.+)',
            text, re.IGNORECASE,
        )
        if run_match:
            intent.action = "run"
            intent.intent_type = "capability"
            intent.confidence = 0.90
            intent.parameters["command"] = run_match.group(1).strip()
            return intent

        # --- Default: chat ---
        intent.action = "chat"
        intent.intent_type = "conversation"
        intent.confidence = 0.4
        return intent

    # ------------------------------------------------------------------
    # LLM parse (fallback)
    # ------------------------------------------------------------------

    async def llm_parse(
        self, user_input: str, fast_intent: Intent
    ) -> Intent:
        """Parse intent using the LLM when fast_parse is uncertain.

        Sends a compact structured prompt to the LLM and parses
        the JSON response. On failure, returns fast_intent with
        a bumped confidence.

        Args:
            user_input: Raw user text.
            fast_intent: The result from fast_parse (used as fallback).

        Returns:
            An ``Intent`` populated from the LLM response.
        """
        system_prompt = (
            "You are an intent parser. Extract structured intent from user input.\n"
            "Return ONLY valid JSON, no explanation, no markdown.\n\n"
            "Available actions: read, write, run, search, fetch, summarize, "
            "delete, list, create, move, explain, remember, recall, chat\n\n"
            "Return this exact JSON structure:\n"
            "{\n"
            '  "action": "one of the actions above",\n'
            '  "entities": ["list of paths, URLs, or key nouns"],\n'
            '  "parameters": {},\n'
            '  "complexity": "simple or multi_step or complex",\n'
            '  "requires_planning": false,\n'
            '  "intent_type": "capability or conversation or memory or system_command",\n'
            '  "confidence": 0.85\n'
            "}"
        )

        messages = [{"role": "user", "content": user_input}]

        raw_response = await self._ollama.chat(messages, system=system_prompt)

        intent = Intent(raw_text=user_input, raw_llm_response=raw_response)

        # Parse JSON from LLM response
        try:
            parsed = self._extract_json(raw_response)
            if parsed is None:
                log.warning("LLM returned no valid JSON")
                fast_intent.confidence = 0.5
                return fast_intent

            # Map fields with validation
            action = parsed.get("action", "chat")
            intent.action = action if action in _VALID_ACTIONS else "chat"

            entities = parsed.get("entities", [])
            intent.entities = entities if isinstance(entities, list) else []

            params = parsed.get("parameters", {})
            intent.parameters = params if isinstance(params, dict) else {}

            complexity = parsed.get("complexity", "simple")
            intent.complexity = complexity if complexity in _VALID_COMPLEXITIES else "simple"

            intent.requires_planning = bool(parsed.get("requires_planning", False))

            itype = parsed.get("intent_type", "conversation")
            intent.intent_type = itype if itype in _VALID_INTENT_TYPES else "conversation"

            confidence = parsed.get("confidence", 0.7)
            intent.confidence = max(0.0, min(1.0, float(confidence)))

            log.debug(
                "LLM parse result",
                data={
                    "action": intent.action,
                    "confidence": intent.confidence,
                    "intent_type": intent.intent_type,
                },
            )

        except Exception as exc:
            log.warning("LLM parse JSON failed", data={"error": str(exc)})
            fast_intent.confidence = 0.5
            return fast_intent

        return intent

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Extract paths, URLs, quoted strings, and code blocks.

        Args:
            text: Raw user input.

        Returns:
            A deduplicated list of extracted entity strings.
        """
        entities: List[str] = []

        # File paths
        for match in _RE_FILE_PATH.finditer(text):
            path = match.group(1).rstrip(".,;:!?)")
            if len(path) > 1:
                entities.append(path)

        # URLs
        for match in _RE_URL.finditer(text):
            entities.append(match.group(0).rstrip(".,;:!?)"))

        # Quoted strings
        for match in _RE_QUOTED.finditer(text):
            entities.append(match.group(1))

        # Code blocks
        for match in _RE_CODE_BLOCK.finditer(text):
            entities.append(match.group(1).strip())

        # Deduplicate while preserving order
        seen: set = set()
        result: List[str] = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)

        return result

    # ------------------------------------------------------------------
    # JSON extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract a JSON object from LLM output.

        Strips markdown code fences and attempts multiple parse
        strategies for robustness.

        Args:
            text: Raw LLM response.

        Returns:
            Parsed dict, or None if no valid JSON found.
        """
        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (fences)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(cleaned[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None
