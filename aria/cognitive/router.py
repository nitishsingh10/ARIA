"""
aria/cognitive/router.py — Hybrid rule+LLM tool selector for ARIA.

Given an ``Intent``, selects the best capability and extracts its
parameters. Uses a 4-step cascade:
    1. Non-capability intents (conversation / memory) → immediate.
    2. Deterministic rules → sub-millisecond.
    3. Semantic search in the registry → fast.
    4. LLM routing → last resort, only when steps 1-3 fail.

Example:
    >>> router = Router(config.aria, ollama, registry, memory)
    >>> result = await router.route(intent)
    >>> result.capability_name
    'read_file'
    >>> result.routing_method
    'rule'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aria.capabilities.base import Capability
from aria.capabilities.registry import CapabilityRegistry
from aria.cognitive.intent import Intent
from aria.cognitive.rules import RoutingRules
from aria.config import AriaConfig
from aria.core.logger import get_logger
from aria.llm.client import OllamaClient
from aria.memory.memory_manager import MemoryManager

log = get_logger("cognitive.router")


# ---------------------------------------------------------------------------
# RouterResult
# ---------------------------------------------------------------------------

@dataclass
class RouterResult:
    """The outcome of routing an intent to a capability.

    Attributes:
        capability_name: Registry name, or None for conversation.
        parameters: Extracted parameters for the capability.
        confidence: Routing confidence (0.0–1.0).
        routing_method: How the route was determined:
            ``"rule"``, ``"llm"``, ``"semantic"``, ``"conversation"``,
            or ``"memory"``.
        alternatives: Other candidate capability names.
        reasoning: Why this capability was chosen.
        requires_planning: Pass-through from Intent.
    """

    capability_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    routing_method: str = "conversation"
    alternatives: List[str] = field(default_factory=list)
    reasoning: str = ""
    requires_planning: bool = False


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    """Hybrid capability router — rules first, LLM last.

    Args:
        config: The ``AriaConfig`` section.
        ollama_client: An initialised ``OllamaClient``.
        registry: The ``CapabilityRegistry`` with all registered tools.
        memory_manager: The ``MemoryManager`` instance.
    """

    def __init__(
        self,
        config: AriaConfig,
        ollama_client: OllamaClient,
        registry: CapabilityRegistry,
        memory_manager: MemoryManager,
    ) -> None:
        self._config = config
        self._ollama = ollama_client
        self._registry = registry
        self._memory = memory_manager
        self.rules = RoutingRules()

    # ------------------------------------------------------------------
    # Main routing entry point
    # ------------------------------------------------------------------

    async def route(self, intent: Intent) -> RouterResult:
        """Route an intent to the best capability.

        Cascade:
            1. Non-capability intents → immediate return.
            2. Deterministic rules → if match with confidence > 0.85.
            3. Semantic search → if single/strong match.
            4. LLM routing → last resort.
            5. Parameter completion → fill missing required fields.

        Args:
            intent: A parsed ``Intent`` object.

        Returns:
            A ``RouterResult`` with the selected capability and params.
        """
        # ── STEP 1: Handle non-capability intents ────────────────
        if intent.intent_type == "conversation":
            log.info(
                "[Router] conversation (no tool)",
                data={"action": intent.action, "confidence": 1.0},
            )
            return RouterResult(
                capability_name=None,
                routing_method="conversation",
                confidence=1.0,
                reasoning="Intent classified as conversation",
                requires_planning=intent.requires_planning,
            )

        if intent.intent_type == "memory":
            log.info(
                "[Router] memory operation",
                data={"action": intent.action, "confidence": 1.0},
            )
            return RouterResult(
                capability_name=None,
                parameters={"text": intent.raw_text},
                routing_method="memory",
                confidence=1.0,
                reasoning=f"Memory operation: {intent.action}",
                requires_planning=False,
            )

        # ── STEP 2: Try deterministic rules ──────────────────────
        rule_match = self.rules.match(intent.raw_text, intent)
        if rule_match:
            rule, params = rule_match
            # Verify capability exists in registry
            if rule.capability_name in self._registry:
                # Add complete_parameters logic
                params = self._complete_parameters(rule.capability_name, intent, params)
                log.info(
                    f"[Router] {rule.capability_name} via rule (conf=0.95)",
                    data={"rule": rule.name, "params": params},
                )
                return RouterResult(
                    capability_name=rule.capability_name,
                    parameters=params,
                    confidence=0.95,
                    routing_method="rule",
                    reasoning=f"Matched rule: {rule.name}",
                    requires_planning=intent.requires_planning,
                )
            else:
                log.warning(
                    f"Rule matched but capability not in registry: {rule.capability_name}",
                )

        # ── STEP 3: Semantic search in registry ──────────────────
        search_query = intent.action
        if intent.entities:
            search_query += " " + " ".join(intent.entities[:3])

        candidates = self._registry.search(search_query)

        if len(candidates) == 1:
            cap = candidates[0]
            params = self._complete_parameters(cap.name, intent, {})
            log.info(
                f"[Router] {cap.name} via semantic (conf=0.80)",
                data={"query": search_query},
            )
            return RouterResult(
                capability_name=cap.name,
                parameters=params,
                confidence=0.80,
                routing_method="semantic",
                reasoning=f"Single semantic match for '{search_query}'",
                alternatives=[],
                requires_planning=intent.requires_planning,
            )

        if len(candidates) > 1:
            # Score each candidate by tag overlap with intent
            scored = [
                (cap, self._score_candidate(cap, intent))
                for cap in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            top_cap, top_score = scored[0]
            alternatives = [c.name for c, _ in scored[1:4]]

            if top_score > 0.3:
                params = self._complete_parameters(top_cap.name, intent, {})
                conf = min(0.85, 0.6 + top_score)
                log.info(
                    f"[Router] {top_cap.name} via semantic (conf={conf:.2f})",
                    data={
                        "query": search_query,
                        "score": round(top_score, 3),
                        "alternatives": alternatives,
                    },
                )
                return RouterResult(
                    capability_name=top_cap.name,
                    parameters=params,
                    confidence=conf,
                    routing_method="semantic",
                    reasoning=f"Top semantic match (score={top_score:.2f})",
                    alternatives=alternatives,
                    requires_planning=intent.requires_planning,
                )

        # ── STEP 4: LLM routing (last resort) ────────────────────
        try:
            llm_result = await self._llm_route(intent, candidates)
            if llm_result.capability_name:
                llm_result.parameters = self._complete_parameters(
                    llm_result.capability_name, intent, llm_result.parameters
                )
                log.info(
                    f"[Router] {llm_result.capability_name} via llm "
                    f"(conf={llm_result.confidence:.2f})",
                    data={"reasoning": llm_result.reasoning},
                )
                return llm_result
        except Exception as exc:
            log.warning("LLM routing failed", data={"error": str(exc)})

        # ── FALLBACK: Treat as conversation ──────────────────────
        log.info(
            "[Router] null via conversation (conf=1.00)",
            data={"reason": "No capability matched"},
        )
        return RouterResult(
            capability_name=None,
            routing_method="conversation",
            confidence=1.0,
            reasoning="No matching capability found, treating as conversation",
            requires_planning=intent.requires_planning,
        )

    # ------------------------------------------------------------------
    # LLM routing
    # ------------------------------------------------------------------

    async def _llm_route(
        self,
        intent: Intent,
        candidates: List[Capability],
    ) -> RouterResult:
        """Use the LLM to select a capability when rules/semantic fail.

        Sends a compact prompt with only candidate tool names and
        descriptions. Strictly validates the response.

        Args:
            intent: The parsed Intent.
            candidates: Candidates from semantic search (may be empty).

        Returns:
            A RouterResult from the LLM's selection.
        """
        # Use all tools if no candidates from semantic search
        tools = candidates if candidates else self._registry.list_all()

        # Build compact tool list (name + description only)
        tool_lines = [f"- {c.name}: {c.description}" for c in tools]
        tool_str = "\n".join(tool_lines)

        system_prompt = (
            "You are a tool router. Pick the best tool for the user's "
            "intent. Return ONLY valid JSON, no explanation.\n\n"
            f"Available tools:\n{tool_str}\n\n"
            "Return:\n"
            "{\n"
            '  "capability_name": "exact_tool_name or null",\n'
            '  "parameters": {},\n'
            '  "confidence": 0.85,\n'
            '  "reasoning": "one sentence",\n'
            '  "alternatives": []\n'
            "}"
        )

        user_msg = (
            f"User intent: action={intent.action}, "
            f"entities={intent.entities}\n"
            f"User input: \"{intent.raw_text}\""
        )

        messages = [{"role": "user", "content": user_msg}]
        raw = await self._ollama.chat(messages, system=system_prompt)

        # Parse JSON response
        parsed = self._extract_json(raw)
        if parsed is None:
            return RouterResult(
                capability_name=None,
                routing_method="conversation",
                confidence=1.0,
                reasoning="LLM returned invalid JSON",
            )

        cap_name = parsed.get("capability_name")

        # Validate capability exists in registry
        if cap_name and cap_name not in self._registry:
            log.warning(
                f"LLM hallucinated capability: {cap_name}",
                data={"raw": raw[:200]},
            )
            cap_name = None

        confidence = parsed.get("confidence", 0.7)
        confidence = max(0.0, min(1.0, float(confidence)))

        return RouterResult(
            capability_name=cap_name,
            parameters=parsed.get("parameters", {}),
            confidence=confidence,
            routing_method="llm" if cap_name else "conversation",
            reasoning=parsed.get("reasoning", ""),
            alternatives=parsed.get("alternatives", []),
            requires_planning=intent.requires_planning,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_candidate(capability: Capability, intent: Intent) -> float:
        """Score a capability against an intent by tag/action overlap.

        Computes Jaccard similarity between the intent's action +
        keywords and the capability's tags.

        Args:
            capability: A candidate capability.
            intent: The parsed intent.

        Returns:
            A float score (0.0–1.0).
        """
        # Build intent tags from action + entities
        intent_tags: set = {intent.action.lower()}
        for entity in intent.entities:
            # Add individual words from entities as tags
            for word in entity.lower().split("/"):
                word = word.strip()
                if word and len(word) > 1:
                    intent_tags.add(word)

        cap_tags = set(t.lower() for t in capability.tags)

        if not intent_tags or not cap_tags:
            return 0.0

        intersection = intent_tags & cap_tags
        union = intent_tags | cap_tags

        score = len(intersection) / len(union) if union else 0.0

        # Boost if action matches any tag exactly
        if intent.action.lower() in cap_tags:
            score = min(1.0, score + 0.15)

        return score

    # ------------------------------------------------------------------
    # Parameter completion
    # ------------------------------------------------------------------

    def _complete_parameters(self, capability_name: str, intent: Intent, partial_params: dict) -> dict:
        """Fill missing capability parameters securely from intent rules."""
        capability = self._registry.get(capability_name)
        if not capability:
            return partial_params

        try:
            schema = capability.input_schema.model_json_schema()
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])
        except Exception:
            return partial_params

        for field in required_fields:
            if field not in partial_params:
                # Try to fill from intent
                if field == "path":
                    # Check parameters first
                    if "path" in intent.parameters:
                        partial_params["path"] = intent.parameters["path"]
                    elif "name" in intent.parameters:
                        partial_params["path"] = intent.parameters["name"]
                    elif intent.entities:
                        partial_params["path"] = intent.entities[0]
                elif field == "content":
                    if "content" in intent.parameters:
                        partial_params["content"] = intent.parameters["content"]
                    else:
                        partial_params["content"] = ""  # default empty
                elif field == "command":
                    if intent.entities:
                        partial_params["command"] = " ".join(intent.entities)
                elif field == "url":
                    urls = [e for e in intent.entities if e.startswith("http")]
                    if urls:
                        partial_params["url"] = urls[0]
                elif field == "query":
                    partial_params["query"] = intent.raw_text
                elif field == "code":
                    if "code" in intent.parameters:
                        partial_params["code"] = intent.parameters["code"]
                        
        return partial_params

    # ------------------------------------------------------------------
    # JSON helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract a JSON object from LLM output.

        Args:
            text: Raw LLM response.

        Returns:
            Parsed dict, or None if no valid JSON found.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(cleaned[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None
