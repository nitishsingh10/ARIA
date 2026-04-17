"""
aria/memory/memory_manager.py — Unified memory interface for ARIA.

The ``MemoryManager`` is the single entry point for all memory
operations in ARIA. It coordinates the vector store (long-term
semantic memory), knowledge graph (structured facts), and session
context (ephemeral conversation state).

Usage:
    >>> from aria.memory import MemoryManager
    >>> mm = MemoryManager(config.aria, ollama_client)
    >>> await mm.remember("Nitish prefers dark themes")
    >>> result = await mm.recall("user preferences")
    >>> print(result.formatted_summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aria.config import AriaConfig
from aria.core.logger import get_logger
from aria.llm.client import OllamaClient
from aria.memory.context import ContextMessage, SessionContext
from aria.memory.knowledge_graph import Entity, KnowledgeGraph, Relation
from aria.memory.vector_store import MemoryResult, VectorStore

log = get_logger("memory.manager")


# ---------------------------------------------------------------------------
# RecallResult
# ---------------------------------------------------------------------------

@dataclass
class RecallResult:
    """Combined results from a memory recall operation.

    Attributes:
        query: The original query string.
        vector_results: Results grouped by vector collection.
        graph_entities: Relevant entities from the knowledge graph.
        graph_relations: Relevant relations from the knowledge graph.
        formatted_summary: A human-readable string summarising the recall.
    """

    query: str = ""
    vector_results: Dict[str, List[MemoryResult]] = field(default_factory=dict)
    graph_entities: List[Entity] = field(default_factory=list)
    graph_relations: List[Relation] = field(default_factory=list)
    formatted_summary: str = ""


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Unified memory interface — the only memory API the rest of ARIA uses.

    Combines:
        - **VectorStore**: Persistent semantic similarity search.
        - **KnowledgeGraph**: Structured entity-relation facts.
        - **SessionContext**: Ephemeral in-memory conversation state.

    Args:
        config: The ``AriaConfig`` section.
        ollama_client: An initialised ``OllamaClient`` for embeddings.
    """

    def __init__(
        self,
        config: AriaConfig,
        ollama_client: OllamaClient,
    ) -> None:
        """Initialise the memory manager and all sub-systems.

        Args:
            config: ARIA runtime config.
            ollama_client: Ollama client for embeddings.
        """
        self._config = config
        self._ollama = ollama_client

        # Sub-system initialisation (all safe — errors are caught internally)
        try:
            self.vector_store = VectorStore(config, ollama_client)
        except Exception as exc:
            log.error("VectorStore init failed — running without vector memory",
                      data={"error": str(exc)})
            self.vector_store = None  # type: ignore[assignment]

        self.knowledge = KnowledgeGraph(config)
        self.context = SessionContext()

        log.info(
            "MemoryManager initialised",
            data={"session_id": self.context.session_id},
        )

    # ------------------------------------------------------------------
    # remember — store something in memory
    # ------------------------------------------------------------------

    async def remember(
        self,
        text: str,
        source: str = "conversation",
        metadata: Optional[dict] = None,
    ) -> str:
        """Store text in the appropriate vector collection and extract facts.

        Args:
            text: The text to remember.
            source: Origin — ``"conversation"``, ``"document"``, or
                ``"user_input"`` (determines which collection to use).
            metadata: Additional metadata for the vector store.

        Returns:
            The vector store document ID.
        """
        meta = metadata or {}
        meta["session_id"] = self.context.session_id
        meta["timestamp"] = datetime.now(timezone.utc).isoformat()

        doc_id = ""
        if self.vector_store:
            try:
                if source == "document":
                    doc_id = await self.vector_store.add_document(
                        text, source=meta.get("file_path", source), metadata=meta
                    )
                elif source == "fact":
                    doc_id = await self.vector_store.add_fact(
                        text,
                        subject=meta.get("subject", "general"),
                        confidence=meta.get("confidence", 1.0),
                        source=source,
                    )
                else:
                    doc_id = await self.vector_store.add_conversation(text, meta)
            except Exception as exc:
                log.error("remember: vector store failed", data={"error": str(exc)})

        # Also extract entities/relations from the text
        try:
            extraction = self.knowledge.extract_and_store(text)
            if extraction["entities_added"] or extraction["relations_added"]:
                log.debug("Knowledge extracted", data=extraction)
        except Exception as exc:
            log.error("remember: knowledge extraction failed", data={"error": str(exc)})

        return doc_id

    # ------------------------------------------------------------------
    # recall — retrieve from memory
    # ------------------------------------------------------------------

    async def recall(self, query: str, top_k: int = 5) -> RecallResult:
        """Search all memory systems for relevant information.

        Queries the vector store (all collections) and the knowledge
        graph, then produces a combined, ranked result.

        Args:
            query: The search query.
            top_k: Maximum results per vector collection.

        Returns:
            A ``RecallResult`` with all found memories.
        """
        result = RecallResult(query=query)

        # --- Vector search ---
        if self.vector_store:
            try:
                result.vector_results = await self.vector_store.search_all(
                    query, top_k=top_k
                )
            except Exception as exc:
                log.error("recall: vector search failed", data={"error": str(exc)})

        # --- Knowledge graph search ---
        try:
            query_lower = query.lower()
            for entity in self.knowledge.entities.values():
                if (
                    entity.name.lower() in query_lower
                    or query_lower in entity.name.lower()
                    or any(
                        query_lower in str(v).lower()
                        for v in entity.attributes.values()
                    )
                ):
                    result.graph_entities.append(entity)
                    result.graph_relations.extend(
                        self.knowledge.get_relations(entity.id)
                    )
        except Exception as exc:
            log.error("recall: graph search failed", data={"error": str(exc)})

        # --- Build formatted summary ---
        result.formatted_summary = self._format_recall(result)

        return result

    async def recall_for_prompt(self, query: str) -> str:
        """Generate a memory context string ready for system prompt injection.

        Args:
            query: The user's latest message.

        Returns:
            A formatted string like:
            ``"Relevant memory:\\n- ...\\n\\nWhat I know about you:\\n- ..."``
        """
        result = await self.recall(query, top_k=3)

        parts: List[str] = []

        # Vector memories
        all_memories: List[MemoryResult] = []
        for memories in result.vector_results.values():
            all_memories.extend(memories)
        all_memories.sort(key=lambda m: m.score, reverse=True)

        if all_memories:
            lines = ["Relevant memory:"]
            for mem in all_memories[:5]:
                # Truncate long texts
                text = mem.text[:200].replace("\n", " ")
                lines.append(f"- [{mem.collection}] {text}")
            parts.append("\n".join(lines))

        # User profile
        profile = self.knowledge.get_user_profile()
        profile_lines: List[str] = []
        if profile.get("name") and profile["name"] != "user":
            profile_lines.append(f"User's name: {profile['name']}")
        if profile.get("projects"):
            profile_lines.append(f"Working on: {', '.join(profile['projects'])}")
        if profile.get("tools"):
            profile_lines.append(f"Uses: {', '.join(profile['tools'])}")
        if profile.get("preferences"):
            profile_lines.append(f"Prefers: {', '.join(profile['preferences'])}")
        if profile.get("facts"):
            for k, v in list(profile["facts"].items())[:5]:
                if k != "role":
                    profile_lines.append(f"{k}: {v}")

        if profile_lines:
            parts.append("What I know about you:\n" + "\n".join(f"- {l}" for l in profile_lines))

        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Conversation tracking
    # ------------------------------------------------------------------

    async def store_conversation_turn(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Record a full conversation turn in all memory systems.

        1. Adds both messages to the SessionContext.
        2. Stores the user message in the vector store.
        3. Stores a summary of the exchange in the vector store.
        4. Runs entity extraction on both messages.

        Args:
            user_message: The user's text.
            assistant_response: ARIA's reply.
        """
        session_id = self.context.session_id

        # 1. Session context
        self.context.add_user_message(user_message)
        self.context.add_assistant_message(assistant_response)

        # 2-3. Vector store
        if self.vector_store:
            try:
                await self.vector_store.add_conversation(
                    user_message,
                    metadata={
                        "role": "user",
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                # Store a combined summary for better recall
                summary = f"User: {user_message}\nARIA: {assistant_response[:500]}"
                await self.vector_store.add_conversation(
                    summary,
                    metadata={
                        "role": "summary",
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception as exc:
                log.error("store_conversation_turn: vector failed", data={"error": str(exc)})

        # 4. Knowledge extraction
        try:
            self.knowledge.extract_and_store(user_message)
            self.knowledge.extract_and_store(assistant_response)
        except Exception as exc:
            log.error("store_conversation_turn: extraction failed", data={"error": str(exc)})

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def get_context_for_llm(self, last_n: int = 10) -> List[Dict[str, str]]:
        """Get the most recent conversation context in LLM format.

        Args:
            last_n: Number of recent messages.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts.
        """
        return self.context.to_llm_format(last_n)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def new_session(self) -> str:
        """End the current session and start a fresh one.

        The old session's messages are summarised and stored in the
        vector store (synchronously-safe best-effort).

        Returns:
            The new session ID.
        """
        old_summary = self.context.get_summary()
        old_messages = self.context.get_messages()

        log.info("Starting new session", data=old_summary)

        # Create new context
        self.context = SessionContext()

        # Store old session summary in vector store (best effort)
        if self.vector_store and old_messages:
            try:
                # Build a textual summary of the old session
                msg_texts = [
                    f"{m.role}: {m.content[:100]}" for m in old_messages[-10:]
                ]
                summary_text = (
                    f"Session summary ({old_summary['message_count']} messages, "
                    f"{old_summary['duration_seconds']}s):\n"
                    + "\n".join(msg_texts)
                )
                # Use synchronous add via the embedding function's loop trick
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        loop.run_until_complete(
                            self.vector_store.add_conversation(
                                summary_text,
                                metadata={
                                    "role": "session_summary",
                                    "session_id": old_summary["session_id"],
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                        )
                except RuntimeError:
                    pass  # Can't store from sync context, skip
            except Exception as exc:
                log.warning("Failed to store old session summary", data={"error": str(exc)})

        return self.context.session_id

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Get combined statistics from all memory sub-systems.

        Returns:
            A dict with vector_store, knowledge_graph, and context stats.
        """
        vs_stats = self.vector_store.stats() if self.vector_store else {}
        kg_entities = len(self.knowledge.entities)
        kg_relations = len(self.knowledge.relations)

        return {
            "vector_store": vs_stats,
            "knowledge_graph": {
                "entities": kg_entities,
                "relations": kg_relations,
            },
            "context": self.context.get_summary(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_recall(result: RecallResult) -> str:
        """Build a human-readable summary string from recall results.

        Args:
            result: The raw RecallResult.

        Returns:
            A multi-line formatted string.
        """
        lines: List[str] = [f"Memory recall for: \"{result.query}\"", ""]

        # Vector results
        total_vector = sum(len(v) for v in result.vector_results.values())
        if total_vector:
            lines.append(f"── Vector Memory ({total_vector} results) ──")
            for coll_name, results in result.vector_results.items():
                if results:
                    lines.append(f"  [{coll_name}]")
                    for r in results:
                        text_preview = r.text[:120].replace("\n", " ")
                        lines.append(f"    • (score={r.score}) {text_preview}")
            lines.append("")

        # Graph entities
        if result.graph_entities:
            lines.append(f"── Knowledge Graph ({len(result.graph_entities)} entities) ──")
            for entity in result.graph_entities:
                attrs = ", ".join(f"{k}={v}" for k, v in entity.attributes.items())
                attrs_str = f" ({attrs})" if attrs else ""
                lines.append(f"  • {entity.name} [{entity.entity_type}]{attrs_str}")
            lines.append("")

        # Graph relations
        if result.graph_relations:
            lines.append(f"── Relations ({len(result.graph_relations)}) ──")
            for rel in result.graph_relations:
                lines.append(
                    f"  • {rel.subject_id[:8]}… —[{rel.predicate}]→ "
                    f"{rel.object_id[:8]}… (conf={rel.confidence})"
                )
            lines.append("")

        if total_vector == 0 and not result.graph_entities:
            lines.append("  No relevant memories found.")

        return "\n".join(lines)
