"""
aria/memory/vector_store.py — ChromaDB-backed semantic vector memory.

Provides persistent long-term memory for ARIA. Stores and retrieves
conversations, documents, and facts using embedding-based similarity
search.

Three collections are maintained:
    - **conversations**: Chat history and summaries.
    - **documents**: Files and content ARIA has read.
    - **facts**: Extracted facts about the user and world.

Embedding strategy:
    - Uses ChromaDB's built-in DefaultEmbeddingFunction for all
      collection operations (reliable, local, no Ollama dependency).
    - This ensures memory always works even when Ollama is offline.

Example:
    >>> from aria.memory.vector_store import VectorStore
    >>> vs = VectorStore(config.aria, ollama_client)
    >>> doc_id = await vs.add_fact("Nitish uses Python", subject="Nitish")
    >>> results = await vs.search("who uses python", collection="facts")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from aria.config import AriaConfig
from aria.core.logger import get_logger
from aria.llm.client import OllamaClient

log = get_logger("memory.vector_store")

# Collection names
_CONVERSATIONS = "conversations"
_DOCUMENTS = "documents"
_FACTS = "facts"
_ALL_COLLECTIONS = [_CONVERSATIONS, _DOCUMENTS, _FACTS]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryResult:
    """A single result from a vector similarity search.

    Attributes:
        doc_id: The unique document identifier.
        text: The stored text content.
        metadata: Associated metadata dict.
        score: Cosine similarity score (0–1, higher is better).
        collection: The collection this result came from.
    """

    doc_id: str
    text: str
    metadata: dict
    score: float
    collection: str


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """ChromaDB-backed persistent semantic memory store.

    Manages three collections (conversations, documents, facts) and
    provides add/search/delete/recent operations with ChromaDB's
    built-in embedding function for maximum reliability.

    Args:
        config: The ARIA config section (``AriaConfig``).
        ollama_client: An initialised ``OllamaClient`` (stored for
            future use but not used for embeddings currently).
    """

    def __init__(self, config: AriaConfig, ollama_client: OllamaClient) -> None:
        """Initialise the vector store.

        Creates the ChromaDB persistent client and ensures all three
        collections exist.

        Args:
            config: ARIA runtime config.
            ollama_client: Ollama client reference (stored for future use).
        """
        self._ollama = ollama_client
        chroma_path = Path(config.memory_dir).expanduser() / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        try:
            self._chroma = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )
        except Exception as exc:
            log.error(
                "Failed to initialise ChromaDB",
                data={"error": str(exc), "path": str(chroma_path)},
            )
            raise

        # Ensure collections exist (using ChromaDB's built-in embedding)
        self._collections: Dict[str, Any] = {}
        for name in _ALL_COLLECTIONS:
            try:
                self._collections[name] = self._chroma.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                log.error(
                    f"Failed to create collection '{name}'",
                    data={"error": str(exc)},
                )

        log.info(
            "VectorStore initialised",
            data={"path": str(chroma_path), "collections": list(self._collections.keys())},
        )

    # ------------------------------------------------------------------
    # Add methods
    # ------------------------------------------------------------------

    async def add_conversation(
        self,
        text: str,
        metadata: dict,
        doc_id: Optional[str] = None,
    ) -> str:
        """Store a conversation message or summary.

        Args:
            text: The conversation text.
            metadata: Must include ``role``, ``timestamp``, ``session_id``.
            doc_id: Optional explicit ID (auto-generated UUID if None).

        Returns:
            The document ID.
        """
        doc_id = doc_id or str(uuid.uuid4())
        metadata = self._sanitize_metadata(metadata)
        try:
            self._collections[_CONVERSATIONS].add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
            )
            log.debug("Added conversation memory", data={"doc_id": doc_id})
        except Exception as exc:
            log.error("Failed to add conversation", data={"error": str(exc)})
        return doc_id

    async def add_document(
        self,
        content: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a document's content for later retrieval.

        Args:
            content: The document text.
            source: File path or URL the content came from.
            metadata: Additional metadata.

        Returns:
            The document ID.
        """
        doc_id = str(uuid.uuid4())
        meta = self._sanitize_metadata(metadata or {})
        meta["source"] = source
        meta["added_at"] = datetime.now(timezone.utc).isoformat()
        try:
            self._collections[_DOCUMENTS].add(
                documents=[content],
                metadatas=[meta],
                ids=[doc_id],
            )
            log.debug("Added document memory", data={"doc_id": doc_id, "source": source})
        except Exception as exc:
            log.error("Failed to add document", data={"error": str(exc)})
        return doc_id

    async def add_fact(
        self,
        fact: str,
        subject: str,
        confidence: float = 1.0,
        source: Optional[str] = None,
    ) -> str:
        """Store an extracted fact.

        Args:
            fact: The fact text.
            subject: Who or what the fact is about.
            confidence: Confidence score (0–1).
            source: Where the fact was learned from.

        Returns:
            The document ID.
        """
        doc_id = str(uuid.uuid4())
        meta: dict = {
            "subject": subject,
            "confidence": confidence,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        if source:
            meta["source"] = source
        try:
            self._collections[_FACTS].add(
                documents=[fact],
                metadatas=[meta],
                ids=[doc_id],
            )
            log.debug("Added fact", data={"doc_id": doc_id, "subject": subject})
        except Exception as exc:
            log.error("Failed to add fact", data={"error": str(exc)})
        return doc_id

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        collection: str = _CONVERSATIONS,
        top_k: int = 8,
        min_score: float = 0.2,
        metadata_filter: Optional[dict] = None,
    ) -> List[MemoryResult]:
        """Semantic search within a single collection.

        Args:
            query: The search query text.
            collection: Collection name to search.
            top_k: Maximum number of results.
            min_score: Minimum cosine similarity (0–1).
            metadata_filter: Optional ChromaDB where filter.

        Returns:
            A list of ``MemoryResult`` ordered by score (descending).
        """
        if collection not in self._collections:
            log.warning(f"Unknown collection: {collection}")
            return []

        try:
            coll = self._collections[collection]

            # Don't query empty collections
            if coll.count() == 0:
                return []

            kwargs: dict = {
                "query_texts": [query],
                "n_results": min(top_k, coll.count()),
            }
            if metadata_filter:
                kwargs["where"] = metadata_filter

            results = coll.query(**kwargs)
            return self._parse_results(results, collection, min_score)
        except Exception as exc:
            log.error(
                f"Search failed in '{collection}'",
                data={"error": str(exc)},
            )
            return []

    async def search_all(
        self,
        query: str,
        top_k: int = 8,
        min_score: float = 0.2,
    ) -> Dict[str, List[MemoryResult]]:
        """Search across all three collections.

        Args:
            query: The search query text.
            top_k: Max results per collection.

        Returns:
            A dict mapping collection name to result lists.
        """
        grouped: Dict[str, List[MemoryResult]] = {}
        for name in _ALL_COLLECTIONS:
            grouped[name] = await self.search(
                query, collection=name, top_k=top_k, min_score=min_score
            )
        return grouped

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete(self, doc_id: str, collection: str) -> bool:
        """Delete a document by ID from a collection.

        Args:
            doc_id: The document identifier.
            collection: The collection to delete from.

        Returns:
            True if deletion succeeded.
        """
        if collection not in self._collections:
            return False
        try:
            self._collections[collection].delete(ids=[doc_id])
            log.debug("Deleted memory", data={"doc_id": doc_id, "collection": collection})
            return True
        except Exception as exc:
            log.error("Delete failed", data={"error": str(exc)})
            return False

    # ------------------------------------------------------------------
    # Recent
    # ------------------------------------------------------------------

    async def get_recent(
        self,
        collection: str = _CONVERSATIONS,
        limit: int = 20,
        session_id: Optional[str] = None,
    ) -> List[MemoryResult]:
        """Retrieve the most recently added items from a collection.

        Args:
            collection: Collection name.
            limit: Maximum items to return.
            session_id: If provided, filter to this session only.

        Returns:
            A list of ``MemoryResult`` sorted by recency.
        """
        if collection not in self._collections:
            return []

        try:
            coll = self._collections[collection]
            kwargs: dict = {"limit": limit}
            if session_id:
                kwargs["where"] = {"session_id": session_id}

            # ChromaDB .get() returns all matching, ordered by insertion
            raw = coll.get(**kwargs)
            results: List[MemoryResult] = []
            if raw and raw.get("ids"):
                ids = raw["ids"]
                docs = raw.get("documents", [])
                metas = raw.get("metadatas", [])
                for i, doc_id in enumerate(ids):
                    results.append(
                        MemoryResult(
                            doc_id=doc_id,
                            text=docs[i] if i < len(docs) else "",
                            metadata=metas[i] if i < len(metas) else {},
                            score=1.0,
                            collection=collection,
                        )
                    )
            return results
        except Exception as exc:
            log.error("get_recent failed", data={"error": str(exc)})
            return []

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return item counts for all collections.

        Returns:
            A dict mapping collection name to document count.
        """
        counts: dict = {}
        for name, coll in self._collections.items():
            try:
                counts[name] = coll.count()
            except Exception:
                counts[name] = -1
        return counts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """Ensure all metadata values are ChromaDB-compatible types.

        ChromaDB only accepts str, int, float, bool metadata values.

        Args:
            meta: Raw metadata dict.

        Returns:
            Sanitized metadata with values coerced to str where needed.
        """
        clean: dict = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                continue  # skip None values
            else:
                clean[k] = str(v)
        return clean

    @staticmethod
    def _parse_results(
        raw: dict,
        collection: str,
        min_score: float,
    ) -> List[MemoryResult]:
        """Parse ChromaDB query results into MemoryResult objects.

        Args:
            raw: The raw ChromaDB query response.
            collection: The collection name.
            min_score: Minimum score filter.

        Returns:
            Filtered and sorted list of MemoryResult.
        """
        results: List[MemoryResult] = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: score = 1 - (distance / 2)
            distance = distances[i] if i < len(distances) else 1.0
            score = 1.0 - (distance / 2.0)

            if score < min_score:
                continue

            results.append(
                MemoryResult(
                    doc_id=doc_id,
                    text=docs[i] if i < len(docs) else "",
                    metadata=metas[i] if i < len(metas) else {},
                    score=round(score, 4),
                    collection=collection,
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)
