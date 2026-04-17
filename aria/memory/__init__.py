"""
aria.memory — Persistent memory system for ARIA.

Provides three layers of memory:
    - **VectorStore**: ChromaDB-backed semantic similarity search
      across conversations, documents, and facts.
    - **KnowledgeGraph**: JSON-persisted structured entity-relation
      graph with rule-based extraction.
    - **SessionContext**: Ephemeral in-memory conversation tracker.

The **MemoryManager** is the single entry point — never import
the sub-modules directly in application code.

Usage:
    >>> from aria.memory import MemoryManager
    >>> mm = MemoryManager(config.aria, ollama_client)
    >>> await mm.remember("Nitish uses Python and VS Code")
    >>> result = await mm.recall("what tools does the user have")
    >>> print(result.formatted_summary)
"""

from aria.memory.context import ContextMessage, SessionContext
from aria.memory.knowledge_graph import Entity, KnowledgeGraph, Relation
from aria.memory.memory_manager import MemoryManager, RecallResult
from aria.memory.vector_store import MemoryResult, VectorStore

__all__ = [
    # Primary interface
    "MemoryManager",
    "RecallResult",
    # Context
    "SessionContext",
    "ContextMessage",
    # Vector store
    "VectorStore",
    "MemoryResult",
    # Knowledge graph
    "KnowledgeGraph",
    "Entity",
    "Relation",
]
