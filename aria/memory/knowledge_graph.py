"""
aria/memory/knowledge_graph.py — Personal knowledge graph for ARIA.

Stores structured facts about the user, their projects, preferences,
and relationships between entities. Persists as a simple JSON file
(no graph DB dependency).

The graph supports rule-based entity extraction from natural language
text — no LLM required.

Example:
    >>> kg = KnowledgeGraph(config.aria)
    >>> kg.add_entity("Nitish", "person", {"role": "developer"})
    >>> kg.add_relation(user_id, "works_on", project_id)
    >>> kg.extract_and_store("I use VS Code and prefer dark themes")
    >>> profile = kg.get_user_profile()
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aria.config import AriaConfig
from aria.core.logger import get_logger

log = get_logger("memory.knowledge_graph")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A node in the knowledge graph.

    Attributes:
        id: Unique identifier (UUID).
        name: Display name (e.g. ``"Nitish"``, ``"ARIA project"``).
        entity_type: Category — ``"person"``, ``"project"``, ``"file"``,
            ``"concept"``, ``"tool"``, ``"preference"``.
        attributes: Free-form key-value attributes.
        created_at: When the entity was first added (UTC ISO string).
        updated_at: When the entity was last modified (UTC ISO string).
    """

    id: str = ""
    name: str = ""
    entity_type: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Relation:
    """An edge in the knowledge graph connecting two entities.

    Attributes:
        id: Unique identifier (UUID).
        subject_id: The source entity's ID.
        predicate: Relationship type — ``"works_on"``, ``"prefers"``,
            ``"owns"``, ``"located_at"``, ``"uses"``, ``"knows"``.
        object_id: The target entity's ID or a plain string value.
        confidence: Confidence score (0–1).
        created_at: When the relation was created (UTC ISO string).
    """

    id: str = ""
    subject_id: str = ""
    predicate: str = ""
    object_id: str = ""
    confidence: float = 1.0
    created_at: str = ""


# ---------------------------------------------------------------------------
# Extraction patterns (rule-based, no LLM)
# ---------------------------------------------------------------------------

# Each pattern yields: (regex pattern, entity_type, predicate)
_STOP_WORDS = {"and", "i", "who", "am", "is", "from", "in", "at", "the", "a", "an",
                "but", "or", "so", "my", "we", "he", "she", "it", "that", "this",
                "im", "iam", "currently", "also", "here"}
_NAME_PATTERNS = [
    (r"(?:my name is|i(?:'| a)m called|call me)\s+([a-z]+(?:\s+[a-z]+)?)", "person", "knows"),
]
_USE_PATTERNS = [
    (r"(?:i use|i(?:'| a)m using|i work with)\s+(.+?)(?:\s+(?:for|to|and|$))", "tool", "uses"),
]
_PREFER_PATTERNS = [
    (r"(?:i prefer|i like|i love|my (?:fav(?:ou?rite)?) is)\s+(.+?)(?:\.|,|$)", "preference", "prefers"),
]
_PROJECT_PATTERNS = [
    (r"(?:i(?:'| a)m (?:working on|building|developing|creating))\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3}?)(?:\s+(?:in|with|using|and|for|on)|[.,;!]|$)", "project", "works_on"),
]
_ATTRIBUTE_PATTERNS = [
    (r"my\s+(\w+)\s+is\s+(.+?)(?:\.|,|$)", None, None),  # "my X is Y" → attribute
]


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """JSON-persisted personal knowledge graph.

    Stores entities and relations as in-memory dicts/lists, serialised
    to a JSON file on ``save()``.

    Args:
        config: The ``AriaConfig`` section with ``data_dir``.
    """

    def __init__(self, config: AriaConfig) -> None:
        """Initialise the knowledge graph.

        Loads existing data from disk or creates an empty graph.

        Args:
            config: ARIA runtime config.
        """
        data_dir = Path(config.data_dir).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "knowledge_graph.json"

        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # Ensure a "user" entity always exists (ARIA's owner)
        self._user_id: str = ""
        self.load()

        if not self._user_id:
            user = self.add_entity("user", "person", {"role": "owner"})
            self._user_id = user.id
            self.save()

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Add a new entity, de-duplicating by name + type.

        If an entity with the same name and type already exists, the
        existing one is returned (with attributes merged).

        Args:
            name: Display name.
            entity_type: Entity category.
            attributes: Optional key-value attributes.

        Returns:
            The new or existing ``Entity``.
        """
        existing = self.find_entity(name, entity_type)
        if existing:
            if attributes:
                existing.attributes.update(attributes)
                existing.updated_at = datetime.now(timezone.utc).isoformat()
            return existing

        now = datetime.now(timezone.utc).isoformat()
        entity = Entity(
            id=str(uuid.uuid4()),
            name=name.strip(),
            entity_type=entity_type,
            attributes=attributes or {},
            created_at=now,
            updated_at=now,
        )
        self.entities[entity.id] = entity
        log.debug(
            f"Added entity: {name} ({entity_type})",
            data={"entity_id": entity.id},
        )
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID.

        Args:
            entity_id: The entity's UUID.

        Returns:
            The ``Entity`` or None.
        """
        return self.entities.get(entity_id)

    def find_entity(
        self, name: str, entity_type: Optional[str] = None
    ) -> Optional[Entity]:
        """Find an entity by name and optional type.

        Args:
            name: Entity name (case-insensitive).
            entity_type: Optional type filter.

        Returns:
            The first matching ``Entity``, or None.
        """
        name_lower = name.strip().lower()
        for entity in self.entities.values():
            if entity.name.lower() == name_lower:
                if entity_type is None or entity.entity_type == entity_type:
                    return entity
        return None

    def update_entity(
        self, entity_id: str, attributes: Dict[str, Any]
    ) -> Optional[Entity]:
        """Merge new attributes into an existing entity.

        Args:
            entity_id: The entity's UUID.
            attributes: Key-value pairs to merge.

        Returns:
            The updated ``Entity``, or None if not found.
        """
        entity = self.entities.get(entity_id)
        if entity is None:
            return None
        entity.attributes.update(attributes)
        entity.updated_at = datetime.now(timezone.utc).isoformat()
        return entity

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float = 1.0,
    ) -> Relation:
        """Add a new relation between two entities.

        Args:
            subject_id: Source entity ID.
            predicate: Relationship type.
            object_id: Target entity ID (or plain string value).
            confidence: Confidence score (0–1).

        Returns:
            The new ``Relation``.
        """
        # Check for duplicate
        for rel in self.relations:
            if (
                rel.subject_id == subject_id
                and rel.predicate == predicate
                and rel.object_id == object_id
            ):
                rel.confidence = max(rel.confidence, confidence)
                return rel

        relation = Relation(
            id=str(uuid.uuid4()),
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.relations.append(relation)
        log.debug(
            f"Added relation: {predicate}",
            data={"subject": subject_id, "object": object_id},
        )
        return relation

    def get_relations(
        self,
        entity_id: str,
        predicate: Optional[str] = None,
    ) -> List[Relation]:
        """Get all relations involving an entity (as subject or object).

        Args:
            entity_id: The entity's UUID.
            predicate: Optional predicate filter.

        Returns:
            Matching relations.
        """
        results: List[Relation] = []
        for rel in self.relations:
            if rel.subject_id != entity_id and rel.object_id != entity_id:
                continue
            if predicate and rel.predicate != predicate:
                continue
            results.append(rel)
        return results

    # ------------------------------------------------------------------
    # Entity extraction (rule-based)
    # ------------------------------------------------------------------

    def extract_and_store(self, text: str) -> dict:
        """Extract entities and relations from text using rules.

        Detects patterns like:
            - ``"my name is X"`` → person entity + knows relation
            - ``"I use X"``      → tool entity + uses relation
            - ``"I prefer X"``   → preference entity + prefers relation
            - ``"I am working on X"`` → project entity + works_on relation
            - ``"my X is Y"``    → attribute update on the user entity

        Args:
            text: A natural-language string to extract from.

        Returns:
            A dict ``{"entities_added": N, "relations_added": N}``.
        """
        entities_added = 0
        relations_added = 0
        text_lower = text.lower()

        # --- Name detection ---
        for pattern, etype, pred in _NAME_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                raw_name = match.group(1).strip()
                # Strip trailing stop words: "nitish and" → "nitish"
                words = raw_name.split()
                while words and words[-1] in _STOP_WORDS:
                    words.pop()
                if not words:
                    continue
                name = " ".join(words).title()
                entity = self.add_entity(name, etype)
                if entity.created_at == entity.updated_at:
                    entities_added += 1
                # Update the user entity name
                user = self.entities.get(self._user_id)
                if user and user.name == "user":
                    user.name = name
                    user.updated_at = datetime.now(timezone.utc).isoformat()
                self.add_relation(self._user_id, pred, entity.id)
                relations_added += 1

        # --- Tool usage ---
        for pattern, etype, pred in _USE_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                items = [s.strip() for s in match.group(1).split(",")]
                for item in items:
                    item = item.strip().rstrip(".")
                    if len(item) < 2 or len(item) > 50:
                        continue
                    entity = self.add_entity(item.title(), etype)
                    if entity.created_at == entity.updated_at:
                        entities_added += 1
                    self.add_relation(self._user_id, pred, entity.id)
                    relations_added += 1

        # --- Preferences ---
        for pattern, etype, pred in _PREFER_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                item = match.group(1).strip().rstrip(".")
                if len(item) < 2 or len(item) > 50:
                    continue
                entity = self.add_entity(item.title(), etype)
                if entity.created_at == entity.updated_at:
                    entities_added += 1
                self.add_relation(self._user_id, pred, entity.id)
                relations_added += 1

        # --- Projects ---
        for pattern, etype, pred in _PROJECT_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                item = match.group(1).strip().rstrip(".")
                if len(item) < 2 or len(item) > 80:
                    continue
                entity = self.add_entity(item.title(), etype)
                if entity.created_at == entity.updated_at:
                    entities_added += 1
                self.add_relation(self._user_id, pred, entity.id)
                relations_added += 1

        # --- Attribute detection: "my X is Y" ---
        _SKIP_ATTR_KEYS = {"name", "names"}
        for pattern, _, _ in _ATTRIBUTE_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                key = match.group(1).strip()
                value = match.group(2).strip().rstrip(".")
                if key and value and len(key) < 30 and key not in _SKIP_ATTR_KEYS:
                    user = self.entities.get(self._user_id)
                    if user:
                        user.attributes[key] = value
                        user.updated_at = datetime.now(timezone.utc).isoformat()

        if entities_added > 0 or relations_added > 0:
            self.save()

        return {
            "entities_added": entities_added,
            "relations_added": relations_added,
        }

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[Relation]:
        """Flexible triple-pattern query; any field can be None = wildcard.

        Args:
            subject: Subject entity ID filter (or None).
            predicate: Predicate filter (or None).
            obj: Object entity ID filter (or None).

        Returns:
            All matching relations.
        """
        results: List[Relation] = []
        for rel in self.relations:
            if subject and rel.subject_id != subject:
                continue
            if predicate and rel.predicate != predicate:
                continue
            if obj and rel.object_id != obj:
                continue
            results.append(rel)
        return results

    def get_user_profile(self) -> dict:
        """Build a summary of everything known about the user.

        Returns:
            A dict with name, preferences, projects, tools, and
            raw facts (attributes).
        """
        user = self.entities.get(self._user_id)
        if not user:
            return {"name": "unknown", "preferences": [], "projects": [], "tools": [], "facts": {}}

        prefs: List[str] = []
        projects: List[str] = []
        tools: List[str] = []

        for rel in self.relations:
            if rel.subject_id != self._user_id:
                continue
            target = self.entities.get(rel.object_id)
            target_name = target.name if target else rel.object_id

            if rel.predicate == "prefers":
                prefs.append(target_name)
            elif rel.predicate == "works_on":
                projects.append(target_name)
            elif rel.predicate == "uses":
                tools.append(target_name)

        return {
            "name": user.name,
            "preferences": prefs,
            "projects": projects,
            "tools": tools,
            "facts": dict(user.attributes),
        }

    @property
    def user_id(self) -> str:
        """The entity ID of the user (ARIA's owner).

        Returns:
            The user entity's UUID.
        """
        return self._user_id

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Serialise the graph to a JSON file on disk."""
        data = {
            "entities": {eid: asdict(e) for eid, e in self.entities.items()},
            "relations": [asdict(r) for r in self.relations],
            "user_id": self._user_id,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            log.debug("Knowledge graph saved", data={"path": str(self._path)})
        except Exception as exc:
            log.error("Failed to save knowledge graph", data={"error": str(exc)})

    def load(self) -> None:
        """Load the graph from a JSON file on disk.

        If the file is missing or corrupt, reset to an empty graph
        with a warning.
        """
        if not self._path.exists():
            log.info("No existing knowledge graph found, starting fresh")
            return

        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # Restore entities
            for eid, edata in data.get("entities", {}).items():
                self.entities[eid] = Entity(**edata)

            # Restore relations
            for rdata in data.get("relations", []):
                self.relations.append(Relation(**rdata))

            self._user_id = data.get("user_id", "")

            log.info(
                "Knowledge graph loaded",
                data={
                    "entities": len(self.entities),
                    "relations": len(self.relations),
                },
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning(
                "Knowledge graph file is corrupt, resetting to empty",
                data={"error": str(exc), "path": str(self._path)},
            )
            self.entities = {}
            self.relations = []
            self._user_id = ""
