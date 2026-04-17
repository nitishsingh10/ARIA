"""
aria/capabilities/registry.py — Central capability registry for ARIA.

The ``CapabilityRegistry`` is the single authoritative store through
which all capabilities are registered, discovered, and invoked at
runtime.  Never call ``Capability.execute()`` directly — always go
through the registry.

Usage:
    >>> from aria.capabilities.registry import GLOBAL_REGISTRY
    >>> GLOBAL_REGISTRY.register(my_capability)
    >>> result = await GLOBAL_REGISTRY.execute("read_file", {"path": "/tmp/x"})
"""

from __future__ import annotations

from typing import Any

from aria.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityOutput,
)
from aria.core.logger import get_logger

log = get_logger("capabilities.registry")


class CapabilityRegistry:
    """Central store for all registered ARIA capabilities.

    Provides registration, lookup (by name or tag), keyword search,
    LLM tool-spec generation, and a unified ``execute`` entry point.
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._capabilities: dict[str, Capability] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, capability: Capability) -> None:
        """Register a capability instance.

        Args:
            capability: A fully-initialised ``Capability`` subclass.

        Raises:
            ValueError: If a capability with the same name is already
                registered.
        """
        if capability.name in self._capabilities:
            raise ValueError(
                f"Capability '{capability.name}' is already registered."
            )
        self._capabilities[capability.name] = capability
        log.debug(
            f"Registered capability: {capability.name}",
            data={"tags": capability.tags},
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Capability | None:
        """Retrieve a capability by exact name.

        Args:
            name: The snake_case capability name.

        Returns:
            The matching ``Capability``, or ``None``.
        """
        return self._capabilities.get(name)

    def list_all(self) -> list[Capability]:
        """Return all registered capabilities.

        Returns:
            A list of ``Capability`` instances, sorted by name.
        """
        return sorted(self._capabilities.values(), key=lambda c: c.name)

    def list_by_tag(self, tag: str) -> list[Capability]:
        """Return capabilities that carry a specific tag.

        Args:
            tag: The tag to filter by (case-insensitive).

        Returns:
            A list of matching capabilities, sorted by name.
        """
        tag_lower = tag.lower()
        return sorted(
            [c for c in self._capabilities.values() if tag_lower in c.tags],
            key=lambda c: c.name,
        )

    def search(self, query: str) -> list[Capability]:
        """Simple keyword search across name, description, and tags.

        Args:
            query: One or more search terms (space-separated).

        Returns:
            Capabilities whose name, description, or tags contain
            **any** of the query terms, sorted by name.
        """
        terms = query.lower().split()
        results: list[Capability] = []
        for cap in self._capabilities.values():
            searchable = (
                f"{cap.name} {cap.description} {' '.join(cap.tags)}"
            ).lower()
            if any(t in searchable for t in terms):
                results.append(cap)
        return sorted(results, key=lambda c: c.name)

    # ------------------------------------------------------------------
    # Tool specs for LLM
    # ------------------------------------------------------------------

    def to_tool_specs(self) -> list[dict[str, Any]]:
        """Generate LLM tool-calling specs for every registered capability.

        Returns:
            A list of JSON-schema dicts suitable for the ``tools``
            parameter of an LLM chat request.
        """
        return [c.to_tool_spec() for c in self.list_all()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self, name: str, input_data: dict[str, Any]
    ) -> CapabilityOutput:
        """Validate and execute a capability by name.

        This is the **only** sanctioned way to run a capability at
        runtime.

        Args:
            name: The snake_case capability name.
            input_data: A dict of input parameters.

        Returns:
            A ``CapabilityOutput`` with results or error info.

        Raises:
            CapabilityError: If no capability with ``name`` exists.
        """
        cap = self.get(name)
        if cap is None:
            raise CapabilityError(
                message=f"Unknown capability: '{name}'",
                capability_name=name,
                input_data=input_data,
            )

        log.info(
            f"Executing capability: {name}",
            data={"input_keys": list(input_data.keys())},
        )
        return await cap.run(input_data)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of registered capabilities."""
        return len(self._capabilities)

    def __contains__(self, name: str) -> bool:
        """Check if a capability name is registered."""
        return name in self._capabilities

    def __repr__(self) -> str:
        """Return a dev-friendly representation."""
        return (
            f"<CapabilityRegistry capabilities={len(self._capabilities)}>"
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
GLOBAL_REGISTRY = CapabilityRegistry()
