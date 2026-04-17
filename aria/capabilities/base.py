"""
aria/capabilities/base.py — Capability contract system for ARIA.

Defines the abstract base class that every capability must implement,
along with standardised input/output models and error types.

Every capability is a deterministic, real action — never a hallucination.
The contract ensures consistent validation, logging, timing, and error
handling across all capabilities.

Example:
    >>> class ReadFileInput(CapabilityInput):
    ...     path: str
    ...
    >>> class ReadFileCapability(Capability):
    ...     name = "read_file"
    ...     description = "Read the contents of a file"
    ...     input_schema = ReadFileInput
    ...     tags = ["system", "file", "read"]
    ...     async def execute(self, input_data):
    ...         ...
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from aria.core.logger import get_logger

log = get_logger("capabilities.base")


# ---------------------------------------------------------------------------
# Input / Output contracts
# ---------------------------------------------------------------------------


class CapabilityInput(BaseModel):
    """Base model for all capability inputs.

    Subclass this to define the input schema for a specific capability.
    Pydantic validates all fields on instantiation.
    """

    class Config:
        """Pydantic configuration for capability inputs."""

        extra = "forbid"


class CapabilityOutput(BaseModel):
    """Standardised output returned by every capability execution.

    Attributes:
        success: Whether the capability executed without error.
        data: The result payload (capability-specific).
        error: Human-readable error message, if any.
        execution_time_ms: Wall-clock duration in milliseconds.
        capability_name: The ``name`` of the capability that produced this.
    """

    success: bool = Field(description="Whether the execution succeeded.")
    data: Any = Field(default=None, description="Result payload.")
    error: Optional[str] = Field(
        default=None, description="Error message if success is False."
    )
    execution_time_ms: float = Field(
        default=0.0, description="Execution duration in milliseconds."
    )
    capability_name: str = Field(
        default="", description="Name of the capability that ran."
    )


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class CapabilityError(Exception):
    """Raised when a capability fails at the framework level.

    Carries structured context about the failure for logging and
    debugging.

    Attributes:
        capability_name: Name of the capability that errored.
        input_data: The input dict that was provided.
        original_error: The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        capability_name: str = "",
        input_data: Any = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialise a CapabilityError.

        Args:
            message: Human-readable description of the error.
            capability_name: Name of the failing capability.
            input_data: The raw input dict that caused the failure.
            original_error: The original exception, if applicable.
        """
        super().__init__(message)
        self.capability_name = capability_name
        self.input_data = input_data
        self.original_error = original_error


# ---------------------------------------------------------------------------
# Abstract base capability
# ---------------------------------------------------------------------------


class Capability(ABC):
    """Abstract base class for all ARIA capabilities.

    Every concrete capability **must** define the following class
    attributes and implement the ``execute`` method:

    Class Attributes:
        name: A unique snake_case identifier (e.g. ``"read_file"``).
        description: A natural-language description used by the router
            to decide when to invoke this capability.
        input_schema: The ``CapabilityInput`` subclass that defines
            accepted parameters.
        tags: A list of lowercase tags for filtering/search
            (e.g. ``["system", "file", "read"]``).
        requires_confirmation: If ``True``, destructive operations
            must receive explicit ``confirm=True`` before executing.
        is_sandboxed: If ``True``, the capability runs in an isolated
            environment (subprocess, container, etc.).
    """

    name: str = ""
    description: str = ""
    input_schema: type[CapabilityInput] = CapabilityInput
    tags: list[str] = []
    requires_confirmation: bool = False
    is_sandboxed: bool = False

    @abstractmethod
    async def execute(self, input_data: CapabilityInput) -> CapabilityOutput:
        """Execute the capability with validated input.

        Subclasses must override this method with real, deterministic
        logic.  Never fabricate results.

        Args:
            input_data: Validated input conforming to ``input_schema``.

        Returns:
            A ``CapabilityOutput`` with the result or error details.
        """
        ...

    async def run(self, raw_input: dict[str, Any]) -> CapabilityOutput:
        """Validate input, execute, time, and wrap errors.

        This is the standard entry point called by the registry.
        It handles:
            1. Input validation via the Pydantic ``input_schema``.
            2. Timing the execution.
            3. Catching all exceptions and returning them as
               ``CapabilityOutput(success=False, ...)``.

        Args:
            raw_input: A dictionary of input parameters.

        Returns:
            A ``CapabilityOutput``, always — even on failure.
        """
        start = time.perf_counter()
        try:
            validated = self.input_schema(**raw_input)
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1_000
            log.warning(
                f"Input validation failed for {self.name}",
                data={"error": str(exc)},
            )
            return CapabilityOutput(
                success=False,
                error=f"Input validation error: {exc}",
                execution_time_ms=round(duration, 2),
                capability_name=self.name,
            )

        try:
            result = await self.execute(validated)
            duration = (time.perf_counter() - start) * 1_000
            result.execution_time_ms = round(duration, 2)
            result.capability_name = self.name
            log.info(
                f"Capability executed: {self.name}",
                data={
                    "success": result.success,
                    "duration_ms": result.execution_time_ms,
                },
            )
            return result
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1_000
            log.error(
                f"Capability failed: {self.name}",
                data={"error": str(exc), "duration_ms": round(duration, 2)},
            )
            return CapabilityOutput(
                success=False,
                error=str(exc),
                execution_time_ms=round(duration, 2),
                capability_name=self.name,
            )

    def to_tool_spec(self) -> dict[str, Any]:
        """Return this capability as a JSON-schema tool specification.

        The returned dict follows the standard LLM tool-calling format
        used by OpenAI-compatible APIs and ARIA's own router.

        Returns:
            A dict with ``name``, ``description``, and ``parameters``
            keys, where ``parameters`` is a JSON Schema object.

        Example::

            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": { ... },
                    "required": [ ... ]
                }
            }
        """
        schema = self.input_schema.model_json_schema()
        # Strip pydantic metadata keys that the LLM doesn't need.
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }
