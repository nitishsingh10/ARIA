"""
ARIA Router Output Schema
=========================

This is the EXACT JSON schema the fine-tuned router model must always output.
Every training example uses this schema as the output. This file is the
contract — never deviate from it.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ─── Known capabilities ──────────────────────────────────────────────
KNOWN_CAPABILITIES: list[str] = [
    # FILE
    "read_file", "write_file", "append_file", "delete_file",
    "copy_file", "move_file", "search_files", "file_info",
    # FOLDER
    "list_directory", "create_directory", "delete_directory", "folder_tree",
    # PROCESS
    "run_command", "list_processes", "get_process", "kill_process",
    # EXECUTE
    "run_python", "run_in_docker",
    # EXTERNAL
    "fetch_url", "get_system_info", "get_current_time", "get_env_var",
]

CAPABILITY_SET: set[str] = set(KNOWN_CAPABILITIES)

# ─── Action ↔ Capability mapping ─────────────────────────────────────
ACTION_VALUES = frozenset({
    "read", "write", "append", "delete", "copy", "move", "search",
    "list", "create", "run", "execute", "fetch", "info", "kill",
    "remember", "recall", "explain", "chat", "help",
})

COMPLEXITY_VALUES = frozenset({"simple", "multi_step", "complex"})

INTENT_TYPE_VALUES = frozenset({
    "capability", "conversation", "memory", "system_command",
})


class IntentOutput(BaseModel):
    """Parsed intent from user input."""

    action: str = Field(
        ...,
        description=(
            "One of: read | write | append | delete | copy | move | search | "
            "list | create | run | execute | fetch | info | kill | remember | "
            "recall | explain | chat | help"
        ),
    )
    complexity: str = Field(
        ...,
        description="simple | multi_step | complex",
    )
    requires_planning: bool = Field(
        ...,
        description="True ONLY for multi_step or complex",
    )
    intent_type: str = Field(
        ...,
        description="capability | conversation | memory | system_command",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Router confidence score 0.0–1.0",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ACTION_VALUES:
            raise ValueError(f"Invalid action '{v}'. Must be one of {sorted(ACTION_VALUES)}")
        return v

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in COMPLEXITY_VALUES:
            raise ValueError(f"Invalid complexity '{v}'. Must be one of {sorted(COMPLEXITY_VALUES)}")
        return v

    @field_validator("intent_type")
    @classmethod
    def validate_intent_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in INTENT_TYPE_VALUES:
            raise ValueError(f"Invalid intent_type '{v}'. Must be one of {sorted(INTENT_TYPE_VALUES)}")
        return v


class RoutingOutput(BaseModel):
    """Where to route the request."""

    capability_name: str | None = Field(
        None,
        description=(
            "Exact name from 22 capabilities OR null. "
            "Null means: handle as conversation/memory."
        ),
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Extracted parameters matching the capability's input schema.",
    )
    fallback: str = Field(
        default="conversation",
        description="Safety fallback — always 'conversation'.",
    )

    @field_validator("capability_name")
    @classmethod
    def validate_capability_name(cls, v: str | None) -> str | None:
        if v is not None and v not in CAPABILITY_SET:
            raise ValueError(
                f"Unknown capability '{v}'. Must be one of {sorted(CAPABILITY_SET)} or null."
            )
        return v


class RouterOutput(BaseModel):
    """
    Complete router output schema.

    This is the top-level model that every training example's ``output``
    field must deserialize to.
    """

    intent: IntentOutput
    routing: RoutingOutput
    entities: list[str] = Field(
        default_factory=list,
        description="Extracted nouns: paths, URLs, names, quoted strings, code snippets.",
    )
    reasoning: str = Field(
        ...,
        max_length=100,
        description="ONE sentence: why this routing was chosen. Max 100 chars, no newlines.",
    )

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        v = v.replace("\n", " ").strip()
        if len(v) > 100:
            v = v[:97] + "..."
        return v

    def to_training_output(self) -> str:
        """Return compact JSON string (no indent) for the training ``output`` field."""
        return self.model_dump_json(exclude_none=False)

    @classmethod
    def from_training_output(cls, raw: str) -> "RouterOutput":
        """Parse a compact JSON string back into a RouterOutput."""
        data = json.loads(raw)
        return cls.model_validate(data)
