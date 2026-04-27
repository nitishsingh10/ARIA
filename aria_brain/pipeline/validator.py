"""
Dataset Validator
==================
Validates every training example against the RouterOutput schema
and content-quality rules.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from aria_brain.schema.examples import TrainingExample, Dataset
from aria_brain.schema.router_output import (
    RouterOutput, CAPABILITY_SET, ACTION_VALUES,
    COMPLEXITY_VALUES, INTENT_TYPE_VALUES,
)

# ── PII detection patterns ──────────────────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_SSN_RE   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# ── Required params per capability ──────────────────────────────────
_REQUIRED_PARAMS: dict[str, list[str]] = {
    "read_file": ["path"], "write_file": ["path"],
    "append_file": ["path"], "delete_file": ["path"],
    "copy_file": ["source", "dest"], "move_file": ["source", "dest"],
    "search_files": ["pattern"], "file_info": ["path"],
    "list_directory": [], "create_directory": ["directory"],
    "delete_directory": ["directory"], "folder_tree": [],
    "run_command": ["command"], "list_processes": [],
    "get_process": ["pid"], "kill_process": ["pid"],
    "run_python": ["code"], "run_in_docker": ["image"],
    "fetch_url": ["url"], "get_system_info": [],
    "get_current_time": [], "get_env_var": ["var"],
}


@dataclass
class ValidationResult:
    """Result of validating a single example."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetValidationReport:
    """Aggregate validation report for a full dataset."""
    total: int = 0
    valid: int = 0
    invalid: int = 0
    errors: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0


class DatasetValidator:
    """Validates training examples against the RouterOutput schema and quality rules."""

    def validate_example(self, ex: TrainingExample, index: int = 0) -> ValidationResult:
        result = ValidationResult()

        # 1. Input length
        if not ex.input or len(ex.input) < 5:
            result.errors.append("Input too short (< 5 chars)")
            result.is_valid = False
        elif len(ex.input) > 500:
            result.errors.append("Input too long (> 500 chars)")
            result.is_valid = False

        # 2. Output is valid JSON
        try:
            parsed = json.loads(ex.output)
        except (json.JSONDecodeError, TypeError):
            result.errors.append("Output is not valid JSON")
            result.is_valid = False
            return result

        # 3. Output matches RouterOutput schema
        try:
            ro = RouterOutput.model_validate(parsed)
        except Exception as e:
            result.errors.append(f"Output schema validation failed: {e}")
            result.is_valid = False
            return result

        # 4. capability_name check
        cap = ro.routing.capability_name
        if cap is not None and cap not in CAPABILITY_SET:
            result.errors.append(f"Unknown capability_name: {cap}")
            result.is_valid = False

        # 5. Confidence range
        if not (0.0 <= ro.intent.confidence <= 1.0):
            result.errors.append(f"Confidence out of range: {ro.intent.confidence}")
            result.is_valid = False

        # 6. Required parameters
        if cap is not None and cap in _REQUIRED_PARAMS:
            required = _REQUIRED_PARAMS[cap]
            for param in required:
                if param not in ro.routing.parameters:
                    result.warnings.append(f"Missing recommended param '{param}' for {cap}")

        # 7. Reasoning length
        if len(ro.reasoning) > 100:
            result.warnings.append("Reasoning exceeds 100 chars")

        # 8. PII check
        combined = ex.input + " " + ex.output
        if _EMAIL_RE.search(combined):
            result.errors.append("Contains email address (PII)")
            result.is_valid = False
        if _PHONE_RE.search(combined):
            result.warnings.append("May contain phone number (PII)")
        if _SSN_RE.search(combined):
            result.errors.append("Contains SSN-like pattern (PII)")
            result.is_valid = False

        # 9. Action ↔ intent_type consistency
        if ro.intent.intent_type == "capability" and cap is None:
            # Multi-step planning can have capability intent but null cap
            if not ro.intent.requires_planning:
                result.warnings.append("capability intent_type but capability_name is null (and not planning)")

        return result

    def validate_dataset(self, dataset: Dataset) -> DatasetValidationReport:
        report = DatasetValidationReport(total=len(dataset.examples))

        for i, ex in enumerate(dataset.examples):
            vr = self.validate_example(ex, index=i)
            if vr.is_valid:
                report.valid += 1
            else:
                report.invalid += 1
                report.errors.append({
                    "example_id": i,
                    "error_message": "; ".join(vr.errors),
                })
            report.warnings.extend(vr.warnings)

        report.quality_score = round(report.valid / report.total, 4) if report.total else 0.0
        return report
