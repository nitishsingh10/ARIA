"""
aria/planner/step.py
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    step_number: int
    description: str
    capability_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    is_optional: bool = False
    timeout: int = 30
    on_failure: str = "stop"
    retry_count: int = 0
    output_key: Optional[str] = None
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class StepResult:
    step_id: str
    step_number: int
    capability_name: Optional[str]
    success: bool
    output: Any
    error: Optional[str]
    duration_ms: float
    skipped: bool = False


@dataclass
class Plan:
    goal: str
    steps: List[PlanStep]
    requires_confirmation: bool
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def estimated_duration_s(self) -> int:
        return sum(s.timeout for s in self.steps)

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def summary(self) -> str:
        lines = [
            f"Plan: {self.goal}",
            f"Steps: {len(self.steps)}"
        ]
        for step in sorted(self.steps, key=lambda x: x.step_number):
            cap = f"[{step.capability_name}]" if step.capability_name else "[reasoning]"
            lines.append(f"{step.step_number}. {cap} {step.description}")
        return "\n".join(lines)
