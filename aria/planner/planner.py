"""
aria/planner/planner.py
"""
import json
from typing import Any

from aria.cognitive.intent import Intent
from aria.cognitive.router import RouterResult
from aria.core.logger import get_logger
from aria.planner.step import Plan, PlanStep
from aria.planner.templates import TemplateRegistry


class Planner:
    def __init__(self, config: Any, ollama_client: Any, registry: Any, memory_manager: Any):
        self.config = config
        self.ollama_client = ollama_client
        self.registry = registry
        self.memory_manager = memory_manager
        self.templates = TemplateRegistry()
        self.logger = get_logger("planner")

    async def create_plan(self, intent: Intent, router_result: RouterResult) -> Plan:
        # STEP 1: Try template matching
        template = self.templates.match(intent, intent.raw_text)
        if template:
            self.logger.info(f"Template matched: {template.name}")
            steps = template.build(intent.raw_text, intent, {})
            return Plan(
                goal=intent.raw_text,
                steps=steps,
                requires_confirmation=self._check_confirmation(steps)
            )

        # STEP 2: Check if planning is needed
        if not intent.requires_planning and not router_result.requires_planning:
            self.logger.info("Fast path triggered, no planning required.")
            step = PlanStep(
                step_number=1,
                description=intent.raw_text,
                capability_name=router_result.capability_name,
                parameters=router_result.parameters,
            )
            return Plan(
                goal=intent.raw_text,
                steps=[step],
                requires_confirmation=self._check_confirmation([step])
            )

        # STEP 3: LLM planning (for complex/unknown tasks)
        self.logger.info("Falling back to LLM for planning.")
        memory_context = await self.memory_manager.recall_for_prompt(intent.raw_text)

        system_prompt = f"""You are a task planner for ARIA. Break the user's goal into concrete steps using available tools.
Return ONLY a JSON array of steps, no explanation.

Available tools:
{json.dumps(self.registry.to_tool_specs(), indent=2)}

Rules:
- Maximum 8 steps per plan
- Each step uses exactly one tool OR is a "reasoning" step (capability_name: null)
- Reasoning steps use the LLM to transform or analyze data from previous steps
- Use output_key to pass data between steps
- Mark steps that can fail gracefully as is_optional: true

Return format (JSON array):
[
  {{
    "step_number": 1,
    "description": "...",
    "capability_name": "read_file",
    "parameters": {{"path": "..."}},
    "depends_on": [],
    "is_optional": false,
    "on_failure": "stop",
    "output_key": "file_content",
    "timeout": 30
  }}
]"""
        messages = [{"role": "user", "content": f"Goal: {intent.raw_text}\nContext: {memory_context}"}]

        response = await self.ollama_client.chat(messages=messages, system=system_prompt)

        steps = []
        try:
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]

            raw_steps = json.loads(text.strip())
            for idx, rs in enumerate(raw_steps):
                if idx >= 8:
                    break

                cap_name = rs.get("capability_name")
                if cap_name and not self.registry.search(cap_name):
                    cap_name = None

                step = PlanStep(
                    step_number=rs.get("step_number", idx + 1),
                    description=rs.get("description", "Reasoning step"),
                    capability_name=cap_name,
                    parameters=rs.get("parameters", {}),
                    depends_on=rs.get("depends_on", []),
                    is_optional=rs.get("is_optional", False),
                    on_failure=rs.get("on_failure", "stop"),
                    output_key=rs.get("output_key"),
                    timeout=rs.get("timeout", 30)
                )
                steps.append(step)

            if not steps:
                raise ValueError("Parsed JSON resulted in 0 steps.")

        except Exception as e:
            self.logger.warning(f"Failed to parse LLM plan: {e}. Falling back to 1-step plan.")
            step = PlanStep(
                step_number=1,
                description=intent.raw_text,
                capability_name=router_result.capability_name,
                parameters=router_result.parameters,
            )
            steps = [step]

        return Plan(
            goal=intent.raw_text,
            steps=steps,
            requires_confirmation=self._check_confirmation(steps)
        )

    def _check_confirmation(self, steps: list[PlanStep]) -> bool:
        for s in steps:
            if s.capability_name:
                results = self.registry.search(s.capability_name)
                # Ensure exact match in case of overlapping names
                cap = next((c for c in results if c.name == s.capability_name), None)
                if cap and getattr(cap, "requires_confirmation", False):
                    self.logger.warning(f"Plan requires confirmation due to capability '{s.capability_name}'")
                    return True
        return False

    def explain_plan(self, plan: Plan) -> str:
        s = plan.estimated_duration_s
        return f"I'll accomplish this in {len(plan.steps)} steps:\n(Estimated time: ~{s}s)\n\n" + plan.summary()
