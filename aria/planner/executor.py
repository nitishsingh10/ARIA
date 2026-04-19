"""
aria/planner/executor.py
"""
import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from aria.core.logger import get_logger
from aria.planner.step import Plan, PlanStep, StepResult


@dataclass
class ExecutionResult:
    plan_id: str
    goal: str
    success: bool
    results: List[StepResult]
    context: Dict[str, Any]
    final_output: Any
    duration_ms: float
    failed_at_step: Optional[int] = None

    def summary(self) -> str:
        s_count = len(self.results)
        ans = str(self.final_output)[:200] if self.final_output else ""
        return f"Completed {s_count} steps in {self.duration_ms:.1f}ms\nResult: {ans}"


class PlanExecutor:
    def __init__(self, config: Any, ollama_client: Any, registry: Any, memory_manager: Any):
        self.config = config
        self.ollama_client = ollama_client
        self.registry = registry
        self.memory_manager = memory_manager
        self.logger = get_logger("executor")

    def _resolve_placeholders(self, data: Any, context: Dict[str, Any]) -> Any:
        if isinstance(data, str):
            import re
            def get_val(key_path: str) -> Any:
                parts = key_path.split(".")
                v = context.get(parts[0], f"{{{key_path}}}")
                if v == f"{{{key_path}}}":
                    return v
                for part in parts[1:]:
                    if isinstance(v, dict):
                        v = v.get(part, f"{{{key_path}}}")
                    elif hasattr(v, part):
                        v = getattr(v, part, f"{{{key_path}}}")
                    else:
                        return f"{{{key_path}}}"
                return v

            # If it's a single exact match, we can return the raw object (e.g. dict)
            exact_match = re.fullmatch(r"\{([a-zA-Z0-9_.]+)\}", data.strip())
            if exact_match:
                # Need to return the raw object
                res = get_val(exact_match.group(1))
                if str(res) != f"{{{exact_match.group(1)}}}":
                    return res
                
            # Otherwise we replace occurrences within a string
            def replacer(match: re.Match) -> str:
                res = get_val(match.group(1))
                return str(res)
                
            return re.sub(r"\{([a-zA-Z0-9_.]+)\}", replacer, data)
        elif isinstance(data, dict):
            return {k: self._resolve_placeholders(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_placeholders(item, context) for item in data]
        return data

    async def execute(self, plan: Plan, on_step_complete: Optional[Callable[[PlanStep, StepResult], Any]] = None) -> ExecutionResult:
        context: Dict[str, Any] = {}
        results: List[StepResult] = []
        start_time = asyncio.get_event_loop().time()

        for step in sorted(plan.steps, key=lambda x: x.step_number):
            can_run = True
            for dep_id in step.depends_on:
                dep_result = next((r for r in results if r.step_id == dep_id), None)
                if dep_result and not dep_result.success:
                    if step.is_optional:
                        results.append(StepResult(
                            step_id=step.step_id,
                            step_number=step.step_number,
                            capability_name=step.capability_name,
                            success=False,
                            output=None,
                            error="Dependency failed",
                            duration_ms=0,
                            skipped=True
                        ))
                        can_run = False
                        break
                    else:
                        return ExecutionResult(
                            plan_id=plan.plan_id,
                            goal=plan.goal,
                            success=False,
                            results=results,
                            context=context,
                            final_output=context.get("final_output"),
                            duration_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                            failed_at_step=step.step_number
                        )

            if not can_run:
                continue

            desc = step.description
            cap = step.capability_name or "reasoning"
            self.logger.info(f"Step {step.step_number}/{len(plan.steps)}: {desc} (using {cap})")

            resolved_params = self._resolve_placeholders(step.parameters, context)
            resolved_desc = self._resolve_placeholders(step.description, context)

            attempt = 0
            success = False
            output = None
            error = None
            step_start = asyncio.get_event_loop().time()

            while attempt <= step.retry_count:
                try:
                    if step.capability_name:
                        result = await asyncio.wait_for(
                            self.registry.execute(step.capability_name, resolved_params),
                            timeout=step.timeout
                        )
                        success = result.success
                        output = result.data
                        error = result.error
                    else:
                        prompt = resolved_desc
                        messages = [{"role": "user", "content": prompt}]
                        sys_prompt = "You are ARIA. Be concise and precise. Return only the requested output."
                        response = await asyncio.wait_for(
                            self.ollama_client.chat(messages=messages, system=sys_prompt),
                            timeout=step.timeout
                        )
                        success = True
                        output = response
                        error = None

                    if success:
                        if step.output_key:
                            context[step.output_key] = output
                        break
                except asyncio.TimeoutError:
                    error = f"Step timed out after {step.timeout}s"
                    success = False
                except Exception as e:
                    error = str(e)
                    success = False

                attempt += 1
                if attempt <= step.retry_count:
                    await asyncio.sleep(2 ** attempt)

            step_duration = (asyncio.get_event_loop().time() - step_start) * 1000

            step_result = StepResult(
                step_id=step.step_id,
                step_number=step.step_number,
                capability_name=step.capability_name,
                success=success,
                output=output,
                error=error if not success else None,
                duration_ms=step_duration,
                skipped=False
            )
            results.append(step_result)

            self.logger.info(f"Step {step.step_number} complete: {'OK' if success else 'FAILED'} in {step_duration:.1f}ms")

            if on_step_complete:
                if asyncio.iscoroutinefunction(on_step_complete):
                    await on_step_complete(step, step_result)
                else:
                    on_step_complete(step, step_result)

            await self.memory_manager.remember(
                f"Executed step: {step.description}. Result: {'success' if success else error}",
                source="execution",
                metadata={"plan_id": plan.plan_id, "step": step.step_number}
            )

            if not success and step.on_failure == "stop":
                break

        overall_success = all((r.success or r.skipped) for r in results) if results else True
        total_duration = (asyncio.get_event_loop().time() - start_time) * 1000

        last_out = next((r.output for r in reversed(results) if r.success), None)
        final_output = context.get("final_output", last_out)

        return ExecutionResult(
            plan_id=plan.plan_id,
            goal=plan.goal,
            success=overall_success,
            results=results,
            context=context,
            final_output=final_output,
            duration_ms=total_duration,
            failed_at_step=next((r.step_number for r in results if not r.success and not r.skipped), None)
        )
