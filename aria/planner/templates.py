"""
aria/planner/templates.py
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aria.cognitive.intent import Intent
from aria.planner.step import PlanStep


@dataclass
class PlanTemplate:
    name: str
    description: str
    trigger_keywords: List[str]

    def build(self, goal: str, intent: Intent, context: Dict[str, Any]) -> List[PlanStep]:
        raise NotImplementedError


class ReadAndSummarizeTemplate(PlanTemplate):
    def __init__(self):
        super().__init__(
            name="read_and_summarize",
            description="Reads a file and summarizes it",
            trigger_keywords=["summarize", "summary", "what does", "explain this file", "read and"],
        )

    def build(self, goal: str, intent: Intent, context: Dict[str, Any]) -> List[PlanStep]:
        path = intent.parameters.get("path") or intent.raw_text.split()[-1]
        step1 = PlanStep(
            step_number=1,
            description=f"Read the file: {path}",
            capability_name="read_file",
            parameters={"path": path},
            output_key="read_output",
        )
        step2 = PlanStep(
            step_number=2,
            description="Summarize the content: {read_output}",
            capability_name=None,
            depends_on=[step1.step_id],
        )
        return [step1, step2]


class AnalyzeAndFixTemplate(PlanTemplate):
    def __init__(self):
        super().__init__(
            name="analyze_and_fix",
            description="Analyzes and fixes code",
            trigger_keywords=["fix", "debug", "find bug", "repair", "correct", "analyze and fix"],
        )

    def build(self, goal: str, intent: Intent, context: Dict[str, Any]) -> List[PlanStep]:
        path = intent.parameters.get("path") or intent.raw_text.split()[-1]
        step1 = PlanStep(
            step_number=1,
            description=f"Read {path}",
            capability_name="read_file",
            parameters={"path": path},
            output_key="read_content",
        )
        step2 = PlanStep(
            step_number=2,
            description="Analyze for bugs: {read_content}",
            capability_name=None,
            output_key="fixed_content",
            depends_on=[step1.step_id],
        )
        step3 = PlanStep(
            step_number=3,
            description="Write fixed content",
            capability_name="write_file",
            parameters={"path": path, "content": "{fixed_content}"},
            depends_on=[step2.step_id],
        )
        step4 = PlanStep(
            step_number=4,
            description="Run the code",
            capability_name="run_command", # We use run_python OR run_command, the template spec says "run_python(code) OR run_command". For files, run_command might be easier: `python3 {path}`. Wait, "python3 path" won't work cleanly everywhere. If the file is Python, `run_python` might take code. We'll use run_command as simple.
            parameters={"command": f"python3 {path}"},
            depends_on=[step3.step_id],
            is_optional=True,
        )
        return [step1, step2, step3, step4]


class FetchAndSaveTemplate(PlanTemplate):
    def __init__(self):
        super().__init__(
            name="fetch_and_save",
            description="Gets URL and saves to file",
            trigger_keywords=["fetch and save", "download and store", "get url and save", "scrape and save", "fetch"],
        )

    def build(self, goal: str, intent: Intent, context: Dict[str, Any]) -> List[PlanStep]:
        parts = intent.raw_text.split()
        url = next((p for p in parts if p.startswith("http")), "https://example.com")
        path = parts[-1] if " to " in intent.raw_text else "/tmp/fetched.md"

        step1 = PlanStep(
            step_number=1,
            description=f"Fetch {url}",
            capability_name="fetch_url",
            parameters={"url": url},
            output_key="fetched_content",
        )
        step2 = PlanStep(
            step_number=2,
            description=f"Save to {path}",
            capability_name="write_file",
            parameters={"path": path, "content": "{fetched_content.content}"},
            depends_on=[step1.step_id],
        )
        return [step1, step2]


class CreateAndRunTemplate(PlanTemplate):
    def __init__(self):
        super().__init__(
            name="create_and_run",
            description="Write and execute script",
            trigger_keywords=["create and run", "write and execute", "make a script", "write a python"],
        )

    def build(self, goal: str, intent: Intent, context: Dict[str, Any]) -> List[PlanStep]:
        path = intent.parameters.get("path") or "/tmp/script.py"
        step1 = PlanStep(
            step_number=1,
            description=f"Write Python code for: {goal}",
            capability_name=None,
            output_key="script_code",
        )
        step2 = PlanStep(
            step_number=2,
            description=f"Save code to {path}",
            capability_name="write_file",
            parameters={"path": path, "content": "{script_code}"},
            depends_on=[step1.step_id],
        )
        step3 = PlanStep(
            step_number=3,
            description="Execute script",
            capability_name="run_python",
            parameters={"code": "{script_code}"},
            depends_on=[step2.step_id],
        )
        return [step1, step2, step3]


class SearchAndReportTemplate(PlanTemplate):
    def __init__(self):
        super().__init__(
            name="search_and_report",
            description="Lookup and report",
            trigger_keywords=["search and report", "find and summarize", "look up and explain"],
        )

    def build(self, goal: str, intent: Intent, context: Dict[str, Any]) -> List[PlanStep]:
        url = next((p for p in intent.raw_text.split() if p.startswith("http")), "https://example.com")
        step1 = PlanStep(
            step_number=1,
            description=f"Search {url}",
            capability_name="fetch_url",
            parameters={"url": url},
            output_key="search_result",
        )
        step2 = PlanStep(
            step_number=2,
            description="Report on findings: {search_result}",
            capability_name=None,
            output_key="report_content",
            depends_on=[step1.step_id],
        )
        step3 = PlanStep(
            step_number=3,
            description="Save report",
            capability_name="write_file",
            parameters={"path": "report.md", "content": "{report_content}"},
            depends_on=[step2.step_id],
        )
        return [step1, step2, step3]


class TemplateRegistry:
    def __init__(self):
        self.templates = [
            ReadAndSummarizeTemplate(),
            AnalyzeAndFixTemplate(),
            FetchAndSaveTemplate(),
            CreateAndRunTemplate(),
            SearchAndReportTemplate(),
        ]

    def match(self, intent: Intent, goal: str) -> Optional[PlanTemplate]:
        text_lower = intent.raw_text.lower()
        action_lower = intent.action.lower() if intent.action else ""
        for template in self.templates:
            for keyword in template.trigger_keywords:
                if keyword in text_lower or keyword in action_lower:
                    return template
        return None
