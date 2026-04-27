"""
Multi-Step Generator
=====================
Examples where requires_planning=True — the user's request implies
two or more sequential operations.
"""
from __future__ import annotations
import random
from aria_brain.generators.base_generator import BaseGenerator
from aria_brain.schema.examples import TrainingExample

_PATHS = [
    "/tmp/test.txt", "~/documents/notes.md", "./README.md",
    "~/projects/main.py", "config.yaml", "data/output.json",
    "./src/app.py", "requirements.txt", "report.md",
]
_URLS = [
    "https://example.com", "https://news.ycombinator.com",
    "https://api.github.com/repos", "https://httpbin.org/get",
]
_DIRS = [
    "/tmp", "~/documents", "./src", "~/workspace", ".", "./tests",
]
_LANGUAGES = ["Spanish", "French", "Japanese", "German", "Hindi"]
_EXTENSIONS = ["py", "txt", "json", "md", "yaml", "js", "csv"]
_TASKS = [
    "calculate fibonacci", "sort a list", "parse JSON",
    "scrape a webpage", "convert CSV to JSON", "generate a UUID",
]
_PATTERNS = ["TODO", "FIXME", "error", "import", "def ", "class "]

MULTI_STEP_TEMPLATES: list[dict] = [
    # ── read + transform ────────────────────────────────────────────
    {"input": "read {path} and summarize it",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "read {path} and count the lines",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "read {path} and tell me if there are any errors",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "open {path} and extract all URLs from it",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "read {path} and convert it to JSON",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "read {path} and format it nicely",
     "action": "read", "params": {"source_path": "{path}"}},
    # ── fetch + save ────────────────────────────────────────────────
    {"input": "fetch {url} and save it to {path}",
     "action": "fetch", "params": {"url": "{url}", "path": "{path}"}},
    {"input": "download {url} and store the content in {path}",
     "action": "fetch", "params": {"url": "{url}", "path": "{path}"}},
    {"input": "grab {url} and write the output to {path}",
     "action": "fetch", "params": {"url": "{url}", "path": "{path}"}},
    {"input": "fetch {url}, extract the main content, and save a summary",
     "action": "fetch", "params": {"url": "{url}"}},
    {"input": "download {url} and parse the JSON response",
     "action": "fetch", "params": {"url": "{url}"}},
    # ── analyze + fix ───────────────────────────────────────────────
    {"input": "analyze {path} and fix any bugs",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "read {path} and refactor the code",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "check {path} for syntax errors and fix them",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "review {path} and suggest improvements",
     "action": "read", "params": {"source_path": "{path}"}},
    # ── create + run ────────────────────────────────────────────────
    {"input": "create a python script that does {task} and run it",
     "action": "execute", "params": {"task": "{task}"}},
    {"input": "write a script to {task} and execute it",
     "action": "execute", "params": {"task": "{task}"}},
    {"input": "generate a python program for {task} then test it",
     "action": "execute", "params": {"task": "{task}"}},
    {"input": "make a script that will {task} and save the output",
     "action": "execute", "params": {"task": "{task}"}},
    # ── search + action ─────────────────────────────────────────────
    {"input": "search for {pattern} in {directory} and show me the results",
     "action": "search", "params": {"pattern": "{pattern}", "directory": "{directory}"}},
    {"input": "find all {ext} files in {directory} and count them",
     "action": "search", "params": {"pattern": "*.{ext}", "directory": "{directory}"}},
    {"input": "search for {pattern} in {directory} and replace it with something else",
     "action": "search", "params": {"pattern": "{pattern}", "directory": "{directory}"}},
    {"input": "find all files containing {pattern} and delete the empty ones",
     "action": "search", "params": {"pattern": "{pattern}", "directory": "."}},
    {"input": "list all {ext} files in {directory} and delete them",
     "action": "delete", "params": {"pattern": "*.{ext}", "directory": "{directory}"}},
    # ── translate / convert ─────────────────────────────────────────
    {"input": "read {path}, translate it to {language}, save to {outpath}",
     "action": "read", "params": {"source_path": "{path}", "language": "{language}", "output_path": "{outpath}"}},
    {"input": "read {path} and translate the content to {language}",
     "action": "read", "params": {"source_path": "{path}", "language": "{language}"}},
    {"input": "convert {path} from YAML to JSON and save it",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "transform {path} into markdown format",
     "action": "read", "params": {"source_path": "{path}"}},
    # ── folder + file ops ───────────────────────────────────────────
    {"input": "create a folder {directory} and copy {path} into it",
     "action": "create", "params": {"directory": "{directory}", "source_path": "{path}"}},
    {"input": "make a new directory {directory} and move all python files there",
     "action": "create", "params": {"directory": "{directory}"}},
    {"input": "create {directory} and initialize a git repo in it",
     "action": "create", "params": {"directory": "{directory}"}},
    {"input": "set up a project folder at {directory} with a README and .gitignore",
     "action": "create", "params": {"directory": "{directory}"}},
    # ── read + append ───────────────────────────────────────────────
    {"input": "read {path} and append a timestamp to the end",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "read {path}, add line numbers, and save it back",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "open {path} and add a header comment at the top",
     "action": "read", "params": {"source_path": "{path}"}},
    # ── system + report ─────────────────────────────────────────────
    {"input": "check system info and save a report to {path}",
     "action": "info", "params": {"output_path": "{path}"}},
    {"input": "get the current time and log it to {path}",
     "action": "info", "params": {"output_path": "{path}"}},
    {"input": "list all processes and save the output to {path}",
     "action": "list", "params": {"output_path": "{path}"}},
    {"input": "show the folder tree and export it to {path}",
     "action": "list", "params": {"output_path": "{path}"}},
    # ── copy + modify ───────────────────────────────────────────────
    {"input": "copy {path} to backup and then modify the original",
     "action": "copy", "params": {"source_path": "{path}"}},
    {"input": "back up {path} then delete the original",
     "action": "copy", "params": {"source_path": "{path}"}},
    {"input": "duplicate {path} and rename the copy",
     "action": "copy", "params": {"source_path": "{path}"}},
    # ── docker multi-step ───────────────────────────────────────────
    {"input": "run tests in a docker container and show me the results",
     "action": "execute", "params": {}},
    {"input": "build the project in docker and deploy it",
     "action": "execute", "params": {}},
    {"input": "spin up a container, run the tests, and clean up",
     "action": "execute", "params": {}},
    # ── process management ──────────────────────────────────────────
    {"input": "find the process using port 8080 and kill it",
     "action": "kill", "params": {"port": "8080"}},
    {"input": "list all node processes and terminate them",
     "action": "kill", "params": {}},
    {"input": "check what's running on port 3000 and stop it",
     "action": "kill", "params": {"port": "3000"}},
    # ── env + config ────────────────────────────────────────────────
    {"input": "check my PATH variable and add /usr/local/bin if missing",
     "action": "info", "params": {"var": "PATH"}},
    {"input": "read the .env file and set the variables",
     "action": "read", "params": {"source_path": ".env"}},
    # ── compound operations ─────────────────────────────────────────
    {"input": "clean up the project: delete node_modules, clear cache, and rebuild",
     "action": "delete", "params": {}},
    {"input": "set up the project: create folders, install deps, and run tests",
     "action": "create", "params": {}},
    {"input": "deploy the app: build, test, and push to production",
     "action": "execute", "params": {}},
    {"input": "archive the project: compress all files and move to backup",
     "action": "move", "params": {}},
    {"input": "audit the system: check disk, memory, and running processes",
     "action": "info", "params": {}},
    {"input": "prepare release: bump version, update changelog, and tag",
     "action": "write", "params": {}},
    {"input": "migrate data: read from old format, convert, and write to new location",
     "action": "read", "params": {}},
    {"input": "bootstrap environment: check python version, create venv, install requirements",
     "action": "execute", "params": {}},
    {"input": "generate report: collect system info, list files, and write summary to {path}",
     "action": "info", "params": {"output_path": "{path}"}},
    {"input": "scan for vulnerabilities: search for hardcoded secrets in {directory}",
     "action": "search", "params": {"directory": "{directory}"}},
    {"input": "organize photos: create date folders and move images into them",
     "action": "create", "params": {}},
    {"input": "sync configs: read local config, fetch remote config, and merge them",
     "action": "read", "params": {}},
    {"input": "refactor module: read {path}, split into multiple files, and test",
     "action": "read", "params": {"source_path": "{path}"}},
    {"input": "backup and rotate logs: copy logs to archive and delete old ones",
     "action": "copy", "params": {}},
    {"input": "benchmark the system: run tests, collect metrics, and save report",
     "action": "execute", "params": {}},
    {"input": "lint the project: find style issues in {directory} and auto-fix them",
     "action": "search", "params": {"directory": "{directory}"}},
    {"input": "update dependencies: read requirements, check for updates, and install",
     "action": "read", "params": {}},
    {"input": "debug the error: read the log file, find the traceback, and explain it",
     "action": "read", "params": {}},
    {"input": "initialize workspace: create {directory}, clone repo, and set up env",
     "action": "create", "params": {"directory": "{directory}"}},
    {"input": "publish package: build dist, run tests, and upload to PyPI",
     "action": "execute", "params": {}},
    {"input": "compress {directory} into a zip file and upload it",
     "action": "execute", "params": {"directory": "{directory}"}},
    {"input": "read all yaml files in {directory} and validate their syntax",
     "action": "read", "params": {"directory": "{directory}"}},
    {"input": "fetch {url}, parse the HTML, and extract all links",
     "action": "fetch", "params": {"url": "{url}"}},
    {"input": "check disk usage, find large files, and suggest what to delete",
     "action": "info", "params": {}},
    {"input": "monitor process {pid}: check CPU usage every second for 10 seconds",
     "action": "info", "params": {"pid": "{pid}"}},
    {"input": "rename all {ext} files in {directory} to lowercase",
     "action": "move", "params": {"directory": "{directory}"}},
]


class MultiStepGenerator(BaseGenerator):
    """Generate training examples for multi-step planning intents."""

    def generate_all(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for tmpl in MULTI_STEP_TEMPLATES:
            filled = self._fill_template(tmpl)
            examples.append(filled)
        self.rng.shuffle(examples)
        return examples

    def _fill_template(self, tmpl: dict) -> TrainingExample:
        raw_input: str = tmpl["input"]
        params: dict = dict(tmpl.get("params", {}))

        # Fill placeholders in the input string
        replacements: dict[str, str] = {}
        if "{path}" in raw_input or "{path}" in str(params):
            replacements["path"] = self.rng.choice(_PATHS)
        if "{outpath}" in raw_input:
            replacements["outpath"] = self.rng.choice(_PATHS)
        if "{url}" in raw_input or "{url}" in str(params):
            replacements["url"] = self.rng.choice(_URLS)
        if "{directory}" in raw_input or "{directory}" in str(params):
            replacements["directory"] = self.rng.choice(_DIRS)
        if "{language}" in raw_input:
            replacements["language"] = self.rng.choice(_LANGUAGES)
        if "{ext}" in raw_input or "{ext}" in str(params):
            replacements["ext"] = self.rng.choice(_EXTENSIONS)
        if "{task}" in raw_input:
            replacements["task"] = self.rng.choice(_TASKS)
        if "{pattern}" in raw_input or "{pattern}" in str(params):
            replacements["pattern"] = self.rng.choice(_PATTERNS)
        if "{pid}" in raw_input:
            replacements["pid"] = str(self.rng.randint(1000, 9999))

        user_input = raw_input.format_map(_SafeDict(replacements))

        # Resolve param values
        resolved_params: dict = {}
        for k, v in params.items():
            if isinstance(v, str):
                resolved_params[k] = v.format_map(_SafeDict(replacements))
            else:
                resolved_params[k] = v

        entities = [v for v in resolved_params.values() if isinstance(v, str) and v]
        complexity = "multi_step"
        is_compound = any(kw in raw_input for kw in ["and", "then", ","])
        if sum(1 for kw in ["and", "then", ","] if kw in raw_input) >= 2:
            complexity = "complex"

        ro = self._build_router_output(
            capability_name=None,
            parameters=resolved_params,
            action=tmpl["action"],
            complexity=complexity,
            requires_planning=True,
            intent_type="capability",
            confidence=round(self.rng.uniform(0.75, 0.92), 2),
            entities=entities,
            reasoning="Multi-step request requires planner orchestration.",
        )

        difficulty = "hard" if complexity == "complex" else "medium"
        return self._make_example(user_input, ro, "multi_step", difficulty)


class _SafeDict(dict):
    """dict subclass that returns the key as '{key}' for missing keys."""
    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
