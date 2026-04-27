"""
Capability Generator
====================
Template-based training examples for all 22 ARIA capabilities.
Each capability has 15 input templates with parameter generators.
"""
from __future__ import annotations
import random
from aria_brain.generators.base_generator import BaseGenerator
from aria_brain.schema.examples import TrainingExample

# ── Reusable parameter pools ────────────────────────────────────────
_PATHS = [
    "/tmp/test.txt", "~/documents/notes.md", "./README.md",
    "/home/user/data.csv", "~/projects/main.py", "/etc/hosts",
    "config.yaml", "data/output.json", "~/Downloads/report.pdf",
    "./src/app.py", "/var/log/syslog", "~/Desktop/todo.txt",
    "package.json", "requirements.txt", ".env",
]
_DIRS = [
    "/tmp", "~/documents", "./src", "/home/user/projects",
    "~/workspace", "/var/log", "~/Desktop", "./build",
    "~/Downloads", "/opt/data", ".", "./tests", "~/code",
]
_URLS = [
    "https://example.com", "https://news.ycombinator.com",
    "https://api.github.com/repos", "https://httpbin.org/get",
    "https://jsonplaceholder.typicode.com/posts",
    "https://raw.githubusercontent.com/user/repo/main/README.md",
    "http://localhost:8080/api/status", "https://google.com",
]
_CONTENTS = [
    "Hello World", "test content", "my notes here",
    "the result data", "print('hello')", "TODO: fix this",
    "line one\\nline two", "# Heading\\nSome text",
]
_PATTERNS = [
    "*.py", "*.txt", "TODO", "error", "import os",
    "function", "class ", "def ", "*.json", "*.md",
]
_COMMANDS = [
    "ls -la", "echo hello", "pwd", "whoami", "date",
    "df -h", "uname -a", "ps aux", "top -l 1", "cat /etc/hosts",
    "python --version", "node --version", "git status",
    "pip list", "npm list",
]
_PYTHON_CODE = [
    "print('hello world')", "import os; print(os.getcwd())",
    "for i in range(10): print(i)", "import sys; print(sys.version)",
    "print(2 + 2)", "import json; print(json.dumps({'a': 1}))",
]
_DOCKER_IMAGES = [
    "python:3.11", "node:18", "ubuntu:22.04", "alpine:latest",
    "postgres:15", "redis:7",
]
_ENV_VARS = [
    "HOME", "PATH", "USER", "SHELL", "LANG", "EDITOR",
    "TERM", "PYTHONPATH", "NODE_ENV", "API_KEY",
]
_EXTENSIONS = ["py", "txt", "json", "md", "yaml", "js", "ts", "csv"]
_PIDS = ["1234", "5678", "42", "9999", "100", "3001", "8080"]

CAPABILITY_TEMPLATES: dict[str, dict] = {
    # ────────────────── FILE ──────────────────
    "read_file": {
        "templates": [
            "read {path}", "open {path}", "show me the contents of {path}",
            "what's in {path}", "cat {path}", "display {path}",
            "print the file {path}", "read the file at {path}",
            "show {path}", "open and read {path}",
            "get the contents of {path}", "load {path}",
            "what does {path} contain", "view {path}",
            "read the contents of {path}",
        ],
        "param_generators": {"path": lambda: random.choice(_PATHS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "read", "complexity": "simple",
    },
    "write_file": {
        "templates": [
            "write {content} to {path}", "create a file named {path} containing {content}",
            "save {content} to {path}", "create {path} with content {content}",
            "make a new file at {path}", "write to {path}: {content}",
            "create a text file called {path}", "save this to {path}: {content}",
            "create {path} and put {content} in it", "write a file named {path}",
            "make {path} with {content}", "create file {path}",
            "write the following to {path}: {content}", "store {content} in {path}",
            "output {content} to {path}",
        ],
        "param_generators": {
            "path": lambda: random.choice(_PATHS),
            "content": lambda: random.choice(_CONTENTS),
        },
        "difficulty": "easy", "intent_type": "capability",
        "action": "write", "complexity": "simple",
    },
    "append_file": {
        "templates": [
            "append {content} to {path}", "add {content} to the end of {path}",
            "add a line to {path}: {content}", "append to {path}: {content}",
            "write {content} at the end of {path}", "add {content} to {path}",
            "put {content} at the bottom of {path}", "extend {path} with {content}",
            "add text to {path}", "insert {content} at the end of {path}",
            "append a new line to {path} saying {content}",
            "tack {content} onto {path}", "add to file {path}: {content}",
            "append line {content} to {path}", "update {path} by adding {content}",
        ],
        "param_generators": {
            "path": lambda: random.choice(_PATHS),
            "content": lambda: random.choice(_CONTENTS),
        },
        "difficulty": "easy", "intent_type": "capability",
        "action": "append", "complexity": "simple",
    },
    "delete_file": {
        "templates": [
            "delete {path}", "remove {path}", "delete the file {path}",
            "remove the file at {path}", "trash {path}", "get rid of {path}",
            "erase {path}", "delete file {path}", "rm {path}",
            "remove file {path}", "destroy {path}", "wipe {path}",
            "drop the file {path}", "purge {path}", "unlink {path}",
        ],
        "param_generators": {"path": lambda: random.choice(_PATHS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "delete", "complexity": "simple",
    },
    "copy_file": {
        "templates": [
            "copy {source} to {dest}", "cp {source} {dest}",
            "duplicate {source} as {dest}", "copy the file {source} to {dest}",
            "make a copy of {source} at {dest}", "clone {source} to {dest}",
            "replicate {source} into {dest}", "copy file {source} to {dest}",
            "back up {source} to {dest}", "create a copy of {source} named {dest}",
            "copy {source} into {dest}", "duplicate file {source} to {dest}",
            "save a copy of {source} as {dest}", "mirror {source} to {dest}",
            "cp file {source} to {dest}",
        ],
        "param_generators": {
            "source": lambda: random.choice(_PATHS),
            "dest": lambda: random.choice(_PATHS),
        },
        "difficulty": "easy", "intent_type": "capability",
        "action": "copy", "complexity": "simple",
    },
    "move_file": {
        "templates": [
            "move {source} to {dest}", "mv {source} {dest}",
            "rename {source} to {dest}", "move the file {source} to {dest}",
            "relocate {source} to {dest}", "transfer {source} to {dest}",
            "move file {source} to {dest}", "shift {source} to {dest}",
            "put {source} in {dest}", "move {source} into {dest}",
            "rename file {source} to {dest}", "mv file {source} {dest}",
            "relocate file {source} to {dest}", "send {source} to {dest}",
            "transfer file {source} to {dest}",
        ],
        "param_generators": {
            "source": lambda: random.choice(_PATHS),
            "dest": lambda: random.choice(_PATHS),
        },
        "difficulty": "easy", "intent_type": "capability",
        "action": "move", "complexity": "simple",
    },
    "search_files": {
        "templates": [
            "search for {pattern} in {directory}",
            "find {pattern} in {directory}",
            "grep {pattern} in {directory}",
            "look for {pattern} in {directory}",
            "search {directory} for {pattern}",
            "find files matching {pattern} in {directory}",
            "find all files containing {pattern} in {directory}",
            "search files in {directory} for {pattern}",
            "locate {pattern} in {directory}",
            "scan {directory} for {pattern}",
            "find all {pattern} files in {directory}",
            "look for files with {pattern} in {directory}",
            "search {directory} matching {pattern}",
            "hunt for {pattern} in {directory}",
            "grep -r {pattern} {directory}",
        ],
        "param_generators": {
            "pattern": lambda: random.choice(_PATTERNS),
            "directory": lambda: random.choice(_DIRS),
        },
        "difficulty": "medium", "intent_type": "capability",
        "action": "search", "complexity": "simple",
    },
    "file_info": {
        "templates": [
            "get info about {path}", "file info for {path}",
            "show details of {path}", "what are the details of {path}",
            "stat {path}", "get metadata for {path}",
            "show file info for {path}", "describe {path}",
            "tell me about the file {path}", "file details {path}",
            "properties of {path}", "how big is {path}",
            "when was {path} modified", "check {path} info",
            "show the stats of {path}",
        ],
        "param_generators": {"path": lambda: random.choice(_PATHS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "info", "complexity": "simple",
    },
    # ────────────────── FOLDER ──────────────────
    "list_directory": {
        "templates": [
            "list {directory}", "ls {directory}", "show contents of {directory}",
            "what's in {directory}", "list files in {directory}",
            "dir {directory}", "show {directory}", "list the directory {directory}",
            "show me what's in {directory}", "list all files in {directory}",
            "what files are in {directory}", "browse {directory}",
            "display contents of {directory}", "ls -la {directory}",
            "show folder {directory}",
        ],
        "param_generators": {"directory": lambda: random.choice(_DIRS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "list", "complexity": "simple",
    },
    "create_directory": {
        "templates": [
            "create directory {directory}", "mkdir {directory}",
            "make a folder called {directory}", "create folder {directory}",
            "new directory {directory}", "create a new folder {directory}",
            "make directory {directory}", "add folder {directory}",
            "set up directory {directory}", "make a new directory at {directory}",
            "create the folder {directory}", "mkdir -p {directory}",
            "create dir {directory}", "new folder {directory}",
            "create a directory named {directory}",
        ],
        "param_generators": {"directory": lambda: random.choice(_DIRS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "create", "complexity": "simple",
    },
    "delete_directory": {
        "templates": [
            "delete directory {directory}", "rmdir {directory}",
            "remove the folder {directory}", "delete folder {directory}",
            "remove directory {directory}", "wipe the directory {directory}",
            "trash folder {directory}", "get rid of directory {directory}",
            "rm -rf {directory}", "delete dir {directory}",
            "erase folder {directory}", "destroy directory {directory}",
            "remove dir {directory}", "nuke {directory}",
            "purge the folder {directory}",
        ],
        "param_generators": {"directory": lambda: random.choice(_DIRS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "delete", "complexity": "simple",
    },
    "folder_tree": {
        "templates": [
            "show tree of {directory}", "tree {directory}",
            "folder tree for {directory}", "display the tree of {directory}",
            "show directory structure of {directory}",
            "show me a tree of {directory}", "visualize {directory} structure",
            "show folder hierarchy of {directory}", "tree view of {directory}",
            "list tree of {directory}", "print tree of {directory}",
            "show the structure of {directory}", "directory tree {directory}",
            "map out {directory}", "show the folder tree for {directory}",
        ],
        "param_generators": {"directory": lambda: random.choice(_DIRS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "list", "complexity": "simple",
    },
    # ────────────────── PROCESS ──────────────────
    "run_command": {
        "templates": [
            "run {command}", "execute {command}", "run command {command}",
            "shell {command}", "exec {command}", "run this: {command}",
            "execute command {command}", "terminal {command}",
            "bash {command}", "run in terminal: {command}",
            "execute this command: {command}", "do {command}",
            "run the command {command}", "perform {command}",
            "issue the command {command}",
        ],
        "param_generators": {"command": lambda: random.choice(_COMMANDS)},
        "difficulty": "medium", "intent_type": "capability",
        "action": "run", "complexity": "simple",
    },
    "list_processes": {
        "templates": [
            "list processes", "show running processes",
            "what processes are running", "ps", "show all processes",
            "list running tasks", "what's running on my machine",
            "show active processes", "display processes",
            "list all running processes", "get running processes",
            "show me the process list", "what tasks are active",
            "list system processes", "show current processes",
        ],
        "param_generators": {},
        "difficulty": "easy", "intent_type": "capability",
        "action": "list", "complexity": "simple",
    },
    "get_process": {
        "templates": [
            "get process {pid}", "show process {pid}",
            "process info for {pid}", "what is process {pid}",
            "details of process {pid}", "check process {pid}",
            "info on PID {pid}", "status of process {pid}",
            "describe process {pid}", "show me process {pid}",
            "get info on process {pid}", "process {pid} details",
            "what's process {pid} doing", "look up process {pid}",
            "check on PID {pid}",
        ],
        "param_generators": {"pid": lambda: random.choice(_PIDS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "info", "complexity": "simple",
    },
    "kill_process": {
        "templates": [
            "kill process {pid}", "stop process {pid}",
            "terminate process {pid}", "end process {pid}",
            "kill PID {pid}", "kill {pid}", "stop PID {pid}",
            "terminate PID {pid}", "force stop process {pid}",
            "shut down process {pid}", "abort process {pid}",
            "cancel process {pid}", "kill -9 {pid}",
            "end task {pid}", "force quit process {pid}",
        ],
        "param_generators": {"pid": lambda: random.choice(_PIDS)},
        "difficulty": "medium", "intent_type": "capability",
        "action": "kill", "complexity": "simple",
    },
    # ────────────────── EXECUTE ──────────────────
    "run_python": {
        "templates": [
            "run python: {code}", "execute python: {code}",
            "run this python code: {code}", "python {code}",
            "run python script: {code}", "execute this: {code}",
            "eval {code}", "python -c '{code}'",
            "run the following python: {code}", "execute python code: {code}",
            "evaluate {code} in python", "run python snippet {code}",
            "compute {code} with python", "python run {code}",
            "execute the python snippet: {code}",
        ],
        "param_generators": {"code": lambda: random.choice(_PYTHON_CODE)},
        "difficulty": "medium", "intent_type": "capability",
        "action": "execute", "complexity": "simple",
    },
    "run_in_docker": {
        "templates": [
            "run {command} in docker {image}",
            "docker run {image} {command}",
            "execute {command} inside docker container {image}",
            "run in container {image}: {command}",
            "docker exec {command} on {image}",
            "spin up {image} and run {command}",
            "launch {image} container with {command}",
            "run {command} in a {image} container",
            "use docker {image} to run {command}",
            "containerized run of {command} in {image}",
            "execute in docker image {image}: {command}",
            "run inside {image}: {command}",
            "docker {image} exec {command}",
            "start {image} and execute {command}",
            "run {command} using {image} image",
        ],
        "param_generators": {
            "command": lambda: random.choice(_COMMANDS),
            "image": lambda: random.choice(_DOCKER_IMAGES),
        },
        "difficulty": "medium", "intent_type": "capability",
        "action": "execute", "complexity": "simple",
    },
    # ────────────────── EXTERNAL ──────────────────
    "fetch_url": {
        "templates": [
            "fetch {url}", "get {url}", "download {url}",
            "curl {url}", "grab {url}", "open {url}",
            "load {url}", "request {url}", "visit {url}",
            "retrieve {url}", "fetch the URL {url}",
            "get the page at {url}", "download from {url}",
            "pull content from {url}", "access {url}",
        ],
        "param_generators": {"url": lambda: random.choice(_URLS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "fetch", "complexity": "simple",
    },
    "get_system_info": {
        "templates": [
            "get system info", "show system information",
            "what's my system info", "system details",
            "show my system specs", "describe my system",
            "what hardware am I running", "system status",
            "show machine info", "get computer details",
            "what system am I on", "my system configuration",
            "system specifications", "show system details",
            "get my machine info",
        ],
        "param_generators": {},
        "difficulty": "easy", "intent_type": "capability",
        "action": "info", "complexity": "simple",
    },
    "get_current_time": {
        "templates": [
            "what time is it", "current time", "get the time",
            "what's the time", "show me the time", "time now",
            "tell me the time", "what is the current time",
            "what's the current date and time", "show the clock",
            "display current time", "what time do you have",
            "get current time", "what day is it", "date and time",
        ],
        "param_generators": {},
        "difficulty": "easy", "intent_type": "capability",
        "action": "info", "complexity": "simple",
    },
    "get_env_var": {
        "templates": [
            "get env var {var}", "show environment variable {var}",
            "what is ${var}", "echo ${var}", "get {var} env variable",
            "show me the value of {var}", "environment variable {var}",
            "what's the {var} variable", "print env {var}",
            "read environment variable {var}", "check {var} variable",
            "get the value of {var}", "env {var}",
            "show {var} from environment", "display env var {var}",
        ],
        "param_generators": {"var": lambda: random.choice(_ENV_VARS)},
        "difficulty": "easy", "intent_type": "capability",
        "action": "info", "complexity": "simple",
    },
}

# ── Map capability → primary parameter names for entity extraction ──
_CAP_PARAM_KEYS: dict[str, list[str]] = {
    name: list(cfg["param_generators"].keys())
    for name, cfg in CAPABILITY_TEMPLATES.items()
}


class CapabilityGenerator(BaseGenerator):
    """Generate template-based training examples for all 22 capabilities."""

    def generate_all(self, n_per_capability: int = 60) -> list[TrainingExample]:
        """Generate *n_per_capability* examples for every capability."""
        all_examples: list[TrainingExample] = []
        for cap_name in CAPABILITY_TEMPLATES:
            all_examples.extend(self.generate_for_capability(cap_name, n_per_capability))
        self.rng.shuffle(all_examples)
        return all_examples

    def generate_for_capability(
        self, name: str, n: int = 60
    ) -> list[TrainingExample]:
        """Generate *n* examples for a single capability."""
        cfg = CAPABILITY_TEMPLATES[name]
        templates: list[str] = cfg["templates"]
        param_gens: dict = cfg["param_generators"]
        examples: list[TrainingExample] = []

        for i in range(n):
            template = templates[i % len(templates)]
            params = {k: gen() for k, gen in param_gens.items()}
            user_input = template.format(**params)

            entities = [v for v in params.values() if v]
            reasoning = f"User wants to {cfg['action']} via {name}."

            router_output = self._build_router_output(
                capability_name=name,
                parameters=params,
                action=cfg["action"],
                complexity=cfg["complexity"],
                requires_planning=False,
                intent_type=cfg["intent_type"],
                confidence=round(self.rng.uniform(0.88, 0.99), 2),
                entities=entities,
                reasoning=reasoning,
            )
            examples.append(
                self._make_example(
                    user_input=user_input,
                    router_output=router_output,
                    capability=name,
                    difficulty=cfg["difficulty"],
                    source="template",
                )
            )
        return examples
