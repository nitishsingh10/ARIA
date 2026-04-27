"""
Edge-Case Generator
====================
Hard examples — ambiguous, destructive, complex NL param extraction,
memory ops, and typo/casual phrasings.
"""
from __future__ import annotations
from aria_brain.generators.base_generator import BaseGenerator
from aria_brain.schema.examples import TrainingExample


class EdgeCaseGenerator(BaseGenerator):
    """Generates hard-difficulty examples across five edge-case categories."""

    # ── CATEGORY 1: Ambiguous → conversation fallback ───────────────
    _AMBIGUOUS = [
        "do the thing", "handle it", "you know what I mean",
        "same as before", "open it", "run that", "fix the problem",
        "help me with my project", "do what I asked earlier",
        "take care of this", "process the data", "make it work",
        "update everything", "check on that", "handle my files",
        "do the usual", "just do it", "finish what you started",
        "continue", "go ahead", "you decide", "whatever works",
        "sort it out", "make things right", "do your thing",
        "handle the situation", "deal with it", "take over",
        "figure it out", "work your magic", "idk just help",
        "do something useful", "fix stuff", "solve my problem",
        "make it happen", "I need help", "what should I do",
        "just handle everything", "clean things up",
        "get things done", "finish the job", "wrap it up",
        "take care of business", "manage this for me",
        "sort everything out", "do the right thing",
        "run the usual routine", "perform the operation",
        "apply the changes", "complete the task",
    ]

    # ── CATEGORY 2: Destructive operations ──────────────────────────
    _DESTRUCTIVE = [
        ("delete everything in /tmp", "delete_file", {"path": "/tmp/*", "confirm": False}),
        ("remove all files in my documents", "delete_file", {"path": "~/documents/*", "confirm": False}),
        ("wipe the directory", "delete_directory", {"directory": ".", "confirm": False}),
        ("kill all processes", "kill_process", {"pid": "all", "confirm": False}),
        ("delete the project folder", "delete_directory", {"directory": "./project", "confirm": False}),
        ("erase everything on my desktop", "delete_file", {"path": "~/Desktop/*", "confirm": False}),
        ("rm -rf /", "delete_directory", {"directory": "/", "confirm": False}),
        ("destroy the build folder", "delete_directory", {"directory": "./build", "confirm": False}),
        ("purge all log files", "delete_file", {"path": "/var/log/*", "confirm": False}),
        ("nuke the temp directory", "delete_directory", {"directory": "/tmp", "confirm": False}),
        ("remove all .pyc files", "delete_file", {"path": "**/*.pyc", "confirm": False}),
        ("delete my entire home directory", "delete_directory", {"directory": "~", "confirm": False}),
        ("wipe the database folder", "delete_directory", {"directory": "./db", "confirm": False}),
        ("clear all cached data", "delete_file", {"path": ".cache/*", "confirm": False}),
        ("delete every file in downloads", "delete_file", {"path": "~/Downloads/*", "confirm": False}),
        ("remove the entire project", "delete_directory", {"directory": "./", "confirm": False}),
        ("trash all temporary files", "delete_file", {"path": "/tmp/*", "confirm": False}),
        ("eliminate the old backups", "delete_file", {"path": "~/backups/*", "confirm": False}),
        ("obliterate the test data", "delete_directory", {"directory": "./test_data", "confirm": False}),
        ("shred the config files", "delete_file", {"path": "./config/*", "confirm": False}),
        ("clean out the output folder", "delete_directory", {"directory": "./output", "confirm": False}),
        ("remove all node_modules", "delete_directory", {"directory": "node_modules", "confirm": False}),
        ("delete the .git directory", "delete_directory", {"directory": ".git", "confirm": False}),
        ("wipe the dist folder clean", "delete_directory", {"directory": "./dist", "confirm": False}),
        ("terminate all background jobs", "kill_process", {"pid": "all", "confirm": False}),
        ("kill every running process", "kill_process", {"pid": "all", "confirm": False}),
        ("force stop all services", "kill_process", {"pid": "all", "confirm": False}),
        ("destroy all test fixtures", "delete_directory", {"directory": "./fixtures", "confirm": False}),
        ("purge the entire workspace", "delete_directory", {"directory": "~/workspace", "confirm": False}),
        ("remove all generated files", "delete_file", {"path": "./generated/*", "confirm": False}),
    ]

    # ── CATEGORY 3: Complex NL parameter extraction ─────────────────
    _COMPLEX_NL = [
        ("can you check what time it is in Tokyo", "get_current_time", {"timezone": "Asia/Tokyo"}, "info"),
        ("I need to see what's running on my machine", "list_processes", {}, "list"),
        ("could you grab the content from https://news.ycombinator.com", "fetch_url", {"url": "https://news.ycombinator.com"}, "fetch"),
        ("show me a tree of the current directory up to 2 levels", "folder_tree", {"directory": ".", "depth": 2}, "list"),
        ("find all python files in my home directory", "search_files", {"pattern": "*.py", "directory": "~"}, "search"),
        ("create a folder called 'myproject' inside ~/workspace", "create_directory", {"directory": "~/workspace/myproject"}, "create"),
        ("append a new line to my notes file saying 'meeting at 3pm'", "append_file", {"path": "notes.txt", "content": "meeting at 3pm"}, "append"),
        ("tell me the size of the README file", "file_info", {"path": "README.md"}, "info"),
        ("copy my resume to the backup folder", "copy_file", {"source": "resume.pdf", "dest": "backup/resume.pdf"}, "copy"),
        ("move the old logs to archive", "move_file", {"source": "logs/", "dest": "archive/logs/"}, "move"),
        ("what environment does my shell use", "get_env_var", {"var": "SHELL"}, "info"),
        ("check how much disk space I have left", "run_command", {"command": "df -h"}, "run"),
        ("run a quick python check to see my python version", "run_python", {"code": "import sys; print(sys.version)"}, "execute"),
        ("download the front page of hacker news for me", "fetch_url", {"url": "https://news.ycombinator.com"}, "fetch"),
        ("list everything inside the source code folder recursively", "folder_tree", {"directory": "./src"}, "list"),
        ("write a simple hello world program to test.py", "write_file", {"path": "test.py", "content": "print('hello world')"}, "write"),
        ("I want to know when this config file was last changed", "file_info", {"path": "config.yaml"}, "info"),
        ("search my project for any TODO comments", "search_files", {"pattern": "TODO", "directory": "."}, "search"),
        ("pull up my PATH environment variable", "get_env_var", {"var": "PATH"}, "info"),
        ("launch an ubuntu container and check the os version", "run_in_docker", {"image": "ubuntu:22.04", "command": "cat /etc/os-release"}, "execute"),
        ("spin up a python container and install numpy", "run_in_docker", {"image": "python:3.11", "command": "pip install numpy"}, "execute"),
        ("make a new folder for my experiments under projects", "create_directory", {"directory": "~/projects/experiments"}, "create"),
        ("find any files containing the word 'error' in the logs directory", "search_files", {"pattern": "error", "directory": "./logs"}, "search"),
        ("please show me what my USER env variable is set to", "get_env_var", {"var": "USER"}, "info"),
        ("read the first file in my downloads folder", "read_file", {"path": "~/Downloads/"}, "read"),
        ("I need the system information for debugging", "get_system_info", {}, "info"),
        ("rename my old_config.yaml to config.yaml.bak", "move_file", {"source": "old_config.yaml", "dest": "config.yaml.bak"}, "move"),
        ("what command would show all listening ports", "run_command", {"command": "lsof -i -P -n"}, "run"),
        ("get info about the process with PID 3001", "get_process", {"pid": "3001"}, "info"),
        ("save a note that says 'deploy v2 tomorrow' to deploy_notes.txt", "write_file", {"path": "deploy_notes.txt", "content": "deploy v2 tomorrow"}, "write"),
        ("look at the package.json to see my dependencies", "read_file", {"path": "package.json"}, "read"),
        ("create a hidden folder called .secrets in my home dir", "create_directory", {"directory": "~/.secrets"}, "create"),
        ("remove that old backup file from last week", "delete_file", {"path": "backup_old.tar.gz", "confirm": False}, "delete"),
        ("can you tell me what OS I am running", "get_system_info", {}, "info"),
        ("check if there's a process hogging port 8080", "run_command", {"command": "lsof -i :8080"}, "run"),
        ("I'd like to see the directory listing of /var/log", "list_directory", {"directory": "/var/log"}, "list"),
        ("copy all my configs to a safe location", "copy_file", {"source": "config/", "dest": "backup/config/"}, "copy"),
        ("fetch the API status from localhost", "fetch_url", {"url": "http://localhost:8080/api/status"}, "fetch"),
        ("execute 'git status' in my terminal", "run_command", {"command": "git status"}, "run"),
        ("add a reminder to my todo list: buy groceries", "append_file", {"path": "todo.txt", "content": "buy groceries"}, "append"),
        ("I want to check if docker is installed", "run_command", {"command": "docker --version"}, "run"),
        ("browse through the test directory", "list_directory", {"directory": "./tests"}, "list"),
        ("what is the current date and time right now", "get_current_time", {}, "info"),
        ("display the folder structure of my home directory", "folder_tree", {"directory": "~"}, "list"),
        ("find every markdown file in this project", "search_files", {"pattern": "*.md", "directory": "."}, "search"),
        ("get the details about requirements.txt", "file_info", {"path": "requirements.txt"}, "info"),
        ("stop the process that's using PID 9999", "kill_process", {"pid": "9999"}, "kill"),
        ("run python3 -c 'print(42)' for me", "run_python", {"code": "print(42)"}, "execute"),
        ("what is the HOME variable set to on this system", "get_env_var", {"var": "HOME"}, "info"),
        ("move my presentation slides to the shared folder", "move_file", {"source": "slides.pptx", "dest": "shared/slides.pptx"}, "move"),
    ]

    # ── CATEGORY 4: Memory operations ───────────────────────────────
    _MEMORY_REMEMBER = [
        "remember my name is Nitish",
        "note that I prefer dark mode",
        "save the fact that I use Python 3.11",
        "don't forget that my API key expires in June",
        "keep in mind I'm working on ARIA",
        "remember that my project is in ~/workspace/aria",
        "note: I always use vim as my editor",
        "store the fact that I prefer JSON over YAML",
        "remember I have a meeting at 3pm tomorrow",
        "keep track that my dev server runs on port 3000",
        "note that my database is PostgreSQL 15",
        "remember my github username is nitishsingh10",
        "save this: I deploy to AWS us-east-1",
        "don't forget I need to update the API docs",
        "remember that tests should always run before deploy",
        "note that my preferred shell is zsh",
        "keep in mind the staging URL is staging.example.com",
        "remember I'm using Node 18 for this project",
        "save that my backup runs every Sunday at 2am",
        "note my preferred language is Python",
    ]
    _MEMORY_RECALL = [
        "what do you know about me",
        "do you remember my name",
        "recall what I told you about my project",
        "what have I asked you before",
        "what's my name",
        "what editor do I use",
        "do you know my github username",
        "what database am I using",
        "recall my preferred shell",
        "what port does my dev server use",
        "remind me what my deployment target is",
        "what language do I prefer",
        "do you remember what I'm working on",
        "what did I note about my API key",
        "recall my meeting schedule",
        "what do you remember about my setup",
        "tell me what you know about my preferences",
        "what's my github username again",
        "recall the staging URL",
        "what node version am I using",
    ]

    # ── CATEGORY 5: Typos / casual / slang ──────────────────────────
    _TYPO_CASUAL = [
        ("reed the file /tmp/test.txt", "read_file", {"path": "/tmp/test.txt"}, "read"),
        ("lst my documents", "list_directory", {"directory": "~/documents"}, "list"),
        ("wats in this folder", "list_directory", {"directory": "."}, "list"),
        ("plz fetch example.com", "fetch_url", {"url": "https://example.com"}, "fetch"),
        ("yo what time is it", "get_current_time", {}, "info"),
        ("gimme the contents of notes.txt", "read_file", {"path": "notes.txt"}, "read"),
        ("cna you delte this file", "delete_file", {"path": "this", "confirm": False}, "delete"),
        ("shwo me the proccesses", "list_processes", {}, "list"),
        ("runn the comand ls", "run_command", {"command": "ls"}, "run"),
        ("craete a new directry here", "create_directory", {"directory": "."}, "create"),
        ("kil process 1234", "kill_process", {"pid": "1234"}, "kill"),
        ("whats my systm info", "get_system_info", {}, "info"),
        ("cp my file to bakcup", "copy_file", {"source": "my_file", "dest": "backup/"}, "copy"),
        ("serch for errors in logs", "search_files", {"pattern": "errors", "directory": "logs"}, "search"),
        ("moev config.yaml to old_config.yaml", "move_file", {"source": "config.yaml", "dest": "old_config.yaml"}, "move"),
        ("hey can u read readme for me", "read_file", {"path": "README.md"}, "read"),
        ("bruh just delete it already", "delete_file", {"path": "it", "confirm": False}, "delete"),
        ("fyi need to run python script", "run_python", {"code": "script"}, "execute"),
        ("pls show env PATH", "get_env_var", {"var": "PATH"}, "info"),
        ("lmk whats in /tmp", "list_directory", {"directory": "/tmp"}, "list"),
        ("gonna need u to fetch that url real quick", "fetch_url", {"url": "that"}, "fetch"),
        ("aight show me da tree", "folder_tree", {"directory": "."}, "list"),
        ("ngl i need file info on config", "file_info", {"path": "config"}, "info"),
        ("tbh just write hello to test.txt", "write_file", {"path": "test.txt", "content": "hello"}, "write"),
        ("lowkey need to append stuff to notes", "append_file", {"path": "notes", "content": "stuff"}, "append"),
        ("rn i need the time", "get_current_time", {}, "info"),
        ("ayo run docker ubuntu real quick", "run_in_docker", {"image": "ubuntu:22.04", "command": "bash"}, "execute"),
        ("frfr what processes running", "list_processes", {}, "list"),
        ("bet show me system info", "get_system_info", {}, "info"),
        ("no cap read that file rq", "read_file", {"path": "that"}, "read"),
    ]

    def generate_all(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for cat in range(1, 6):
            examples.extend(self.generate_category(cat))
        self.rng.shuffle(examples)
        return examples

    def generate_category(self, category: int) -> list[TrainingExample]:
        dispatch = {
            1: self._gen_ambiguous,
            2: self._gen_destructive,
            3: self._gen_complex_nl,
            4: self._gen_memory,
            5: self._gen_typo_casual,
        }
        return dispatch[category]()

    # ── internal generators ─────────────────────────────────────────

    def _gen_ambiguous(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for text in self._AMBIGUOUS:
            ro = self._build_router_output(
                capability_name=None, parameters={},
                action="chat", complexity="simple",
                intent_type="conversation",
                confidence=round(self.rng.uniform(0.15, 0.45), 2),
                entities=[], reasoning="Input is ambiguous; falling back to conversation.",
            )
            examples.append(self._make_example(text, ro, "conversation", "hard"))
        return examples

    def _gen_destructive(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for text, cap, params, *_ in self._DESTRUCTIVE:
            action = "delete" if "delete" in cap else "kill"
            ro = self._build_router_output(
                capability_name=cap, parameters=params,
                action=action, complexity="simple",
                intent_type="capability",
                confidence=round(self.rng.uniform(0.80, 0.95), 2),
                entities=[v for v in params.values() if isinstance(v, str)],
                reasoning=f"Destructive {action} operation — confirm=false.",
            )
            examples.append(self._make_example(text, ro, cap, "hard"))
        return examples

    def _gen_complex_nl(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for text, cap, params, action in self._COMPLEX_NL:
            ro = self._build_router_output(
                capability_name=cap, parameters=params,
                action=action, complexity="simple",
                intent_type="capability",
                confidence=round(self.rng.uniform(0.78, 0.94), 2),
                entities=[v for v in params.values() if isinstance(v, str)],
                reasoning=f"Extracted params from natural language for {cap}.",
            )
            examples.append(self._make_example(text, ro, cap, "hard"))
        return examples

    def _gen_memory(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for text in self._MEMORY_REMEMBER:
            ro = self._build_router_output(
                capability_name=None, parameters={},
                action="remember", complexity="simple",
                intent_type="memory",
                confidence=round(self.rng.uniform(0.85, 0.98), 2),
                entities=[], reasoning="User wants to store a memory.",
            )
            examples.append(self._make_example(text, ro, "memory", "medium"))
        for text in self._MEMORY_RECALL:
            ro = self._build_router_output(
                capability_name=None, parameters={},
                action="recall", complexity="simple",
                intent_type="memory",
                confidence=round(self.rng.uniform(0.82, 0.96), 2),
                entities=[], reasoning="User wants to recall stored memory.",
            )
            examples.append(self._make_example(text, ro, "memory", "medium"))
        return examples

    def _gen_typo_casual(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for text, cap, params, action in self._TYPO_CASUAL:
            ro = self._build_router_output(
                capability_name=cap, parameters=params,
                action=action, complexity="simple",
                intent_type="capability",
                confidence=round(self.rng.uniform(0.70, 0.90), 2),
                entities=[v for v in params.values() if isinstance(v, str)],
                reasoning=f"Matched despite typo/casual phrasing to {cap}.",
            )
            examples.append(self._make_example(text, ro, cap, "hard"))
        return examples
