"""
Ollama Generator
=================
Uses Gemma4 (via Ollama) to generate diverse training examples
beyond what templates can produce.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

from aria_brain.generators.base_generator import BaseGenerator
from aria_brain.schema.examples import TrainingExample
from aria_brain.schema.router_output import KNOWN_CAPABILITIES

logger = logging.getLogger(__name__)

# ── Capability descriptions for the LLM prompt ─────────────────────
_CAP_DESCRIPTIONS: dict[str, dict] = {
    "read_file":         {"desc": "Read a file's contents",            "params": ["path"]},
    "write_file":        {"desc": "Write content to a file",           "params": ["path", "content"]},
    "append_file":       {"desc": "Append content to a file",          "params": ["path", "content"]},
    "delete_file":       {"desc": "Delete a file",                     "params": ["path"]},
    "copy_file":         {"desc": "Copy a file to another location",   "params": ["source", "dest"]},
    "move_file":         {"desc": "Move/rename a file",                "params": ["source", "dest"]},
    "search_files":      {"desc": "Search for files by pattern",       "params": ["pattern", "directory"]},
    "file_info":         {"desc": "Get metadata about a file",         "params": ["path"]},
    "list_directory":    {"desc": "List contents of a directory",      "params": ["directory"]},
    "create_directory":  {"desc": "Create a new directory",            "params": ["directory"]},
    "delete_directory":  {"desc": "Delete a directory",                "params": ["directory"]},
    "folder_tree":       {"desc": "Show directory tree structure",     "params": ["directory"]},
    "run_command":       {"desc": "Run a shell command",               "params": ["command"]},
    "list_processes":    {"desc": "List running processes",            "params": []},
    "get_process":       {"desc": "Get info about a specific process", "params": ["pid"]},
    "kill_process":      {"desc": "Kill a process by PID",             "params": ["pid"]},
    "run_python":        {"desc": "Execute Python code",               "params": ["code"]},
    "run_in_docker":     {"desc": "Run command in Docker container",   "params": ["image", "command"]},
    "fetch_url":         {"desc": "Fetch content from a URL",          "params": ["url"]},
    "get_system_info":   {"desc": "Get system information",            "params": []},
    "get_current_time":  {"desc": "Get the current date and time",     "params": []},
    "get_env_var":       {"desc": "Get an environment variable",       "params": ["var"]},
}

# Action mapping per capability
_CAP_ACTION: dict[str, str] = {
    "read_file": "read", "write_file": "write", "append_file": "append",
    "delete_file": "delete", "copy_file": "copy", "move_file": "move",
    "search_files": "search", "file_info": "info",
    "list_directory": "list", "create_directory": "create",
    "delete_directory": "delete", "folder_tree": "list",
    "run_command": "run", "list_processes": "list",
    "get_process": "info", "kill_process": "kill",
    "run_python": "execute", "run_in_docker": "execute",
    "fetch_url": "fetch", "get_system_info": "info",
    "get_current_time": "info", "get_env_var": "info",
}


def _load_ollama_url() -> str:
    """Try to read the Ollama base URL from aria.yaml, else fall back."""
    for candidate in [
        Path("aria.yaml"),
        Path("../aria.yaml"),
        Path.home() / ".aria" / "aria.yaml",
    ]:
        if candidate.exists():
            try:
                import yaml
                with open(candidate) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("llm", {}).get("base_url", "http://localhost:11434")
            except Exception:
                pass
    return "http://localhost:11434"


class OllamaGenerator(BaseGenerator):
    """Generate diverse examples using Gemma4 via Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "gemma4:latest",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.base_url = (base_url or _load_ollama_url()).rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=120.0)

    # ── public API ──────────────────────────────────────────────────

    def generate_all(self) -> list[TrainingExample]:
        """Alias for generate_full_dataset with defaults."""
        return self.generate_full_dataset()

    def generate_full_dataset(
        self,
        n_per_capability: int = 50,
        n_conversation: int = 200,
        n_memory: int = 100,
    ) -> list[TrainingExample]:
        all_examples: list[TrainingExample] = []

        for cap_name in KNOWN_CAPABILITIES:
            try:
                batch = self.generate_batch(cap_name, [], n_per_capability)
                all_examples.extend(batch)
                logger.info("Generated %d examples for %s", len(batch), cap_name)
            except Exception as e:
                logger.warning("Failed to generate for %s: %s", cap_name, e)

        try:
            conv = self.generate_conversation_examples(n_conversation)
            all_examples.extend(conv)
            logger.info("Generated %d conversation examples", len(conv))
        except Exception as e:
            logger.warning("Failed conversation generation: %s", e)

        try:
            mem = self._generate_memory_examples(n_memory)
            all_examples.extend(mem)
            logger.info("Generated %d memory examples", len(mem))
        except Exception as e:
            logger.warning("Failed memory generation: %s", e)

        self.rng.shuffle(all_examples)
        return all_examples

    def generate_batch(
        self,
        capability_name: str,
        existing_inputs: list[str],
        n: int = 20,
    ) -> list[TrainingExample]:
        cap_info = _CAP_DESCRIPTIONS.get(capability_name)
        if cap_info is None:
            raise ValueError(f"Unknown capability: {capability_name}")

        prompt = self._build_prompt(capability_name, cap_info, existing_inputs, n)
        raw = self._call_ollama(prompt)
        inputs = self._parse_json_array(raw)

        examples: list[TrainingExample] = []
        action = _CAP_ACTION[capability_name]
        for user_input in inputs[:n]:
            if not user_input or len(user_input) < 5:
                continue
            ro = self._build_router_output(
                capability_name=capability_name,
                parameters={},
                action=action,
                complexity="simple",
                intent_type="capability",
                confidence=round(self.rng.uniform(0.82, 0.96), 2),
                entities=[],
                reasoning=f"LLM-generated input routed to {capability_name}.",
            )
            examples.append(
                self._make_example(user_input, ro, capability_name, "medium", "llm_generated")
            )
        return examples

    def generate_conversation_examples(self, n: int = 200) -> list[TrainingExample]:
        prompt = (
            f"Generate {n} diverse user inputs that are general conversation, "
            "questions, or requests for explanations — NOT tool/command invocations. "
            "Examples: greetings, asking for explanations, opinions, coding help, "
            "philosophical questions, jokes, etc.\n\n"
            "Rules:\n"
            "- Each input must be unique and varied\n"
            "- Mix casual, formal, and technical phrasings\n"
            "- Include greetings, farewells, thank-yous\n"
            "- Include 'explain X', 'what is X', 'how does X work'\n"
            "- Do NOT include any file/folder/process operations\n\n"
            'Return ONLY a JSON array of strings: ["input1", "input2", ...]'
        )
        raw = self._call_ollama(prompt)
        inputs = self._parse_json_array(raw)

        examples: list[TrainingExample] = []
        for user_input in inputs[:n]:
            if not user_input or len(user_input) < 3:
                continue
            ro = self._build_router_output(
                capability_name=None,
                parameters={},
                action="chat",
                complexity="simple",
                intent_type="conversation",
                confidence=round(self.rng.uniform(0.85, 0.98), 2),
                entities=[],
                reasoning="General conversation — no tool needed.",
            )
            examples.append(
                self._make_example(user_input, ro, "conversation", "easy", "llm_generated")
            )
        return examples

    # ── private ─────────────────────────────────────────────────────

    def _generate_memory_examples(self, n: int = 100) -> list[TrainingExample]:
        prompt = (
            f"Generate {n} diverse user inputs that are about remembering or recalling "
            "personal information. Half should be 'remember' type (storing info), "
            "half should be 'recall' type (retrieving info).\n\n"
            "Remember examples: 'remember my name is X', 'note that I prefer Y'\n"
            "Recall examples: 'what's my name', 'do you remember my preference'\n\n"
            'Return ONLY a JSON array of strings: ["input1", "input2", ...]'
        )
        raw = self._call_ollama(prompt)
        inputs = self._parse_json_array(raw)

        examples: list[TrainingExample] = []
        for i, user_input in enumerate(inputs[:n]):
            if not user_input or len(user_input) < 5:
                continue
            is_remember = i < len(inputs) // 2
            action = "remember" if is_remember else "recall"
            ro = self._build_router_output(
                capability_name=None,
                parameters={},
                action=action,
                complexity="simple",
                intent_type="memory",
                confidence=round(self.rng.uniform(0.82, 0.96), 2),
                entities=[],
                reasoning=f"User wants to {action} personal information.",
            )
            examples.append(
                self._make_example(user_input, ro, "memory", "medium", "llm_generated")
            )
        return examples

    def _build_prompt(
        self,
        capability_name: str,
        cap_info: dict,
        existing_inputs: list[str],
        n: int,
    ) -> str:
        avoid = ""
        if existing_inputs:
            sample = existing_inputs[:5]
            avoid = f"- Do NOT repeat these existing inputs: {sample}\n"

        return (
            f"You are generating training data for an AI router. "
            f"Generate {n} diverse user inputs that should route to the "
            f"'{capability_name}' tool.\n\n"
            f"Tool description: {cap_info['desc']}\n"
            f"Required parameters: {cap_info['params']}\n\n"
            f"Rules:\n"
            f"- Each input must be unique\n"
            f"- Vary the phrasing significantly\n"
            f"- Include casual, formal, and technical phrasings\n"
            f"- Include some with typos or slang\n"
            f"- Include paths/URLs/names as entities where appropriate\n"
            f"{avoid}\n"
            f'Return ONLY a JSON array of strings: ["input1", "input2", ...]'
        )

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 4096},
        }
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except httpx.HTTPError as e:
            logger.error("Ollama request failed: %s", e)
            raise

    def _parse_json_array(self, raw: str) -> list[str]:
        raw = raw.strip()
        # Find the JSON array in the response
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            logger.warning("No JSON array found in response")
            return []
        try:
            arr = json.loads(raw[start : end + 1])
            if isinstance(arr, list):
                return [str(item) for item in arr if item]
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON array: %s", e)
        return []
