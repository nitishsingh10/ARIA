# ARIA Brain — Router Fine-Tuning Data Pipeline

Training data generation pipeline for fine-tuning **Qwen2.5 1.5B** as ARIA's router model.

## What This Does

The router model learns ONE job: **natural language input → structured JSON routing decision**.

It handles:
- Intent classification (19 action types)
- Capability selection from 22 tools
- Parameter extraction from natural language
- Complexity assessment (simple / multi_step / complex)
- Ambiguous input → fallback to conversation

## Quick Start

```bash
cd aria_brain
pip install -r requirements.txt

# Generate dataset WITHOUT LLM (fast, deterministic)
python scripts/generate.py --skip-llm --output-dir ./data

# Generate FULL dataset with Gemma4 via Ollama
python scripts/generate.py --output-dir ./data

# Validate an existing dataset
python scripts/validate.py ./data/train.jsonl

# Show dataset statistics
python scripts/stats.py ./data/train.jsonl
```

## Target Dataset Composition

| Segment           | Count    | Source              |
|--------------------|----------|---------------------|
| Per capability     | ~120 each| Template + LLM + Aug|
| Conversation       | ~500     | Edge cases + LLM    |
| Memory             | ~280     | Edge cases + LLM    |
| Multi-step         | ~80      | Templates           |
| Edge cases         | ~200     | Hand-crafted        |
| **Total**          | **3,000–3,500** |              |

## Output Format (Alpaca)

```json
{
  "instruction": "You are ARIA's router. Analyze the user input and output a JSON routing decision.",
  "input": "read ~/notes.md",
  "output": "{\"intent\":{\"action\":\"read\",\"complexity\":\"simple\",\"requires_planning\":false,\"intent_type\":\"capability\",\"confidence\":0.95},\"routing\":{\"capability_name\":\"read_file\",\"parameters\":{\"path\":\"~/notes.md\"},\"fallback\":\"conversation\"},\"entities\":[\"~/notes.md\"],\"reasoning\":\"User wants to read via read_file.\"}"
}
```

## 22 Capabilities

| Group    | Capabilities |
|----------|-------------|
| FILE     | read_file, write_file, append_file, delete_file, copy_file, move_file, search_files, file_info |
| FOLDER   | list_directory, create_directory, delete_directory, folder_tree |
| PROCESS  | run_command, list_processes, get_process, kill_process |
| EXECUTE  | run_python, run_in_docker |
| EXTERNAL | fetch_url, get_system_info, get_current_time, get_env_var |
