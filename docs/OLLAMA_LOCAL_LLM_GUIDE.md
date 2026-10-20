# CiteGuard: Local LLM Configuration Guide (Ollama)

## 1. Overview

CiteGuard supports running zero-cost, local open-weights Large Language Models (e.g. `llama3.2`, `mistral`, `phi3`, `gemma2`) via **Ollama**. 

This integration is completely optional and non-intrusive. Offline testing and CI/CD pipelines continue to default to the fast, deterministic `MockProvider`.

---

## 2. Prerequisites & Installation

To run a local LLM with CiteGuard:

1. Download and install Ollama from [ollama.com](https://ollama.com/).
2. Pull your preferred local model (e.g. `llama3.2` or `mistral`):
   ```bash
   ollama pull llama3.2
   ```
3. Start the Ollama local service (default port: `11434`):
   ```bash
   ollama serve
   ```

---

## 3. Environment Variables Configuration

Configure environment variables in PowerShell or bash:

```powershell
# Windows (PowerShell)
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.2"
$env:OLLAMA_BASE_URL="http://localhost:11434/v1/chat/completions"
```

```bash
# Linux / macOS (Bash)
export LLM_PROVIDER="ollama"
export OLLAMA_MODEL="llama3.2"
export OLLAMA_BASE_URL="http://localhost:11434/v1/chat/completions"
```

---

## 4. Running CiteGuard CLI with Local Ollama

Run scientific claim verification using the local model:

```bash
python scripts/citeguard_cli.py \
  --claim "MicroRNA-21 regulates apoptosis in cancer cells" \
  --provider ollama \
  --pretty
```

---

## 5. Architecture & Safety Guarantees

- **No Hard-coded Keys**: Local LLMs run locally without requiring external API keys.
- **Zero Automatic Downloads**: CiteGuard will **NEVER** automatically download multi-GB model weights. The user must manually run `ollama pull <model>` when desired.
- **Offline Fallback**: Unit tests use mocked HTTP endpoints (`tests/verification/test_ollama_provider.py`) ensuring zero dependency on active local servers during test execution.
