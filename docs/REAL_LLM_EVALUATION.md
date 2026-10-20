# CiteGuard: Real LLM Provider Evaluation & Security Protocol Report

## 1. Objective

This document specifies the setup, provider architecture, security protocols, and execution status of live Large Language Model (LLM) integrations for **CiteGuard**.

---

## 2. Supported Live Providers

CiteGuard supports multiple LLM backbones via `src/verification/providers/`:

1. **OpenAI Provider** (`src/verification/providers/openai_provider.py`)
   - Models: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`
   - Config: Environment variable `OPENAI_API_KEY`
2. **Anthropic Provider** (`src/verification/providers/anthropic_provider.py`)
   - Models: `claude-3-5-sonnet`, `claude-3-haiku`
   - Config: Environment variable `ANTHROPIC_API_KEY`
3. **Custom / Local API Provider** (`src/verification/providers/custom_provider.py`)
   - Models: Local vLLM, Ollama, or OpenAI-compatible endpoints
   - Config: Environment variables `CUSTOM_API_KEY`, `CUSTOM_BASE_URL`
4. **Mock Offline Provider** (`src/verification/providers/mock_provider.py`)
   - Deterministic offline provider for zero-cost testing and automated CI/CD runs.

---

## 3. Credential Security Guidelines

- **No Hard-coded Keys**: Credentials must never be placed in source files, tests, notebooks, scripts, or markdown documentation.
- **Environment Isolation**: Keys must be set exclusively via standard shell environment variables (e.g. `$env:OPENAI_API_KEY="sk-..."`).
- **Gitignore Protection**: `.env` and `.env.local` files are strictly gitignored in `.gitignore`.
- **Offline Safety Fallback**: If `LLM_PROVIDER=openai` is selected but `OPENAI_API_KEY` is not present, `get_provider()` automatically falls back to `MockProvider` unless `fallback_to_mock=False` is explicitly set.

---

## 4. Current Live LLM Verification Status

```
REAL LLM VALIDATION: NOT RUN
REASON: API credentials unavailable in execution environment
```

### Audit Findings:
- Environment variable check for `OPENAI_API_KEY`: **Not Set**
- Environment variable check for `ANTHROPIC_API_KEY`: **Not Set**
- Environment variable check for `CUSTOM_API_KEY`: **Not Set**

No live network API calls were performed. All offline benchmark evaluations and unit tests continue to pass 100%.
