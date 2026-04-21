# Extracter SiliconFlow Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch `extracter` from the disabled DeepSeek native API config to SiliconFlow's OpenAI-compatible endpoint with model `deepseek-ai/DeepSeek-V3.2`.

**Architecture:** Keep the existing `LLMClient` request shape and only change configuration ownership. `extracter/.env` becomes SiliconFlow-specific, while `extracter/configs.py` reads `SILICONFLOW_*` keys into the existing `LLMConfig` object so the pipeline code stays unchanged.

**Tech Stack:** Python, pytest, urllib, dataclasses

---

### Task 1: Lock the SiliconFlow contract with tests

**Files:**
- Create: `tests/extracter/test_configs.py`
- Test: `tests/extracter/test_configs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_runtime_config_reads_siliconflow_settings(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# LLM_BASE_URL=https://api.deepseek.com",
                "# LLM_API_KEY=sk-deepseek-disabled",
                "# LLM_MODEL=deepseek-chat",
                "SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1",
                "SILICONFLOW_API_KEY=sk-siliconflow-active",
                "SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.2",
            ]
        ),
        encoding="utf-8",
    )

    config = build_runtime_config(stage="generate", env_file=env_file)

    assert config.llm.base_url == "https://api.siliconflow.cn/v1"
    assert config.llm.api_key == "sk-siliconflow-active"
    assert config.llm.model == "deepseek-ai/DeepSeek-V3.2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/extracter/test_configs.py::test_build_runtime_config_reads_siliconflow_settings -v`
Expected: FAIL because `build_runtime_config()` still reads `LLM_*` keys.

- [ ] **Step 3: Write minimal implementation**

```python
llm = LLMConfig(
    base_url=env_values.get("SILICONFLOW_BASE_URL"),
    api_key=env_values.get("SILICONFLOW_API_KEY"),
    model=env_values.get("SILICONFLOW_MODEL"),
    timeout=_int_env(env_values, "LLM_TIMEOUT", 120),
    max_retries=_int_env(env_values, "LLM_MAX_RETRIES", 2),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/extracter/test_configs.py::test_build_runtime_config_reads_siliconflow_settings -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/extracter/test_configs.py extracter/configs.py
git commit -m "feat: switch extracter config to siliconflow"
```

### Task 2: Update local env and keep request URL behavior stable

**Files:**
- Modify: `extracter/.env`
- Test: `tests/extracter/test_configs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_chat_completions_url_appends_endpoint_once() -> None:
    assert _resolve_chat_completions_url("https://api.siliconflow.cn/v1") == (
        "https://api.siliconflow.cn/v1/chat/completions"
    )
    assert _resolve_chat_completions_url("https://api.siliconflow.cn/v1/chat/completions") == (
        "https://api.siliconflow.cn/v1/chat/completions"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/extracter/test_configs.py::test_resolve_chat_completions_url_appends_endpoint_once -v`
Expected: PASS if current helper already behaves correctly; keep it as regression coverage while updating `.env`.

- [ ] **Step 3: Write minimal implementation**

```dotenv
# LLM_BASE_URL=https://api.deepseek.com
# LLM_API_KEY=sk-your-deepseek-key
# LLM_MODEL=deepseek-chat
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_API_KEY=sk-your-siliconflow-key
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/extracter/test_configs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add extracter/.env tests/extracter/test_configs.py
git commit -m "chore: update extracter env for siliconflow"
```
