from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INFERENCE_CONFIG_PATH = "SFT/configs/inference_config.yaml"
DEFAULT_EVAL_INPUT_PATH = "SFT/data/test.jsonl"
DEFAULT_ENV_FILE = "extracter/.env"


@dataclass(frozen=True)
class BackendConfig:
    type: str
    model: str | None
    base_url: str | None
    api_key: str | None
    api_key_env: str | None
    base_model_path: Path | None
    adapter_path: Path | None


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    max_new_tokens: int
    timeout: int
    max_retries: int


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    torch_dtype: str
    load_in_4bit: bool
    trust_remote_code: bool
    eval_batch_size: int


@dataclass(frozen=True)
class InferenceConfig:
    backend: BackendConfig
    generation: GenerationConfig
    runtime: RuntimeConfig
    config_path: Path
    env_file: Path | None
    default_eval_input_path: Path
    data_dict_path: Path


def load_inference_config(path: str | Path = DEFAULT_INFERENCE_CONFIG_PATH) -> InferenceConfig:
    resolved_path = Path(path).resolve()
    raw = _load_yaml(resolved_path)
    config_dir = resolved_path.parent
    env_file_value = raw.get("env_file", DEFAULT_ENV_FILE)
    env_file = _resolve_path(config_dir, env_file_value) if env_file_value else None
    env_values = _load_env_file(env_file) if env_file is not None else {}

    backend_section = raw.get("backend", {})
    generation_section = raw.get("generation", {})
    runtime_section = raw.get("runtime", {})

    api_key_env = backend_section.get("api_key_env")
    resolved_api_key = _resolve_string(backend_section.get("api_key"), env_values=env_values)
    if resolved_api_key is None and api_key_env:
        resolved_api_key = env_values.get(api_key_env, os.environ.get(api_key_env))

    backend = BackendConfig(
        type=str(backend_section.get("type", "openai_compat")),
        model=_resolve_string(backend_section.get("model"), env_values=env_values),
        base_url=_resolve_string(backend_section.get("base_url"), env_values=env_values),
        api_key=resolved_api_key,
        api_key_env=api_key_env,
        base_model_path=_resolve_optional_path(config_dir, backend_section.get("base_model_path")),
        adapter_path=_resolve_optional_path(config_dir, backend_section.get("adapter_path")),
    )
    generation = GenerationConfig(
        temperature=float(generation_section.get("temperature", 0.2)),
        max_new_tokens=int(generation_section.get("max_new_tokens", 1024)),
        timeout=int(generation_section.get("timeout", 120)),
        max_retries=int(generation_section.get("max_retries", 2)),
    )
    runtime = RuntimeConfig(
        device=str(runtime_section.get("device", "auto")),
        torch_dtype=str(runtime_section.get("torch_dtype", "bfloat16")),
        load_in_4bit=bool(runtime_section.get("load_in_4bit", False)),
        trust_remote_code=bool(runtime_section.get("trust_remote_code", True)),
        eval_batch_size=max(1, int(runtime_section.get("eval_batch_size", 1))),
    )

    project_root = resolved_path.parent.parent.parent
    default_eval_input = _resolve_path(
        project_root,
        raw.get("default_eval_input_path", DEFAULT_EVAL_INPUT_PATH),
    )
    return InferenceConfig(
        backend=backend,
        generation=generation,
        runtime=runtime,
        config_path=resolved_path,
        env_file=env_file,
        default_eval_input_path=default_eval_input,
        data_dict_path=project_root / "extracter/data_dict.md",
    )


def resolve_api_key(config: InferenceConfig) -> str | None:
    if config.backend.api_key:
        return config.backend.api_key
    if config.backend.api_key_env:
        return os.environ.get(config.backend.api_key_env)
    return None


def _resolve_optional_path(base_dir: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(base_dir, value)


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    resolved = Path(value)
    if resolved.is_absolute():
        return resolved
    return (base_dir / resolved).resolve()


def _resolve_string(value: Any, *, env_values: dict[str, str]) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return str(value)
    if value.startswith("${") and value.endswith("}"):
        return env_values.get(value[2:-1], os.environ.get(value[2:-1]))
    return value


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the inference config. Install pyyaml in the runtime environment."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid inference config format: {path}")
    return payload
