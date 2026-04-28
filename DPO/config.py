from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_DPO_CONFIG_PATH = "DPO/configs/dpo_config.yaml"


@dataclass(frozen=True)
class DPOPaths:
    base_model_path: Path
    sft_adapter_path: Path
    train_file: Path
    candidates_file: Path
    candidate_summary_file: Path
    validated_candidates_file: Path
    validation_summary_file: Path
    data_dict_file: Path
    dpo_train_file: Path
    hard_cases_file: Path
    preference_summary_file: Path
    dpo_adapter_path: Path
    val_file: Path
    test_file: Path
    eval_report_file: Path
    case_studies_file: Path


@dataclass(frozen=True)
class CandidateGenerationConfig:
    group_size: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    overwrite: bool
    batch_size: int
    device: str
    torch_dtype: str
    load_in_4bit: bool
    trust_remote_code: bool


@dataclass(frozen=True)
class ValidationConfig:
    require_backtest: bool
    ic_horizon: int
    overwrite: bool


@dataclass(frozen=True)
class PreferenceConfig:
    primary_metric: str
    allow_single_success_pair: bool
    min_abs_ic_gap: float
    overwrite: bool


@dataclass(frozen=True)
class TrainingConfig:
    trainer: str
    beta: float
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_prompt_length: int
    max_length: int
    seed: int
    overwrite: bool
    device: str
    torch_dtype: str
    load_in_4bit: bool
    trust_remote_code: bool
    gradient_checkpointing: bool
    logging_steps: int
    save_steps: int


@dataclass(frozen=True)
class EvaluationConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    device: str
    torch_dtype: str
    load_in_4bit: bool
    trust_remote_code: bool
    batch_size: int


@dataclass(frozen=True)
class DPOConfig:
    paths: DPOPaths
    candidate_generation: CandidateGenerationConfig
    validation: ValidationConfig
    preference: PreferenceConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    config_path: Path


def load_dpo_config(path: str | Path = DEFAULT_DPO_CONFIG_PATH) -> DPOConfig:
    resolved_path = Path(path).resolve()
    raw = _load_yaml(resolved_path)
    config_dir = resolved_path.parent
    project_root = config_dir.parent.parent

    paths_section = raw.get("paths", {})
    generation_section = raw.get("candidate_generation", {})

    paths = DPOPaths(
        base_model_path=_resolve_path(project_root, paths_section.get("base_model_path", "SFT/model/base")),
        sft_adapter_path=_resolve_path(
            project_root,
            paths_section.get("sft_adapter_path", "SFT/trained/qwen3_0_6b_sft_lora"),
        ),
        train_file=_resolve_path(project_root, paths_section.get("train_file", "SFT/data/train.jsonl")),
        candidates_file=_resolve_path(
            project_root,
            paths_section.get("candidates_file", "DPO/data/candidates.jsonl"),
        ),
        candidate_summary_file=_resolve_path(
            project_root,
            paths_section.get("candidate_summary_file", "DPO/data/candidate_summary.json"),
        ),
        validated_candidates_file=_resolve_path(
            project_root,
            paths_section.get("validated_candidates_file", "DPO/data/validated_candidates.jsonl"),
        ),
        validation_summary_file=_resolve_path(
            project_root,
            paths_section.get("validation_summary_file", "DPO/data/validation_summary.json"),
        ),
        data_dict_file=_resolve_path(
            project_root,
            paths_section.get("data_dict_file", "extracter/data_dict.md"),
        ),
        dpo_train_file=_resolve_path(
            project_root,
            paths_section.get("dpo_train_file", "DPO/data/dpo_train.jsonl"),
        ),
        hard_cases_file=_resolve_path(
            project_root,
            paths_section.get("hard_cases_file", "DPO/data/hard_cases.jsonl"),
        ),
        preference_summary_file=_resolve_path(
            project_root,
            paths_section.get("preference_summary_file", "DPO/data/preference_summary.json"),
        ),
        dpo_adapter_path=_resolve_path(
            project_root,
            paths_section.get("dpo_adapter_path", "DPO/trained/qwen3_0_6b_dpo_lora"),
        ),
        val_file=_resolve_path(project_root, paths_section.get("val_file", "SFT/data/val.jsonl")),
        test_file=_resolve_path(project_root, paths_section.get("test_file", "SFT/data/test.jsonl")),
        eval_report_file=_resolve_path(
            project_root,
            paths_section.get("eval_report_file", "DPO/output/eval_report.json"),
        ),
        case_studies_file=_resolve_path(
            project_root,
            paths_section.get("case_studies_file", "DPO/output/case_studies.md"),
        ),
    )
    generation = CandidateGenerationConfig(
        group_size=max(1, int(generation_section.get("group_size", 4))),
        temperature=float(generation_section.get("temperature", 0.8)),
        top_p=float(generation_section.get("top_p", 0.9)),
        max_new_tokens=int(generation_section.get("max_new_tokens", 1024)),
        seed=int(generation_section.get("seed", 42)),
        overwrite=bool(generation_section.get("overwrite", False)),
        batch_size=max(1, int(generation_section.get("batch_size", 1))),
        device=str(generation_section.get("device", "auto")),
        torch_dtype=str(generation_section.get("torch_dtype", "bfloat16")),
        load_in_4bit=bool(generation_section.get("load_in_4bit", False)),
        trust_remote_code=bool(generation_section.get("trust_remote_code", True)),
    )
    validation_section = raw.get("validation", {})
    validation = ValidationConfig(
        require_backtest=bool(validation_section.get("require_backtest", True)),
        ic_horizon=int(validation_section.get("ic_horizon", 1)),
        overwrite=bool(validation_section.get("overwrite", False)),
    )
    preference_section = raw.get("preference", {})
    preference = PreferenceConfig(
        primary_metric=str(preference_section.get("primary_metric", "abs_ic_1")),
        allow_single_success_pair=bool(preference_section.get("allow_single_success_pair", True)),
        min_abs_ic_gap=float(preference_section.get("min_abs_ic_gap", 0.0)),
        overwrite=bool(preference_section.get("overwrite", False)),
    )
    training_section = raw.get("training", {})
    training = TrainingConfig(
        trainer=str(training_section.get("trainer", "DPOTrainer")),
        beta=float(training_section.get("beta", 0.1)),
        learning_rate=float(training_section.get("learning_rate", 0.000005)),
        num_train_epochs=float(training_section.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training_section.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_section.get("gradient_accumulation_steps", 8)),
        max_prompt_length=int(training_section.get("max_prompt_length", 1536)),
        max_length=int(training_section.get("max_length", 3072)),
        seed=int(training_section.get("seed", 42)),
        overwrite=bool(training_section.get("overwrite", False)),
        device=str(training_section.get("device", "auto")),
        torch_dtype=str(training_section.get("torch_dtype", "bfloat16")),
        load_in_4bit=bool(training_section.get("load_in_4bit", False)),
        trust_remote_code=bool(training_section.get("trust_remote_code", True)),
        gradient_checkpointing=bool(training_section.get("gradient_checkpointing", True)),
        logging_steps=int(training_section.get("logging_steps", 10)),
        save_steps=int(training_section.get("save_steps", 100)),
    )
    evaluation_section = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        temperature=float(evaluation_section.get("temperature", 0.0)),
        top_p=float(evaluation_section.get("top_p", 1.0)),
        max_new_tokens=int(evaluation_section.get("max_new_tokens", 1024)),
        seed=int(evaluation_section.get("seed", 42)),
        device=str(evaluation_section.get("device", "auto")),
        torch_dtype=str(evaluation_section.get("torch_dtype", "bfloat16")),
        load_in_4bit=bool(evaluation_section.get("load_in_4bit", False)),
        trust_remote_code=bool(evaluation_section.get("trust_remote_code", True)),
        batch_size=max(1, int(evaluation_section.get("batch_size", 1))),
    )
    return DPOConfig(
        paths=paths,
        candidate_generation=generation,
        validation=validation,
        preference=preference,
        training=training,
        evaluation=evaluation,
        config_path=resolved_path,
    )


def override_candidate_generation(
    config: DPOConfig,
    *,
    group_size: int | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    overwrite: bool | None = None,
) -> DPOConfig:
    updates: dict[str, Any] = {}
    if group_size is not None:
        updates["group_size"] = max(1, group_size)
    if seed is not None:
        updates["seed"] = seed
    if temperature is not None:
        updates["temperature"] = temperature
    if top_p is not None:
        updates["top_p"] = top_p
    if max_new_tokens is not None:
        updates["max_new_tokens"] = max_new_tokens
    if overwrite is not None:
        updates["overwrite"] = overwrite
    return replace(
        config,
        candidate_generation=replace(config.candidate_generation, **updates),
    )


def override_validation(
    config: DPOConfig,
    *,
    require_backtest: bool | None = None,
    ic_horizon: int | None = None,
    overwrite: bool | None = None,
) -> DPOConfig:
    updates: dict[str, Any] = {}
    if require_backtest is not None:
        updates["require_backtest"] = require_backtest
    if ic_horizon is not None:
        updates["ic_horizon"] = ic_horizon
    if overwrite is not None:
        updates["overwrite"] = overwrite
    return replace(config, validation=replace(config.validation, **updates))


def override_preference(
    config: DPOConfig,
    *,
    primary_metric: str | None = None,
    allow_single_success_pair: bool | None = None,
    min_abs_ic_gap: float | None = None,
    overwrite: bool | None = None,
) -> DPOConfig:
    updates: dict[str, Any] = {}
    if primary_metric is not None:
        updates["primary_metric"] = primary_metric
    if allow_single_success_pair is not None:
        updates["allow_single_success_pair"] = allow_single_success_pair
    if min_abs_ic_gap is not None:
        updates["min_abs_ic_gap"] = min_abs_ic_gap
    if overwrite is not None:
        updates["overwrite"] = overwrite
    return replace(config, preference=replace(config.preference, **updates))


def override_training(
    config: DPOConfig,
    *,
    beta: float | None = None,
    learning_rate: float | None = None,
    num_train_epochs: float | None = None,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    overwrite: bool | None = None,
) -> DPOConfig:
    updates: dict[str, Any] = {}
    if beta is not None:
        updates["beta"] = beta
    if learning_rate is not None:
        updates["learning_rate"] = learning_rate
    if num_train_epochs is not None:
        updates["num_train_epochs"] = num_train_epochs
    if per_device_train_batch_size is not None:
        updates["per_device_train_batch_size"] = per_device_train_batch_size
    if gradient_accumulation_steps is not None:
        updates["gradient_accumulation_steps"] = gradient_accumulation_steps
    if overwrite is not None:
        updates["overwrite"] = overwrite
    return replace(config, training=replace(config.training, **updates))


def override_evaluation(
    config: DPOConfig,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    seed: int | None = None,
) -> DPOConfig:
    updates: dict[str, Any] = {}
    if temperature is not None:
        updates["temperature"] = temperature
    if top_p is not None:
        updates["top_p"] = top_p
    if max_new_tokens is not None:
        updates["max_new_tokens"] = max_new_tokens
    if seed is not None:
        updates["seed"] = seed
    return replace(config, evaluation=replace(config.evaluation, **updates))


def _resolve_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"DPO config file not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load DPO config.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid DPO config format: {path}")
    return payload
