from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

from SFT.inference_config import BackendConfig, GenerationConfig, InferenceConfig, RuntimeConfig
from SFT.io_utils import ensure_directory, read_jsonl, write_json

from .config import DPOConfig


class DPOTrainingError(RuntimeError):
    pass


class EvaluationError(RuntimeError):
    pass


class DPOTrainerRunner(Protocol):
    def train(self, config: DPOConfig, samples: list[dict[str, Any]]) -> dict[str, Any]:
        pass


class ModelEvaluator(Protocol):
    def evaluate_model(
        self,
        *,
        model_name: str,
        adapter_path: Path,
        split_name: str,
        data_file: Path,
        config: DPOConfig,
    ) -> dict[str, Any]:
        pass


class TRLDPOTrainerRunner:
    def train(self, config: DPOConfig, samples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer

        model_kwargs: dict[str, Any] = {"trust_remote_code": config.training.trust_remote_code}
        if config.training.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_resolve_torch_dtype(config.training.torch_dtype),
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = _resolve_torch_dtype(config.training.torch_dtype)
            if config.training.device == "auto":
                model_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(
            config.paths.base_model_path,
            trust_remote_code=config.training.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy_base = AutoModelForCausalLM.from_pretrained(config.paths.base_model_path, **model_kwargs)
        ref_base = AutoModelForCausalLM.from_pretrained(config.paths.base_model_path, **model_kwargs)
        policy = PeftModel.from_pretrained(policy_base, config.paths.sft_adapter_path, is_trainable=True)
        reference = PeftModel.from_pretrained(ref_base, config.paths.sft_adapter_path, is_trainable=False)

        train_rows = [
            {
                "prompt": _format_prompt(tokenizer, sample["prompt"]),
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
            }
            for sample in samples
        ]
        train_dataset = Dataset.from_list(train_rows)
        training_args = TRLDPOConfig(
            output_dir=str(config.paths.dpo_adapter_path),
            beta=config.training.beta,
            learning_rate=config.training.learning_rate,
            num_train_epochs=config.training.num_train_epochs,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_prompt_length=config.training.max_prompt_length,
            max_length=config.training.max_length,
            seed=config.training.seed,
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            gradient_checkpointing=config.training.gradient_checkpointing,
            report_to=[],
        )
        trainer = DPOTrainer(
            model=policy,
            ref_model=reference,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        train_result = trainer.train()
        trainer.save_model(str(config.paths.dpo_adapter_path))
        tokenizer.save_pretrained(config.paths.dpo_adapter_path)
        metrics = getattr(train_result, "metrics", None)
        return dict(metrics) if isinstance(metrics, dict) else {}


class StructuralEvaluator:
    def evaluate_model(
        self,
        *,
        model_name: str,
        adapter_path: Path,
        split_name: str,
        data_file: Path,
        config: DPOConfig,
    ) -> dict[str, Any]:
        from SFT.evaluator import evaluate_records
        from SFT.inference_backends import build_inference_backend

        inference_config = _build_inference_config(config, adapter_path, data_file)
        backend = build_inference_backend(inference_config)
        result = evaluate_records(
            read_jsonl(data_file),
            backend=backend,
            inference_config=inference_config,
        )
        return {
            **result.summary,
            "sample_count": result.summary.get("sample_count", 0),
            "backtest_metrics_available": False,
        }


def run_dpo_training(
    config: DPOConfig,
    *,
    trainer: DPOTrainerRunner | None = None,
) -> dict[str, Any]:
    _validate_training_paths(config)
    samples = _load_dpo_train_samples(config.paths.dpo_train_file)
    ensure_directory(config.paths.dpo_adapter_path)
    trainer = trainer or TRLDPOTrainerRunner()
    train_metrics = trainer.train(config, samples)
    manifest = _build_manifest(config, samples, train_metrics)
    write_json(config.paths.dpo_adapter_path / "manifest.json", manifest)
    return manifest


def run_dpo_evaluation(
    config: DPOConfig,
    *,
    evaluator: ModelEvaluator | None = None,
) -> dict[str, Any]:
    _validate_evaluation_paths(config)
    evaluator = evaluator or StructuralEvaluator()
    ensure_directory(config.paths.eval_report_file.parent)
    ensure_directory(config.paths.case_studies_file.parent)

    failures: list[dict[str, str]] = []
    metrics: dict[str, dict[str, Any]] = {}
    for split_name, data_file in (("val", config.paths.val_file), ("test", config.paths.test_file)):
        split_metrics: dict[str, Any] = {}
        for model_name, adapter_path in (
            ("sft_baseline", config.paths.sft_adapter_path),
            ("dpo_model", config.paths.dpo_adapter_path),
        ):
            try:
                split_metrics[model_name] = evaluator.evaluate_model(
                    model_name=model_name,
                    adapter_path=adapter_path,
                    split_name=split_name,
                    data_file=data_file,
                    config=config,
                )
            except Exception as exc:
                failures.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "stage": "evaluation",
                        "error": _format_error(exc),
                    }
                )
                raise EvaluationError(f"{model_name} evaluation failed on {split_name}: {exc}") from exc
        split_metrics["delta"] = _metric_delta(split_metrics["sft_baseline"], split_metrics["dpo_model"])
        metrics[split_name] = split_metrics

    report = {
        "models": {
            "sft_baseline": {
                "base_model_path": str(config.paths.base_model_path),
                "adapter_path": str(config.paths.sft_adapter_path),
            },
            "dpo_model": {
                "base_model_path": str(config.paths.base_model_path),
                "adapter_path": str(config.paths.dpo_adapter_path),
            },
        },
        "eval_data": {
            "val_file": str(config.paths.val_file),
            "test_file": str(config.paths.test_file),
            "num_val": _count_jsonl(config.paths.val_file),
            "num_test": _count_jsonl(config.paths.test_file),
        },
        "generation_config": {
            "temperature": config.evaluation.temperature,
            "top_p": config.evaluation.top_p,
            "max_new_tokens": config.evaluation.max_new_tokens,
            "seed": config.evaluation.seed,
        },
        "metrics": metrics,
        "backtest_metrics_available": _backtest_available(metrics),
        "failures": failures,
    }
    write_json(config.paths.eval_report_file, report)
    _write_case_studies(config.paths.case_studies_file, report)
    return report


def _validate_training_paths(config: DPOConfig) -> None:
    if not config.paths.base_model_path.exists():
        raise DPOTrainingError(f"Base model path not found: {config.paths.base_model_path}")
    if not config.paths.sft_adapter_path.exists():
        raise DPOTrainingError(f"SFT adapter path not found: {config.paths.sft_adapter_path}")
    if not config.paths.dpo_train_file.exists():
        raise DPOTrainingError(f"DPO train file not found: {config.paths.dpo_train_file}")
    if config.paths.dpo_adapter_path == config.paths.sft_adapter_path:
        raise DPOTrainingError("dpo_adapter_path must differ from sft_adapter_path")
    if config.paths.dpo_adapter_path.exists() and any(config.paths.dpo_adapter_path.iterdir()) and not config.training.overwrite:
        raise DPOTrainingError(f"DPO adapter output directory already exists: {config.paths.dpo_adapter_path}")


def _validate_evaluation_paths(config: DPOConfig) -> None:
    if not config.paths.base_model_path.exists():
        raise EvaluationError(f"Base model path not found: {config.paths.base_model_path}")
    if not config.paths.sft_adapter_path.exists():
        raise EvaluationError(f"SFT adapter path not found: {config.paths.sft_adapter_path}")
    if not config.paths.dpo_adapter_path.exists():
        raise EvaluationError(f"DPO adapter path not found: {config.paths.dpo_adapter_path}")
    if not config.paths.val_file.exists():
        raise EvaluationError(f"Validation file not found: {config.paths.val_file}")
    if not config.paths.test_file.exists():
        raise EvaluationError(f"Test file not found: {config.paths.test_file}")


def _load_dpo_train_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_jsonl(path), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DPOTrainingError(f"Invalid DPO train JSONL at line {line_number}: {exc.msg}") from exc
        if "prompt" not in sample:
            raise DPOTrainingError(f"line {line_number}: missing prompt")
        if "chosen" not in sample:
            raise DPOTrainingError(f"line {line_number}: missing chosen")
        if "rejected" not in sample:
            raise DPOTrainingError(f"line {line_number}: missing rejected")
        if not isinstance(sample["chosen"], str):
            raise DPOTrainingError(f"line {line_number}: chosen must be a string")
        if not isinstance(sample["rejected"], str):
            raise DPOTrainingError(f"line {line_number}: rejected must be a string")
        if not isinstance(sample["prompt"], list):
            raise DPOTrainingError(f"line {line_number}: prompt must be a message list")
        samples.append(sample)
    if not samples:
        raise DPOTrainingError("DPO train file contains no samples")
    return samples


def _build_manifest(
    config: DPOConfig,
    samples: list[dict[str, Any]],
    train_metrics: dict[str, Any],
) -> dict[str, Any]:
    preference_rules = sorted(
        {
            sample.get("metadata", {}).get("preference_rule")
            for sample in samples
            if isinstance(sample.get("metadata"), dict) and sample["metadata"].get("preference_rule")
        }
    )
    group_sizes = [
        sample.get("metadata", {}).get("candidate_group_size")
        for sample in samples
        if isinstance(sample.get("metadata"), dict)
    ]
    return {
        "base_model_path": str(config.paths.base_model_path),
        "sft_adapter_path": str(config.paths.sft_adapter_path),
        "dpo_adapter_path": str(config.paths.dpo_adapter_path),
        "merged": False,
        "reference_model": "base_model_path + sft_adapter_path",
        "dpo_train_file": str(config.paths.dpo_train_file),
        "num_train_samples": len(samples),
        "candidate_group_size": _most_common(group_sizes),
        "preference_rules": preference_rules,
        "trainer_framework": "trl",
        "trainer": config.training.trainer,
        "training_config": {
            "beta": config.training.beta,
            "learning_rate": config.training.learning_rate,
            "num_train_epochs": config.training.num_train_epochs,
            "per_device_train_batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "seed": config.training.seed,
        },
        "train_metrics": train_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_inference_config(config: DPOConfig, adapter_path: Path, data_file: Path) -> InferenceConfig:
    return InferenceConfig(
        backend=BackendConfig(
            type="local_hf",
            model=None,
            base_url=None,
            api_key=None,
            api_key_env=None,
            base_model_path=config.paths.base_model_path,
            adapter_path=adapter_path,
        ),
        generation=GenerationConfig(
            temperature=config.evaluation.temperature,
            max_new_tokens=config.evaluation.max_new_tokens,
            timeout=120,
            max_retries=0,
        ),
        runtime=RuntimeConfig(
            device=config.evaluation.device,
            torch_dtype=config.evaluation.torch_dtype,
            load_in_4bit=config.evaluation.load_in_4bit,
            trust_remote_code=config.evaluation.trust_remote_code,
            eval_batch_size=config.evaluation.batch_size,
        ),
        config_path=config.config_path,
        env_file=None,
        default_eval_input_path=data_file,
        data_dict_path=config.paths.data_dict_file,
    )


def _format_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            pass
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\n\nassistant:"


def _metric_delta(baseline: dict[str, Any], dpo: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    keys = set(baseline) | set(dpo)
    for key in keys:
        base_value = baseline.get(key)
        dpo_value = dpo.get(key)
        if isinstance(base_value, (int, float)) and isinstance(dpo_value, (int, float)):
            delta[key] = dpo_value - base_value
        elif isinstance(base_value, dict) and isinstance(dpo_value, dict):
            nested = _metric_delta(base_value, dpo_value)
            if nested:
                delta[key] = nested
    return delta


def _backtest_available(metrics: dict[str, dict[str, Any]]) -> bool:
    for split_metrics in metrics.values():
        for model_name in ("sft_baseline", "dpo_model"):
            if split_metrics.get(model_name, {}).get("backtest_metrics_available") is False:
                return False
    return True


def _write_case_studies(path: Path, report: dict[str, Any]) -> None:
    content = [
        "# DPO Case Studies",
        "",
        "This file is generated as a lightweight review index.",
        "",
        f"- Validation samples: {report['eval_data']['num_val']}",
        f"- Test samples: {report['eval_data']['num_test']}",
        f"- Backtest metrics available: {report['backtest_metrics_available']}",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in read_jsonl(path) if line.strip())


def _most_common(values: list[Any]) -> Any:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _format_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {message}"


def _resolve_torch_dtype(dtype_name: str):
    import torch

    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")
