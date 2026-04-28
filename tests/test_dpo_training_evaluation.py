from __future__ import annotations

import json
from pathlib import Path

import pytest

from DPO.config import load_dpo_config
from DPO.training import (
    DPOTrainingError,
    EvaluationError,
    run_dpo_evaluation,
    run_dpo_training,
)


class FakeTrainer:
    def __init__(self, *, fail_reference: bool = False) -> None:
        self.fail_reference = fail_reference
        self.called = False

    def train(self, config, samples: list[dict]) -> dict:
        if self.fail_reference:
            raise DPOTrainingError("reference model load failed")
        self.called = True
        config.paths.dpo_adapter_path.mkdir(parents=True, exist_ok=True)
        (config.paths.dpo_adapter_path / "adapter_config.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return {"train_loss": 0.1, "num_train_samples": len(samples)}


class FakeEvaluator:
    def evaluate_model(self, *, model_name: str, adapter_path: Path, split_name: str, data_file: Path, config) -> dict:
        count = len([line for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()])
        offset = 0.1 if model_name == "dpo_model" else 0.0
        return {
            "sample_count": count,
            "valid_json_rate": 0.7 + offset,
            "backtest_success_rate": 0.2 + offset,
            "ic_1": {"mean_abs": 0.03 + offset, "median_abs": 0.02 + offset},
            "backtest_metrics_available": True,
        }


class NoBacktestEvaluator(FakeEvaluator):
    def evaluate_model(self, *, model_name: str, adapter_path: Path, split_name: str, data_file: Path, config) -> dict:
        payload = super().evaluate_model(
            model_name=model_name,
            adapter_path=adapter_path,
            split_name=split_name,
            data_file=data_file,
            config=config,
        )
        payload["backtest_metrics_available"] = False
        return payload


def test_dpo_adapter_equal_sft_adapter_fails_training(tmp_path: Path) -> None:
    dpo_train_file = _write_dpo_train(tmp_path, [_pair("sample_001")])
    config = load_dpo_config(
        _write_config(tmp_path, dpo_train_file, dpo_adapter_same_as_sft=True)
    )

    with pytest.raises(DPOTrainingError, match="must differ"):
        run_dpo_training(config, trainer=FakeTrainer())


def test_missing_chosen_fails_training_data_check(tmp_path: Path) -> None:
    pair = _pair("sample_001")
    del pair["chosen"]
    dpo_train_file = _write_dpo_train(tmp_path, [pair])
    config = load_dpo_config(_write_config(tmp_path, dpo_train_file))

    with pytest.raises(DPOTrainingError, match="missing chosen"):
        run_dpo_training(config, trainer=FakeTrainer())


def test_non_string_chosen_fails_training_data_check(tmp_path: Path) -> None:
    pair = _pair("sample_001")
    pair["chosen"] = {"reasoning": "dict is invalid"}
    dpo_train_file = _write_dpo_train(tmp_path, [pair])
    config = load_dpo_config(_write_config(tmp_path, dpo_train_file))

    with pytest.raises(DPOTrainingError, match="chosen must be a string"):
        run_dpo_training(config, trainer=FakeTrainer())


def test_training_writes_manifest(tmp_path: Path) -> None:
    dpo_train_file = _write_dpo_train(
        tmp_path,
        [_pair("sample_001", rule="highest_abs_ic_1_vs_lowest_abs_ic_1")],
    )
    config = load_dpo_config(_write_config(tmp_path, dpo_train_file))

    manifest = run_dpo_training(config, trainer=FakeTrainer())

    saved = json.loads((config.paths.dpo_adapter_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["base_model_path"] == str(config.paths.base_model_path)
    assert saved["sft_adapter_path"] == str(config.paths.sft_adapter_path)
    assert saved["dpo_adapter_path"] == str(config.paths.dpo_adapter_path)
    assert saved["merged"] is False
    assert saved["reference_model"] == "base_model_path + sft_adapter_path"
    assert saved["trainer"] == "DPOTrainer"
    assert saved["num_train_samples"] == 1
    assert saved["preference_rules"] == ["highest_abs_ic_1_vs_lowest_abs_ic_1"]
    assert manifest == saved


def test_reference_model_failure_stops_training(tmp_path: Path) -> None:
    dpo_train_file = _write_dpo_train(tmp_path, [_pair("sample_001")])
    config = load_dpo_config(_write_config(tmp_path, dpo_train_file))

    with pytest.raises(DPOTrainingError, match="reference model load failed"):
        run_dpo_training(config, trainer=FakeTrainer(fail_reference=True))


def test_evaluation_report_compares_same_val_and_test_counts(tmp_path: Path) -> None:
    dpo_train_file = _write_dpo_train(tmp_path, [_pair("sample_001")])
    config = load_dpo_config(_write_config(tmp_path, dpo_train_file, val_count=2, test_count=3))
    config.paths.dpo_adapter_path.mkdir(parents=True, exist_ok=True)

    report = run_dpo_evaluation(config, evaluator=FakeEvaluator())

    assert report["eval_data"]["num_val"] == 2
    assert report["eval_data"]["num_test"] == 3
    assert set(report["metrics"]["val"]) == {"sft_baseline", "dpo_model", "delta"}
    assert set(report["metrics"]["test"]) == {"sft_baseline", "dpo_model", "delta"}
    assert report["metrics"]["val"]["sft_baseline"]["sample_count"] == 2
    assert report["metrics"]["val"]["dpo_model"]["sample_count"] == 2
    assert report["metrics"]["test"]["sft_baseline"]["sample_count"] == 3
    assert report["metrics"]["test"]["dpo_model"]["sample_count"] == 3
    assert report["metrics"]["val"]["delta"]["valid_json_rate"] == pytest.approx(0.1)
    saved = json.loads(config.paths.eval_report_file.read_text(encoding="utf-8"))
    assert saved["models"]["dpo_model"]["adapter_path"] == str(config.paths.dpo_adapter_path)


def test_evaluation_marks_backtest_unavailable(tmp_path: Path) -> None:
    dpo_train_file = _write_dpo_train(tmp_path, [_pair("sample_001")])
    config = load_dpo_config(_write_config(tmp_path, dpo_train_file))
    config.paths.dpo_adapter_path.mkdir(parents=True, exist_ok=True)

    report = run_dpo_evaluation(config, evaluator=NoBacktestEvaluator())

    assert report["backtest_metrics_available"] is False
    assert config.paths.case_studies_file.exists()


def _write_config(
    tmp_path: Path,
    dpo_train_file: Path,
    *,
    dpo_adapter_same_as_sft: bool = False,
    val_count: int = 1,
    test_count: int = 1,
) -> Path:
    base_model_path = tmp_path / "base_model"
    sft_adapter_path = tmp_path / "sft_adapter"
    dpo_adapter_path = sft_adapter_path if dpo_adapter_same_as_sft else tmp_path / "dpo_adapter"
    base_model_path.mkdir()
    sft_adapter_path.mkdir()
    val_file = _write_split(tmp_path / "val.jsonl", val_count)
    test_file = _write_split(tmp_path / "test.jsonl", test_count)
    config_path = tmp_path / "dpo_config.yaml"
    config_path.write_text(
        f"""
paths:
  base_model_path: {base_model_path}
  sft_adapter_path: {sft_adapter_path}
  dpo_adapter_path: {dpo_adapter_path}
  train_file: {tmp_path / "train.jsonl"}
  candidates_file: {tmp_path / "candidates.jsonl"}
  candidate_summary_file: {tmp_path / "candidate_summary.json"}
  validated_candidates_file: {tmp_path / "validated_candidates.jsonl"}
  validation_summary_file: {tmp_path / "validation_summary.json"}
  data_dict_file: {tmp_path / "data_dict.md"}
  dpo_train_file: {dpo_train_file}
  hard_cases_file: {tmp_path / "hard_cases.jsonl"}
  preference_summary_file: {tmp_path / "preference_summary.json"}
  val_file: {val_file}
  test_file: {test_file}
  eval_report_file: {tmp_path / "eval_report.json"}
  case_studies_file: {tmp_path / "case_studies.md"}
candidate_generation:
  group_size: 4
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 128
  seed: 42
  overwrite: true
validation:
  require_backtest: true
  ic_horizon: 1
  overwrite: true
preference:
  primary_metric: abs_ic_1
  allow_single_success_pair: true
  min_abs_ic_gap: 0.0
  overwrite: true
training:
  trainer: DPOTrainer
  beta: 0.1
  learning_rate: 0.000005
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_prompt_length: 1536
  max_length: 3072
  seed: 42
  overwrite: true
evaluation:
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 1024
  seed: 42
""".strip(),
        encoding="utf-8",
    )
    return config_path


def _write_dpo_train(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "dpo_train.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _write_split(path: Path, count: int) -> Path:
    rows = [
        {"prompt": [{"role": "user", "content": f"prompt {index}"}], "metadata": {"sample_id": f"sample_{index:03d}"}}
        for index in range(count)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _pair(sample_id: str, *, rule: str = "single_backtest_success_vs_hard_failure") -> dict:
    assistant_json = json.dumps(
        {
            "reasoning": "reason",
            "factor_formula": "close",
            "factor_python": "def compute_factor(close):\n    return close",
            "required_inputs": ["close"],
            "inavailable_inputs": [],
        },
        ensure_ascii=False,
    )
    return {
        "prompt": [{"role": "user", "content": "prompt"}],
        "chosen": assistant_json,
        "rejected": assistant_json,
        "metadata": {
            "sample_id": sample_id,
            "candidate_group_size": 4,
            "preference_rule": rule,
        },
    }
