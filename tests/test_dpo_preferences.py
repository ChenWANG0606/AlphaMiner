from __future__ import annotations

import json
from pathlib import Path

import pytest

from DPO.config import load_dpo_config
from DPO.preferences import PreferenceBuildError, run_preference_building


def test_two_success_candidates_choose_highest_and_lowest_abs_ic(tmp_path: Path) -> None:
    validated_file = _write_validated(
        tmp_path,
        [
            _record("sample_001", 0, abs_ic=0.03),
            _record("sample_001", 1, abs_ic=0.10),
            _record("sample_001", 2, abs_ic=0.01),
        ],
    )
    config = load_dpo_config(_write_config(tmp_path, validated_file))

    summary = run_preference_building(config)

    [pair] = _read_jsonl(config.paths.dpo_train_file)
    assert pair["metadata"]["chosen_candidate_id"] == "sample_001_cand_01"
    assert pair["metadata"]["rejected_candidate_id"] == "sample_001_cand_02"
    assert pair["metadata"]["preference_rule"] == "highest_abs_ic_1_vs_lowest_abs_ic_1"
    assert pair["metadata"]["abs_ic_1_gap"] == pytest.approx(0.09)
    chosen = json.loads(pair["chosen"])
    rejected = json.loads(pair["rejected"])
    assert list(chosen) == [
        "reasoning",
        "factor_formula",
        "factor_python",
        "required_inputs",
        "inavailable_inputs",
    ]
    assert "candidate_id" not in chosen
    assert "metadata" not in rejected
    assert summary["num_pairs"] == 1
    assert summary["rule_counts"]["highest_abs_ic_1_vs_lowest_abs_ic_1"] == 1
    assert summary["mean_abs_ic_1_gap"] == pytest.approx(0.09)


def test_equal_abs_ic_uses_candidate_index_stably(tmp_path: Path) -> None:
    validated_file = _write_validated(
        tmp_path,
        [
            _record("sample_001", 2, abs_ic=0.05),
            _record("sample_001", 0, abs_ic=0.05),
        ],
    )
    config = load_dpo_config(_write_config(tmp_path, validated_file))

    run_preference_building(config)

    [pair] = _read_jsonl(config.paths.dpo_train_file)
    assert pair["metadata"]["chosen_candidate_id"] == "sample_001_cand_00"
    assert pair["metadata"]["rejected_candidate_id"] == "sample_001_cand_02"


def test_single_success_uses_highest_severity_failure_as_rejected(tmp_path: Path) -> None:
    validated_file = _write_validated(
        tmp_path,
        [
            _record("sample_001", 0, abs_ic=0.08),
            _record("sample_001", 1, success=False, failure_type="backtest_error", severity=1),
            _record("sample_001", 2, success=False, failure_type="invalid_json", severity=8),
        ],
    )
    config = load_dpo_config(_write_config(tmp_path, validated_file))

    summary = run_preference_building(config)

    [pair] = _read_jsonl(config.paths.dpo_train_file)
    assert pair["metadata"]["chosen_candidate_id"] == "sample_001_cand_00"
    assert pair["metadata"]["rejected_candidate_id"] == "sample_001_cand_02"
    assert pair["metadata"]["preference_rule"] == "single_backtest_success_vs_hard_failure"
    assert pair["metadata"]["abs_ic_1_gap"] is None
    assert summary["rule_counts"]["single_backtest_success_vs_hard_failure"] == 1


def test_single_success_disabled_goes_to_hard_case(tmp_path: Path) -> None:
    validated_file = _write_validated(
        tmp_path,
        [
            _record("sample_001", 0, abs_ic=0.08),
            _record("sample_001", 1, success=False, failure_type="invalid_json", severity=8),
        ],
    )
    config = load_dpo_config(
        _write_config(tmp_path, validated_file, allow_single_success_pair=False)
    )

    summary = run_preference_building(config)

    assert _read_jsonl(config.paths.dpo_train_file) == []
    [hard_case] = _read_jsonl(config.paths.hard_cases_file)
    assert hard_case["hard_case_reason"] == "single_success_pair_disabled"
    assert "payload" not in hard_case
    assert summary["num_hard_cases"] == 1


def test_no_success_goes_to_hard_case(tmp_path: Path) -> None:
    validated_file = _write_validated(
        tmp_path,
        [
            _record("sample_001", 0, success=False, failure_type="invalid_json", severity=8),
            _record("sample_001", 1, success=False, failure_type="backtest_error", severity=1),
        ],
    )
    config = load_dpo_config(_write_config(tmp_path, validated_file))

    run_preference_building(config)

    [hard_case] = _read_jsonl(config.paths.hard_cases_file)
    assert hard_case["hard_case_reason"] == "no_backtest_success_candidate"
    assert hard_case["failure_type_counts"] == {"invalid_json": 1, "backtest_error": 1}


def test_non_train_split_goes_to_hard_case(tmp_path: Path) -> None:
    validated_file = _write_validated(
        tmp_path,
        [
            _record("sample_001", 0, abs_ic=0.10, source_split="val"),
            _record("sample_001", 1, abs_ic=0.01, source_split="val"),
        ],
    )
    config = load_dpo_config(_write_config(tmp_path, validated_file))

    run_preference_building(config)

    [hard_case] = _read_jsonl(config.paths.hard_cases_file)
    assert hard_case["hard_case_reason"] == "non_train_source_split"


def test_min_abs_ic_gap_filters_small_gap(tmp_path: Path) -> None:
    validated_file = _write_validated(
        tmp_path,
        [
            _record("sample_001", 0, abs_ic=0.05),
            _record("sample_001", 1, abs_ic=0.04),
        ],
    )
    config = load_dpo_config(_write_config(tmp_path, validated_file, min_abs_ic_gap=0.02))

    summary = run_preference_building(config)

    assert _read_jsonl(config.paths.dpo_train_file) == []
    [hard_case] = _read_jsonl(config.paths.hard_cases_file)
    assert hard_case["hard_case_reason"] == "min_abs_ic_gap_not_met"
    assert summary["mean_abs_ic_1_gap"] is None


def test_existing_output_without_overwrite_fails(tmp_path: Path) -> None:
    validated_file = _write_validated(
        tmp_path,
        [_record("sample_001", 0, abs_ic=0.10), _record("sample_001", 1, abs_ic=0.01)],
    )
    config = load_dpo_config(_write_config(tmp_path, validated_file, overwrite=False))
    config.paths.dpo_train_file.parent.mkdir(parents=True, exist_ok=True)
    config.paths.dpo_train_file.write_text("", encoding="utf-8")

    with pytest.raises(PreferenceBuildError, match="already exists"):
        run_preference_building(config)


def _write_config(
    tmp_path: Path,
    validated_file: Path,
    *,
    allow_single_success_pair: bool = True,
    min_abs_ic_gap: float = 0.0,
    overwrite: bool = True,
) -> Path:
    config_path = tmp_path / "dpo_config.yaml"
    config_path.write_text(
        f"""
paths:
  base_model_path: {tmp_path / "base_model"}
  sft_adapter_path: {tmp_path / "sft_adapter"}
  train_file: {tmp_path / "train.jsonl"}
  candidates_file: {tmp_path / "candidates.jsonl"}
  candidate_summary_file: {tmp_path / "candidate_summary.json"}
  validated_candidates_file: {validated_file}
  validation_summary_file: {tmp_path / "validation_summary.json"}
  data_dict_file: {tmp_path / "data_dict.md"}
  dpo_train_file: {tmp_path / "dpo_train.jsonl"}
  hard_cases_file: {tmp_path / "hard_cases.jsonl"}
  preference_summary_file: {tmp_path / "preference_summary.json"}
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
  allow_single_success_pair: {str(allow_single_success_pair).lower()}
  min_abs_ic_gap: {min_abs_ic_gap}
  overwrite: {str(overwrite).lower()}
""".strip(),
        encoding="utf-8",
    )
    return config_path


def _write_validated(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "validated_candidates.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _record(
    sample_id: str,
    candidate_index: int,
    *,
    abs_ic: float | None = None,
    success: bool = True,
    failure_type: str | None = None,
    severity: int = 0,
    source_split: str = "train",
) -> dict:
    ic_value = abs_ic if abs_ic is not None else None
    if success and abs_ic is None:
        raise ValueError("abs_ic is required for successful records")
    return {
        "sample_id": sample_id,
        "group_id": sample_id,
        "candidate_id": f"{sample_id}_cand_{candidate_index:02d}",
        "candidate_index": candidate_index,
        "group_size": 4,
        "prompt": [{"role": "user", "content": f"prompt {sample_id}"}],
        "payload": _payload(f"candidate {candidate_index}"),
        "validation": {"errors": []},
        "backtest": {
            "success": success,
            "ic_1": ic_value,
            "abs_ic_1": abs_ic,
            "rank_ic_1": 0.02 if success else None,
            "ir_1": 0.3 if success else None,
            "error": None if success else failure_type,
        },
        "failure_type": failure_type,
        "failure_severity": severity,
        "metadata": {
            "source_split": source_split,
            "report_title": f"report {sample_id}",
            "base_model_path": "base",
            "sft_adapter_path": "adapter",
        },
    }


def _payload(reasoning: str) -> dict:
    return {
        "reasoning": reasoning,
        "factor_formula": "close / volume",
        "factor_python": "def compute_factor(close, volume):\n    return close / volume",
        "required_inputs": ["close", "volume"],
        "inavailable_inputs": [],
        "candidate_id": "must_not_be_serialized",
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
