from __future__ import annotations

import json
from pathlib import Path

import pytest

from DPO.config import load_dpo_config
from DPO.validation import ValidationError, run_validation


class FakeBacktester:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def run(self, payload: dict, *, ic_horizon: int) -> dict:
        self.calls += 1
        if self.fail:
            raise RuntimeError("cannot backtest")
        return {
            "success": True,
            f"ic_{ic_horizon}": -0.12,
            f"abs_ic_{ic_horizon}": 0.12,
            f"rank_ic_{ic_horizon}": 0.03,
            f"ir_{ic_horizon}": 0.5,
            "error": None,
        }


VALID_PAYLOAD = {
    "reasoning": "close over volume",
    "factor_formula": "close / volume",
    "factor_python": "def compute_factor(close, volume):\n    return close / volume",
    "required_inputs": ["close", "volume"],
    "inavailable_inputs": [],
}


def test_valid_candidate_passes_validation_and_backtest(tmp_path: Path) -> None:
    candidates_file = _write_candidates(tmp_path, [_candidate("sample_001", _payload())])
    config = load_dpo_config(_write_config(tmp_path, candidates_file))
    backtester = FakeBacktester()

    summary = run_validation(config, backtester=backtester)

    [row] = _read_jsonl(config.paths.validated_candidates_file)
    assert row["payload"] == _payload()
    assert row["validation"]["valid_json"] is True
    assert row["validation"]["required_keys"] is True
    assert row["validation"]["python_ast_parse"] is True
    assert row["validation"]["required_inputs_arg_match"] is True
    assert row["validation"]["whitelist_compliance"] is True
    assert row["backtest"]["success"] is True
    assert row["backtest"]["ic_1"] == -0.12
    assert row["backtest"]["abs_ic_1"] == 0.12
    assert row["failure_type"] is None
    assert row["failure_severity"] == 0
    assert backtester.calls == 1
    assert summary["backtest_success_count"] == 1
    assert summary["ic_1"]["mean_abs"] == 0.12


@pytest.mark.parametrize(
    ("payload", "failure_type", "severity"),
    [
        (None, "invalid_json", 8),
        (
            {
                "reasoning": "ok",
                "factor_formula": "close / volume",
                "required_inputs": ["close", "volume"],
                "inavailable_inputs": [],
            },
            "missing_required_keys",
            7,
        ),
        (
            {
                **VALID_PAYLOAD,
                "required_inputs": "close",
            },
            "invalid_field_type",
            6,
        ),
        (
            {
                **VALID_PAYLOAD,
                "factor_python": "def compute_factor(close):\n    return close /",
            },
            "python_ast_error",
            5,
        ),
        (
            {
                **VALID_PAYLOAD,
                "factor_python": "def compute_factor(close):\n    return close",
                "required_inputs": ["volume"],
            },
            "arg_mismatch",
            4,
        ),
        (
            {
                **VALID_PAYLOAD,
                "factor_formula": "paused / close",
                "factor_python": "def compute_factor(paused, close):\n    return paused / close",
                "required_inputs": ["paused", "close"],
            },
            "whitelist_violation",
            3,
        ),
    ],
)
def test_validation_failures_do_not_enter_backtest(
    tmp_path: Path,
    payload: dict | None,
    failure_type: str,
    severity: int,
) -> None:
    candidates_file = _write_candidates(tmp_path, [_candidate("sample_001", payload)])
    config = load_dpo_config(_write_config(tmp_path, candidates_file))
    backtester = FakeBacktester()

    summary = run_validation(config, backtester=backtester)

    [row] = _read_jsonl(config.paths.validated_candidates_file)
    assert row["failure_type"] == failure_type
    assert row["failure_severity"] == severity
    assert row["backtest"]["success"] is False
    assert row["backtest"]["ic_1"] is None
    assert backtester.calls == 0
    assert summary["failure_type_counts"][failure_type] == 1


def test_backtest_error_keeps_metrics_empty(tmp_path: Path) -> None:
    candidates_file = _write_candidates(tmp_path, [_candidate("sample_001", _payload())])
    config = load_dpo_config(_write_config(tmp_path, candidates_file))

    summary = run_validation(config, backtester=FakeBacktester(fail=True))

    [row] = _read_jsonl(config.paths.validated_candidates_file)
    assert row["failure_type"] == "backtest_error"
    assert row["failure_severity"] == 1
    assert row["backtest"] == {
        "success": False,
        "ic_1": None,
        "abs_ic_1": None,
        "rank_ic_1": None,
        "ir_1": None,
        "error": "RuntimeError: cannot backtest",
    }
    assert summary["backtest_success_count"] == 0
    assert summary["ic_1"] is None


def test_missing_candidate_identity_is_validator_error(tmp_path: Path) -> None:
    candidate = _candidate("sample_001", _payload())
    del candidate["candidate_id"]
    candidates_file = _write_candidates(tmp_path, [candidate])
    config = load_dpo_config(_write_config(tmp_path, candidates_file))
    backtester = FakeBacktester()

    run_validation(config, backtester=backtester)

    [row] = _read_jsonl(config.paths.validated_candidates_file)
    assert row["failure_type"] == "validator_error"
    assert row["failure_severity"] == 2
    assert "missing_candidate_field:candidate_id" in row["validation"]["errors"]
    assert row["backtest"]["success"] is False
    assert backtester.calls == 0


def test_existing_output_without_overwrite_fails(tmp_path: Path) -> None:
    candidates_file = _write_candidates(tmp_path, [_candidate("sample_001", _payload())])
    config = load_dpo_config(_write_config(tmp_path, candidates_file, overwrite=False))
    config.paths.validated_candidates_file.parent.mkdir(parents=True, exist_ok=True)
    config.paths.validated_candidates_file.write_text("", encoding="utf-8")

    with pytest.raises(ValidationError, match="already exists"):
        run_validation(config, backtester=FakeBacktester())


def test_require_backtest_false_marks_backtest_skipped(tmp_path: Path) -> None:
    candidates_file = _write_candidates(tmp_path, [_candidate("sample_001", _payload())])
    config = load_dpo_config(
        _write_config(tmp_path, candidates_file, require_backtest=False)
    )

    run_validation(config, backtester=FakeBacktester())

    [row] = _read_jsonl(config.paths.validated_candidates_file)
    assert row["failure_type"] == "backtest_error"
    assert row["backtest"]["error"] == "backtest_skipped_by_config"


def _write_config(
    tmp_path: Path,
    candidates_file: Path,
    *,
    overwrite: bool = True,
    require_backtest: bool = True,
) -> Path:
    data_dict_file = tmp_path / "data_dict.md"
    data_dict_file.write_text(
        """
### 表名: price
| 列名 | 描述 |
| ------ | ------ |
| close | close price |
| volume | volume |
| paused | paused |
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "dpo_config.yaml"
    config_path.write_text(
        f"""
paths:
  base_model_path: {tmp_path / "base_model"}
  sft_adapter_path: {tmp_path / "sft_adapter"}
  train_file: {tmp_path / "train.jsonl"}
  candidates_file: {candidates_file}
  candidate_summary_file: {tmp_path / "candidate_summary.json"}
  validated_candidates_file: {tmp_path / "validated_candidates.jsonl"}
  validation_summary_file: {tmp_path / "validation_summary.json"}
  data_dict_file: {data_dict_file}
candidate_generation:
  group_size: 4
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 128
  seed: 42
  overwrite: true
validation:
  require_backtest: {str(require_backtest).lower()}
  ic_horizon: 1
  overwrite: {str(overwrite).lower()}
""".strip(),
        encoding="utf-8",
    )
    return config_path


def _write_candidates(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _candidate(sample_id: str, payload: dict | None) -> dict:
    return {
        "sample_id": sample_id,
        "group_id": sample_id,
        "candidate_id": f"{sample_id}_cand_00",
        "candidate_index": 0,
        "group_size": 1,
        "prompt": [{"role": "user", "content": "prompt"}],
        "raw_output": json.dumps(payload, ensure_ascii=False) if payload is not None else "bad",
        "parsed_json": payload,
        "parse_error": None if payload is not None else "invalid_json: no object",
        "generation_error": None,
        "metadata": {
            "source_split": "train",
            "report_title": "report",
            "base_model_path": "base",
            "sft_adapter_path": "adapter",
        },
    }


def _payload() -> dict:
    return dict(VALID_PAYLOAD)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
