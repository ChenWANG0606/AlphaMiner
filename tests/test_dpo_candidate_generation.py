from __future__ import annotations

import json
from pathlib import Path

import pytest

from DPO.candidate_generation import CandidateGenerationError, run_candidate_generation
from DPO.config import load_dpo_config


class FakeGenerator:
    def __init__(self, outputs: list[str | Exception]) -> None:
        self._outputs = list(outputs)

    def generate(self, prompt: list[dict[str, str]]) -> str:
        output = self._outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def test_generates_candidates_and_summary(tmp_path: Path) -> None:
    train_file = _write_train_file(tmp_path, ["sample_001", "sample_002"])
    config_path = _write_config(tmp_path, train_file, group_size=2, overwrite=False)
    config = load_dpo_config(config_path)
    generator = FakeGenerator(
        [
            _valid_output("alpha"),
            _valid_output("beta"),
            _valid_output("gamma"),
            _valid_output("delta"),
        ]
    )

    summary = run_candidate_generation(config, generator=generator)

    rows = _read_jsonl(config.paths.candidates_file)
    assert len(rows) == 4
    assert [row["candidate_id"] for row in rows] == [
        "sample_001_cand_00",
        "sample_001_cand_01",
        "sample_002_cand_00",
        "sample_002_cand_01",
    ]
    assert rows[0]["metadata"]["source_split"] == "train"
    assert rows[0]["metadata"]["base_model_path"] == str(config.paths.base_model_path)
    assert rows[0]["metadata"]["sft_adapter_path"] == str(config.paths.sft_adapter_path)
    assert rows[0]["parsed_json"] == {
        "reasoning": "alpha",
        "factor_formula": "close",
        "factor_python": "def compute_factor(close):\n    return close",
        "required_inputs": ["close"],
        "inavailable_inputs": [],
    }
    assert summary["actual_candidate_count"] == 4
    assert summary["generation_success_count"] == 4
    assert summary["parse_success_count"] == 4
    saved_summary = json.loads(config.paths.candidate_summary_file.read_text(encoding="utf-8"))
    assert saved_summary["actual_candidate_count"] == len(rows)


def test_duplicate_sample_id_fails(tmp_path: Path) -> None:
    train_file = _write_train_file(tmp_path, ["sample_001", "sample_001"])
    config_path = _write_config(tmp_path, train_file)
    config = load_dpo_config(config_path)

    with pytest.raises(CandidateGenerationError, match="Duplicate sample_id"):
        run_candidate_generation(config, generator=FakeGenerator([]))


def test_invalid_json_keeps_raw_output_and_parse_error(tmp_path: Path) -> None:
    train_file = _write_train_file(tmp_path, ["sample_001"])
    config_path = _write_config(tmp_path, train_file)
    config = load_dpo_config(config_path)

    summary = run_candidate_generation(config, generator=FakeGenerator(["not json"]))

    [row] = _read_jsonl(config.paths.candidates_file)
    assert row["raw_output"] == "not json"
    assert row["parsed_json"] is None
    assert row["parse_error"].startswith("invalid_json:")
    assert row["generation_error"] is None
    assert summary["parse_error_count"] == 1
    assert summary["error_type_counts"]["invalid_json"] == 1


def test_generation_error_keeps_candidate_record(tmp_path: Path) -> None:
    train_file = _write_train_file(tmp_path, ["sample_001"])
    config_path = _write_config(tmp_path, train_file)
    config = load_dpo_config(config_path)

    summary = run_candidate_generation(config, generator=FakeGenerator([RuntimeError("boom")]))

    [row] = _read_jsonl(config.paths.candidates_file)
    assert row["raw_output"] == ""
    assert row["parsed_json"] is None
    assert row["parse_error"] is None
    assert row["generation_error"] == "RuntimeError: boom"
    assert summary["generation_error_count"] == 1
    assert summary["error_type_counts"]["generation_error"] == 1


def test_existing_output_without_overwrite_fails(tmp_path: Path) -> None:
    train_file = _write_train_file(tmp_path, ["sample_001"])
    config_path = _write_config(tmp_path, train_file, overwrite=False)
    config = load_dpo_config(config_path)
    config.paths.candidates_file.parent.mkdir(parents=True, exist_ok=True)
    config.paths.candidates_file.write_text("", encoding="utf-8")

    with pytest.raises(CandidateGenerationError, match="already exists"):
        run_candidate_generation(config, generator=FakeGenerator([]))


def _write_config(
    tmp_path: Path,
    train_file: Path,
    *,
    group_size: int = 1,
    overwrite: bool = True,
) -> Path:
    base_model_path = tmp_path / "base_model"
    sft_adapter_path = tmp_path / "sft_adapter"
    base_model_path.mkdir()
    sft_adapter_path.mkdir()
    config_path = tmp_path / "dpo_config.yaml"
    config_path.write_text(
        f"""
paths:
  base_model_path: {base_model_path}
  sft_adapter_path: {sft_adapter_path}
  train_file: {train_file}
  candidates_file: {tmp_path / "candidates.jsonl"}
  candidate_summary_file: {tmp_path / "candidate_summary.json"}
candidate_generation:
  group_size: {group_size}
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 128
  seed: 42
  overwrite: {str(overwrite).lower()}
""".strip(),
        encoding="utf-8",
    )
    return config_path


def _write_train_file(tmp_path: Path, sample_ids: list[str]) -> Path:
    train_file = tmp_path / "train.jsonl"
    rows = []
    for sample_id in sample_ids:
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": f"user {sample_id}"},
                ],
                "completion": [{"role": "assistant", "content": _valid_output("gold")}],
                "metadata": {
                    "sample_id": sample_id,
                    "report_title": f"title {sample_id}",
                    "report_date": "2025-01-01",
                    "broker": "broker",
                },
            }
        )
    train_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return train_file


def _valid_output(reasoning: str) -> str:
    return json.dumps(
        {
            "reasoning": reasoning,
            "factor_formula": "close",
            "factor_python": "def compute_factor(close):\n    return close",
            "required_inputs": ["close"],
            "inavailable_inputs": [],
            "sample_id": "must_not_be_kept",
        },
        ensure_ascii=False,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
