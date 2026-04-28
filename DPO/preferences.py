from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
from typing import Any

from SFT.io_utils import ensure_directory, read_jsonl, write_json, write_jsonl
from SFT.prompt_builder import GENERATION_OUTPUT_FIELDS

from .config import DPOConfig


class PreferenceBuildError(RuntimeError):
    pass


def run_preference_building(config: DPOConfig) -> dict[str, Any]:
    _validate_stage_paths(config)
    records = _load_validated_records(config.paths.validated_candidates_file)
    groups = _group_records(records)

    ensure_directory(config.paths.dpo_train_file.parent)
    ensure_directory(config.paths.hard_cases_file.parent)
    ensure_directory(config.paths.preference_summary_file.parent)

    pairs: list[dict[str, Any]] = []
    hard_cases: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        pair, hard_case = _build_group_preference(group_id, groups[group_id], config)
        if pair is not None:
            pairs.append(pair)
        if hard_case is not None:
            hard_cases.append(hard_case)

    write_jsonl(config.paths.dpo_train_file, pairs)
    write_jsonl(config.paths.hard_cases_file, hard_cases)
    summary = _build_summary(config, groups, pairs, hard_cases)
    write_json(config.paths.preference_summary_file, summary)
    return summary


def _build_group_preference(
    group_id: str,
    records: list[dict[str, Any]],
    config: DPOConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    consistency_error = _group_consistency_error(records)
    if consistency_error is not None:
        return None, _build_hard_case(records, consistency_error)

    successes = [record for record in records if record.get("backtest", {}).get("success") is True]
    if len(successes) >= 2:
        return _build_multi_success_pair(records, successes, config)
    if len(successes) == 1:
        return _build_single_success_pair(records, successes[0], config)
    return None, _build_hard_case(records, "no_backtest_success_candidate")


def _build_multi_success_pair(
    records: list[dict[str, Any]],
    successes: list[dict[str, Any]],
    config: DPOConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    metric_name = config.preference.primary_metric
    sorted_successes = sorted(
        successes,
        key=lambda record: (-_metric_value(record, metric_name), int(record.get("candidate_index", 0))),
    )
    chosen = sorted_successes[0]
    rejected = sorted(
        successes,
        key=lambda record: (_metric_value(record, metric_name), -int(record.get("candidate_index", 0))),
    )[0]
    gap = _metric_value(chosen, metric_name) - _metric_value(rejected, metric_name)
    if gap < config.preference.min_abs_ic_gap:
        return None, _build_hard_case(records, "min_abs_ic_gap_not_met")
    return _serialize_pair(
        records=records,
        chosen=chosen,
        rejected=rejected,
        preference_rule=f"highest_{metric_name}_vs_lowest_{metric_name}",
        metric_gap=gap,
        metric_name=metric_name,
    )


def _build_single_success_pair(
    records: list[dict[str, Any]],
    chosen: dict[str, Any],
    config: DPOConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not config.preference.allow_single_success_pair:
        return None, _build_hard_case(records, "single_success_pair_disabled")
    failures = [record for record in records if record.get("backtest", {}).get("success") is not True]
    if not failures:
        return None, _build_hard_case(records, "single_success_without_rejected")
    rejected = sorted(
        failures,
        key=lambda record: (-int(record.get("failure_severity") or 0), int(record.get("candidate_index", 0))),
    )[0]
    return _serialize_pair(
        records=records,
        chosen=chosen,
        rejected=rejected,
        preference_rule="single_backtest_success_vs_hard_failure",
        metric_gap=None,
        metric_name=config.preference.primary_metric,
    )


def _serialize_pair(
    *,
    records: list[dict[str, Any]],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    preference_rule: str,
    metric_gap: float | None,
    metric_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        chosen_content = _assistant_json(chosen.get("payload"))
        rejected_content = _assistant_json(rejected.get("payload"))
    except (TypeError, ValueError):
        return None, _build_hard_case(records, "assistant_json_serialization_failed")

    first = records[0]
    metadata = first.get("metadata") if isinstance(first.get("metadata"), dict) else {}
    return (
        {
            "prompt": first.get("prompt"),
            "chosen": chosen_content,
            "rejected": rejected_content,
            "metadata": {
                "sample_id": first.get("sample_id"),
                "group_id": first.get("group_id"),
                "report_title": metadata.get("report_title"),
                "source_split": metadata.get("source_split"),
                "candidate_group_size": len(records),
                "chosen_candidate_id": chosen.get("candidate_id"),
                "rejected_candidate_id": rejected.get("candidate_id"),
                "chosen_metric": _metric_payload(chosen, metric_name),
                "rejected_metric": _metric_payload(rejected, metric_name),
                "preference_rule": preference_rule,
                f"{metric_name}_gap": metric_gap,
            },
        },
        None,
    )


def _assistant_json(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    ordered = {}
    for field_name in GENERATION_OUTPUT_FIELDS:
        if field_name not in payload:
            raise ValueError(f"missing payload field: {field_name}")
        ordered[field_name] = payload[field_name]
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _metric_payload(record: dict[str, Any], metric_name: str) -> dict[str, Any]:
    backtest = record.get("backtest") if isinstance(record.get("backtest"), dict) else {}
    suffix = metric_name.removeprefix("abs_")
    if suffix.startswith("ic_"):
        horizon = suffix.split("_", 1)[1]
    else:
        horizon = "1"
    return {
        f"ic_{horizon}": backtest.get(f"ic_{horizon}"),
        f"abs_ic_{horizon}": backtest.get(f"abs_ic_{horizon}"),
        f"rank_ic_{horizon}": backtest.get(f"rank_ic_{horizon}"),
        f"ir_{horizon}": backtest.get(f"ir_{horizon}"),
    }


def _group_consistency_error(records: list[dict[str, Any]]) -> str | None:
    if not records:
        return "empty_group"
    if any(not record.get("group_id") or not record.get("sample_id") or not record.get("prompt") for record in records):
        return "missing_group_required_field"
    sample_ids = {json.dumps(record.get("sample_id"), ensure_ascii=False) for record in records}
    prompts = {json.dumps(record.get("prompt"), ensure_ascii=False, sort_keys=True) for record in records}
    source_splits = {_source_split(record) for record in records}
    if len(sample_ids) != 1:
        return "inconsistent_sample_id"
    if len(prompts) != 1:
        return "inconsistent_prompt"
    if source_splits != {"train"}:
        return "non_train_source_split"
    return None


def _build_hard_case(records: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    first = records[0] if records else {}
    metadata = first.get("metadata") if isinstance(first.get("metadata"), dict) else {}
    failure_type_counts = Counter(
        record.get("failure_type")
        for record in records
        if record.get("failure_type") is not None
    )
    return {
        "sample_id": first.get("sample_id"),
        "group_id": first.get("group_id"),
        "report_title": metadata.get("report_title"),
        "source_split": metadata.get("source_split"),
        "candidate_count": len(records),
        "backtest_success_count": sum(
            1 for record in records if record.get("backtest", {}).get("success") is True
        ),
        "failure_type_counts": dict(failure_type_counts),
        "hard_case_reason": reason,
        "candidate_ids": [record.get("candidate_id") for record in records],
    }


def _build_summary(
    config: DPOConfig,
    groups: dict[str, list[dict[str, Any]]],
    pairs: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_name = config.preference.primary_metric
    gap_key = f"{metric_name}_gap"
    gaps = [
        pair["metadata"].get(gap_key)
        for pair in pairs
        if isinstance(pair["metadata"].get(gap_key), (int, float))
    ]
    source_split_counts = Counter(
        _source_split(records[0]) if records else None
        for records in groups.values()
    )
    source_split_counts.pop(None, None)
    return {
        "stage": "preference_pair_building",
        "input_file": str(config.paths.validated_candidates_file),
        "dpo_train_file": str(config.paths.dpo_train_file),
        "hard_cases_file": str(config.paths.hard_cases_file),
        "num_groups": len(groups),
        "num_pairs": len(pairs),
        "num_hard_cases": len(hard_cases),
        "rule_counts": dict(Counter(pair["metadata"]["preference_rule"] for pair in pairs)),
        "hard_case_reason_counts": dict(Counter(case["hard_case_reason"] for case in hard_cases)),
        f"mean_{metric_name}_gap": sum(gaps) / len(gaps) if gaps else None,
        f"median_{metric_name}_gap": median(gaps) if gaps else None,
        "source_split_counts": dict(source_split_counts),
    }


def _metric_value(record: dict[str, Any], metric_name: str) -> float:
    value = record.get("backtest", {}).get(metric_name)
    if not isinstance(value, (int, float)):
        return float("-inf")
    return float(value)


def _source_split(record: dict[str, Any]) -> Any:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return metadata.get("source_split")


def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        group_id = record.get("group_id") or f"__missing_group_{index}"
        groups[str(group_id)].append(record)
    return groups


def _load_validated_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_jsonl(path), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreferenceBuildError(f"Invalid validated JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise PreferenceBuildError(f"Validated row must be an object at line {line_number}")
        records.append(record)
    return records


def _validate_stage_paths(config: DPOConfig) -> None:
    if not config.paths.validated_candidates_file.exists():
        raise PreferenceBuildError(f"Validated candidates file not found: {config.paths.validated_candidates_file}")
    for output_path in (
        config.paths.dpo_train_file,
        config.paths.hard_cases_file,
        config.paths.preference_summary_file,
    ):
        if output_path.exists() and not config.preference.overwrite:
            raise PreferenceBuildError(f"Output file already exists: {output_path}")
