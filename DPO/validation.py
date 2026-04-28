from __future__ import annotations

from collections import Counter
import ast
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any, Protocol

from extracter.parser.data_dict_parser import DataDictionary, load_data_dictionary
from extracter.validation.result_validation import validate_generated_sample
from SFT.io_utils import ensure_directory, read_jsonl, write_json, write_jsonl
from SFT.prompt_builder import GENERATION_OUTPUT_FIELDS

from .config import DPOConfig


FAILURE_SEVERITY = {
    "invalid_json": 8,
    "missing_required_keys": 7,
    "invalid_field_type": 6,
    "python_ast_error": 5,
    "arg_mismatch": 4,
    "whitelist_violation": 3,
    "validator_error": 2,
    "backtest_error": 1,
}


class ValidationError(RuntimeError):
    pass


class Backtester(Protocol):
    def run(self, payload: dict[str, Any], *, ic_horizon: int) -> dict[str, Any]:
        pass


class LocalBacktester:
    def run(self, payload: dict[str, Any], *, ic_horizon: int) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parent.parent
        backtest_dir = project_root / "backtest"
        with _temporary_sys_path(backtest_dir):
            run_backtest = importlib.import_module("run_backtest")
            factor_backtest = importlib.import_module("factor_backtest")

        data = run_backtest.load_data()
        local_env: dict[str, Any] = {}
        exec(payload["factor_python"], {}, local_env)
        factor_func = local_env.get("compute_factor")
        if not callable(factor_func):
            raise RuntimeError("compute_factor not found")

        inputs = {
            field_name: data[field_name]
            for field_name in payload["required_inputs"]
            if field_name in data
        }
        missing = set(payload["required_inputs"]) - set(inputs)
        if missing:
            raise RuntimeError("missing backtest fields: " + ", ".join(sorted(missing)))

        factor_value = factor_func(**inputs)
        result = factor_backtest.analyze_factor(factor_value, periods=(ic_horizon,))
        if result is None:
            raise RuntimeError("analyze_factor returned None")

        ic_value = _read_metric(result.ic_summary, ic_horizon, "IC_Mean")
        rank_ic_value = _read_metric(result.rank_ic_summary, ic_horizon, "IC_Mean")
        ir_value = _read_metric(result.ic_summary, ic_horizon, "IR")
        return {
            "success": True,
            f"ic_{ic_horizon}": ic_value,
            f"abs_ic_{ic_horizon}": abs(ic_value) if ic_value is not None else None,
            f"rank_ic_{ic_horizon}": rank_ic_value,
            f"ir_{ic_horizon}": ir_value,
            "error": None,
        }


def run_validation(
    config: DPOConfig,
    *,
    backtester: Backtester | None = None,
) -> dict[str, Any]:
    _validate_stage_paths(config)
    data_dictionary = load_data_dictionary(config.paths.data_dict_file)
    candidates = _load_candidates(config.paths.candidates_file)
    backtester = backtester or LocalBacktester()

    ensure_directory(config.paths.validated_candidates_file.parent)
    ensure_directory(config.paths.validation_summary_file.parent)

    records = [
        _validate_candidate(
            candidate=candidate,
            config=config,
            data_dictionary=data_dictionary,
            backtester=backtester,
        )
        for candidate in candidates
    ]
    write_jsonl(config.paths.validated_candidates_file, records)
    summary = _build_summary(config, records)
    write_json(config.paths.validation_summary_file, summary)
    return summary


def _validate_candidate(
    *,
    candidate: dict[str, Any],
    config: DPOConfig,
    data_dictionary: DataDictionary,
    backtester: Backtester,
) -> dict[str, Any]:
    payload = candidate.get("parsed_json")
    validation = _empty_validation()
    failure_type: str | None = None
    failure_severity = 0
    identity_errors = _candidate_identity_errors(candidate)
    for error in identity_errors:
        _add_error(validation, error)

    if not isinstance(payload, dict):
        failure_type = "invalid_json"
        _add_error(validation, "invalid_json: parsed_json must be an object")
    else:
        validation["valid_json"] = True
        failure_type = _validate_payload(payload, validation, data_dictionary)
        if failure_type is None and identity_errors:
            failure_type = "validator_error"

    if failure_type is not None:
        failure_severity = FAILURE_SEVERITY[failure_type]
        backtest = _empty_backtest(
            config.validation.ic_horizon,
            error=f"skipped_due_to_{failure_type}",
        )
    else:
        backtest, failure_type, failure_severity = _run_backtest(
            payload=payload,
            config=config,
            backtester=backtester,
        )

    return {
        "sample_id": candidate.get("sample_id"),
        "group_id": candidate.get("group_id"),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_index": candidate.get("candidate_index"),
        "group_size": candidate.get("group_size"),
        "prompt": candidate.get("prompt"),
        "payload": payload if isinstance(payload, dict) else None,
        "validation": validation,
        "backtest": backtest,
        "failure_type": failure_type,
        "failure_severity": failure_severity,
        "metadata": _build_metadata(candidate),
    }


def _validate_payload(
    payload: dict[str, Any],
    validation: dict[str, Any],
    data_dictionary: DataDictionary,
) -> str | None:
    missing = [field for field in GENERATION_OUTPUT_FIELDS if field not in payload]
    if missing:
        _add_error(validation, "missing_required_keys: " + ",".join(missing))
        return "missing_required_keys"
    validation["required_keys"] = True

    type_errors = _field_type_errors(payload)
    if type_errors:
        for error in type_errors:
            _add_error(validation, "invalid_field_type: " + error)
        return "invalid_field_type"
    validation["valid_field_types"] = True

    try:
        tree = ast.parse(payload["factor_python"])
    except SyntaxError as exc:
        _add_error(validation, f"python_ast_error: {exc.msg}")
        return "python_ast_error"
    validation["python_ast_parse"] = True

    if not _has_single_compute_factor_with_args(tree, payload["required_inputs"]):
        _add_error(validation, "arg_mismatch: compute_factor args must match required_inputs")
        return "arg_mismatch"
    validation["required_inputs_arg_match"] = True

    validator_errors = validate_generated_sample(
        {"inspiration": "dpo_candidate", **payload},
        data_dictionary,
    )
    if _has_whitelist_error(validator_errors):
        for error in validator_errors:
            _add_error(validation, "validator: " + error)
        return "whitelist_violation"
    validation["whitelist_compliance"] = True

    if validator_errors:
        for error in validator_errors:
            _add_error(validation, "validator: " + error)
        return "validator_error"
    validation["validator_pass"] = True
    return None


def _run_backtest(
    *,
    payload: dict[str, Any],
    config: DPOConfig,
    backtester: Backtester,
) -> tuple[dict[str, Any], str | None, int]:
    horizon = config.validation.ic_horizon
    if not config.validation.require_backtest:
        return (
            _empty_backtest(horizon, error="backtest_skipped_by_config"),
            "backtest_error",
            FAILURE_SEVERITY["backtest_error"],
        )
    try:
        raw_result = backtester.run(payload, ic_horizon=horizon)
    except Exception as exc:
        return (
            _empty_backtest(horizon, error=_format_error(exc)),
            "backtest_error",
            FAILURE_SEVERITY["backtest_error"],
        )

    backtest = _normalize_backtest_result(raw_result, horizon)
    if not backtest["success"]:
        return backtest, "backtest_error", FAILURE_SEVERITY["backtest_error"]
    return backtest, None, 0


def _field_type_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in ("reasoning", "factor_formula", "factor_python"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field_name} must be a non-empty string")
    for field_name in ("required_inputs", "inavailable_inputs"):
        value = payload.get(field_name)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            errors.append(f"{field_name} must be a list of strings")
    return errors


def _has_single_compute_factor_with_args(tree: ast.AST, required_inputs: list[str]) -> bool:
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_factor"
    ]
    if len(functions) != 1:
        return False
    args = [arg.arg for arg in functions[0].args.args]
    return args == required_inputs


def _has_whitelist_error(errors: list[str]) -> bool:
    markers = (
        "unsupported field",
        "cannot contain paused",
        "cannot reference paused",
        "not listed in required_inputs",
    )
    return any(any(marker in error for marker in markers) for error in errors)


def _empty_validation() -> dict[str, Any]:
    return {
        "valid_json": False,
        "required_keys": False,
        "valid_field_types": False,
        "validator_pass": False,
        "python_ast_parse": False,
        "required_inputs_arg_match": False,
        "whitelist_compliance": False,
        "errors": [],
    }


def _candidate_identity_errors(candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in ("sample_id", "group_id", "candidate_id"):
        if not candidate.get(field_name):
            errors.append(f"missing_candidate_field:{field_name}")
    return errors


def _empty_backtest(horizon: int, *, error: str | None) -> dict[str, Any]:
    return {
        "success": False,
        f"ic_{horizon}": None,
        f"abs_ic_{horizon}": None,
        f"rank_ic_{horizon}": None,
        f"ir_{horizon}": None,
        "error": error,
    }


def _normalize_backtest_result(raw_result: dict[str, Any], horizon: int) -> dict[str, Any]:
    backtest = _empty_backtest(horizon, error=raw_result.get("error"))
    backtest["success"] = bool(raw_result.get("success"))
    for key in (f"ic_{horizon}", f"abs_ic_{horizon}", f"rank_ic_{horizon}", f"ir_{horizon}"):
        backtest[key] = raw_result.get(key)
    ic_key = f"ic_{horizon}"
    abs_key = f"abs_ic_{horizon}"
    if backtest[abs_key] is None and backtest[ic_key] is not None:
        backtest[abs_key] = abs(backtest[ic_key])
    return backtest


def _build_summary(config: DPOConfig, records: list[dict[str, Any]]) -> dict[str, Any]:
    horizon = config.validation.ic_horizon
    failure_counts = Counter(
        record["failure_type"]
        for record in records
        if record["failure_type"] is not None
    )
    backtest_successes = [
        record for record in records if record["backtest"].get("success") is True
    ]
    abs_ic_values = [
        record["backtest"].get(f"abs_ic_{horizon}")
        for record in backtest_successes
        if isinstance(record["backtest"].get(f"abs_ic_{horizon}"), (int, float))
    ]
    rank_ic_values = [
        record["backtest"].get(f"rank_ic_{horizon}")
        for record in backtest_successes
        if isinstance(record["backtest"].get(f"rank_ic_{horizon}"), (int, float))
    ]
    num_candidates = len(records)
    return {
        "stage": "validation_and_backtest",
        "input_file": str(config.paths.candidates_file),
        "output_file": str(config.paths.validated_candidates_file),
        "num_candidates": num_candidates,
        "valid_json_count": sum(record["validation"]["valid_json"] for record in records),
        "required_keys_count": sum(record["validation"]["required_keys"] for record in records),
        "python_ast_parse_count": sum(record["validation"]["python_ast_parse"] for record in records),
        "validator_pass_count": sum(record["validation"]["validator_pass"] for record in records),
        "backtest_success_count": len(backtest_successes),
        "backtest_success_rate": len(backtest_successes) / num_candidates if num_candidates else 0.0,
        "failure_type_counts": dict(failure_counts),
        f"ic_{horizon}": _abs_distribution(abs_ic_values),
        f"rank_ic_{horizon}": _distribution(rank_ic_values),
    }


def _abs_distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    sorted_values = sorted(values)
    return {
        "mean_abs": sum(values) / len(values),
        "median_abs": median(values),
        "p25_abs": _percentile(sorted_values, 0.25),
        "p75_abs": _percentile(sorted_values, 0.75),
    }


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "median": median(values),
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_jsonl(path), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid candidates JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValidationError(f"Candidate row must be an object at line {line_number}")
        rows.append(row)
    return rows


def _validate_stage_paths(config: DPOConfig) -> None:
    if not config.paths.candidates_file.exists():
        raise ValidationError(f"Candidates file not found: {config.paths.candidates_file}")
    if not config.paths.data_dict_file.exists():
        raise ValidationError(f"Data dictionary file not found: {config.paths.data_dict_file}")
    if config.paths.validated_candidates_file.exists() and not config.validation.overwrite:
        raise ValidationError(f"Output file already exists: {config.paths.validated_candidates_file}")
    if config.paths.validation_summary_file.exists() and not config.validation.overwrite:
        raise ValidationError(f"Summary file already exists: {config.paths.validation_summary_file}")


def _build_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    return {
        "source_split": metadata.get("source_split"),
        "report_title": metadata.get("report_title"),
        "base_model_path": metadata.get("base_model_path"),
        "sft_adapter_path": metadata.get("sft_adapter_path"),
    }


def _add_error(validation: dict[str, Any], error: str) -> None:
    validation["errors"].append(error)


def _format_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {message}"


def _read_metric(summary: Any, horizon: int, column: str) -> float | None:
    try:
        value = summary.loc[horizon, column]
    except Exception:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@contextmanager
def _temporary_sys_path(path: Path):
    value = str(path)
    added = value not in sys.path
    if added:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(value)
            except ValueError:
                pass
