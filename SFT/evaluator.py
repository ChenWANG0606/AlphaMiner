from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from extracter.parser.data_dict_parser import load_data_dictionary
from extracter.validation.result_validation import validate_generated_sample

from .inference_backends import BackendResponse, InferenceBackend
from .inference_config import InferenceConfig
from .progress import stage_progress
from .prompt_builder import (
    GENERATION_OUTPUT_FIELDS,
    extract_inspiration_from_messages,
    normalize_completion_content,
    normalize_generation_payload,
)


@dataclass(frozen=True)
class EvaluationResult:
    records: list[dict[str, Any]]
    summary: dict[str, Any]


def evaluate_records(
    lines: list[str],
    *,
    backend: InferenceBackend,
    inference_config: InferenceConfig,
) -> EvaluationResult:
    data_dictionary = load_data_dictionary(inference_config.data_dict_path)
    dataset_rows = [json.loads(raw_line) for raw_line in lines if raw_line.strip()]
    records: list[dict[str, Any]] = []
    failure_type_breakdown: dict[str, int] = {}
    batch_size = inference_config.runtime.eval_batch_size

    metric_totals = {
        "valid_json": 0,
        "required_keys": 0,
        "validator_pass": 0,
        "python_ast_parse": 0,
        "required_inputs_arg_match": 0,
        "whitelist_compliance": 0,
        "required_inputs_exact_match": 0,
        "inavailable_inputs_exact_match": 0,
        "factor_formula_exact_match": 0,
    }

    with stage_progress(total=len(dataset_rows), desc="Eval Samples", unit="sample") as progress_bar:
        for batch_rows in _chunked(dataset_rows, batch_size):
            prompt_messages_batch = [
                _normalize_messages(dataset_row.get("prompt"))
                for dataset_row in batch_rows
            ]
            try:
                batch_responses = backend.generate_batch(prompt_messages_batch)
            except Exception as exc:
                batch_responses = _build_batch_error_responses(
                    batch_size=len(batch_rows),
                    error_text=f"backend_error: {exc}",
                    inference_config=inference_config,
                )

            if len(batch_responses) != len(batch_rows):
                batch_responses = _build_batch_error_responses(
                    batch_size=len(batch_rows),
                    error_text=(
                        "backend_error: backend returned "
                        f"{len(batch_responses)} responses for {len(batch_rows)} prompts"
                    ),
                    inference_config=inference_config,
                )

            for dataset_row, response in zip(batch_rows, batch_responses):
                reference_payload, reference_errors = _load_reference_payload(dataset_row)
                metadata = _extract_metadata(dataset_row, reference_payload)
                prediction_payload, prediction_errors, metrics = _evaluate_prediction(
                    response=response,
                    reference_payload=reference_payload,
                    inspiration=metadata["inspiration"],
                    data_dictionary=data_dictionary,
                )

                errors = [*reference_errors, *prediction_errors]
                for metric_name, value in metrics.items():
                    if metric_name not in metric_totals:
                        continue
                    metric_totals[metric_name] += int(value)
                for error_text in errors:
                    error_key = error_text.split(":", 1)[0]
                    failure_type_breakdown[error_key] = failure_type_breakdown.get(error_key, 0) + 1

                records.append(
                    {
                        "sample_id": metadata["sample_id"],
                        "report_title": metadata["report_title"],
                        "report_date": metadata["report_date"],
                        "broker": metadata["broker"],
                        "inspiration": metadata["inspiration"],
                        "prediction": prediction_payload,
                        "reference": reference_payload,
                        "metrics": metrics,
                        "errors": errors,
                        "raw_response": response.raw_text,
                    }
                )

            progress_bar.advance(f"batch={len(batch_rows)}", amount=len(batch_rows))

    sample_count = len(records)
    summary = {
        "backend_type": records[0]["metrics"].get("backend_type") if records else inference_config.backend.type,
        "model_name": records[0]["metrics"].get("model_name") if records else (inference_config.backend.model or "local_hf"),
        "sample_count": sample_count,
        "success_count": sum(record["metrics"]["valid_json"] for record in records),
        "valid_json_rate": _ratio(metric_totals["valid_json"], sample_count),
        "required_keys_rate": _ratio(metric_totals["required_keys"], sample_count),
        "validator_pass_rate": _ratio(metric_totals["validator_pass"], sample_count),
        "python_ast_parse_rate": _ratio(metric_totals["python_ast_parse"], sample_count),
        "required_inputs_arg_match_rate": _ratio(metric_totals["required_inputs_arg_match"], sample_count),
        "whitelist_compliance_rate": _ratio(metric_totals["whitelist_compliance"], sample_count),
        "required_inputs_exact_match_rate": _ratio(metric_totals["required_inputs_exact_match"], sample_count),
        "inavailable_inputs_exact_match_rate": _ratio(metric_totals["inavailable_inputs_exact_match"], sample_count),
        "factor_formula_exact_match_rate": _ratio(metric_totals["factor_formula_exact_match"], sample_count),
        "failure_type_breakdown": dict(sorted(failure_type_breakdown.items())),
    }
    return EvaluationResult(records=records, summary=summary)


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("Dataset row prompt must be a list of messages.")
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str):
            normalized.append({"role": role, "content": content})
    return normalized


def _load_reference_payload(dataset_row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    completion = dataset_row.get("completion")
    if not isinstance(completion, list) or not completion:
        return _empty_payload(), ["missing_completion"]

    first_message = completion[0]
    if not isinstance(first_message, dict):
        return _empty_payload(), ["invalid_completion_message"]

    content = first_message.get("content")
    if not isinstance(content, str):
        return _empty_payload(), ["invalid_completion_content"]

    return normalize_completion_content(content)


def _extract_metadata(dataset_row: dict[str, Any], reference_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dataset_row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    completion = dataset_row.get("completion")
    completion_payload: dict[str, Any] = {}
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        content = completion[0].get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    completion_payload = parsed
            except json.JSONDecodeError:
                completion_payload = {}

    prompt_messages = dataset_row.get("prompt") if isinstance(dataset_row.get("prompt"), list) else []
    return {
        "sample_id": metadata.get("sample_id") or completion_payload.get("sample_id"),
        "report_title": metadata.get("report_title") or completion_payload.get("report_title"),
        "report_date": metadata.get("report_date") or completion_payload.get("report_date"),
        "broker": metadata.get("broker") or completion_payload.get("broker"),
        "inspiration": (
            metadata.get("inspiration")
            or completion_payload.get("inspiration")
            or extract_inspiration_from_messages(prompt_messages)
        ),
    }


def _evaluate_prediction(
    *,
    response: BackendResponse,
    reference_payload: dict[str, Any],
    inspiration: str | None,
    data_dictionary: Any,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    prediction_payload, normalization_errors = normalize_generation_payload(response.parsed_json)
    errors: list[str] = []

    if response.parse_error:
        errors.append(response.parse_error)
    errors.extend(normalization_errors)

    validator_errors = _run_validator(
        payload=prediction_payload,
        inspiration=inspiration,
        data_dictionary=data_dictionary,
    )
    errors.extend(f"validator:{item}" for item in validator_errors)

    python_parse_ok = _is_python_parse_ok(prediction_payload.get("factor_python", ""))
    arg_match_ok = _is_required_inputs_arg_match(prediction_payload)
    whitelist_ok = _is_whitelist_compliant(validator_errors)

    metrics: dict[str, Any] = {
        "backend_type": response.backend_type,
        "model_name": response.model_name,
        "valid_json": response.parsed_json is not None,
        "required_keys": not any(item.startswith(("missing_field", "invalid_type")) for item in normalization_errors),
        "validator_pass": len(validator_errors) == 0,
        "python_ast_parse": python_parse_ok,
        "required_inputs_arg_match": arg_match_ok,
        "whitelist_compliance": whitelist_ok,
        "required_inputs_exact_match": prediction_payload["required_inputs"] == reference_payload["required_inputs"],
        "inavailable_inputs_exact_match": prediction_payload["inavailable_inputs"] == reference_payload["inavailable_inputs"],
        "factor_formula_exact_match": prediction_payload["factor_formula"].strip() == reference_payload["factor_formula"].strip(),
    }
    return prediction_payload, errors, metrics


def _run_validator(*, payload: dict[str, Any], inspiration: str | None, data_dictionary: Any) -> list[str]:
    validator_payload = {
        "inspiration": inspiration or "",
        **payload,
    }
    return validate_generated_sample(validator_payload, data_dictionary)


def _is_python_parse_ok(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _is_required_inputs_arg_match(payload: dict[str, Any]) -> bool:
    code = payload.get("factor_python", "")
    required_inputs = payload.get("required_inputs", [])
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    compute_factor = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_factor":
            compute_factor = node
            break
    if compute_factor is None:
        return False
    arg_names = [arg.arg for arg in compute_factor.args.args]
    return arg_names == required_inputs


def _is_whitelist_compliant(validator_errors: list[str]) -> bool:
    whitelist_markers = (
        "unsupported field",
        "cannot contain paused",
        "cannot reference paused",
        "not listed in required_inputs",
    )
    return not any(any(marker in error for marker in whitelist_markers) for error in validator_errors)


def _empty_payload() -> dict[str, Any]:
    return {
        field_name: [] if field_name.endswith("inputs") else ""
        for field_name in GENERATION_OUTPUT_FIELDS
    }


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 4)


def _chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [
        items[index : index + chunk_size]
        for index in range(0, len(items), chunk_size)
    ]


def _build_batch_error_responses(
    *,
    batch_size: int,
    error_text: str,
    inference_config: InferenceConfig,
) -> list[BackendResponse]:
    return [
        BackendResponse(
            raw_text="",
            parsed_json=None,
            parse_error=error_text,
            backend_type=inference_config.backend.type,
            model_name=inference_config.backend.model or "local_hf",
        )
        for _ in range(batch_size)
    ]
