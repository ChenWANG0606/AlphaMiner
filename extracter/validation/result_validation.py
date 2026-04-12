from __future__ import annotations

import ast
import re

from ..parser.data_dict_parser import DataDictionary


REQUIRED_KEYS = {
    "inspiration",
    "reasoning",
    "factor_formula",
    "factor_python",
    "required_inputs",
    "inavailable_inputs",
}
FORBIDDEN_TEXT_PHRASES = (
    "研报",
    "报告",
    "文中",
    "作者认为",
    "根据研报",
    "本报告",
    "本文",
    "这一章节",
)
FORMULA_SIGNAL_TOKENS = ("=", "/", "*", "(", ")", "log", "mean", "std", "rank", "corr", "cov", "max", "min", "sum", "delta", "shift", "rolling")
FORMULA_RESERVED_NAMES = {
    "abs",
    "corr",
    "cov",
    "delta",
    "exp",
    "log",
    "max",
    "mean",
    "median",
    "min",
    "rank",
    "rolling",
    "shift",
    "std",
    "sum",
}
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def validate_generated_sample(sample: dict, data_dictionary: DataDictionary) -> list[str]:
    errors: list[str] = []
    missing_keys = sorted(key for key in REQUIRED_KEYS if key not in sample)
    if missing_keys:
        errors.append(f"Missing required keys: {', '.join(missing_keys)}")
        return errors

    for key in ("inspiration", "reasoning", "factor_formula", "factor_python"):
        value = sample.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty string.")

    for key in ("inspiration", "reasoning", "factor_formula"):
        value = sample.get(key, "")
        if isinstance(value, str):
            _validate_text_field(key, value, errors)

    required_inputs = sample.get("required_inputs")
    if not isinstance(required_inputs, list) or any(not isinstance(item, str) for item in required_inputs):
        errors.append("required_inputs must be a list of strings.")
        required_inputs = []
    inavailable_inputs = sample.get("inavailable_inputs")
    if not isinstance(inavailable_inputs, list) or any(not isinstance(item, str) for item in inavailable_inputs):
        errors.append("inavailable_inputs must be a list of strings.")

    for field_name in required_inputs:
        if field_name == "paused":
            errors.append("required_inputs cannot contain paused.")
        elif not data_dictionary.has_field(field_name):
            errors.append(f"required_inputs contains unsupported field: {field_name}")

    factor_formula = sample.get("factor_formula", "")
    if isinstance(factor_formula, str) and factor_formula.strip():
        errors.extend(_validate_factor_formula(factor_formula, required_inputs, data_dictionary))

    factor_python = sample.get("factor_python", "")
    if "print(" in factor_python or "logging." in factor_python:
        errors.append("factor_python cannot contain print or logging.")

    try:
        tree = ast.parse(factor_python)
    except SyntaxError as exc:
        errors.append(f"factor_python is not valid Python: {exc}")
        return errors

    compute_factor = _find_compute_factor(tree)
    if compute_factor is None:
        errors.append("factor_python must define a function named compute_factor.")
        return errors

    arg_names = [arg.arg for arg in compute_factor.args.args]
    if set(arg_names) != set(required_inputs):
        errors.append("Function arguments must exactly match required_inputs.")

    forbidden_names = {"paused"}
    referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for arg_name in arg_names:
        if arg_name not in referenced_names:
            errors.append(f"Function argument is unused: {arg_name}")
    if forbidden_names & referenced_names:
        errors.append("factor_python cannot reference paused.")

    return errors


def _find_compute_factor(tree: ast.AST) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_factor":
            return node
    return None


def _validate_text_field(key: str, value: str, errors: list[str]) -> None:
    for phrase in FORBIDDEN_TEXT_PHRASES:
        if phrase in value:
            errors.append(f"{key} cannot contain source-style phrasing: {phrase}")
            break


def _validate_factor_formula(
    formula: str,
    required_inputs: list[str],
    data_dictionary: DataDictionary,
) -> list[str]:
    errors: list[str] = []
    normalized = formula.strip()
    lowered = normalized.lower()
    has_symbolic_expression = any(token in lowered for token in FORMULA_SIGNAL_TOKENS)
    has_additive_expression = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*[-+]\s*[A-Za-z0-9_]", normalized) is not None
    if not has_symbolic_expression and not has_additive_expression:
        errors.append("factor_formula must contain an explicit mathematical expression or definition.")

    identifiers = {
        identifier
        for identifier in IDENTIFIER_PATTERN.findall(normalized)
        if identifier.lower() not in FORMULA_RESERVED_NAMES
    }
    if "paused" in identifiers:
        errors.append("factor_formula cannot reference paused.")

    recognized_fields = {identifier for identifier in identifiers if data_dictionary.has_field(identifier)}
    unsupported_fields = sorted(field for field in recognized_fields if field not in required_inputs)
    if unsupported_fields:
        errors.append(
            "factor_formula references fields that are not listed in required_inputs: "
            + ", ".join(unsupported_fields)
        )

    return errors
