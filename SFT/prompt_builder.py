from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import re
from textwrap import dedent
from typing import Any

from extracter.parser.data_dict_parser import DataDictionary
from extracter.parser.data_dict_parser import load_data_dictionary


GENERATION_OUTPUT_FIELDS = (
    "reasoning",
    "factor_formula",
    "factor_python",
    "required_inputs",
    "inavailable_inputs",
)

SYSTEM_PROMPT = dedent(
    """
    你是量化因子生成助手，目标是根据给定现象构造“可计算单因子”。

    你必须严格遵守以下规则：
    - 只能输出 JSON，不要输出 markdown，不要输出额外解释
    - 顶层 JSON 只能包含 reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs
    - 这是因子定义生成任务，不是文本总结任务
    - 只可使用给定字段白名单中的字段，禁止发明文档外字段
    - 仅依赖日度数据
    - 禁止分钟、tick 或其他日内数据
    - reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs 五者必须前后一致
    - reasoning 必须直接解释因子逻辑与定义
    - factor_formula 必须单独给出清晰、可读的数学表达式或定义
    - factor_formula 中的变量名优先使用 required_inputs 中的英文字段名，并尽量与数据字典字段一致
    - required_inputs 只能列出实际参与计算的字段，不能多写，也不能漏写
    - factor_python 必须是单个 compute_factor 函数, 默认环境中已导入 pandas as pd 和 numpy as np
    - compute_factor 的参数必须与 required_inputs 完全一致，顺序也必须一致
    - 输入的每个参数都是 DataFrame，index=日期，columns=股票；输出也必须是同结构 DataFrame
    - 绝对禁止逐股票 for 循环，绝对禁止对 DataFrame 遍历 `for t in range(...)`，也不允许使用 `df[t]` 这种整数列索引,只允许 pandas / numpy 风格的宽表向量化计算
    - 绝对禁止引入未来函数！计算只能使用历史数据，严禁在 shift() 中使用负数（如 shift(-1)）
    - 因子公式必须具有现实的金融意义，绝不允许出现 `A - A`、`A / A`、`A * 0` 等导致计算结果为常数的无意义数学运算
    - 注意除法安全！在 factor_python 中进行除法运算时，必须处理分母为 0 的情况（如将 0 替换为 NaN），防止产生 inf 异常
    - 由于回测总数据量只有1年（252天），所有时间序列参数（包括但不限于 .rolling(window=...), .shift(periods=...), .pct_change(periods=...)）绝对不得超过 22！即使原始逻辑中明确要求“年度”、“12个月”或“252天”等长周期，也必须强制将其缩短或近似为22天以内
    - 所有计算必须能处理 NaN
    - 禁止 print、logging 和代码注释
    - 若原始逻辑依赖白名单外字段，应先做合理近似；无法近似时再写入 inavailable_inputs
    - assistant 输出中不得包含 sample_id、report_title、report_date、broker、inspiration 等元信息
    """
).strip()

_CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_INSPIRATION_PATTERN = re.compile(
    r"现象描述(?:\s*/\s*研究观察)?：\s*(.*?)\s*(?:允许使用的字段白名单|可用字段)：",
    re.DOTALL,
)
_FIELD_SPLIT_PATTERN = re.compile(r"[.,，:\s：]+")


def build_allowed_fields_text(data_dictionary: DataDictionary) -> str:
    lines: list[str] = []
    for table_name in sorted(data_dictionary.tables):
        table = data_dictionary.tables[table_name]
        fields = [
            field_name
            for field_name in table.fields
            if data_dictionary.has_field(field_name)
        ]
        lines.append(f"- {table_name}: {', '.join(fields)}")
    return "\n".join(lines)


def build_user_prompt_from_inspiration(inspiration: str, allowed_fields_text: str) -> str:
    normalized_inspiration = inspiration or "无"
    return dedent(
        f"""
        任务：根据给定现象与约束，生成单因子定义结果。

        输出格式必须是 JSON 对象，形如：
        {{
          "reasoning": "...",
          "factor_formula": "...",
          "factor_python": "def compute_factor(...):\\n    ...",
          "required_inputs": ["field_a", "field_b"],
          "inavailable_inputs": []
        }}

        现象描述 / 研究观察：
        {normalized_inspiration}

        允许使用的字段白名单：
        {allowed_fields_text}

        生成要求：
        1. 仅输出 reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs 五个字段
        2. reasoning 只解释金融逻辑与定义，不要复述来源
        3. factor_formula 必须给出明确数学定义，并与 reasoning、factor_python 一致
        4. required_inputs 必须只列出实际使用、且在白名单内的字段
        5. factor_python 必须是向量化宽表实现，函数名固定为 compute_factor
        6. 若原始逻辑依赖不可用字段，请优先做合理近似；确实无法替代时再写入 inavailable_inputs
        7. 只使用日频数据，不得使用 paused，不得输出额外文字
        """
    ).strip()


def build_user_prompt(sample: dict[str, Any], allowed_fields_text: str) -> str:
    return build_user_prompt_from_inspiration(sample.get("inspiration", ""), allowed_fields_text)


def build_generation_payload(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "reasoning": sample.get("reasoning", ""),
        "factor_formula": sample.get("factor_formula", ""),
        "factor_python": sample.get("factor_python", ""),
        "required_inputs": sample.get("required_inputs", []),
        "inavailable_inputs": sample.get("inavailable_inputs", []),
    }


def build_assistant_content(sample: dict[str, Any]) -> str:
    return json.dumps(build_generation_payload(sample), ensure_ascii=False)


def build_prompt_messages(sample: dict[str, Any], allowed_fields_text: str) -> list[dict[str, str]]:
    return build_inference_messages(
        inspiration=sample.get("inspiration", ""),
        allowed_fields_text=allowed_fields_text,
    )


def build_inference_messages(*, inspiration: str, allowed_fields_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt_from_inspiration(inspiration, allowed_fields_text),
        },
    ]


def build_completion_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": build_assistant_content(sample),
        }
    ]


def build_prompt_completion_record(sample: dict[str, Any], allowed_fields_text: str) -> dict[str, Any]:
    return {
        "prompt": build_prompt_messages(sample, allowed_fields_text),
        "completion": build_completion_messages(sample),
        "metadata": {
            "sample_id": sample.get("sample_id"),
            "report_title": sample.get("report_title"),
            "report_date": sample.get("report_date"),
            "broker": sample.get("broker"),
            "inspiration": sample.get("inspiration"),
            "class": sample.get("class"),
            "version": sample.get("version"),
            "length_input": sample.get("length_input"),
            "length_output": sample.get("length_output"),
        },
    }


def parse_model_output(content: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(extract_json_object_text(content))
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc.msg}"
    normalized, errors = normalize_generation_payload(payload)
    if errors:
        return normalized, "; ".join(errors)
    return normalized, None


def normalize_generation_payload(
    payload: Any,
    *,
    allow_extra_keys: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    normalized = {
        "reasoning": "",
        "factor_formula": "",
        "factor_python": "",
        "required_inputs": [],
        "inavailable_inputs": [],
    }
    errors: list[str] = []

    if not isinstance(payload, dict):
        return normalized, ["payload_must_be_object"]

    unexpected_keys = sorted(key for key in payload if key not in GENERATION_OUTPUT_FIELDS)
    if unexpected_keys and not allow_extra_keys:
        errors.append("unexpected_keys:" + ",".join(unexpected_keys))

    for field_name in ("reasoning", "factor_formula", "factor_python"):
        value = payload.get(field_name)
        if isinstance(value, str):
            normalized[field_name] = value
        elif value is None:
            errors.append(f"missing_field:{field_name}")
        else:
            normalized[field_name] = str(value)
            errors.append(f"invalid_type:{field_name}")

    normalized["required_inputs"], required_errors = _normalize_string_list(
        payload.get("required_inputs"),
        field_name="required_inputs",
    )
    normalized["inavailable_inputs"], unavailable_errors = _normalize_string_list(
        payload.get("inavailable_inputs"),
        field_name="inavailable_inputs",
    )
    errors.extend(required_errors)
    errors.extend(unavailable_errors)

    return normalized, errors


def normalize_completion_content(content: str) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        parsed_payload, parse_error = parse_model_output(content)
        if parsed_payload is None:
            return {
                "reasoning": "",
                "factor_formula": "",
                "factor_python": "",
                "required_inputs": [],
                "inavailable_inputs": [],
            }, [parse_error or "invalid_completion_content"]
        errors = [parse_error] if parse_error else []
        return parsed_payload, errors
    return normalize_generation_payload(payload, allow_extra_keys=True)


def extract_json_object_text(content: str) -> str:
    if not isinstance(content, str):
        raise json.JSONDecodeError("Content is not a string.", str(content), 0)

    match = _CODE_BLOCK_PATTERN.search(content)
    if match is not None:
        return match.group(1)

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("No JSON object found.", content, 0)
    return content[start : end + 1]


def extract_inspiration_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = _INSPIRATION_PATTERN.search(content)
        if match is not None:
            return match.group(1).strip()
    return None


def _normalize_string_list(value: Any, *, field_name: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if value is None:
        return [], [f"missing_field:{field_name}"]
    if isinstance(value, list):
        return _sanitize_field_name_items(value), errors
    return _sanitize_field_name_items([value]), [f"invalid_type:{field_name}"]


def _sanitize_field_name_items(items: list[Any]) -> list[str]:
    allowed_fields = _load_allowed_factor_fields()
    sanitized: list[str] = []
    for item in items:
        raw_value = str(item)
        normalized_value = raw_value.strip()
        if not normalized_value:
            continue
        tokens = [
            token.strip()
            for token in _FIELD_SPLIT_PATTERN.split(normalized_value)
            if token.strip()
        ]
        matched_tokens = [token for token in tokens if token in allowed_fields]
        if matched_tokens:
            sanitized.extend(matched_tokens)
        else:
            sanitized.append(raw_value)
    return sanitized


@lru_cache(maxsize=1)
def _load_allowed_factor_fields() -> frozenset[str]:
    project_root = Path(__file__).resolve().parent.parent
    data_dictionary = load_data_dictionary(project_root / "extracter/data_dict.md")
    return data_dictionary.allowed_factor_fields
