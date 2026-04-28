from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Protocol

from SFT.io_utils import ensure_directory, read_jsonl, write_json, write_jsonl
from SFT.prompt_builder import GENERATION_OUTPUT_FIELDS, extract_json_object_text

from .config import DPOConfig


class CandidateGenerationError(RuntimeError):
    pass


class CandidateGenerator(Protocol):
    def generate(self, prompt: list[dict[str, str]]) -> str:
        pass


class LocalHFCandidateGenerator:
    def __init__(self, config: DPOConfig) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None
        self._seeded = False

    def generate(self, prompt: list[dict[str, str]]) -> str:
        import torch

        model, tokenizer = self._load_model_and_tokenizer()
        self._seed_once(torch)

        prompt_text = self._render_prompt_text(tokenizer, prompt)
        encoded = tokenizer(prompt_text, return_tensors="pt")
        device = self._resolve_device(torch)
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if hasattr(value, "to")
        }
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._config.candidate_generation.max_new_tokens,
            "do_sample": self._config.candidate_generation.temperature > 0,
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = self._config.candidate_generation.temperature
            generate_kwargs["top_p"] = self._config.candidate_generation.top_p

        with torch.inference_mode():
            outputs = model.generate(**model_inputs, **generate_kwargs)

        prompt_width = model_inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_width:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _seed_once(self, torch_module: Any) -> None:
        if self._seeded:
            return
        torch_module.manual_seed(self._config.candidate_generation.seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(self._config.candidate_generation.seed)
        self._seeded = True

    def _load_model_and_tokenizer(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        generation = self._config.candidate_generation
        model_kwargs: dict[str, Any] = {"trust_remote_code": generation.trust_remote_code}
        if generation.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_resolve_torch_dtype(generation.torch_dtype),
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = _resolve_torch_dtype(generation.torch_dtype)
            if generation.device == "auto":
                model_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(
            self._config.paths.base_model_path,
            trust_remote_code=generation.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self._config.paths.base_model_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(model, self._config.paths.sft_adapter_path)

        if generation.device != "auto" and not generation.load_in_4bit:
            model = model.to(generation.device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def _render_prompt_text(self, tokenizer: Any, prompt: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except TypeError:
                pass
        return "\n\n".join(f"{message['role']}: {message['content']}" for message in prompt) + "\n\nassistant:"

    def _resolve_device(self, torch_module: Any) -> str:
        if self._config.candidate_generation.device != "auto":
            return self._config.candidate_generation.device
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"


def run_candidate_generation(
    config: DPOConfig,
    *,
    generator: CandidateGenerator | None = None,
) -> dict[str, Any]:
    _validate_stage_paths(config)
    samples = _load_train_samples(config.paths.train_file)
    generator = generator or LocalHFCandidateGenerator(config)

    ensure_directory(config.paths.candidates_file.parent)
    ensure_directory(config.paths.candidate_summary_file.parent)

    rows: list[dict[str, Any]] = []
    generation_success_count = 0
    generation_error_count = 0
    parse_success_count = 0
    parse_error_count = 0
    error_type_counts: Counter[str] = Counter()

    for sample in samples:
        sample_id = sample["metadata"]["sample_id"]
        for candidate_index in range(config.candidate_generation.group_size):
            try:
                raw_output = generator.generate(sample["prompt"])
                generation_error = None
                generation_success_count += 1
            except Exception as exc:
                raw_output = ""
                generation_error = _format_error(exc)
                generation_error_count += 1
                error_type_counts["generation_error"] += 1

            parsed_json = None
            parse_error = None
            if generation_error is None:
                parsed_json, parse_error = parse_candidate_output(raw_output)
                if parse_error is None:
                    parse_success_count += 1
                else:
                    parse_error_count += 1
                    error_type_counts["invalid_json"] += 1

            rows.append(
                _build_candidate_record(
                    sample=sample,
                    raw_output=raw_output,
                    parsed_json=parsed_json,
                    parse_error=parse_error,
                    generation_error=generation_error,
                    candidate_index=candidate_index,
                    config=config,
                )
            )

    write_jsonl(config.paths.candidates_file, rows)
    summary = {
        "stage": "candidate_generation",
        "input_file": str(config.paths.train_file),
        "output_file": str(config.paths.candidates_file),
        "base_model_path": str(config.paths.base_model_path),
        "sft_adapter_path": str(config.paths.sft_adapter_path),
        "group_size": config.candidate_generation.group_size,
        "num_input_samples": len(samples),
        "target_candidate_count": len(samples) * config.candidate_generation.group_size,
        "actual_candidate_count": len(rows),
        "generation_success_count": generation_success_count,
        "generation_error_count": generation_error_count,
        "parse_success_count": parse_success_count,
        "parse_error_count": parse_error_count,
        "parse_success_rate": parse_success_count / len(rows) if rows else 0.0,
        "error_type_counts": dict(error_type_counts),
        "seed": config.candidate_generation.seed,
    }
    write_json(config.paths.candidate_summary_file, summary)
    return summary


def parse_candidate_output(content: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(extract_json_object_text(content))
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "invalid_json: payload_must_be_object"
    return {
        field_name: payload[field_name]
        for field_name in GENERATION_OUTPUT_FIELDS
        if field_name in payload
    }, None


def _validate_stage_paths(config: DPOConfig) -> None:
    if not config.paths.base_model_path.exists():
        raise CandidateGenerationError(f"Base model path not found: {config.paths.base_model_path}")
    if not config.paths.sft_adapter_path.exists():
        raise CandidateGenerationError(f"SFT adapter path not found: {config.paths.sft_adapter_path}")
    if not config.paths.train_file.exists():
        raise CandidateGenerationError(f"Train file not found: {config.paths.train_file}")
    if config.paths.candidates_file.exists() and not config.candidate_generation.overwrite:
        raise CandidateGenerationError(f"Output file already exists: {config.paths.candidates_file}")
    if config.paths.candidate_summary_file.exists() and not config.candidate_generation.overwrite:
        raise CandidateGenerationError(f"Summary file already exists: {config.paths.candidate_summary_file}")


def _load_train_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for line_number, line in enumerate(read_jsonl(path), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateGenerationError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
        prompt = sample.get("prompt")
        metadata = sample.get("metadata")
        if not _is_message_list(prompt):
            raise CandidateGenerationError(f"Missing or invalid prompt at line {line_number}")
        if not isinstance(metadata, dict) or not metadata.get("sample_id"):
            raise CandidateGenerationError(f"Missing metadata.sample_id at line {line_number}")
        sample_id = str(metadata["sample_id"])
        if sample_id in seen_sample_ids:
            raise CandidateGenerationError(f"Duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        sample["metadata"]["sample_id"] = sample_id
        samples.append(sample)
    return samples


def _is_message_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for message in value:
        if not isinstance(message, dict):
            return False
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            return False
    return True


def _build_candidate_record(
    *,
    sample: dict[str, Any],
    raw_output: str,
    parsed_json: dict[str, Any] | None,
    parse_error: str | None,
    generation_error: str | None,
    candidate_index: int,
    config: DPOConfig,
) -> dict[str, Any]:
    metadata = sample["metadata"]
    sample_id = metadata["sample_id"]
    generation_config = {
        "temperature": config.candidate_generation.temperature,
        "top_p": config.candidate_generation.top_p,
        "max_new_tokens": config.candidate_generation.max_new_tokens,
        "seed": config.candidate_generation.seed,
    }
    return {
        "sample_id": sample_id,
        "group_id": sample_id,
        "candidate_id": f"{sample_id}_cand_{candidate_index:02d}",
        "candidate_index": candidate_index,
        "group_size": config.candidate_generation.group_size,
        "prompt": sample["prompt"],
        "raw_output": raw_output,
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "generation_error": generation_error,
        "metadata": {
            "report_title": metadata.get("report_title"),
            "report_date": metadata.get("report_date"),
            "broker": metadata.get("broker"),
            "source_split": "train",
            "base_model_path": str(config.paths.base_model_path),
            "sft_adapter_path": str(config.paths.sft_adapter_path),
            "generation_config": generation_config,
        },
    }


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
