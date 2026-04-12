from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .configs import (
    DEFAULT_CONTEXT_MODE,
    DEFAULT_ENV_FILE,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_FACTORS_PER_REPORT,
    DEFAULT_MAX_QPS,
    DEFAULT_MAX_SAMPLES_GENERATION,
    build_runtime_config,
)
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extracter pipeline entrypoint.")
    parser.add_argument(
        "--stage",
        choices=("discovery", "generate"),
        required=True,
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--context-mode",
        choices=("section", "full_text"),
        default=DEFAULT_CONTEXT_MODE,
        help="Generation context mode.",
    )
    parser.add_argument(
        "--env-file",
        default=f"extracter/{DEFAULT_ENV_FILE}",
        help="Path to env file.",
    )
    parser.add_argument(
        "--output-path",
        default="extracter/output",
        help="Directory for pipeline outputs.",
    )
    parser.add_argument(
        "--max-factors-per-report",
        type=int,
        default=DEFAULT_MAX_FACTORS_PER_REPORT,
        help="Prompt hint for the target factor count per report; no local truncation is applied.",
    )
    parser.add_argument(
        "--max-samples-generation",
        type=int,
        default=DEFAULT_MAX_SAMPLES_GENERATION,
        help="Discovery-stage upper bound for candidate reports.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum number of in-flight generation workers.",
    )
    parser.add_argument(
        "--max-qps",
        type=float,
        default=DEFAULT_MAX_QPS,
        help="Maximum global LLM request rate.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = build_runtime_config(
        stage=args.stage,
        context_mode=args.context_mode,
        env_file=args.env_file,
        output_path=args.output_path,
        max_factors_per_report=args.max_factors_per_report,
        max_samples_generation=args.max_samples_generation,
        max_concurrency=args.max_concurrency,
        max_qps=args.max_qps,
    )
    result = run_pipeline(config)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
