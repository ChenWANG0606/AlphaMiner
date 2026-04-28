from __future__ import annotations

import argparse
import json

from .candidate_generation import run_candidate_generation
from .config import (
    DEFAULT_DPO_CONFIG_PATH,
    load_dpo_config,
    override_candidate_generation,
    override_evaluation,
    override_preference,
    override_training,
    override_validation,
)
from .preferences import run_preference_building
from .training import run_dpo_evaluation, run_dpo_training
from .validation import run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO pipeline entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-candidates", help="Run Phase 1 candidate generation.")
    generate.add_argument("--config", default=DEFAULT_DPO_CONFIG_PATH, help="DPO YAML config path.")
    generate.add_argument("--group-size", type=int, default=None)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--temperature", type=float, default=None)
    generate.add_argument("--top-p", type=float, default=None)
    generate.add_argument("--max-new-tokens", type=int, default=None)
    generate.add_argument("--overwrite", action="store_true", default=None)
    validate = subparsers.add_parser("validate-candidates", help="Run Phase 2 validation and backtest.")
    validate.add_argument("--config", default=DEFAULT_DPO_CONFIG_PATH, help="DPO YAML config path.")
    validate.add_argument("--require-backtest", action=argparse.BooleanOptionalAction, default=None)
    validate.add_argument("--ic-horizon", type=int, default=None)
    validate.add_argument("--overwrite", action="store_true", default=None)
    preferences = subparsers.add_parser("build-preferences", help="Run Phase 3 preference pair building.")
    preferences.add_argument("--config", default=DEFAULT_DPO_CONFIG_PATH, help="DPO YAML config path.")
    preferences.add_argument("--primary-metric", default=None)
    preferences.add_argument("--allow-single-success-pair", action=argparse.BooleanOptionalAction, default=None)
    preferences.add_argument("--min-abs-ic-gap", type=float, default=None)
    preferences.add_argument("--overwrite", action="store_true", default=None)
    train = subparsers.add_parser("train", help="Run Phase 4 DPO training.")
    train.add_argument("--config", default=DEFAULT_DPO_CONFIG_PATH, help="DPO YAML config path.")
    train.add_argument("--beta", type=float, default=None)
    train.add_argument("--learning-rate", type=float, default=None)
    train.add_argument("--num-train-epochs", type=float, default=None)
    train.add_argument("--per-device-train-batch-size", type=int, default=None)
    train.add_argument("--gradient-accumulation-steps", type=int, default=None)
    train.add_argument("--overwrite", action="store_true", default=None)
    evaluate = subparsers.add_parser("evaluate", help="Run Phase 4 SFT vs DPO evaluation.")
    evaluate.add_argument("--config", default=DEFAULT_DPO_CONFIG_PATH, help="DPO YAML config path.")
    evaluate.add_argument("--temperature", type=float, default=None)
    evaluate.add_argument("--top-p", type=float, default=None)
    evaluate.add_argument("--max-new-tokens", type=int, default=None)
    evaluate.add_argument("--seed", type=int, default=None)
    run_all = subparsers.add_parser("run-all", help="Run Phase 4 training then evaluation.")
    run_all.add_argument("--config", default=DEFAULT_DPO_CONFIG_PATH, help="DPO YAML config path.")
    run_all.add_argument("--overwrite", action="store_true", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate-candidates":
        config = load_dpo_config(args.config)
        config = override_candidate_generation(
            config,
            group_size=args.group_size,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
        summary = run_candidate_generation(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-candidates":
        config = load_dpo_config(args.config)
        config = override_validation(
            config,
            require_backtest=args.require_backtest,
            ic_horizon=args.ic_horizon,
            overwrite=args.overwrite,
        )
        summary = run_validation(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-preferences":
        config = load_dpo_config(args.config)
        config = override_preference(
            config,
            primary_metric=args.primary_metric,
            allow_single_success_pair=args.allow_single_success_pair,
            min_abs_ic_gap=args.min_abs_ic_gap,
            overwrite=args.overwrite,
        )
        summary = run_preference_building(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        config = load_dpo_config(args.config)
        config = override_training(
            config,
            beta=args.beta,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            overwrite=args.overwrite,
        )
        manifest = run_dpo_training(config)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate":
        config = load_dpo_config(args.config)
        config = override_evaluation(
            config,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        report = run_dpo_evaluation(config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-all":
        config = load_dpo_config(args.config)
        config = override_training(config, overwrite=args.overwrite)
        manifest = run_dpo_training(config)
        report = run_dpo_evaluation(config)
        print(json.dumps({"manifest": manifest, "eval_report": report}, ensure_ascii=False, indent=2))
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
