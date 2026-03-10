from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_PRESETS: dict[str, list[dict[str, Any]]] = {
    "quick": [
        {"name": "gru_48x1", "overrides": {"model": {"name": "gru", "hidden_dim": 48, "num_layers": 1, "dropout": 0.1}}},
        {"name": "gru_64x2", "overrides": {"model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3}}},
        {"name": "lstm_48x1", "overrides": {"model": {"name": "lstm", "hidden_dim": 48, "num_layers": 1, "dropout": 0.1}}},
        {
            "name": "attention_64x2",
            "overrides": {
                "model": {
                    "name": "attention",
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "attention_heads": 4,
                    "ff_multiplier": 2,
                }
            },
        },
    ],
    "standard": [
        {"name": "gru_48x1", "overrides": {"model": {"name": "gru", "hidden_dim": 48, "num_layers": 1, "dropout": 0.1}}},
        {"name": "gru_64x2", "overrides": {"model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3}}},
        {"name": "gru_80x2", "overrides": {"model": {"name": "gru", "hidden_dim": 80, "num_layers": 2, "dropout": 0.2}}},
        {"name": "lstm_48x1", "overrides": {"model": {"name": "lstm", "hidden_dim": 48, "num_layers": 1, "dropout": 0.1}}},
        {"name": "lstm_64x2", "overrides": {"model": {"name": "lstm", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3}}},
        {
            "name": "attention_64x2",
            "overrides": {
                "model": {
                    "name": "attention",
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "attention_heads": 4,
                    "ff_multiplier": 2,
                }
            },
        },
        {
            "name": "attention_80x2",
            "overrides": {
                "model": {
                    "name": "attention",
                    "hidden_dim": 80,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "attention_heads": 4,
                    "ff_multiplier": 2,
                }
            },
        },
        {"name": "gru_64x2_seq10", "overrides": {"sequence": {"seq_len": 10}, "model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.2}}},
        {"name": "gru_64x2_seq30", "overrides": {"sequence": {"seq_len": 30}, "model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3, "max_seq_len": 64}}},
    ],
    "full": [
        {"name": "gru_48x1", "overrides": {"model": {"name": "gru", "hidden_dim": 48, "num_layers": 1, "dropout": 0.1}}},
        {"name": "gru_64x1", "overrides": {"model": {"name": "gru", "hidden_dim": 64, "num_layers": 1, "dropout": 0.1}}},
        {"name": "gru_64x2", "overrides": {"model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3}}},
        {"name": "gru_80x2", "overrides": {"model": {"name": "gru", "hidden_dim": 80, "num_layers": 2, "dropout": 0.2}}},
        {"name": "gru_64x2_lr5e4", "overrides": {"model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3, "lr": 0.0005}}},
        {"name": "gru_64x2_lr1e3", "overrides": {"model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3, "lr": 0.001}}},
        {"name": "lstm_48x1", "overrides": {"model": {"name": "lstm", "hidden_dim": 48, "num_layers": 1, "dropout": 0.1}}},
        {"name": "lstm_64x2", "overrides": {"model": {"name": "lstm", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3}}},
        {"name": "lstm_80x2", "overrides": {"model": {"name": "lstm", "hidden_dim": 80, "num_layers": 2, "dropout": 0.2}}},
        {
            "name": "attention_64x2",
            "overrides": {
                "model": {
                    "name": "attention",
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "attention_heads": 4,
                    "ff_multiplier": 2,
                }
            },
        },
        {
            "name": "attention_80x2",
            "overrides": {
                "model": {
                    "name": "attention",
                    "hidden_dim": 80,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "attention_heads": 4,
                    "ff_multiplier": 2,
                }
            },
        },
        {
            "name": "attention_96x2",
            "overrides": {
                "model": {
                    "name": "attention",
                    "hidden_dim": 96,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "attention_heads": 4,
                    "ff_multiplier": 2,
                }
            },
        },
        {"name": "gru_64x2_seq10", "overrides": {"sequence": {"seq_len": 10}, "model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.2}}},
        {"name": "gru_64x2_seq30", "overrides": {"sequence": {"seq_len": 30}, "model": {"name": "gru", "hidden_dim": 64, "num_layers": 2, "dropout": 0.3}}},
        {"name": "attention_64x2_seq30", "overrides": {"sequence": {"seq_len": 30}, "model": {"name": "attention", "hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "attention_heads": 4, "ff_multiplier": 2}}},
    ],
}


@dataclass
class ExperimentResult:
    experiment_name: str
    run_name: str
    status: str
    return_code: int
    config_path: str
    log_path: str
    run_dir: str | None = None
    model_name: str | None = None
    seq_len: int | None = None
    hidden_dim: int | None = None
    num_layers: int | None = None
    dropout: float | None = None
    lr: float | None = None
    valid_ic: float | None = None
    test_ic: float | None = None
    best_valid_ic: float | None = None
    backtest_relative_return: float | None = None
    backtest_cumulative_return: float | None = None
    backtest_win_rate: float | None = None
    backtest_sharpe: float | None = None
    as_of_date: str | None = None
    time_block_shuffle: bool | None = None


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run multiple model architectures and hyperparameter experiments.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--preset", type=str, default="standard", choices=sorted(DEFAULT_PRESETS.keys()))
    parser.add_argument("--experiment-file", type=Path, default=None, help="Optional YAML/JSON file with custom experiments.")
    parser.add_argument("--run-prefix", type=str, default="batch")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Where to store temporary configs, logs, and summary.")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--shuffle-time-blocks", action="store_true")
    parser.add_argument("--shuffle-block-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def slugify(text: str) -> str:
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "experiment"


def load_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.experiment_file is None:
        experiments = deepcopy(DEFAULT_PRESETS[args.preset])
    else:
        payload_text = args.experiment_file.read_text(encoding="utf-8")
        if args.experiment_file.suffix.lower() == ".json":
            payload = json.loads(payload_text)
        else:
            payload = yaml.safe_load(payload_text)
        experiments = payload.get("experiments", payload)
        if not isinstance(experiments, list):
            raise ValueError("Experiment file must contain a list or an 'experiments' list.")
        experiments = deepcopy(experiments)

    if args.max_experiments is not None:
        experiments = experiments[: int(args.max_experiments)]
    return experiments


def build_batch_dir(args: argparse.Namespace) -> Path:
    if args.batch_dir is not None:
        batch_dir = args.batch_dir.resolve()
    else:
        batch_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = Path("outputs") / "batch_runs" / f"{args.run_prefix}_{batch_id}"
    (batch_dir / "configs").mkdir(parents=True, exist_ok=True)
    (batch_dir / "logs").mkdir(parents=True, exist_ok=True)
    return batch_dir


def derive_batch_id(batch_dir: Path, run_prefix: str) -> str:
    prefix = f"{run_prefix}_"
    name = batch_dir.name
    return name[len(prefix) :] if name.startswith(prefix) else name


def build_experiment_config(
    base_config: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(base_config)
    deep_update(config, experiment.get("overrides", {}))
    return config


def extract_metrics(summary: dict[str, Any], experiment_config: dict[str, Any]) -> dict[str, Any]:
    backtest_metrics = summary.get("backtest_metrics", {})
    model_cfg = experiment_config.get("model", {})
    sequence_cfg = experiment_config.get("sequence", {})
    return {
        "run_dir": summary.get("run_dir"),
        "model_name": summary.get("model_name"),
        "seq_len": sequence_cfg.get("seq_len", summary.get("seq_len")),
        "hidden_dim": model_cfg.get("hidden_dim"),
        "num_layers": model_cfg.get("num_layers"),
        "dropout": model_cfg.get("dropout"),
        "lr": model_cfg.get("lr"),
        "valid_ic": summary.get("valid_ic"),
        "test_ic": summary.get("test_ic"),
        "best_valid_ic": summary.get("best_valid_ic"),
        "backtest_relative_return": backtest_metrics.get("relative_return"),
        "backtest_cumulative_return": backtest_metrics.get("cumulative_return"),
        "backtest_win_rate": backtest_metrics.get("win_rate"),
        "backtest_sharpe": backtest_metrics.get("sharpe_annualized"),
        "as_of_date": summary.get("as_of_date"),
        "time_block_shuffle": summary.get("time_block_shuffle"),
    }


def resolve_run_name(run_root: Path, requested_name: str) -> str:
    candidate = str(requested_name)
    if not (run_root / candidate).exists():
        return candidate
    suffix = 2
    while (run_root / f"{requested_name}_v{suffix}").exists():
        suffix += 1
    return f"{requested_name}_v{suffix}"


def run_experiment(
    python_exe: str,
    repo_root: Path,
    batch_dir: Path,
    config_path: Path,
    experiment_config: dict[str, Any],
    run_name: str,
    args: argparse.Namespace,
    experiment_name: str,
) -> ExperimentResult:
    run_root = repo_root / "outputs" / "runs"
    resolved_run_name = resolve_run_name(run_root, run_name)
    if resolved_run_name != run_name:
        print(f"[Batch] Run dir exists, renamed run_name: {run_name} -> {resolved_run_name}")
    log_path = batch_dir / "logs" / f"{run_name}.log"
    command = [python_exe, "main.py", "--config", str(config_path), "--run-name", resolved_run_name]
    if args.as_of_date:
        command.extend(["--as-of-date", args.as_of_date])
    if args.shuffle_time_blocks:
        command.append("--shuffle-time-blocks")
    if args.shuffle_block_size is not None:
        command.extend(["--shuffle-block-size", str(args.shuffle_block_size)])

    if args.dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(command), encoding="utf-8")
        return ExperimentResult(
            experiment_name=experiment_name,
            run_name=resolved_run_name,
            status="dry_run",
            return_code=0,
            config_path=str(config_path),
            log_path=str(log_path),
            as_of_date=args.as_of_date,
            time_block_shuffle=bool(args.shuffle_time_blocks),
        )

    # 每个实验单独落日志，方便回头定位是哪组参数失败或效果异常。
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(
        f"COMMAND:\n{' '.join(command)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}",
        encoding="utf-8",
    )

    if completed.returncode != 0:
        return ExperimentResult(
            experiment_name=experiment_name,
            run_name=resolved_run_name,
            status="failed",
            return_code=int(completed.returncode),
            config_path=str(config_path),
            log_path=str(log_path),
            as_of_date=args.as_of_date,
            time_block_shuffle=bool(args.shuffle_time_blocks),
        )

    run_dir = repo_root / "outputs" / "runs" / resolved_run_name
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return ExperimentResult(
            experiment_name=experiment_name,
            run_name=resolved_run_name,
            status="missing_summary",
            return_code=0,
            config_path=str(config_path),
            log_path=str(log_path),
            run_dir=str(run_dir.resolve()),
            as_of_date=args.as_of_date,
            time_block_shuffle=bool(args.shuffle_time_blocks),
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = extract_metrics(summary, experiment_config=experiment_config)
    return ExperimentResult(
        experiment_name=experiment_name,
        run_name=resolved_run_name,
        status="completed",
        return_code=0,
        config_path=str(config_path),
        log_path=str(log_path),
        **metrics,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config = load_config(args.config)
    experiments = load_experiments(args)
    if not experiments:
        raise ValueError("No experiments to run.")

    batch_dir = build_batch_dir(args)
    batch_id = derive_batch_id(batch_dir, args.run_prefix)
    print(f"[Batch] Batch dir: {batch_dir.resolve()}")
    print(f"[Batch] Experiments: {len(experiments)}")
    if args.as_of_date:
        print(f"[Batch] As-of date: {args.as_of_date}")
    if args.shuffle_time_blocks:
        print(f"[Batch] Time block shuffle enabled. block_size={args.shuffle_block_size or base_config.get('sequence', {}).get('seq_len', 20)}")

    results: list[ExperimentResult] = []
    batch_manifest: list[dict[str, Any]] = []

    for idx, experiment in enumerate(experiments, start=1):
        experiment_name = str(experiment.get("name", f"exp_{idx:02d}"))
        run_name = f"{args.run_prefix}_{batch_id}_{idx:02d}_{slugify(experiment_name)}"
        experiment_config = build_experiment_config(base_config, experiment)
        config_path = batch_dir / "configs" / f"{run_name}.yaml"
        config_path.write_text(yaml.safe_dump(experiment_config, allow_unicode=True, sort_keys=False), encoding="utf-8")

        batch_manifest.append(
            {
                "experiment_name": experiment_name,
                "run_name": run_name,
                "config_path": str(config_path),
                "overrides": experiment.get("overrides", {}),
            }
        )

        print(f"[Batch] ({idx}/{len(experiments)}) {experiment_name} -> {run_name}")
        result = run_experiment(
            python_exe=args.python_exe,
            repo_root=repo_root,
            batch_dir=batch_dir,
            config_path=config_path,
            experiment_config=experiment_config,
            run_name=run_name,
            args=args,
            experiment_name=experiment_name,
        )
        results.append(result)
        print(
            f"[Batch] status={result.status} "
            f"valid_ic={result.valid_ic} test_ic={result.test_ic} "
            f"log={result.log_path}"
        )

    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results_df = pd.DataFrame([result.__dict__ for result in results])
    if not results_df.empty:
        sort_columns = [column for column in ["status", "best_valid_ic", "valid_ic", "test_ic"] if column in results_df.columns]
        ascending = [True, False, False, False][: len(sort_columns)]
        results_df = results_df.sort_values(sort_columns, ascending=ascending, na_position="last")
    summary_csv = batch_dir / "batch_summary.csv"
    summary_json = batch_dir / "batch_summary.json"
    results_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(results_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print(f"[Batch] Saved summary csv to: {summary_csv.resolve()}")
    print(f"[Batch] Saved summary json to: {summary_json.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Batch] 用户中断，批量实验已停止。")
        raise SystemExit(130)
