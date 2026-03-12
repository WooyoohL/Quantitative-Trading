from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_csv_str(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_int(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_float(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def build_experiment_name(
    model_name: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    seq_len: int,
    lr: float,
    weight_decay: float,
) -> str:
    dropout_tag = str(dropout).replace(".", "")
    lr_tag = f"{lr:.5f}".rstrip("0").rstrip(".").replace(".", "p")
    wd_tag = f"{weight_decay:.5f}".rstrip("0").rstrip(".").replace(".", "p")
    return (
        f"{model_name}_h{hidden_dim}_l{num_layers}_d{dropout_tag}"
        f"_s{seq_len}_lr{lr_tag}_wd{wd_tag}"
    )


def build_experiments(base_config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    base_model = dict(base_config.get("model", {}))
    base_sequence = dict(base_config.get("sequence", {}))

    experiments: list[dict[str, Any]] = []
    for model_name, hidden_dim, num_layers, dropout, seq_len, lr, weight_decay in product(
        args.models,
        args.hidden_dims,
        args.num_layers,
        args.dropouts,
        args.seq_lens,
        args.lrs,
        args.weight_decays,
    ):
        overrides: dict[str, Any] = {
            "sequence": {
                "seq_len": int(seq_len),
            },
            "model": {
                "name": str(model_name),
                "hidden_dim": int(hidden_dim),
                "num_layers": int(num_layers),
                "dropout": float(dropout),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            },
        }
        if str(model_name).lower() == "attention":
            overrides["model"]["attention_heads"] = int(base_model.get("attention_heads", 4))
            overrides["model"]["ff_multiplier"] = int(base_model.get("ff_multiplier", 2))
            overrides["model"]["max_seq_len"] = int(base_model.get("max_seq_len", max(args.seq_lens)))
        elif "attention_heads" in base_model:
            # Keep non-attention models clean so the config diff is readable.
            pass

        experiments.append(
            {
                "name": build_experiment_name(
                    model_name=str(model_name),
                    hidden_dim=int(hidden_dim),
                    num_layers=int(num_layers),
                    dropout=float(dropout),
                    seq_len=int(seq_len),
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                ),
                "overrides": overrides,
            }
        )
    return experiments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a model hyperparameter sweep via batch_experiments.py."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("scripts/experiment_template_model_sweep.yaml"),
        help="Where to write the generated experiment YAML.",
    )
    parser.add_argument("--run-prefix", type=str, default="modelgrid")
    parser.add_argument("--batch-dir", type=Path, default=None)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate the YAML file and do not launch batch_experiments.py.",
    )
    parser.add_argument(
        "--models",
        type=parse_csv_str,
        default=parse_csv_str("gru,lstm,attention"),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_csv_int,
        default=parse_csv_int("64,128,256"),
        help="Comma-separated hidden_dim values.",
    )
    parser.add_argument(
        "--num-layers",
        type=parse_csv_int,
        default=parse_csv_int("1,2"),
        help="Comma-separated num_layers values.",
    )
    parser.add_argument(
        "--dropouts",
        type=parse_csv_float,
        default=parse_csv_float("0.1,0.3,0.5"),
        help="Comma-separated dropout values.",
    )
    parser.add_argument(
        "--seq-lens",
        type=parse_csv_int,
        default=parse_csv_int("20"),
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument(
        "--lrs",
        type=parse_csv_float,
        default=parse_csv_float("0.0008"),
        help="Comma-separated learning rates.",
    )
    parser.add_argument(
        "--weight-decays",
        type=parse_csv_float,
        default=parse_csv_float("0.01"),
        help="Comma-separated weight_decay values.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    base_config = load_config(args.config)
    experiments = build_experiments(base_config, args)

    payload = {"experiments": experiments}
    output_path = args.output_file
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"Generated {len(experiments)} experiments: {output_path}")
    if experiments:
        print(f"First experiment: {experiments[0]['name']}")
        print(f"Last experiment: {experiments[-1]['name']}")

    if args.generate_only:
        return

    command = [
        args.python_exe,
        "scripts/batch_experiments.py",
        "--config",
        str(args.config),
        "--experiment-file",
        str(output_path),
        "--run-prefix",
        str(args.run_prefix),
    ]
    if args.batch_dir is not None:
        command.extend(["--batch-dir", str(args.batch_dir)])
    if args.as_of_date:
        command.extend(["--as-of-date", str(args.as_of_date)])
    if args.dry_run:
        command.append("--dry-run")

    print("Launching batch experiments:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=repo_root)
    raise SystemExit(int(completed.returncode))


if __name__ == "__main__":
    main()
