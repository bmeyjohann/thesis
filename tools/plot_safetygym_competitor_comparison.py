#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "baseline": "Goal-only baseline",
    "own": "Ours",
    "pvp": "PVP",
    "eil": "EIL",
    "hilserl": "HIL-SERL",
    "bc": "BC",
    "behavior_cloning": "BC",
    "hgdagger": "HG-DAgger",
}

INIT_LABELS = {
    "goal_policy": "goal init",
    "scratch": "scratch",
}


def _load_matplotlib():
    if not os.environ.get("MPLCONFIGDIR"):
        path = Path("/tmp/mplconfig-safetygym-competitors")
        path.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        text = line.strip()
        if not text.startswith("{") or not text.endswith("}"):
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _method_from_name(path: Path, row: dict[str, Any]) -> str:
    exp = str(row.get("exp_name") or path.stem).lower()
    for method in ("baseline", "own", "pvp", "eil", "hilserl", "bc", "behavior_cloning", "hgdagger"):
        if f"_{method}_" in exp or exp.startswith(method) or method in path.stem.lower():
            return method
    return path.stem.split("_")[0]


def _init_from_name(path: Path, row: dict[str, Any]) -> str:
    exp = str(row.get("exp_name") or path.stem).lower()
    stem = path.stem.lower()
    text = f"{exp} {stem}"
    if "baseline" in text:
        return "baseline"
    if "goal_policy" in text or "_goal_" in text:
        return "goal_policy"
    if "scratch" in text:
        return "scratch"
    return "unknown"


def _series_key(method: str, init_mode: str) -> str:
    if method == "baseline":
        return "baseline"
    if init_mode in ("goal_policy", "scratch"):
        return f"{method}:{init_mode}"
    return method


def _series_label(method: str, init_mode: str) -> str:
    method_label = METHOD_LABELS.get(method, method)
    if method == "baseline":
        return method_label
    init_label = INIT_LABELS.get(init_mode)
    if init_label:
        return f"{method_label} ({init_label})"
    return method_label


def _collect(log_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.log")):
        for raw in _json_lines(path):
            method = _method_from_name(path, raw)
            init_mode = _init_from_name(path, raw)
            step = raw.get("eval/step", raw.get("train/step", raw.get("step", math.nan)))
            base = {
                "method": method,
                "init_mode": init_mode,
                "series": _series_key(method, init_mode),
                "label": _series_label(method, init_mode),
                "log_file": str(path),
                "step": float(step) if step is not None else math.nan,
            }
            if any(str(k).startswith("eval/") for k in raw):
                row = dict(base)
                for key, value in raw.items():
                    if str(key).startswith("eval/"):
                        row[str(key)[len("eval/") :]] = value
                eval_rows.append(row)
            elif any(str(k).startswith("train/") for k in raw):
                row = dict(base)
                for key, value in raw.items():
                    if str(key).startswith("train/"):
                        row[str(key)[len("train/") :]] = value
                train_rows.append(row)
    eval_rows.sort(key=lambda r: (str(r["series"]), float(r.get("step", 0.0))))
    train_rows.sort(key=lambda r: (str(r["series"]), float(r.get("step", 0.0))))
    return train_rows, eval_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(plt, rows: list[dict[str, Any]], *, key: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    series_keys = sorted({str(row["series"]) for row in rows})
    for series in series_keys:
        sub = [row for row in rows if str(row["series"]) == series and key in row]
        if not sub:
            continue
        xs = [float(row.get("step", 0.0)) for row in sub]
        ys = [float(row.get(key, math.nan)) for row in sub]
        label = str(sub[-1].get("label") or series)
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("environment step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if series_keys:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Safety-Gym competitor comparison from captured JSON logs.")
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    train_rows, eval_rows = _collect(args.log_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "train_metrics.csv", train_rows)
    _write_csv(args.output_dir / "eval_metrics.csv", eval_rows)

    final_rows = []
    for series in sorted({str(row["series"]) for row in eval_rows}):
        sub = [row for row in eval_rows if str(row["series"]) == series]
        if sub:
            final_rows.append(max(sub, key=lambda row: float(row.get("step", 0.0))))
    _write_csv(args.output_dir / "final_eval_summary.csv", final_rows)

    plt = _load_matplotlib()
    _plot_metric(
        plt,
        eval_rows,
        key="first_goal_success_rate",
        ylabel="first-goal success rate",
        output=args.output_dir / "eval_first_goal_success_rate.png",
    )
    _plot_metric(
        plt,
        eval_rows,
        key="episode_cost_sum_mean",
        ylabel="mean native cost",
        output=args.output_dir / "eval_cost_sum_mean.png",
    )
    _plot_metric(
        plt,
        eval_rows,
        key="intervention_fraction_mean",
        ylabel="eval intervention fraction",
        output=args.output_dir / "eval_intervention_fraction.png",
    )
    _plot_metric(
        plt,
        train_rows,
        key="intervention_fraction_mean",
        ylabel="train intervention fraction",
        output=args.output_dir / "train_intervention_fraction.png",
    )
    print(json.dumps({"train_rows": len(train_rows), "eval_rows": len(eval_rows), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
