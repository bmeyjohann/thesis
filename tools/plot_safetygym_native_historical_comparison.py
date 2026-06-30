#!/usr/bin/env python3
"""Collect and plot the native-center-cost SafetyCar comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = (
    ("reference_bc_goal", "BC reference (goal init)"),
    ("own_hybrid_scratch", "Ours (hybrid, scratch)"),
    ("pvp_goal", "PVP (goal init)"),
    ("hilserl_goal", "HIL-SERL (goal init)"),
    ("eil_goal", "EIL (goal init)"),
    ("bc_dagger_goal", "DAgger-style BC (goal init)"),
)


def _plt():
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig-safetygym-native-historical"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _method(text: str) -> tuple[str, str] | None:
    lowered = text.lower()
    for key, label in METHODS:
        if key in lowered:
            return key, label
    return None


def _step(text: str) -> int | None:
    match = re.search(r"(?:step|_)(5000|10000|15000|20000|30000)(?:\D|$)", text.lower())
    return int(match.group(1)) if match else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(plt, rows: list[dict[str, Any]], key: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    for method, label in METHODS:
        sub = sorted((r for r in rows if r["method"] == method and key in r), key=lambda r: r["step"])
        if not sub:
            continue
        ax.plot([r["step"] for r in sub], [r[key] for r in sub], marker="o", label=label)
    ax.set_xlabel("environment step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, Any]] = []
    for output_log in (repo / "logs" / "safetygym_minimal").glob("safetycar_goal1_native_historical_*/wandb/*/files/output.log"):
        found = _method(str(output_log))
        if found is None:
            continue
        method, label = found
        for raw in _json_rows(output_log):
            step = raw.get("train/step")
            if step is None:
                continue
            row = {"method": method, "label": label, "step": int(step)}
            for key, value in raw.items():
                if str(key).startswith("train/"):
                    row[str(key)[6:]] = value
            train_rows.append(row)

    audit_rows: list[dict[str, Any]] = []
    audit_root = repo / "logs" / "safetygym_eval_audits"
    for metrics_path in audit_root.glob("native_policy_*/*metrics.log"):
        found = _method(str(metrics_path))
        step = _step(str(metrics_path))
        if found is None or step is None:
            continue
        rows = _json_rows(metrics_path)
        if not rows:
            continue
        raw = rows[-1]
        method, label = found
        row = {"method": method, "label": label, "step": step, "audit_dir": str(metrics_path.parent)}
        for key, value in raw.items():
            if str(key).startswith("eval/"):
                row[str(key)[5:]] = value
        audit_rows.append(row)

    train_rows.sort(key=lambda r: (r["method"], r["step"]))
    audit_rows.sort(key=lambda r: (r["method"], r["step"]))
    _write_csv(output / "train_metrics.csv", train_rows)
    _write_csv(output / "audit_metrics.csv", audit_rows)
    final = []
    for method, _ in METHODS:
        rows = [row for row in audit_rows if row["method"] == method]
        if rows:
            final.append(max(rows, key=lambda row: row["step"]))
    _write_csv(output / "final_audit_summary.csv", final)

    plt = _plt()
    _plot(plt, audit_rows, "episode_cost_sum_mean", "mean native center cost", output / "audit_cost.png")
    _plot(plt, audit_rows, "first_goal_success_rate", "first-goal success rate", output / "audit_success.png")
    _plot(plt, audit_rows, "mean_first_goal_hit_step", "mean first-goal step", output / "audit_first_goal_step.png")
    _plot(plt, train_rows, "teacher_fraction_steps", "train intervention fraction", output / "train_intervention_fraction.png")
    print(json.dumps({"train_rows": len(train_rows), "audit_rows": len(audit_rows), "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
