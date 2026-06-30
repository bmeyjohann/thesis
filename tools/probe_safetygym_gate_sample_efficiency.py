#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.dataset_io import load_transition_dataset
from safetygym_utils.imitation_teacher import build_window_features, normalize_observations


class GateMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(obs_dim)
        for _ in range(int(max(1, num_layers))):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x)).squeeze(-1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe Safety-Gym intervention-gate sample efficiency.")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--sizes", type=str, default="500,1000,2000,5000,10000,20000,30000")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--context_len", type=int, default=8)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="auto")
    return p


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _stratified_sample(labels: np.ndarray, pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    labels = labels.astype(bool, copy=False)
    pos = pool[labels[pool]]
    neg = pool[~labels[pool]]
    frac_pos = float(labels[pool].mean()) if pool.size else 0.0
    n_pos = min(int(round(n * frac_pos)), int(pos.size))
    n_neg = min(int(n - n_pos), int(neg.size))
    if n_pos < int(round(n * frac_pos)) and neg.size > n_neg:
        n_neg = min(int(n - n_pos), int(neg.size))
    if n_neg < int(n - n_pos) and pos.size > n_pos:
        n_pos = min(int(n - n_neg), int(pos.size))
    out = []
    if n_pos > 0:
        out.append(rng.choice(pos, size=n_pos, replace=False))
    if n_neg > 0:
        out.append(rng.choice(neg, size=n_neg, replace=False))
    if not out:
        return np.asarray([], dtype=np.int64)
    idx = np.concatenate(out).astype(np.int64, copy=False)
    rng.shuffle(idx)
    return idx


def _auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(bool, copy=False)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / max(1, n_pos * n_neg))


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    y = y_true.astype(bool, copy=False)
    pred = prob >= float(threshold)
    tp = int((pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    denom = math.sqrt(max(1.0, float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    mcc = ((tp * tn) - (fp * fn)) / denom
    pearson = float(np.corrcoef(prob, y.astype(np.float32))[0, 1]) if np.std(prob) > 0.0 and np.std(y) > 0.0 else float("nan")
    return {
        "accuracy": float((tp + tn) / max(1, len(y))),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(mcc),
        "pearson": pearson,
        "auroc": _auroc(y, prob),
        "positive_rate_pred": float(pred.mean()),
        "positive_rate_true": float(y.mean()),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _make_svg(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    agg: dict[int, dict[str, float]] = {}
    for size in sorted({int(r["train_size"]) for r in rows}):
        subset = [r for r in rows if int(r["train_size"]) == size]
        agg[size] = {
            "accuracy": float(np.mean([float(r["accuracy"]) for r in subset])),
            "f1": float(np.mean([float(r["f1"]) for r in subset])),
            "mcc": float(np.mean([float(r["mcc"]) for r in subset])),
            "auroc": float(np.mean([float(r["auroc"]) for r in subset])),
        }
    sizes = list(agg)
    w, h = 900, 460
    left, right, top, bottom = 80, 30, 40, 70
    plot_w = w - left - right
    plot_h = h - top - bottom
    min_s, max_s = min(sizes), max(sizes)
    def x(size: int) -> float:
        if min_s == max_s:
            return left + plot_w / 2
        return left + plot_w * ((math.log10(size) - math.log10(min_s)) / (math.log10(max_s) - math.log10(min_s)))
    def y(v: float) -> float:
        return top + plot_h * (1.0 - max(0.0, min(1.0, v)))
    colors = {"accuracy": "#4C78A8", "f1": "#F58518", "mcc": "#54A24B", "auroc": "#B279A2"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<text x="25" y="25" font-family="Arial" font-size="20" font-weight="700">Intervention gate sample efficiency</text>')
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{w-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" font-family="Arial" font-size="12" text-anchor="end">{tick:.2f}</text>')
    for size in sizes:
        xx = x(size)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" stroke="#eee"/>')
        parts.append(f'<text x="{xx:.1f}" y="{h-28}" font-family="Arial" font-size="12" text-anchor="middle">{size}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222"/>')
    for metric, color in colors.items():
        pts = [(x(size), y(agg[size][metric])) for size in sizes]
        point_text = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="3"/>')
        for px, py in pts:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')
    lx = 590
    for i, (metric, color) in enumerate(colors.items()):
        parts.append(f'<rect x="{lx}" y="{55+i*24}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx+20}" y="{67+i*24}" font-family="Arial" font-size="13">{metric}</text>')
    parts.append('<text x="450" y="445" font-family="Arial" font-size="13" text-anchor="middle">training rows, log scale</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_transition_dataset(args.dataset_path)
    observations = np.asarray(data["observations"], dtype=np.float32)
    student_actions = np.asarray(data["student_actions"], dtype=np.float32)
    teacher_intervened = np.asarray(data["teacher_intervened"], dtype=np.bool_).reshape(-1)
    episode_ids = np.asarray(data["episode_ids"], dtype=np.int64).reshape(-1)
    features = build_window_features(
        observations=observations,
        student_actions=student_actions,
        teacher_intervened=teacher_intervened,
        episode_ids=episode_ids,
        context_len=int(args.context_len),
    )
    sizes = _parse_ints(args.sizes)
    seeds = _parse_ints(args.seeds)
    all_idx = np.arange(features.shape[0], dtype=np.int64)
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        shuffled = all_idx.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * float(args.test_fraction))))
        test_idx = shuffled[:n_test]
        pool_idx = shuffled[n_test:]
        y_test_np = teacher_intervened[test_idx].astype(np.float32)
        for size in sizes:
            n = min(int(size), int(pool_idx.size))
            train_idx = _stratified_sample(teacher_intervened, pool_idx, n, rng)
            x_mean = torch.as_tensor(features[train_idx].mean(axis=0, keepdims=True), device=device, dtype=torch.float32)
            x_std = torch.as_tensor(features[train_idx].std(axis=0, keepdims=True) + 1e-6, device=device, dtype=torch.float32)
            x_train = normalize_observations(torch.as_tensor(features[train_idx], device=device, dtype=torch.float32), x_mean, x_std)
            y_train = torch.as_tensor(teacher_intervened[train_idx].astype(np.float32), device=device, dtype=torch.float32)
            x_test = normalize_observations(torch.as_tensor(features[test_idx], device=device, dtype=torch.float32), x_mean, x_std)
            model = GateMLP(features.shape[1], int(args.hidden_dim), int(args.num_layers), float(args.dropout)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
            pos = float(y_train.sum().detach().cpu())
            neg = float(y_train.numel() - pos)
            pos_weight = torch.as_tensor([neg / max(1.0, pos)], device=device, dtype=torch.float32)
            for _step in range(int(args.train_steps)):
                idx = torch.randint(0, int(x_train.shape[0]), (int(args.batch_size),), device=device)
                logits = model(x_train[idx])
                loss = F.binary_cross_entropy_with_logits(logits, y_train[idx], pos_weight=pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
            model.eval()
            with torch.inference_mode():
                prob = torch.sigmoid(model(x_test)).detach().cpu().numpy().astype(np.float64)
            metrics = _metrics(y_test_np, prob, float(args.threshold))
            row: dict[str, float | int | str] = {
                "seed": int(seed),
                "train_size": int(n),
                "test_size": int(test_idx.size),
                "train_positive_rate": float(teacher_intervened[train_idx].mean()),
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    csv_path = output_dir / "gate_sample_efficiency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for size in sorted({int(r["train_size"]) for r in rows}):
        subset = [r for r in rows if int(r["train_size"]) == size]
        summary.append({
            "train_size": size,
            "accuracy_mean": float(np.mean([float(r["accuracy"]) for r in subset])),
            "f1_mean": float(np.mean([float(r["f1"]) for r in subset])),
            "mcc_mean": float(np.mean([float(r["mcc"]) for r in subset])),
            "auroc_mean": float(np.mean([float(r["auroc"]) for r in subset])),
            "recall_mean": float(np.mean([float(r["recall"]) for r in subset])),
            "precision_mean": float(np.mean([float(r["precision"]) for r in subset])),
        })
    summary_path = output_dir / "gate_sample_efficiency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _make_svg(rows, output_dir / "gate_sample_efficiency.svg")
    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), "plot": str(output_dir / "gate_sample_efficiency.svg")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
