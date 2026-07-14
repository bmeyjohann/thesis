#!/usr/bin/env python3
"""Merge an interrupted Unitree metrics history with a checkpoint continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-transitions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_path = args.base / "metrics.jsonl" if args.base.is_dir() else args.base
    continuation_path = args.continuation / "metrics.jsonl" if args.continuation.is_dir() else args.continuation
    base_rows = [row for row in load_rows(base_path) if int(row.get("step", 0)) <= args.checkpoint_step]
    continuation_rows = load_rows(continuation_path)
    for row in continuation_rows:
        row["step"] = int(row.get("step", 0)) + args.checkpoint_step
        row["transitions"] = int(row.get("transitions", 0)) + args.checkpoint_transitions
        row["continued_from_checkpoint"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in [*base_rows, *continuation_rows]:
            handle.write(json.dumps(row, allow_nan=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
