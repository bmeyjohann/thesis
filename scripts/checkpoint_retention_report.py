#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


STEP_RE = re.compile(r"(?:^|_)step(\d+)\.pt$")


@dataclass
class RunInfo:
    family: str
    run_dir: Path
    checkpoint_files: list[Path]
    keep_file: Path
    keep_reason: str
    latest_checkpoint_mtime: float
    total_bytes: int
    keep_bytes: int
    prune_reclaim_bytes: int
    delete_reclaim_bytes: int
    age_days: float
    is_delete_candidate: bool
    action: str

    @property
    def num_checkpoints(self) -> int:
        return len(self.checkpoint_files)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "run_dir": str(self.run_dir),
            "num_checkpoints": self.num_checkpoints,
            "total_gb": round(bytes_to_gb(self.total_bytes), 3),
            "keep_file": self.keep_file.name,
            "keep_reason": self.keep_reason,
            "keep_gb": round(bytes_to_gb(self.keep_bytes), 3),
            "prune_reclaim_gb": round(bytes_to_gb(self.prune_reclaim_bytes), 3),
            "delete_reclaim_gb": round(bytes_to_gb(self.delete_reclaim_bytes), 3),
            "age_days": round(self.age_days, 1),
            "is_delete_candidate": self.is_delete_candidate,
            "action": self.action,
        }


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def find_keep_file(checkpoint_files: Iterable[Path]) -> tuple[Path, str]:
    checkpoint_files = list(checkpoint_files)
    finals = [p for p in checkpoint_files if p.name == "final.pt" or p.name.endswith("_final.pt")]
    if finals:
        return max(finals, key=lambda p: p.stat().st_mtime), "final"

    step_files: list[tuple[int, Path]] = []
    for path in checkpoint_files:
        match = STEP_RE.search(path.name)
        if match:
            step_files.append((int(match.group(1)), path))
    if step_files:
        return max(step_files, key=lambda item: item[0])[1], "latest_step"

    return max(checkpoint_files, key=lambda p: p.stat().st_mtime), "latest_mtime"


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for family_dir in sorted(root.iterdir()):
        if not family_dir.is_dir():
            continue
        for run_dir in sorted(family_dir.iterdir()):
            if run_dir.is_dir():
                yield run_dir


def classify_runs(
    root: Path,
    recent_days: float,
    delete_candidate_days: float,
    delete_keywords: tuple[str, ...],
    now_ts: float,
) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for run_dir in iter_run_dirs(root):
        checkpoint_files = sorted(run_dir.glob("*.pt"))
        if not checkpoint_files:
            continue

        keep_file, keep_reason = find_keep_file(checkpoint_files)
        latest_checkpoint_mtime = max(path.stat().st_mtime for path in checkpoint_files)
        age_days = max(0.0, (now_ts - latest_checkpoint_mtime) / 86400.0)
        total_bytes = sum(path.stat().st_size for path in checkpoint_files)
        keep_bytes = keep_file.stat().st_size
        prune_reclaim_bytes = total_bytes - keep_bytes
        delete_reclaim_bytes = total_bytes
        lower_name = run_dir.name.lower()
        is_delete_candidate = (
            age_days >= delete_candidate_days
            and any(keyword in lower_name for keyword in delete_keywords)
        )

        if age_days < recent_days:
            action = "keep_recent"
        elif is_delete_candidate:
            action = "delete_run_candidate"
        elif len(checkpoint_files) > 1:
            action = "prune_to_canonical"
        else:
            action = "keep_single_old"

        runs.append(
            RunInfo(
                family=run_dir.parent.name,
                run_dir=run_dir,
                checkpoint_files=checkpoint_files,
                keep_file=keep_file,
                keep_reason=keep_reason,
                latest_checkpoint_mtime=latest_checkpoint_mtime,
                total_bytes=total_bytes,
                keep_bytes=keep_bytes,
                prune_reclaim_bytes=prune_reclaim_bytes,
                delete_reclaim_bytes=delete_reclaim_bytes,
                age_days=age_days,
                is_delete_candidate=is_delete_candidate,
                action=action,
            )
        )
    return runs


def summarize_runs(runs: list[RunInfo]) -> dict[str, object]:
    total_bytes = sum(run.total_bytes for run in runs)
    total_prune_reclaim = sum(run.prune_reclaim_bytes for run in runs)

    action_counts = Counter(run.action for run in runs)
    family_totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"runs": 0, "total_gb": 0.0, "prune_reclaim_gb": 0.0}
    )
    for run in runs:
        family_totals[run.family]["runs"] += 1
        family_totals[run.family]["total_gb"] += bytes_to_gb(run.total_bytes)
        family_totals[run.family]["prune_reclaim_gb"] += bytes_to_gb(run.prune_reclaim_bytes)

    combined_reclaim_bytes = 0
    delete_candidate_bytes = 0
    for run in runs:
        if run.action == "delete_run_candidate":
            delete_candidate_bytes += run.delete_reclaim_bytes
            combined_reclaim_bytes += run.delete_reclaim_bytes
        elif run.action == "prune_to_canonical":
            combined_reclaim_bytes += run.prune_reclaim_bytes

    return {
        "run_count": len(runs),
        "total_gb": round(bytes_to_gb(total_bytes), 3),
        "prune_reclaim_gb_if_keep_one_per_run": round(bytes_to_gb(total_prune_reclaim), 3),
        "combined_reclaim_gb_for_policy": round(bytes_to_gb(combined_reclaim_bytes), 3),
        "delete_candidate_total_gb": round(bytes_to_gb(delete_candidate_bytes), 3),
        "action_counts": dict(action_counts),
        "families": {
            family: {
                "runs": values["runs"],
                "total_gb": round(values["total_gb"], 3),
                "prune_reclaim_gb": round(values["prune_reclaim_gb"], 3),
            }
            for family, values in sorted(family_totals.items())
        },
    }


def render_text_report(
    runs: list[RunInfo],
    summary: dict[str, object],
    recent_days: float,
    delete_candidate_days: float,
    top_n: int,
) -> str:
    lines: list[str] = []
    lines.append("Checkpoint Retention Report")
    lines.append("==========================")
    lines.append(f"Root: models")
    lines.append(
        f"Policy: keep all checkpoints for runs newer than {recent_days:g} days; "
        f"otherwise keep only final.pt or the latest step checkpoint."
    )
    lines.append(
        f"Delete candidates: runs older than {delete_candidate_days:g} days whose names contain "
        "explicit ephemeral keywords such as smoke/testing/debug/probe."
    )
    lines.append("")
    lines.append(f"Runs with checkpoints: {summary['run_count']}")
    lines.append(f"Total checkpoint size: {summary['total_gb']:.3f}G")
    lines.append(
        f"Recoverable by keeping one checkpoint per run: "
        f"{summary['prune_reclaim_gb_if_keep_one_per_run']:.3f}G"
    )
    lines.append(
        f"Recoverable under the full policy: {summary['combined_reclaim_gb_for_policy']:.3f}G"
    )
    lines.append(f"Delete-candidate total: {summary['delete_candidate_total_gb']:.3f}G")
    lines.append("")
    lines.append("Action counts:")
    for action, count in sorted(summary["action_counts"].items()):
        lines.append(f"  {action}: {count}")
    lines.append("")
    lines.append("By family:")
    for family, values in summary["families"].items():
        lines.append(
            f"  {family}: runs={values['runs']} total={values['total_gb']:.3f}G "
            f"prune_reclaim={values['prune_reclaim_gb']:.3f}G"
        )

    prune_rows = sorted(
        (run for run in runs if run.action == "prune_to_canonical"),
        key=lambda run: run.prune_reclaim_bytes,
        reverse=True,
    )[:top_n]
    delete_rows = sorted(
        (run for run in runs if run.action == "delete_run_candidate"),
        key=lambda run: run.delete_reclaim_bytes,
        reverse=True,
    )[:top_n]

    lines.append("")
    lines.append(f"Top {len(prune_rows)} prune-to-canonical runs:")
    for run in prune_rows:
        lines.append(
            f"  {bytes_to_gb(run.prune_reclaim_bytes):6.3f}G  {run.run_dir}  "
            f"keep={run.keep_file.name} ({run.keep_reason})  "
            f"checkpoints={run.num_checkpoints}  age={run.age_days:.1f}d"
        )

    lines.append("")
    lines.append(f"Top {len(delete_rows)} delete-run candidates:")
    for run in delete_rows:
        lines.append(
            f"  {bytes_to_gb(run.delete_reclaim_bytes):6.3f}G  {run.run_dir}  "
            f"checkpoints={run.num_checkpoints}  age={run.age_days:.1f}d"
        )

    return "\n".join(lines)


def shell_prune_commands(runs: list[RunInfo]) -> list[str]:
    commands: list[str] = []
    for run in sorted(
        (run for run in runs if run.action == "prune_to_canonical"),
        key=lambda run: str(run.run_dir),
    ):
        delete_paths = [path for path in run.checkpoint_files if path != run.keep_file]
        if not delete_paths:
            continue
        quoted_paths = " ".join(shlex.quote(str(path)) for path in delete_paths)
        commands.append(f"rm -f {quoted_paths}")
    return commands


def shell_delete_commands(runs: list[RunInfo]) -> list[str]:
    return [
        f"rm -rf {shlex.quote(str(run.run_dir))}"
        for run in sorted(
            (run for run in runs if run.action == "delete_run_candidate"),
            key=lambda run: str(run.run_dir),
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run checkpoint retention inventory. "
            "Classifies runs as keep_recent, prune_to_canonical, or delete_run_candidate."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--recent-days", type=float, default=14.0)
    parser.add_argument("--delete-candidate-days", type=float, default=14.0)
    parser.add_argument(
        "--delete-keywords",
        default="smoke,testing,retry,debug,diag,probe",
        help="Comma-separated substrings that mark a run as an explicit delete candidate.",
    )
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", type=Path, help="Write the full report as JSON.")
    parser.add_argument(
        "--shell-prune",
        type=Path,
        help="Write shell commands that remove intermediate checkpoints but keep the canonical one.",
    )
    parser.add_argument(
        "--shell-delete-candidates",
        type=Path,
        help="Write shell commands that delete explicit delete-candidate run directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Checkpoint root does not exist: {root}", file=sys.stderr)
        return 2

    delete_keywords = tuple(
        keyword.strip().lower()
        for keyword in args.delete_keywords.split(",")
        if keyword.strip()
    )
    now_ts = datetime.now(timezone.utc).timestamp()
    runs = classify_runs(
        root=root,
        recent_days=args.recent_days,
        delete_candidate_days=args.delete_candidate_days,
        delete_keywords=delete_keywords,
        now_ts=now_ts,
    )
    summary = summarize_runs(runs)
    print(
        render_text_report(
            runs=runs,
            summary=summary,
            recent_days=args.recent_days,
            delete_candidate_days=args.delete_candidate_days,
            top_n=args.top,
        )
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "recent_days": args.recent_days,
                    "delete_candidate_days": args.delete_candidate_days,
                    "delete_keywords": delete_keywords,
                    "summary": summary,
                    "runs": [run.to_dict() for run in runs],
                },
                indent=2,
            )
            + "\n"
        )

    if args.shell_prune:
        prune_commands = shell_prune_commands(runs)
        args.shell_prune.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(prune_commands) + "\n")

    if args.shell_delete_candidates:
        delete_commands = shell_delete_commands(runs)
        args.shell_delete_candidates.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(delete_commands) + "\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
