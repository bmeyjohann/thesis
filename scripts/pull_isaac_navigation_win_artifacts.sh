#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE'
Usage:
  scripts/pull_isaac_navigation_win_artifacts.sh <windows-run-dir-or-name>
  scripts/pull_isaac_navigation_win_artifacts.sh

Examples:
  scripts/pull_isaac_navigation_win_artifacts.sh
  scripts/pull_isaac_navigation_win_artifacts.sh win_straightline_official_probe
  scripts/pull_isaac_navigation_win_artifacts.sh C:/Data/thesis/logs/isaac_navigation_scripted/win_single_obstacle_teacher_probe
USAGE
  exec "$0" win_straightline_official_probe win_single_obstacle_teacher_probe
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIN_REPO="${WIN_REPO:-/mnt/c/Data/thesis}"
SRC_ROOT="${WIN_REPO}/logs/isaac_navigation_scripted"
DST_ROOT="${REPO_ROOT}/visualizations/isaac_navigation_scripted"

pull_one() {
  local run_ref="$1"
  local src run_name rel copied

  case "${run_ref}" in
    C:/*|C:\\*)
      rel="${run_ref#C:/Data/thesis/}"
      rel="${rel#C:\\Data\\thesis\\}"
      rel="${rel//\\//}"
      src="${WIN_REPO}/${rel}"
      run_name="$(basename "${src}")"
      ;;
    /*)
      src="${run_ref}"
      run_name="$(basename "${src}")"
      ;;
    *)
      src="${SRC_ROOT}/${run_ref}"
      run_name="${run_ref}"
      ;;
  esac

  if [[ ! -d "${src}" ]]; then
    echo "Run directory not found: ${src}" >&2
    return 1
  fi

  mkdir -p "${DST_ROOT}/${run_name}"
  copied=0
  for name in metrics.json trajectories.json trajectories.png; do
    if [[ -f "${src}/${name}" ]]; then
      cp "${src}/${name}" "${DST_ROOT}/${run_name}/${name}"
      copied=$((copied + 1))
    fi
  done

  if [[ -d "${src}/videos" ]]; then
    mkdir -p "${DST_ROOT}/${run_name}/videos"
    find "${src}/videos" -maxdepth 2 -type f \( -name '*.mp4' -o -name '*.gif' \) -exec cp {} "${DST_ROOT}/${run_name}/videos/" \;
  fi

  if [[ -f "${DST_ROOT}/${run_name}/trajectories.json" ]]; then
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_isaac_nav_pull}" \
      /home/benjamin/miniconda3/envs/fasttd3/bin/python "${REPO_ROOT}/tools/make_isaac_navigation_contact_sheet.py" \
        --trajectory_json "${DST_ROOT}/${run_name}/trajectories.json" \
        --output "${DST_ROOT}/${run_name}/trajectory_contact_sheet.png" || true
  fi

  echo "Pulled ${copied} core artifact(s) from ${src}"
  echo "Destination: ${DST_ROOT}/${run_name}"
}

for run_ref in "$@"; do
  pull_one "${run_ref}"
done
