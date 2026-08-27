#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 CHECKPOINT OUTPUT_DIR TRAIN_PARENT_PID TRAIN_CHILD_PID" >&2
    exit 2
fi

checkpoint=$1
output_dir=$2
train_parent_pid=$3
train_child_pid=$4
python=/home/benjamin/miniconda3/envs/fasttd3/bin/python

mkdir -p "$output_dir"
kill -STOP "$train_parent_pid" "$train_child_pid"
resume_training() {
    kill -CONT "$train_parent_pid" "$train_child_pid" 2>/dev/null || true
}
trap resume_training EXIT

for modality in height_scan depth mono_rgb stereo_rgb; do
    echo "[diagnostic] evaluating $modality"
    "$python" eval_unitree_modality_student.py \
        --modality "$modality" \
        --geometry random_rough \
        --material rigid \
        --checkpoint "$checkpoint" \
        --summary-file "$output_dir/$modality.json" \
        --seed 101 \
        --num-envs 32 \
        --steps 300 \
        >"$output_dir/$modality.log" 2>&1
done

cat "$output_dir"/*.json
