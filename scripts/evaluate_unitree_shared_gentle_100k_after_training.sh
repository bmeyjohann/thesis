#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
RUN="$ROOT/artifacts/unitree_multimodal_target/gentle_gru_reconstruction_shared_seed1_100k_20260903_v3"
OUTPUT="$RUN/eval_matrix"
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
TRAIN_SERVICE=unitree-shared-gentle-multimodal-100k.service

while systemctl --user is-active --quiet "$TRAIN_SERVICE"; do
  sleep 300
done

CHECKPOINT="$RUN/student_100000.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training ended without expected checkpoint: $CHECKPOINT" >&2
  exit 2
fi
mkdir -p "$OUTPUT"
for seed in 11 23 37; do
  for modality in height_scan depth mono_rgb stereo_rgb; do
    for geometry in flat random_rough cobblestone stairs stepping_stones; do
      for material in rigid slippery sand_drag; do
        stem="${modality}__${geometry}__${material}__seed${seed}"
        summary="$OUTPUT/${stem}.json"
        [[ -f "$summary" ]] && continue
        "$PYTHON" "$ROOT/eval_unitree_recurrent_student.py" \
          --modality "$modality" \
          --geometry "$geometry" \
          --material "$material" \
          --checkpoint "$CHECKPOINT" \
          --summary-file "$summary" \
          --seed "$seed" \
          --num-envs 32 \
          --steps 300 \
          >"$OUTPUT/${stem}.log" 2>&1
      done
    done
  done
done

"$PYTHON" -c '
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
with (root / "modality_terrain_matrix.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader()
    writer.writerows(rows)
print(root / "modality_terrain_matrix.csv")
' "$OUTPUT"
