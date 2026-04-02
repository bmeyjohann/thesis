#!/usr/bin/env bash
set -euo pipefail

echo "smoke queue sleep: start"
for step in $(seq 0 19); do
  echo "sleep-step=${step}"
  sleep 0.5
done
echo "smoke queue sleep: done"
