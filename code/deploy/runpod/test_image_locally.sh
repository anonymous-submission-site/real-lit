#!/usr/bin/env bash
# End-to-end local test for the RunPod image.
# Builds the image, runs handler.py with a single event, asserts a JSON.
set -euo pipefail

IMAGE="${IMAGE:-n26-tabular:smoke}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "==> Building $IMAGE (this will take 5-10 min the first time)"
docker build -t "$IMAGE" -f "$REPO/deploy/runpod/Dockerfile" "$REPO"

echo "==> Running a single cell inside the container"
# On a CPU-only host (no NVIDIA driver), drop --gpus=all. CUDA-dependent
# imputers/classifiers will warn at startup but the import graph still validates.
docker run --rm --gpus=all \
    -e RUNPOD_TEST_INPUT='{"input":{"experiment_id":1,"run_name":"smoke","csv":"docs/ablation_matrix.csv","time_budget_sec":300}}' \
    "$IMAGE" \
    python -c "
import json, os
from deploy.runpod.handler import handler
event = json.loads(os.environ['RUNPOD_TEST_INPUT'])
out = handler(event)
print(json.dumps(out)[:500])
# handler returns 'ok' on rc==0 or 'rc=N' otherwise; rc=3 is skip-by-design.
assert out['status'] == 'ok' or out['status'] == 'rc=3', out
"
