#!/usr/bin/env bash
set -euo pipefail
python -m kvlab env --config configs/environment.yaml --output outputs/gpu-environment.json
python -m pytest -q
python -m kvlab run --config configs/gpu-smoke.yaml "$@"
