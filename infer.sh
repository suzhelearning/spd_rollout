#!/usr/bin/env bash
set -euo pipefail
ROLLOUT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROLLOUT_ROOT/abc/.venv/bin/python" "$ROLLOUT_ROOT/infer_recording.py" "$@"
