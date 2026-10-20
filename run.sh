#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_DIR/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
if [[ $# -eq 0 ]]; then
  set -- --help
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run --project "$PROJECT_DIR" --locked --no-dev python "$PROJECT_DIR/scripts/analyze_kick.py" "$@"
fi
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/analyze_kick.py" "$@"
fi
if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
  exec "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/scripts/analyze_kick.py" "$@"
fi
printf '%s\n' 'Install uv and run uv sync first. See README.md.' >&2
exit 1
