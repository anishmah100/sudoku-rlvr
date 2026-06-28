#!/usr/bin/env bash
# Helper: source env + run a python module/script. Usage: bash _launch.sh <args...>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$HERE"
source .venv/bin/activate
source env.sh
exec python "$@"
