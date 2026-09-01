#!/usr/bin/env bash
# Create the tested local Python environment for the CW-Net toy demo.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_PREFIX="${CWNET_ENV_PREFIX:-$REPO_ROOT/.venv-codeocean}"
CONDA_COMMAND="${CONDA_EXE:-conda}"

if ! command -v "$CONDA_COMMAND" >/dev/null 2>&1; then
    echo "Conda is required. Install Miniconda or Miniforge and retry." >&2
    exit 1
fi

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
    PYTHON_VERSION="$("$ENV_PREFIX/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$PYTHON_VERSION" != "3.9" ]]; then
        echo "Existing environment uses Python $PYTHON_VERSION; expected 3.9." >&2
        echo "Remove $ENV_PREFIX or set CWNET_ENV_PREFIX to a new path." >&2
        exit 1
    fi
    echo "Reusing existing Python 3.9 environment: $ENV_PREFIX"
else
    "$CONDA_COMMAND" create \
        --prefix "$ENV_PREFIX" \
        --override-channels \
        --channel conda-forge \
        python=3.9 \
        pip=22.1.2 \
        swig \
        -y
fi

"$CONDA_COMMAND" run --prefix "$ENV_PREFIX" \
    python -m pip install --disable-pip-version-check \
    -r "$SCRIPT_DIR/requirements.txt"

echo "Environment ready. Activate it with:"
printf 'conda activate "%s"\n' "$ENV_PREFIX"
