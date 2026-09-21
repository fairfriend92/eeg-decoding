#!/usr/bin/env bash
# Installs the Colab CLI into the repo's .venv.
#
# Pins jupyter_kernel_client to 0.15.0: google-colab-cli==0.6.0 expects a
# `KernelClient` class that was renamed to `JupyterKernelClient` in
# jupyter_kernel_client>=1.0.0, which pip otherwise installs by default and
# breaks `colab exec`/`colab run` with:
#   AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
    python3 -m venv "${REPO_ROOT}/.venv"
fi

"${REPO_ROOT}/.venv/bin/pip" install --quiet --upgrade pip
"${REPO_ROOT}/.venv/bin/pip" install --quiet google-colab-cli "jupyter_kernel_client==0.15.0"

echo "colab CLI installed at .venv/bin/colab"
"${REPO_ROOT}/.venv/bin/colab" version
