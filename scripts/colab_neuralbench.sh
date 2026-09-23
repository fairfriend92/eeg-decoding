#!/usr/bin/env bash
# Runs scripts/neuralbench_eval.py on a Colab session: installs neuralbench
# itself (not in requirements.txt -- a heavy, separate dependency set only
# needed for this investigation, see docs/SESSION_HANDOFF.md), syncs src/
# and this script, then runs it.
#
# Usage:
#   scripts/colab_neuralbench.sh [-s SESSION] [--gpu TYPE] [--skip-sync] -- [neuralbench_eval.py args...]
#
# NeuralBench's DATA_DIR/SAVE_DIR point directly at Drive
# (DRIVE_ROOT/neuralbench_data, DRIVE_ROOT/neuralbench_results, see
# _colab_common.sh), not a VM-local path synced separately -- a download or
# checkpoint is durable the moment it's written, with no separate push step
# to forget (see docs/SESSION_HANDOFF.md for the incident that prompted
# this: a completed download was lost because the sync-back step never ran
# before the session died). CACHE_DIR stays VM-local since it's a fully
# regenerable derived cache, not something worth persisting.
#
# Examples:
#   scripts/colab_neuralbench.sh -s trainer --gpu T4 -- --check-only
#   scripts/colab_neuralbench.sh -s trainer -- --debug
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

NEURALBENCH_DATA_DIR="${DRIVE_ROOT}/neuralbench_data"
NEURALBENCH_SAVE_DIR="${DRIVE_ROOT}/neuralbench_results"

SESSION=""
GPU=""
SKIP_SYNC=0
NB_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --skip-sync) SKIP_SYNC=1; shift ;;
        --) shift; NB_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1 (neuralbench_eval.py args go after --)" >&2; exit 1 ;;
    esac
done

ensure_session
mount_drive

# ~/.neuralbench/config.json lives under the Colab VM's own /root, so it is
# wiped along with everything else whenever a session is lost and recreated
# (see docs/SESSION_HANDOFF.md's Colab-session-churn notes). Writing it
# unconditionally here, every run, is cheap and keeps NeuralBench's paths at
# fixed, known locations instead of falling back to /tmp (which loses
# everything on session loss even more eagerly). USER/ENTITY_NAME/
# PROJECT_NAME are left as fixed literal values rather than something
# derived, since NeuralBench itself treats them as arbitrary identifiers,
# not paths.
"$COLAB" exec "${SESSION_ARGS[@]}" <<PYEOF >/dev/null
import json, os
os.makedirs("/root/.neuralbench", exist_ok=True)
os.makedirs(${NEURALBENCH_DATA_DIR@Q}, exist_ok=True)
os.makedirs(${NEURALBENCH_SAVE_DIR@Q}, exist_ok=True)
cfg = {
    "USER": "eeg-decoding",
    "ENTITY_NAME": "eeg-decoding",
    "PROJECT_NAME": "neuralbench",
    "DATA_DIR": ${NEURALBENCH_DATA_DIR@Q},
    "CACHE_DIR": "/content/neuralbench_cache",
    "SAVE_DIR": ${NEURALBENCH_SAVE_DIR@Q},
    "WANDB_HOST": "",
    "CLUSTER": None,
}
with open("/root/.neuralbench/config.json", "w") as f:
    json.dump(cfg, f)
PYEOF

if [[ "$SKIP_SYNC" -eq 0 ]]; then
    echo "== syncing src/ =="
    upload_tree "${REPO_ROOT}/src" "src"

    echo "== syncing scripts/neuralbench_eval.py =="
    "$COLAB" exec "${SESSION_ARGS[@]}" <<< "import os; os.makedirs(${REMOTE_ROOT@Q} + '/scripts', exist_ok=True)" >/dev/null
    "$COLAB" upload "${SESSION_ARGS[@]}" "${REPO_ROOT}/scripts/neuralbench_eval.py" "${REMOTE_ROOT}/scripts/neuralbench_eval.py"

    echo "== installing neuralbench =="
    "$COLAB" install "${SESSION_ARGS[@]}" neuralbench 'moabb>=1.7.1'
fi

# See colab_train.sh for why this bootstrap (including the sys.modules
# eviction loop) is needed. Chdirs into scripts/, not src/, since
# neuralbench_eval.py itself inserts its sibling src/ onto sys.path.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "${NB_ARGS[@]}" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root = sys.argv[1]
remote_scripts = remote_root + '/scripts'
nb_args = sys.argv[2:]
print("import sys, os")
print(f"sys.path.insert(0, {remote_scripts!r})")
print(f"os.chdir({remote_scripts!r})")
print(
    "for _name, _mod in list(sys.modules.items()):\n"
    "    _file = getattr(_mod, '__file__', None)\n"
    f"    if _file and (_file.startswith({remote_scripts!r}) or _file.startswith({(remote_root + '/src')!r})):\n"
    "        del sys.modules[_name]"
)
print(f"sys.argv = ['neuralbench_eval.py'] + {json.dumps(nb_args)}")
print("exec(compile(open('neuralbench_eval.py').read(), 'neuralbench_eval.py', 'exec'))")
PYEOF

echo "== running neuralbench_eval.py ${NB_ARGS[*]:-} =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout 3600

echo "done"
