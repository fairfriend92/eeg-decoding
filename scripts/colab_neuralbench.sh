#!/usr/bin/env bash
# Runs scripts/neuralbench_eval.py on a Colab session: installs neuralbench
# itself (not in requirements.txt, a heavy, separate dependency set only
# needed for this investigation), syncs src/ and this script, then runs it.
#
# Usage:
#   scripts/colab_neuralbench.sh [-s SESSION] [--gpu TYPE] [--skip-sync] -- [neuralbench_eval.py args...]
#
# NeuralBench's DATA_DIR/SAVE_DIR/CACHE_DIR all point directly at Drive
# (DRIVE_ROOT/neuralbench_data, DRIVE_ROOT/neuralbench_results,
# DRIVE_ROOT/neuralbench_cache, see _colab_common.sh), not a VM-local path
# synced separately. A download or checkpoint is durable the moment it's
# written, with no separate push step to forget: an earlier pull-then-push
# design lost a completed download because the sync-back step never ran
# before the session died. CACHE_DIR was VM-local until a real run showed
# its filtering pass (~22min for dreyer2023's 520 timelines) costs about
# twice the actual training time and gets paid again on every fresh
# session; a completed VM-local cache was copied to Drive by hand once
# instead (heartbeat-wrapped, session confirmed alive throughout), and
# every session from that point on reads straight from the Drive copy
# instead of regenerating it.
#
# A --pretrained-trunk path is always a local file (an src/pretrain_trunk.py
# checkpoint), uploaded to this session automatically before the run, even on
# a session that didn't pretrain it itself.
#
# Examples:
#   scripts/colab_neuralbench.sh -s downloader -- --download --dataset dreyer2023
#   scripts/colab_neuralbench.sh -s trainer --gpu A100 -- --check-only
#   scripts/colab_neuralbench.sh -s trainer -- --debug
#   scripts/colab_neuralbench.sh -s trainer --gpu A100 -- --model reve --dataset dreyer2023
#   scripts/colab_neuralbench.sh -s trainer --gpu A100 -- --model patch_transformer --dataset dreyer2023 \
#       --downstream-wrapper finetune_flatten --pretrained-trunk output/trunk_pretrained/patch_transformer_trunk.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

NEURALBENCH_DATA_DIR="${DRIVE_ROOT}/neuralbench_data"
NEURALBENCH_SAVE_DIR="${DRIVE_ROOT}/neuralbench_results"
NEURALBENCH_CACHE_DIR="${DRIVE_ROOT}/neuralbench_cache"

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

# --download needs no accelerator.
for arg in "${NB_ARGS[@]}"; do
    if [[ "$arg" == "--download" ]]; then
        if [[ -n "$GPU" ]]; then
            echo "== --download requested with --gpu, ignoring --gpu ==" >&2
            GPU=""
        fi
        break
    fi
done

# Deliberately not tuned per invocation. colab exec's --timeout is a hard
# client-side wall-clock ceiling on its own request/reply wait, unrelated
# to the remote kernel and not reset by neuralbench_eval.py's heartbeat:
# hitting it raises a local error while the kernel keeps running whatever
# it was running, unreachable until a fresh exec call is queued behind it.
# Guessing a duration for a training run is a trap either way: too low
# loses the live connection with no way to reconnect to that same output;
# too high just wastes time noticing something actually needs stopping.
# One generous ceiling avoids the guess. To check on a run instead, mount
# Drive on a second, otherwise idle session and list NEURALBENCH_SAVE_DIR's
# checkpoint timestamps; to stop one, use `colab restart-kernel` or
# `colab stop` on the busy session directly rather than a client timeout.
EXEC_TIMEOUT=86400

ensure_session
mount_drive

# A --pretrained-trunk checkpoint is a local output/trunk_pretrained/*.pt file
# from src/pretrain_trunk.py, which download_tree only ever pulls down to the
# local machine, never pushes to Drive. If this session didn't pretrain it
# itself, it has no way to see that path otherwise, so upload it here and
# rewrite the arg to the uploaded remote path.
for i in "${!NB_ARGS[@]}"; do
    if [[ "${NB_ARGS[$i]}" == "--pretrained-trunk" ]]; then
        local_trunk_path="${NB_ARGS[$((i+1))]}"
        remote_trunk_path="${REMOTE_ROOT}/output/trunk_pretrained/$(basename "$local_trunk_path")"
        echo "== uploading pretrained trunk checkpoint =="
        "$COLAB" exec "${SESSION_ARGS[@]}" <<< "import os; os.makedirs(${REMOTE_ROOT@Q} + '/output/trunk_pretrained', exist_ok=True)" >/dev/null
        "$COLAB" upload "${SESSION_ARGS[@]}" "$local_trunk_path" "$remote_trunk_path"
        NB_ARGS[$((i+1))]="$remote_trunk_path"
        break
    fi
done

# ~/.neuralbench/config.json lives under the Colab VM's own /root, so it is
# wiped along with everything else whenever a session is lost and recreated.
# Writing it unconditionally here, every run, is cheap and keeps
# NeuralBench's paths at fixed, known locations instead of falling back to
# /tmp (which loses everything on session loss even more eagerly).
# USER/ENTITY_NAME/PROJECT_NAME are left as fixed literal values rather than
# something derived, since NeuralBench itself treats them as arbitrary
# identifiers, not paths.
"$COLAB" exec "${SESSION_ARGS[@]}" <<PYEOF >/dev/null
import json, os
os.makedirs("/root/.neuralbench", exist_ok=True)
os.makedirs(${NEURALBENCH_DATA_DIR@Q}, exist_ok=True)
os.makedirs(${NEURALBENCH_SAVE_DIR@Q}, exist_ok=True)
os.makedirs(${NEURALBENCH_CACHE_DIR@Q}, exist_ok=True)
cfg = {
    "USER": "eeg-decoding",
    "ENTITY_NAME": "eeg-decoding",
    "PROJECT_NAME": "neuralbench",
    "DATA_DIR": ${NEURALBENCH_DATA_DIR@Q},
    "CACHE_DIR": ${NEURALBENCH_CACHE_DIR@Q},
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
    "$COLAB" install "${SESSION_ARGS[@]}" neuralbench==0.3.1 moabb==1.7.2
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
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout "$EXEC_TIMEOUT"

echo "done"
