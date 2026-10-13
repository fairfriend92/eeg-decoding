#!/usr/bin/env bash
# Fits cross-subject ensemble members (src/cross_subject_members.py) on a fresh
# Colab session, downloads output/DATASET/cross_subject/predictions_MEMBER.npz
# for each and stops the session.
#
# Epoched participants are cached on Drive, so rerunning on a new session
# after an interruption skips the slow epoching.
#
# The run is wrapped in a periodic heartbeat print. The call's --timeout is a
# large client-side ceiling, not an estimate of the run time.
#
# Usage:
#   scripts/colab_cross_subject_members.sh -s SESSION --gpu TYPE --dataset DATASET --members "MEMBER ..." [--config FILE] [--set "KEY=VALUE ..."] [--suffix SUFFIX]
#
# --set passes config overrides through. --suffix is the dump file-name suffix
# those overrides produce (_ea for data.alignment=euclidean), so the download
# finds the right files.
#
# Example:
#   scripts/colab_cross_subject_members.sh -s xsubj --gpu A100 --dataset dreyer2023 --members "patch_transformer"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
DATASET=""
MEMBERS=""
CONFIG=""
SETS=""
SUFFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --members) MEMBERS="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --set) SETS="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$SESSION" || -z "$DATASET" || -z "$MEMBERS" ]] && { echo "-s, --dataset and --members are required" >&2; exit 1; }

ensure_session
mount_drive

echo "== syncing src/ and configs/ =="
upload_tree "${REPO_ROOT}/src" "src"
upload_tree "${REPO_ROOT}/configs" "configs"
# pip runs under a heartbeat: a long silent call is suspected to kill the Colab tunnel.
python3 - "${REPO_ROOT}/requirements.txt" <<'PYEOF' | "$COLAB" exec "${SESSION_ARGS[@]}" --timeout 604800
import json, sys
reqs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
print(f"""import subprocess, sys, threading, time
_stop = threading.Event()
def _heartbeat():
    t0 = time.time()
    while not _stop.wait(30):
        print(f"[heartbeat] pip {{int(time.time() - t0)}}s", flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + {json.dumps(reqs)}, check=True)
finally:
    _stop.set()
print("pip install done", flush=True)""")
PYEOF

REMOTE_OUT="${REMOTE_ROOT}/output/${DATASET}"
echo "import os; os.makedirs('${REMOTE_OUT}', exist_ok=True)" | "$COLAB" exec "${SESSION_ARGS[@]}" >/dev/null

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "$DRIVE_ROOT" "$DATASET" "$MEMBERS" "$CONFIG" "$SETS" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root, drive_data_dir, drive_root, dataset, members, config, sets = sys.argv[1:8]
remote_src = remote_root + '/src'
args = members.split() + ['--dataset', dataset, '--n-jobs', '2', '--cache-dir', drive_root + '/output/' + dataset]
if config:
    args += ['--config', remote_root + '/configs/' + config]
for kv in sets.split():
    args += ['--set', kv]
print(f"""import sys, os, time, threading
os.makedirs({drive_data_dir!r}, exist_ok=True)
os.environ['MNE_DATA'] = {drive_data_dir!r}
sys.path.insert(0, {remote_src!r})
os.chdir({remote_src!r})
for _name, _mod in list(sys.modules.items()):
    _file = getattr(_mod, '__file__', None)
    if _file and _file.startswith({remote_src!r}):
        del sys.modules[_name]
sys.argv = ['cross_subject_members.py'] + {json.dumps(args)}

_stop = threading.Event()
def _heartbeat():
    t0 = time.time()
    while not _stop.wait(30):
        print(f"[heartbeat] {{int(time.time() - t0)}}s", flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()
try:
    exec(compile(open('cross_subject_members.py').read(), 'cross_subject_members.py', 'exec'))
except SystemExit:
    pass
finally:
    _stop.set()""")
PYEOF

echo "== running cross_subject_members.py =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout 604800

mkdir -p "${REPO_ROOT}/output/${DATASET}/cross_subject"
for member in $MEMBERS; do
    echo "== downloading predictions_${member}${SUFFIX}.npz =="
    "$COLAB" download "${SESSION_ARGS[@]}" "${REMOTE_OUT}/cross_subject/predictions_${member}${SUFFIX}.npz" "${REPO_ROOT}/output/${DATASET}/cross_subject/predictions_${member}${SUFFIX}.npz"
done

"$COLAB" stop "${SESSION_ARGS[@]}"
echo "done"
