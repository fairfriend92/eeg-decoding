#!/usr/bin/env bash
# Fits the patch transformer for a Codabench submission (src/export_submission.py)
# on a fresh Colab session, downloads output/DATASET/submission/weights.pt and
# stops the session.
#
# Epoched participants are cached on Drive. The run is wrapped in a periodic
# heartbeat print. The call's --timeout is a large client-side ceiling, not an
# estimate of the run time.
#
# Usage:
#   scripts/colab_export_submission.sh -s SESSION --gpu TYPE --dataset DATASET [--config FILE] [--set "KEY=VALUE ..."] [--all-participants]
#
# --all-participants fits on every participant (sealed phase) and names the
# file weights_all_participants.pt. Without it the preset's test participants
# stay out of the fit and are scored, and the file is weights.pt.
#
# Example:
#   scripts/colab_export_submission.sh -s export --gpu A100 --dataset proteus --config patch_overlap.yaml --set "data.alignment=euclidean data.test_alignment=causal"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
DATASET=""
CONFIG=""
SETS=""
ALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --set) SETS="$2"; shift 2 ;;
        --all-participants) ALL=1; shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$SESSION" || -z "$DATASET" ]] && { echo "-s and --dataset are required" >&2; exit 1; }

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
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "$DRIVE_ROOT" "$DATASET" "$CONFIG" "$SETS" "$ALL" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root, drive_data_dir, drive_root, dataset, config, sets, all_participants = sys.argv[1:8]
name = 'weights_all_participants.pt' if int(all_participants) else 'weights.pt'
remote_src = remote_root + '/src'
args = ['--dataset', dataset, '--n-jobs', '2', '--cache-dir', drive_root + '/output/' + dataset, '--out', remote_root + '/output/' + dataset + '/submission/' + name]
if int(all_participants):
    args += ['--all-participants']
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
sys.argv = ['export_submission.py'] + {json.dumps(args)}

_stop = threading.Event()
def _heartbeat():
    t0 = time.time()
    while not _stop.wait(30):
        print(f"[heartbeat] {{int(time.time() - t0)}}s", flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()
try:
    exec(compile(open('export_submission.py').read(), 'export_submission.py', 'exec'))
except SystemExit:
    pass
finally:
    _stop.set()""")
PYEOF

echo "== running export_submission.py =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout 604800

mkdir -p "${REPO_ROOT}/output/${DATASET}/submission"
WEIGHTS_NAME="weights.pt"
[[ "$ALL" -eq 1 ]] && WEIGHTS_NAME="weights_all_participants.pt"
echo "== downloading ${WEIGHTS_NAME} =="
"$COLAB" download "${SESSION_ARGS[@]}" "${REMOTE_OUT}/submission/${WEIGHTS_NAME}" "${REPO_ROOT}/output/${DATASET}/submission/${WEIGHTS_NAME}"

"$COLAB" stop "${SESSION_ARGS[@]}"
echo "done"
