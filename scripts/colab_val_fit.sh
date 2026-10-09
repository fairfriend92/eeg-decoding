#!/usr/bin/env bash
# Trains validation-slice-excluded models (src/train.py --val-fit) on a fresh
# Colab session, then downloads only the resulting *_valfit.pt checkpoints
# and stops the session. The model's local checkpoint manifest is uploaded
# first, since --val-fit reads it. Run dump_predictions.py locally afterwards.
#
# The run is wrapped in a periodic heartbeat print. The call's --timeout is a
# large client-side ceiling, not an estimate of the run time.
#
# Usage:
#   scripts/colab_val_fit.sh -s SESSION --gpu TYPE --model MODEL --dataset DATASET [--config FILE]
#
# Example:
#   scripts/colab_val_fit.sh -s valfit-conformer --gpu A100 --model conformer --dataset dreyer2023
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
MODEL=""
DATASET=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$SESSION" || -z "$MODEL" || -z "$DATASET" ]] && { echo "-s, --model and --dataset are required" >&2; exit 1; }

ensure_session
mount_drive

echo "== syncing src/ and configs/ =="
upload_tree "${REPO_ROOT}/src" "src"
upload_tree "${REPO_ROOT}/configs" "configs"
"$COLAB" install "${SESSION_ARGS[@]}" -r "${REPO_ROOT}/requirements.txt"

REMOTE_OUT="${REMOTE_ROOT}/output/${DATASET}"
echo "import os; os.makedirs('${REMOTE_OUT}/checkpoints', exist_ok=True)" | "$COLAB" exec "${SESSION_ARGS[@]}" >/dev/null
"$COLAB" upload "${SESSION_ARGS[@]}" "${REPO_ROOT}/output/${DATASET}/${MODEL}_checkpoints.json" "${REMOTE_OUT}/${MODEL}_checkpoints.json"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "$MODEL" "$DATASET" "$CONFIG" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root, drive_data_dir, model, dataset, config = sys.argv[1:6]
remote_src = remote_root + '/src'
args = [model, '--dataset', dataset, '--val-fit']
if config:
    args += ['--config', remote_root + '/configs/' + config]
print(f"""import sys, os, time, threading
os.makedirs({drive_data_dir!r}, exist_ok=True)
os.environ['MNE_DATA'] = {drive_data_dir!r}
sys.path.insert(0, {remote_src!r})
os.chdir({remote_src!r})
for _name, _mod in list(sys.modules.items()):
    _file = getattr(_mod, '__file__', None)
    if _file and _file.startswith({remote_src!r}):
        del sys.modules[_name]
sys.argv = ['train.py'] + {json.dumps(args)}

_stop = threading.Event()
def _heartbeat():
    t0 = time.time()
    while not _stop.wait(30):
        print(f"[heartbeat] {{int(time.time() - t0)}}s", flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()
try:
    exec(compile(open('train.py').read(), 'train.py', 'exec'))
except SystemExit:
    pass
finally:
    _stop.set()""")
PYEOF

echo "== running train.py ${MODEL} --val-fit =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout 604800

echo "== downloading ${MODEL} valfit checkpoints =="
mkdir -p "${REPO_ROOT}/output/${DATASET}/checkpoints"
remote_files="$(
    {
        echo "import glob"
        echo "print('\n'.join(glob.glob('${REMOTE_OUT}/checkpoints/${MODEL}_subject*_valfit.pt')))"
    } | "$COLAB" exec "${SESSION_ARGS[@]}" | grep -F "_valfit.pt"
)"
while IFS= read -r remote_file; do
    [[ -z "$remote_file" ]] && continue
    "$COLAB" download "${SESSION_ARGS[@]}" "$remote_file" "${REPO_ROOT}/output/${DATASET}/checkpoints/$(basename "$remote_file")"
done <<< "$remote_files"

"$COLAB" stop "${SESSION_ARGS[@]}"
echo "done"
