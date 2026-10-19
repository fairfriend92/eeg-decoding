#!/usr/bin/env bash
# Extracts frozen REVE embeddings (src/reve_embeddings.py) on a fresh Colab
# session, downloads output/DATASET/reve_embeddings.npz and stops the session.
# Run src/reve_member.py locally afterwards.
#
# Per-subject embeddings are cached on Drive, so rerunning on a new session
# after an interruption resumes instead of starting over.
#
# The run is wrapped in a periodic heartbeat print. The call's --timeout is a
# large client-side ceiling, not an estimate of the run time.
#
# Usage:
#   scripts/colab_reve_embed.sh -s SESSION --gpu TYPE --dataset DATASET [--config FILE]
#
# Example:
#   scripts/colab_reve_embed.sh -s reve-embed --gpu A100 --dataset dreyer2023
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
DATASET=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$SESSION" || -z "$DATASET" ]] && { echo "-s and --dataset are required" >&2; exit 1; }

prepare_session

echo "== syncing src/ and configs/ =="
upload_tree "${REPO_ROOT}/src" "src"
upload_tree "${REPO_ROOT}/configs" "configs"
"$COLAB" install "${SESSION_ARGS[@]}" -r "${REPO_ROOT}/requirements.txt"

REMOTE_OUT="${REMOTE_ROOT}/output/${DATASET}"
echo "import os; os.makedirs('${REMOTE_OUT}', exist_ok=True)" | "$COLAB" exec "${SESSION_ARGS[@]}" >/dev/null

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "$DRIVE_ROOT" "$DATASET" "$CONFIG" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root, drive_data_dir, drive_root, dataset, config = sys.argv[1:6]
remote_src = remote_root + '/src'
args = ['--dataset', dataset, '--cache-dir', drive_root + '/output/' + dataset + '/reve_embeddings']
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
sys.argv = ['reve_embeddings.py'] + {json.dumps(args)}

_stop = threading.Event()
def _heartbeat():
    t0 = time.time()
    while not _stop.wait(30):
        print(f"[heartbeat] {{int(time.time() - t0)}}s", flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()
try:
    exec(compile(open('reve_embeddings.py').read(), 'reve_embeddings.py', 'exec'))
except SystemExit:
    pass
finally:
    _stop.set()""")
PYEOF

echo "== running reve_embeddings.py =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout 604800

echo "== downloading reve_embeddings.npz =="
mkdir -p "${REPO_ROOT}/output/${DATASET}"
"$COLAB" download "${SESSION_ARGS[@]}" "${REMOTE_OUT}/reve_embeddings.npz" "${REPO_ROOT}/output/${DATASET}/reve_embeddings.npz"

"$COLAB" stop "${SESSION_ARGS[@]}"
echo "done"
