#!/usr/bin/env bash
# Runs src/train.py on a Colab session, forwarding CLI args, after making
# sure src/ has been synced to the VM and requirements.txt installed.
# Downloads output/ back when done.
#
# MOABB's own dataset downloads are redirected to DRIVE_DATA_DIR (see
# _colab_common.sh), not the VM-local data/ dir: a dataset downloaded by
# one session is immediately durable and reused by any later session, with
# no separate sync step.
#
# Usage:
#   scripts/colab_train.sh [-s SESSION] [--gpu TYPE] [--skip-sync] [--script NAME] -- [script args...]
#
# --script runs another src/ entry point (e.g. sweep.py) instead of train.py.
#
# --skip-sync also skips the requirements.txt install; use it for repeat
# runs against a session you've already synced.
#
# Examples:
#   scripts/colab_train.sh -s trainer --gpu T4 -- eegnet --n-folds 5
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
SKIP_SYNC=0
SCRIPT="train.py"
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --skip-sync) SKIP_SYNC=1; shift ;;
        --script) SCRIPT="$2"; shift 2 ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1 (train.py args go after --)" >&2; exit 1 ;;
    esac
done

ensure_session
mount_drive

if [[ "$SKIP_SYNC" -eq 0 ]]; then
    echo "== syncing src/ =="
    upload_tree "${REPO_ROOT}/src" "src"
    upload_tree "${REPO_ROOT}/configs" "configs"

    echo "== installing requirements.txt =="
    "$COLAB" install "${SESSION_ARGS[@]}" -r "${REPO_ROOT}/requirements.txt"
fi

# Build a small bootstrap that sets up sys.path/cwd/argv the way `python
# src/train.py` would locally, then execs the already-uploaded train.py.
# Needed because `colab exec` runs code inside a live kernel process, so
# sys.path[0] isn't the uploaded src/ dir the way it would be for a normal
# `python train.py` invocation, and train.py's flat `from config import
# ...` imports depend on that.
#
# The MNE_DATA override must run before src/ is imported: data_loader.py
# only sets it via os.environ.setdefault, which is a no-op once it is
# already set.
#
# Also evicts any previously-imported project modules from sys.modules:
# the kernel persists across `colab exec` calls, so without this, a
# module re-synced by upload_tree above (e.g. config.py, csp_init.py.
# Anything train.py imports other than itself) keeps resolving to
# whatever copy was already cached in sys.modules by an earlier run in
# this same session, silently ignoring the freshly uploaded file.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "$SCRIPT" "${TRAIN_ARGS[@]}" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root = sys.argv[1]
remote_src = remote_root + '/src'
drive_data_dir = sys.argv[2]
script = sys.argv[3]
train_args = sys.argv[4:]
print("import sys, os")
print(f"os.makedirs({drive_data_dir!r}, exist_ok=True)")
print(f"os.environ['MNE_DATA'] = {drive_data_dir!r}")
print(f"sys.path.insert(0, {remote_src!r})")
print(f"os.chdir({remote_src!r})")
print(
    "for _name, _mod in list(sys.modules.items()):\n"
    "    _file = getattr(_mod, '__file__', None)\n"
    f"    if _file and _file.startswith({remote_src!r}):\n"
    "        del sys.modules[_name]"
)
print(f"sys.argv = [{script!r}] + {json.dumps(train_args)}")
print(f"exec(compile(open({script!r}).read(), {script!r}, 'exec'))")
PYEOF

echo "== running ${SCRIPT} ${TRAIN_ARGS[*]:-} =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout 86400

echo "== downloading output/ =="
download_tree "output"

echo "done"
