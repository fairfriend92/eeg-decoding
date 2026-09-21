#!/usr/bin/env bash
# Runs src/train.py on a Colab session, forwarding CLI args, after making
# sure src/ has been synced to the VM and requirements.txt installed.
# Downloads output/ back when done. Does NOT sync data/ (744M+). Run
# `scripts/colab_sync.sh -s SESSION --data` once per session yourself first.
#
# Usage:
#   scripts/colab_train.sh [-s SESSION] [--gpu TYPE] [--skip-sync] -- [train.py args...]
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
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --skip-sync) SKIP_SYNC=1; shift ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1 (train.py args go after --)" >&2; exit 1 ;;
    esac
done

ensure_session

if [[ "$SKIP_SYNC" -eq 0 ]]; then
    echo "== syncing src/ =="
    upload_tree "${REPO_ROOT}/src" "src"

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
# Also evicts any previously-imported project modules from sys.modules:
# the kernel persists across `colab exec` calls, so without this, a
# module re-synced by upload_tree above (e.g. config.py, csp_init.py --
# anything train.py imports other than itself) keeps resolving to
# whatever copy was already cached in sys.modules by an earlier run in
# this same session, silently ignoring the freshly uploaded file.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "${TRAIN_ARGS[@]}" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root = sys.argv[1]
remote_src = remote_root + '/src'
train_args = sys.argv[2:]
print("import sys, os")
print(f"sys.path.insert(0, {remote_src!r})")
print(f"os.chdir({remote_src!r})")
print(
    "for _name, _mod in list(sys.modules.items()):\n"
    "    _file = getattr(_mod, '__file__', None)\n"
    f"    if _file and _file.startswith({remote_src!r}):\n"
    "        del sys.modules[_name]"
)
print(f"sys.argv = ['train.py'] + {json.dumps(train_args)}")
print("exec(compile(open('train.py').read(), 'train.py', 'exec'))")
PYEOF

echo "== running train.py ${TRAIN_ARGS[*]:-} =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout 3600

echo "== downloading output/ =="
download_tree "output"

echo "done"
