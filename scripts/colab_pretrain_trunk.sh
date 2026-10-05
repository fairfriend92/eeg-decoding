#!/usr/bin/env bash
# Runs src/pretrain_trunk.py on a Colab session, forwarding CLI args, after
# making sure src/ has been synced to the VM and requirements.txt installed.
# Downloads output/ back when done (this is where pretrain_trunk.py writes
# its trunk checkpoint, under output/trunk_pretrained/).
#
# MOABB's own dataset downloads are redirected to DRIVE_DATA_DIR (see
# _colab_common.sh), not the VM-local data/ dir: a dataset downloaded by
# one session is immediately durable and reused by any later session, with
# no separate sync step (same design as colab_neuralbench.sh's
# NEURALBENCH_DATA_DIR).
#
# Usage:
#   scripts/colab_pretrain_trunk.sh [-s SESSION] [--gpu TYPE] [--skip-sync] -- [pretrain_trunk.py args...]
#
# Examples:
#   scripts/colab_pretrain_trunk.sh -s trainer --gpu T4 -- eegnet --source-datasets bci2a scherer2015 --epochs 2
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
SKIP_SYNC=0
TRUNK_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --skip-sync) SKIP_SYNC=1; shift ;;
        --) shift; TRUNK_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1 (pretrain_trunk.py args go after --)" >&2; exit 1 ;;
    esac
done

# Deliberately not tuned per invocation. colab exec's --timeout is a hard
# client-side wall-clock ceiling on its own request/reply wait, unrelated to
# the remote kernel and not reset by pretrain_trunk.py's own per-epoch/
# per-step print output: hitting it raises a local error while the kernel
# keeps running whatever it was running, unreachable until a fresh exec
# call is queued behind it. One generous ceiling avoids guessing a run's
# duration. To check on a run instead, mount Drive on a second, otherwise
# idle session; to stop one, use `colab restart-kernel` or `colab stop` on
# the busy session directly rather than a client timeout.
EXEC_TIMEOUT=86400

ensure_session
mount_drive

if [[ "$SKIP_SYNC" -eq 0 ]]; then
    echo "== syncing src/ =="
    upload_tree "${REPO_ROOT}/src" "src"

    echo "== installing requirements.txt =="
    "$COLAB" install "${SESSION_ARGS[@]}" -r "${REPO_ROOT}/requirements.txt"
fi

# See colab_train.sh for why this bootstrap (including the sys.modules
# eviction loop) is needed. The MNE_DATA override must run before src/ is
# imported: data_loader.py only sets it via os.environ.setdefault, which
# is a no-op once it is already set.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "${TRUNK_ARGS[@]}" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root = sys.argv[1]
remote_src = remote_root + '/src'
drive_data_dir = sys.argv[2]
trunk_args = sys.argv[3:]
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
print(f"sys.argv = ['pretrain_trunk.py'] + {json.dumps(trunk_args)}")
print("exec(compile(open('pretrain_trunk.py').read(), 'pretrain_trunk.py', 'exec'))")
PYEOF

echo "== running pretrain_trunk.py ${TRUNK_ARGS[*]:-} =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout "$EXEC_TIMEOUT"

echo "== downloading output/ =="
download_tree "output"

echo "done"
