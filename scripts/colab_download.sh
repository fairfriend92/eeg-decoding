#!/usr/bin/env bash
# Runs src/download_dataset.py on a CPU-only Colab session, so a dataset is
# on Drive before any GPU run starts. Creates the session without --gpu when
# it does not exist. Does not download output/ (nothing is written there).
#
# MOABB's own dataset downloads are redirected to DRIVE_DATA_DIR (see
# _colab_common.sh), so files fetched here are reused by any later session.
#
# Usage:
#   scripts/colab_download.sh [-s SESSION] [--skip-sync] -- [download_dataset.py args...]
#
# Examples:
#   scripts/colab_download.sh -s downloader -- dreyer2023
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
SKIP_SYNC=0
DOWNLOAD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --skip-sync) SKIP_SYNC=1; shift ;;
        --) shift; DOWNLOAD_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1 (download_dataset.py args go after --)" >&2; exit 1 ;;
    esac
done

# See colab_pretrain_trunk.sh for why this is one large ceiling and not a
# duration estimate.
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
# eviction loop) is needed.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "${DOWNLOAD_ARGS[@]}" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root = sys.argv[1]
remote_src = remote_root + '/src'
drive_data_dir = sys.argv[2]
download_args = sys.argv[3:]
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
print(f"sys.argv = ['download_dataset.py'] + {json.dumps(download_args)}")
print("exec(compile(open('download_dataset.py').read(), 'download_dataset.py', 'exec'))")
PYEOF

echo "== running download_dataset.py ${DOWNLOAD_ARGS[*]:-} =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout "$EXEC_TIMEOUT"

echo "done"
