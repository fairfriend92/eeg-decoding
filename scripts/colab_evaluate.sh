#!/usr/bin/env bash
# Runs src/evaluate.py on a Colab session, scoring output/'s checkpoints on
# the held-out test set. Uploads src/ and output/ and installs
# requirements.txt unless --skip-sync, and downloads output/ back when
# done (updated *_results.json and summary.json).
#
# Usage:
#   scripts/colab_evaluate.sh [-s SESSION] [--gpu TYPE] [--skip-sync] -- {eegnet,conformer} [...]
#
# Examples:
#   scripts/colab_evaluate.sh -s trainer -- eegnet conformer
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
SKIP_SYNC=0
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --skip-sync) SKIP_SYNC=1; shift ;;
        --) shift; EVAL_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1 (evaluate.py args go after --)" >&2; exit 1 ;;
    esac
done

if [[ "${#EVAL_ARGS[@]}" -eq 0 ]]; then
    echo "Usage: $0 [-s SESSION] [--gpu TYPE] [--skip-sync] -- {eegnet,conformer} [...]" >&2
    exit 1
fi

ensure_session

if [[ "$SKIP_SYNC" -eq 0 ]]; then
    echo "== syncing src/ =="
    upload_tree "${REPO_ROOT}/src" "src"

    echo "== syncing output/ (checkpoints) =="
    upload_tree "${REPO_ROOT}/output" "output"

    echo "== installing requirements.txt =="
    "$COLAB" install "${SESSION_ARGS[@]}" -r "${REPO_ROOT}/requirements.txt"
fi

# See colab_train.sh for why this bootstrap (including the sys.modules
# eviction loop) is needed.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
BOOTSTRAP="${SCRATCH}/bootstrap.py"
python3 - "$REMOTE_ROOT" "${EVAL_ARGS[@]}" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root = sys.argv[1]
remote_src = remote_root + '/src'
eval_args = sys.argv[2:]
print("import sys, os")
print(f"sys.path.insert(0, {remote_src!r})")
print(f"os.chdir({remote_src!r})")
print(
    "for _name, _mod in list(sys.modules.items()):\n"
    "    _file = getattr(_mod, '__file__', None)\n"
    f"    if _file and _file.startswith({remote_src!r}):\n"
    "        del sys.modules[_name]"
)
print(f"sys.argv = ['evaluate.py'] + {json.dumps(eval_args)}")
print("exec(compile(open('evaluate.py').read(), 'evaluate.py', 'exec'))")
PYEOF

echo "== running evaluate.py ${EVAL_ARGS[*]:-} =="
"$COLAB" exec "${SESSION_ARGS[@]}" -f "$BOOTSTRAP" --timeout 3600

echo "== downloading output/ =="
download_tree "output"

echo "done"
