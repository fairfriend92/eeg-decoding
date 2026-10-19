#!/usr/bin/env bash
# Fits one neural member per fold on a fresh Colab session, replays the test
# streams window by window under each test-time adaptation variant
# (src/test_time_adapt.py) and downloads
# output/DATASET/cross_subject/predictions_MEMBER<suffix>_tta_VARIANT<tag><fold>.npz
# for none, abn and tent, then stops the session.
#
# The run is wrapped in a periodic heartbeat print. The call's --timeout is a
# large client-side ceiling, not an estimate of the run time.
#
# Usage:
#   scripts/colab_test_time_adapt.sh -s SESSION --gpu TYPE --dataset DATASET --member MEMBER --epochs N [--kfold K] [--config FILE] [--set "KEY=VALUE ..."] [--suffix SUFFIX] [--n-folds N] [--exclude "ID ..."] [--tag TAG] [--lr LR] [--prior-windows K]
#
# --suffix is the dump file-name suffix the --set overrides produce
# (_ea_testcausal for data.alignment=euclidean data.test_alignment=causal).
# --n-folds N cross-validates over N test-participant folds in the one session.
# --exclude leaves participants out of the pool entirely.
#
# Example:
#   scripts/colab_test_time_adapt.sh -s tta --gpu A100 --dataset proteus --member patch_transformer --epochs 8 --config patch_overlap.yaml --set "data.alignment=euclidean data.test_alignment=causal" --suffix _ea_testcausal --n-folds 8 --exclude "4 9"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
GPU=""
DATASET=""
MEMBER=""
CONFIG=""
SETS=""
SUFFIX=""
N_FOLDS=0
KFOLD=0
LR=""
PRIOR=""
EXCLUDE=""
TAG=""
EPOCHS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --member) MEMBER="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --set) SETS="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --n-folds) N_FOLDS="$2"; shift 2 ;;
        --kfold) KFOLD="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --prior-windows) PRIOR="$2"; shift 2 ;;
        --exclude) EXCLUDE="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$SESSION" || -z "$DATASET" || -z "$MEMBER" || ( -z "$EPOCHS" && "$KFOLD" -eq 0 ) ]] && { echo "-s, --dataset and --member are required, and --epochs unless --kfold is set" >&2; exit 1; }

prepare_session

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
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "$DRIVE_ROOT" "$DATASET" "$MEMBER" "$CONFIG" "$SETS" "$N_FOLDS" "$EXCLUDE" "$TAG" "$EPOCHS" "$LR" "$PRIOR" "$KFOLD" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root, drive_data_dir, drive_root, dataset, member, config, sets, n_folds, exclude, tag, epochs, lr, prior, kfold = sys.argv[1:15]
remote_src = remote_root + '/src'
args = ['--member', member, '--dataset', dataset, '--n-jobs', '2', '--cache-dir', drive_root + '/output/' + dataset]
if config:
    args += ['--config', remote_root + '/configs/' + config]
for kv in sets.split():
    args += ['--set', kv]
if int(n_folds):
    args += ['--n-folds', n_folds]
if epochs:
    args += ['--epochs', epochs]
if int(kfold):
    args += ['--kfold', kfold]
if lr:
    args += ['--lr', lr]
if prior:
    args += ['--prior-windows', prior]
if exclude.split():
    args += ['--exclude-subjects'] + exclude.split()
if tag:
    args += ['--tag', tag]
print(f"""import sys, os, time, threading
os.makedirs({drive_data_dir!r}, exist_ok=True)
os.environ['MNE_DATA'] = {drive_data_dir!r}
sys.path.insert(0, {remote_src!r})
os.chdir({remote_src!r})
for _name, _mod in list(sys.modules.items()):
    _file = getattr(_mod, '__file__', None)
    if _file and _file.startswith({remote_src!r}):
        del sys.modules[_name]
sys.argv = ['test_time_adapt.py'] + {json.dumps(args)}

_stop = threading.Event()
def _heartbeat():
    t0 = time.time()
    while not _stop.wait(30):
        print(f"[heartbeat] {{int(time.time() - t0)}}s", flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()
try:
    exec(compile(open('test_time_adapt.py').read(), 'test_time_adapt.py', 'exec'))
except SystemExit:
    pass
finally:
    _stop.set()""")
PYEOF

echo "== running test_time_adapt.py =="
exec_bootstrap "$BOOTSTRAP" 604800

mkdir -p "${REPO_ROOT}/output/${DATASET}/cross_subject"
FOLD_SUFFIXES=("")
[[ "$N_FOLDS" -gt 0 ]] && FOLD_SUFFIXES=($(for ((k = 0; k < N_FOLDS; k++)); do echo "_f${k}"; done))
for variant in none abn tent; do
    for fold in "${FOLD_SUFFIXES[@]}"; do
        name="predictions_${MEMBER}${SUFFIX}_tta_${variant}${TAG}${fold}.npz"
        echo "== downloading ${name} =="
        "$COLAB" download "${SESSION_ARGS[@]}" "${REMOTE_OUT}/cross_subject/${name}" "${REPO_ROOT}/output/${DATASET}/cross_subject/${name}"
    done
done

"$COLAB" stop "${SESSION_ARGS[@]}"
echo "done"
