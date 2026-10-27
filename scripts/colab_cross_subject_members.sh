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
#   scripts/colab_cross_subject_members.sh -s SESSION --gpu TYPE --dataset DATASET --members "MEMBER ..." [--config FILE] [--set "KEY=VALUE ..."] [--suffix SUFFIX] [--n-folds N] [--kfold K] [--early-stop-all] [--ensemble] [--exclude "ID ..."] [--tag TAG] [--epochs N] [--weights]
#
# --set passes config overrides through. --suffix is the dump file-name suffix
# those overrides produce (_ea for data.alignment=euclidean), so the download
# finds the right files. --n-folds N cross-validates over N test-participant
# folds in the one session and downloads every fold's dump (_f0 ... _fN-1).
# --kfold K fits a participant-grouped K-fold ensemble per member instead of
# the single refit, and downloads predictions_MEMBER<suffix>_kfoldK<fold>.npz.
# --early-stop-all fits each member on fitting plus validation participants with
# early stopping instead, and downloads predictions_MEMBER<suffix>_esall<fold>.npz.
# Without --ensemble the single dump's epoch count comes from every training
# participant's windows. --ensemble keeps the ensemble harness layout (epoch
# count from the fitting participants, validation probabilities in the dump).
# --exclude leaves participants out of the pool entirely. --tag is extra
# file-name text before the fold index, which the download then expects.
# Every neural run writes weights files on the VM. --weights downloads them:
# weights_MEMBER<suffix>_kfoldK<tag><fold>.pt (the K models' state dicts) with
# --kfold K, and weights_MEMBER<suffix><tag><fold>.pt (the final refit's state
# dict) for a single dump.
# --epochs N fixes the epoch count of the neural members' final fit and skips
# the early-stopped fit that otherwise sets it.
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
N_FOLDS=0
KFOLD=0
ES_ALL=0
ENSEMBLE=0
EXCLUDE=""
TAG=""
EPOCHS=""
WEIGHTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --members) MEMBERS="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --set) SETS="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --n-folds) N_FOLDS="$2"; shift 2 ;;
        --kfold) KFOLD="$2"; shift 2 ;;
        --early-stop-all) ES_ALL=1; shift ;;
        --ensemble) ENSEMBLE=1; shift ;;
        --exclude) EXCLUDE="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --weights) WEIGHTS=1; shift ;;
        -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$SESSION" || -z "$DATASET" || -z "$MEMBERS" ]] && { echo "-s, --dataset and --members are required" >&2; exit 1; }

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
python3 - "$REMOTE_ROOT" "$DRIVE_DATA_DIR" "$DRIVE_ROOT" "$DATASET" "$MEMBERS" "$CONFIG" "$SETS" "$N_FOLDS" "$KFOLD" "$ES_ALL" "$ENSEMBLE" "$EXCLUDE" "$TAG" "$EPOCHS" > "$BOOTSTRAP" <<'PYEOF'
import sys, json
remote_root, drive_data_dir, drive_root, dataset, members, config, sets, n_folds, kfold, es_all, ensemble, exclude, tag, epochs = sys.argv[1:15]
remote_src = remote_root + '/src'
args = members.split() + ['--dataset', dataset, '--n-jobs', '2', '--cache-dir', drive_root + '/output/' + dataset]
if config:
    args += ['--config', remote_root + '/configs/' + config]
for kv in sets.split():
    args += ['--set', kv]
if int(n_folds):
    args += ['--n-folds', n_folds]
if int(kfold):
    args += ['--kfold', kfold]
if int(es_all):
    args += ['--early-stop-all']
if int(ensemble):
    args += ['--ensemble']
if exclude.split():
    args += ['--exclude-subjects'] + exclude.split()
if tag:
    args += ['--tag', tag]
if epochs:
    args += ['--epochs', epochs]
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
KFOLD_TAG=""
[[ "$KFOLD" -gt 0 ]] && KFOLD_TAG="_kfold${KFOLD}"
[[ "$ES_ALL" -eq 1 ]] && KFOLD_TAG="_esall"
FOLD_SUFFIXES=("")
[[ "$N_FOLDS" -gt 0 ]] && FOLD_SUFFIXES=($(for ((k = 0; k < N_FOLDS; k++)); do echo "_f${k}"; done))
for member in $MEMBERS; do
    for fold in "${FOLD_SUFFIXES[@]}"; do
        echo "== downloading predictions_${member}${SUFFIX}${KFOLD_TAG}${TAG}${fold}.npz =="
        "$COLAB" download "${SESSION_ARGS[@]}" "${REMOTE_OUT}/cross_subject/predictions_${member}${SUFFIX}${KFOLD_TAG}${TAG}${fold}.npz" "${REPO_ROOT}/output/${DATASET}/cross_subject/predictions_${member}${SUFFIX}${KFOLD_TAG}${TAG}${fold}.npz"
    done
done

if [[ "$WEIGHTS" -eq 1 && "$KFOLD" -eq 0 && "$ES_ALL" -eq 0 ]]; then
    for member in $MEMBERS; do
        [[ "$member" =~ ^(eegnet|patch_transformer|conformer)$ ]] || continue
        for fold in "${FOLD_SUFFIXES[@]}"; do
            echo "== downloading weights_${member}${SUFFIX}${TAG}${fold}.pt =="
            "$COLAB" download "${SESSION_ARGS[@]}" "${REMOTE_OUT}/cross_subject/weights_${member}${SUFFIX}${TAG}${fold}.pt" "${REPO_ROOT}/output/${DATASET}/cross_subject/weights_${member}${SUFFIX}${TAG}${fold}.pt"
        done
    done
fi

if [[ "$WEIGHTS" -eq 1 && "$KFOLD" -gt 0 ]]; then
    for member in $MEMBERS; do
        for fold in "${FOLD_SUFFIXES[@]}"; do
            echo "== downloading weights_${member}${SUFFIX}${KFOLD_TAG}${TAG}${fold}.pt =="
            "$COLAB" download "${SESSION_ARGS[@]}" "${REMOTE_OUT}/cross_subject/weights_${member}${SUFFIX}${KFOLD_TAG}${TAG}${fold}.pt" "${REPO_ROOT}/output/${DATASET}/cross_subject/weights_${member}${SUFFIX}${KFOLD_TAG}${TAG}${fold}.pt"
        done
    done
fi

"$COLAB" stop "${SESSION_ARGS[@]}"
echo "done"
