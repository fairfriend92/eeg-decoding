#!/usr/bin/env bash
# Uploads src/ to a running Colab CLI session, preserving the repo's relative
# layout so config.py's REPO_ROOT-based paths resolve correctly on the VM.
# With --data, also uploads the local data/ cache to the Drive data directory
# the pipeline scripts read from.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_colab_common.sh"

SESSION=""
SYNC_DATA=0

usage() {
    echo "Usage: $0 [-s SESSION] [--data]" >&2
    echo "  -s, --session NAME   Target session (omit if only one is active)" >&2
    echo "  --data               Also upload data/ to the Drive data dir (744M+, slow, one-time)" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session) SESSION="$2"; shift 2 ;;
        --data) SYNC_DATA=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

SESSION_ARGS=()
[[ -n "$SESSION" ]] && SESSION_ARGS=(-s "$SESSION")

echo "== syncing src/ =="
upload_tree "${REPO_ROOT}/src" "src"

if [[ "$SYNC_DATA" -eq 1 ]]; then
    echo "== syncing data/ to ${DRIVE_DATA_DIR} (this can take a while) =="
    mount_drive
    upload_tree "${REPO_ROOT}/data" "data" "$DRIVE_DATA_DIR"
else
    echo "skipping data/ (pass --data to upload it)"
fi

echo "done"
