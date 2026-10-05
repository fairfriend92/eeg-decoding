#!/usr/bin/env bash
# Shared helpers for the colab_*.sh scripts. Sourced, not run directly.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB="${REPO_ROOT}/.venv/bin/colab"
REMOTE_ROOT="/content/eeg-decoding"
DRIVE_ROOT="/content/drive/MyDrive/progetti/eeg-decoding"
DRIVE_DATA_DIR="${DRIVE_ROOT}/data"

# Ensures a Colab session exists, creating one (with --gpu, if SESSION and
# GPU are set in the caller's scope) when it does not. Sets SESSION_ARGS.
#
# `colab status -s NAME` exits 0 even when NAME doesn't exist (it just
# prints "not found"), so existence is checked against `colab sessions`'
# "[name] ..." listing instead.
ensure_session() {
    SESSION_ARGS=()
    [[ -n "$SESSION" ]] && SESSION_ARGS=(-s "$SESSION")

    local exists=0
    if [[ -n "$SESSION" ]]; then
        if "$COLAB" sessions 2>/dev/null | grep -qF "[${SESSION}]"; then
            exists=1
        fi
    else
        if "$COLAB" sessions 2>/dev/null | grep -q '^\['; then
            exists=1
        fi
    fi

    if [[ "$exists" -eq 0 ]]; then
        echo "== creating session${SESSION:+ '${SESSION}'}${GPU:+ (--gpu $GPU)} =="
        local new_args=("${SESSION_ARGS[@]}")
        [[ -n "$GPU" ]] && new_args+=(--gpu "$GPU")
        "$COLAB" new "${new_args[@]}"
    fi
}

# Uploads a local directory tree to REMOTE_ROOT/<remote_subdir>, preserving
# structure and skipping __pycache__. Requires SESSION_ARGS to be set.
upload_tree() {
    local local_dir="$1" remote_subdir="$2"
    local remote_base="${REMOTE_ROOT}/${remote_subdir}"
    local file rel remote remote_dirs

    # The Jupyter contents API `colab upload` talks to returns a 500 if the
    # parent directory doesn't already exist, so create the whole remote
    # directory tree up front via a single exec call.
    remote_dirs="$(find "$local_dir" -type d -not -name '__pycache__' -printf '%P\n' | sed "s#^#${remote_base}/#;s#/\$##")"
    {
        echo "import os"
        echo "for d in [${remote_base@Q}] + '''${remote_dirs}'''.splitlines():"
        echo "    if d: os.makedirs(d, exist_ok=True)"
    } | "$COLAB" exec "${SESSION_ARGS[@]}" >/dev/null

    while IFS= read -r -d '' file; do
        rel="${file#"$local_dir"/}"
        remote="${remote_base}/${rel}"
        echo "upload: ${file} -> ${remote}"
        "$COLAB" upload "${SESSION_ARGS[@]}" "$file" "$remote"
    done < <(find "$local_dir" -type f -not -path '*/__pycache__/*' -print0)
}

# Downloads REMOTE_ROOT/<remote_subdir> back to REPO_ROOT/<remote_subdir>.
# Requires SESSION_ARGS to be set. Uses exec + os.walk rather than `colab
# ls`, which prints bare filenames (not full paths) and reports "not
# found" errors on stdout with exit 1. Both are awkward to parse reliably.
download_tree() {
    local remote_subdir="$1"
    local remote_base="${REMOTE_ROOT}/${remote_subdir}"
    local local_dir="${REPO_ROOT}/${remote_subdir}"
    mkdir -p "$local_dir"

    local remote_files
    remote_files="$(
        {
            echo "import os"
            echo "base = ${remote_base@Q}"
            echo "if os.path.isdir(base):"
            echo "    for root, dirs, files in os.walk(base):"
            echo "        for f in files:"
            echo "            print(os.path.join(root, f))"
        } | "$COLAB" exec "${SESSION_ARGS[@]}" | grep -F "$remote_base"
    )"

    local remote_file rel local_path
    while IFS= read -r remote_file; do
        [[ -z "$remote_file" ]] && continue
        rel="${remote_file#"${remote_base}"/}"
        local_path="${local_dir}/${rel}"
        mkdir -p "$(dirname "$local_path")"
        echo "download: ${remote_file} -> ${local_path}"
        "$COLAB" download "${SESSION_ARGS[@]}" "$remote_file" "$local_path"
    done <<< "$remote_files"
}

# Mounts Google Drive on the Colab VM via the CLI's own `drivemount`
# command, which handles the OAuth consent flow itself. Not
# google.colab.drive.mount() run through `colab exec`. That needs a live
# notebook UI for the consent prompt, which a headless `colab exec` call
# doesn't have. Requires SESSION_ARGS to be set.
mount_drive() {
    "$COLAB" drivemount "${SESSION_ARGS[@]}"
}
