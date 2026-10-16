"""Compares cross-subject dumps scored under different test-time alignments.

Reads output/{dataset}/cross_subject/predictions_{member}{suffix}.npz for each
suffix and reports balanced accuracy pooled, averaged over streams (one per
participant and session), and by position of the window in its stream. With
--n-folds it reads the dumps of every test fold instead (suffix + _f{k}) and
pools their rows, so each participant is scored once, and also reports the
mean and spread of the per-fold stream means.
"""

import argparse

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from config import OUTPUT_DIR

WINDOW_BIN = 10    # windows per bin of the by-position table


def stream_positions(subject, session):
    """Position of every row within its stream, rows of a stream being in recording order.

    Parameters
    ----------
    subject, session : ndarray, shape (n_rows,)
        Participant and session label of each row.

    Returns
    -------
    positions : ndarray of int, shape (n_rows,)
        0 for the first window of a stream, 1 for the next, and so on.
    streams : ndarray of int, shape (n_rows,)
        Index of the (subject, session) stream each row belongs to.
    """
    keys = np.char.add(subject.astype(str), "/" + session.astype(str))
    _, streams = np.unique(keys, return_inverse=True)
    positions = np.zeros(len(keys), dtype=int)
    for stream in np.unique(streams):
        idx = np.where(streams == stream)[0]
        positions[idx] = np.arange(len(idx))
    return positions, streams


def load_dump(path_stem, n_folds):
    """Loads one dump, or the concatenation of every fold's dump.

    Parameters
    ----------
    path_stem : Path
        Dump path without the .npz extension and without any fold suffix.
    n_folds : int
        Number of folds. 0 loads the single dump at path_stem.

    Returns
    -------
    dump : dict
        Arrays "y", "probs", "subject" and "session", folds concatenated in order.
    fold_of_row : ndarray of int
        Fold index of every row, all 0 for a single dump.
    """
    paths = [path_stem.parent / f"{path_stem.name}_f{k}.npz" for k in range(n_folds)] if n_folds else [path_stem.parent / f"{path_stem.name}.npz"]
    dumps = [np.load(path) for path in paths]
    dump = {key: np.concatenate([d[key] for d in dumps]) for key in ("y", "probs", "subject", "session")}
    return dump, np.concatenate([np.full(len(d["y"]), k) for k, d in enumerate(dumps)])


def summarize(dump, fold_of_row):
    """Balanced accuracies of one dump.

    Parameters
    ----------
    dump : mapping
        A cross-subject dump with "y", "probs", "subject" and "session".
    fold_of_row : ndarray of int
        Fold index of every row (see load_dump).

    Returns
    -------
    dict
        "pooled" over all rows, "stream_mean" averaged over streams, and
        "fold_means", the stream mean within each fold, and
        "by_position", a list of (first position, balanced accuracy) per bin of
        WINDOW_BIN positions pooled over streams.
    """
    y, pred = dump["y"], dump["probs"].argmax(axis=1)
    positions, streams = stream_positions(dump["subject"], dump["session"])
    by_position = []
    for start in range(0, positions.max() + 1, WINDOW_BIN):
        mask = (positions >= start) & (positions < start + WINDOW_BIN)
        by_position.append((start, balanced_accuracy_score(y[mask], pred[mask])))
    fold_means = [
        float(np.mean([balanced_accuracy_score(y[streams == s], pred[streams == s]) for s in np.unique(streams[fold_of_row == f])]))
        for f in np.unique(fold_of_row)
    ]
    return {
        "fold_means": fold_means,
        "pooled": balanced_accuracy_score(y, pred),
        "stream_mean": float(np.mean([balanced_accuracy_score(y[streams == s], pred[streams == s]) for s in np.unique(streams)])),
        "by_position": by_position,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="proteus")
    parser.add_argument("--member", default="eegnet")
    parser.add_argument("--n-folds", type=int, default=0, help="Read the _f{k} dumps of this many folds.")
    parser.add_argument("--suffixes", nargs="+", default=["", "_ea", "_ea_testcausal", "_ca"], help="Dump file-name suffixes to compare.")
    args = parser.parse_args()

    results = {}
    for suffix in args.suffixes:
        results[suffix] = summarize(*load_dump(OUTPUT_DIR / args.dataset / "cross_subject" / f"predictions_{args.member}{suffix}", args.n_folds))
        folds = results[suffix]["fold_means"]
        print(f"{suffix or 'none':>16}  pooled {results[suffix]['pooled']:.3f}  stream mean {results[suffix]['stream_mean']:.3f}"
              f"  per-fold {np.mean(folds):.3f} +- {np.std(folds):.3f} (n={len(folds)})")
    print("\nbalanced accuracy by window position in its stream")
    n_bins = max(len(r["by_position"]) for r in results.values())
    for b in range(n_bins):
        cells = [f"{r['by_position'][b][1]:.3f}" if b < len(r["by_position"]) else "   - " for r in results.values()]
        print(f"{b * WINDOW_BIN:>4}-{(b + 1) * WINDOW_BIN - 1:<4}  " + "  ".join(cells))
