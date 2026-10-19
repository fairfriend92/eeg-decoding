"""Paired comparison of test-time adaptation variants against causal alignment alone.

Reads the dumps written by test_time_adapt.py for every test fold, scores
balanced accuracy per participant (sessions pooled), and reports each
variant's difference from the "none" variant as mean and standard error over
participants, with the number of participants improved. Also reports the
mean replay time per window.
"""

import argparse

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from config import OUTPUT_DIR

VARIANT_NAMES = ("none", "abn", "tent")


def participant_scores(dataset, member, suffix, variant, tag, n_folds):
    """Balanced accuracy of one variant for every test participant.

    Parameters
    ----------
    dataset, member, suffix, tag : str
        Locate output/{dataset}/cross_subject/predictions_{member}{suffix}_tta_{variant}{tag}_f{k}.npz.
    variant : str
        One of VARIANT_NAMES.
    n_folds : int
        Number of fold dumps to read.

    Returns
    -------
    scores : dict
        Participant id to balanced accuracy.
    seconds : float
        Mean replay seconds per window over the folds.
    """
    scores, seconds = {}, []
    for k in range(n_folds):
        d = np.load(OUTPUT_DIR / dataset / "cross_subject" / f"predictions_{member}{suffix}_tta_{variant}{tag}_f{k}.npz")
        for subject in np.unique(d["subject"]):
            rows = d["subject"] == subject
            scores[int(subject)] = balanced_accuracy_score(d["y"][rows], d["probs"][rows].argmax(axis=1))
        seconds.append(float(d["seconds_per_window"]))
    return scores, float(np.mean(seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="proteus")
    parser.add_argument("--member", default="patch_transformer")
    parser.add_argument("--suffix", default="_ea_testcausal")
    parser.add_argument("--tag", default="_cv8")
    parser.add_argument("--n-folds", type=int, default=8)
    args = parser.parse_args()

    results = {v: participant_scores(args.dataset, args.member, args.suffix, v, args.tag, args.n_folds) for v in VARIANT_NAMES}
    subjects = sorted(results["none"][0])
    base = np.array([results["none"][0][s] for s in subjects])
    print(f"participants: {subjects}")
    for variant in VARIANT_NAMES:
        scores = np.array([results[variant][0][s] for s in subjects])
        diff = scores - base
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        line = f"{variant:5s} mean {scores.mean():.3f} ± {scores.std(ddof=1):.3f}"
        if variant != "none":
            line += f", vs none {diff.mean():+.3f} ± {se:.3f} (SE), improved {(diff > 0).sum()} of {len(diff)}"
        print(f"{line}, {1000 * results[variant][1]:.1f} ms/window")
        print("      per participant: " + ", ".join(f"{s}:{x:.3f}" for s, x in zip(subjects, scores)))
