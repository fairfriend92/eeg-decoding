"""Scores a softmax-average ensemble against its best single member.

Reads the per-trial probabilities written by dump_predictions.py and
reports per-subject and pooled accuracy for each model and for the
probability average, the error overlap between the models, and the paired
per-subject difference of the ensemble against the best single model.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse

import numpy as np

from config import OUTPUT_DIR


def per_subject_accuracy(subject, y, probs):
    """Accuracy of argmax predictions, per subject.

    Parameters
    ----------
    subject, y : ndarray, shape (n_trials,)
        Subject id and label of each trial.
    probs : ndarray, shape (n_trials, n_classes)
        Class probabilities.

    Returns
    -------
    subjects : ndarray
        Sorted unique subject ids.
    acc : ndarray
        Accuracy per subject, in the order of `subjects`.
    """
    subjects = np.unique(subject)
    correct  = probs.argmax(axis=1) == y
    return subjects, np.array([correct[subject == s].mean() for s in subjects])


def paired_difference(a, b):
    """Mean of a - b across subjects and its standard error (ddof=1).

    Parameters
    ----------
    a, b : ndarray
        Per-subject scores in the same subject order.

    Returns
    -------
    float, float
        Mean difference and its standard error.
    """
    d = a - b
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def analyze(dataset, model_names):
    """Reports single-model, ensemble, and overlap statistics.

    Parameters
    ----------
    dataset : str
        Dataset whose output directory holds the predictions_{model}.npz
        files.
    model_names : list of str
        Models to average, at least two.

    Returns
    -------
    dict
        Scores keyed by name: per-subject accuracy arrays for each model
        and for "ensemble", plus "overlap" with pooled error statistics.
    """
    dumps = {m: np.load(OUTPUT_DIR / dataset / f"predictions_{m}.npz") for m in model_names}
    first = dumps[model_names[0]]
    for m in model_names[1:]:
        if not (np.array_equal(first["subject"], dumps[m]["subject"]) and np.array_equal(first["y"], dumps[m]["y"])):
            raise ValueError(f"Predictions for {m} are not aligned with {model_names[0]}.")
    subject, y = first["subject"], first["y"]

    probs = {m: dumps[m]["probs"] for m in model_names}
    probs["ensemble"] = sum(probs[m] for m in model_names) / len(model_names)

    accs = {}
    for name, p in probs.items():
        subjects, accs[name] = per_subject_accuracy(subject, y, p)
        pooled = (p.argmax(axis=1) == y).mean()
        print(f"{name:10s} per-subject mean acc = {accs[name].mean():.3f} +/- {accs[name].std(ddof=1):.3f}, pooled acc = {pooled:.3f}")

    best = max(model_names, key=lambda m: accs[m].mean())
    diff, se = paired_difference(accs["ensemble"], accs[best])
    wins = int((accs["ensemble"] > accs[best]).sum())
    ties = int((accs["ensemble"] == accs[best]).sum())
    print(f"best single: {best}")
    print(f"ensemble - {best}: {diff:+.4f} +/- {se:.4f} (SE, n={len(subjects)}), ensemble better on {wins}, tied on {ties}")

    wrong = {m: probs[m].argmax(axis=1) != y for m in model_names}
    both  = np.logical_and.reduce(list(wrong.values()))
    anyw  = np.logical_or.reduce(list(wrong.values()))
    print(f"errors: both wrong {both.mean():.3f}, at least one wrong {anyw.mean():.3f}, "
          f"overlap (both / at least one) {both.sum() / anyw.sum():.3f}")
    if len(model_names) == 2:
        a, b = (wrong[m] for m in model_names)
        print(f"error correlation (phi) = {np.corrcoef(a, b)[0, 1]:.3f}")

    accs["overlap"] = {"both_wrong": both.mean(), "any_wrong": anyw.mean()}
    return accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--dataset", default="dreyer2023")
    args = parser.parse_args()

    analyze(args.dataset, args.models)
