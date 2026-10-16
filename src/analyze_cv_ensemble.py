"""Scores ensembles of cross-validated cross-subject members.

Reads the per-fold dumps output/{dataset}/cross_subject/predictions_{member}{suffix}_f{k}.npz
of several members (all written with the same suffix) and reports balanced
accuracy, pooled over every test participant and as the mean of the per-fold
scores, for each member, each uniform-average subset, the subset selected on
each fold's validation participants, and the per-participant oracle subset.
The oracle picks the best subset for each test participant from the test labels,
so it bounds what any per-participant selection can reach and is not a result.
"""

import argparse

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from config import OUTPUT_DIR
from ensemble_harness import all_subsets, subset_score


def load_folds(dataset, members, suffix, n_folds):
    """Loads every member's dump for every fold and checks the rows line up.

    Parameters
    ----------
    dataset : str
        Dataset whose output directory holds the dumps.
    members : list of str
        Member names, in the order of the returned probabilities.
    suffix : str
        Dump file-name suffix shared by all members.
    n_folds : int
        Number of test folds.

    Returns
    -------
    list of dict
        One per fold, with the shared arrays "subject", "y", "val_y" and
        "probs" and "val_probs", each of shape (n_members, n_rows, n_classes).

    Raises
    ------
    ValueError
        If a member's rows differ from the first member's in a fold.
    """
    folds = []
    for k in range(n_folds):
        dumps = [np.load(OUTPUT_DIR / dataset / "cross_subject" / f"predictions_{m}{suffix}_f{k}.npz") for m in members]
        for m, dump in zip(members[1:], dumps[1:]):
            for key in ("subject", "session", "y", "val_subject", "val_y"):
                if not np.array_equal(dumps[0][key], dump[key]):
                    raise ValueError(f"Fold {k}: column {key} of {m} is not aligned with {members[0]}.")
        folds.append({
            "subject": dumps[0]["subject"], "y": dumps[0]["y"], "val_y": dumps[0]["val_y"],
            "probs": np.stack([d["probs"] for d in dumps]), "val_probs": np.stack([d["val_probs"] for d in dumps]),
        })
    return folds


def _predict(probs, subset):
    return probs[list(subset)].mean(axis=0).argmax(axis=1)


def evaluate(folds, picker):
    """Pooled and per-fold balanced accuracy of a rule that picks predictions.

    Parameters
    ----------
    folds : list of dict
        As returned by load_folds.
    picker : callable
        picker(fold) returns the predicted class of every test row of the fold.

    Returns
    -------
    pooled : float
        Balanced accuracy over the rows of all folds.
    per_fold : list of float
        Balanced accuracy of each fold.
    """
    preds = [picker(f) for f in folds]
    y = np.concatenate([f["y"] for f in folds])
    pooled = balanced_accuracy_score(y, np.concatenate(preds))
    return pooled, [balanced_accuracy_score(f["y"], p) for f, p in zip(folds, preds)]


def oracle_pick(fold, subsets):
    """Predictions where each test participant uses the subset that scores best on it.

    Parameters
    ----------
    fold : dict
        One element of load_folds' result.
    subsets : list of tuple of int
        Candidate member subsets.

    Returns
    -------
    ndarray of int
        Predicted class of every test row of the fold.
    """
    pred = np.zeros(len(fold["y"]), dtype=int)
    for s in np.unique(fold["subject"]):
        rows = fold["subject"] == s
        scores = [balanced_accuracy_score(fold["y"][rows], _predict(fold["probs"], sub)[rows]) for sub in subsets]
        pred[rows] = _predict(fold["probs"], subsets[int(np.argmax(scores))])[rows]
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members", nargs="+")
    parser.add_argument("--dataset", default="proteus")
    parser.add_argument("--suffix", default="_ea_testcausal")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--criterion", default="nll", choices=["nll", "acc", "bal_acc"], help="Validation criterion for subset selection.")
    args = parser.parse_args()

    folds = load_folds(args.dataset, args.members, args.suffix, args.n_folds)
    subsets = all_subsets(len(args.members))

    def report(name, pooled, per_fold):
        print(f"{name:48s} pooled {pooled:.3f}  per fold {np.mean(per_fold):.3f} +- {np.std(per_fold):.3f}")

    for sub in subsets:
        report("+".join(args.members[i] for i in sub), *evaluate(folds, lambda f, sub=sub: _predict(f["probs"], sub)))

    def validation_pick(fold):
        best = max(subsets, key=lambda sub: subset_score(fold["val_probs"][list(sub)], fold["val_y"], args.criterion))
        return _predict(fold["probs"], best)

    print()
    report(f"selected on validation ({args.criterion}), per fold", *evaluate(folds, validation_pick))
    report("oracle subset per participant (not a result)", *evaluate(folds, lambda f: oracle_pick(f, subsets)))
    report("oracle member per participant (not a result)", *evaluate(folds, lambda f: oracle_pick(f, [(i,) for i in range(len(args.members))])))
