"""Selection and weighting harness over per-trial member probability dumps.

Reads the predictions_{member}.npz files written by dump_predictions.py,
riemann_member.py and fbcsp_member.py. Each member subset is a uniform
probability average. Subsets are chosen on the validation rows only and
scored on the test rows, so selection and scoring data are disjoint.

A group is whatever the "subject" column identifies: one subject for
per-subject datasets, one participant for cross-subject splits. Schemes that
pick a subset per group need validation rows for that group. Schemes that
pick one global subset work when test groups have no validation rows of their
own, as in a cross-subject split.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
from itertools import combinations

import numpy as np

from analyze_ensemble import paired_difference
from config import OUTPUT_DIR


def load_member_dumps(dataset, member_names):
    """Loads and alignment-checks the dumps of several members.

    Parameters
    ----------
    dataset : str
        Dataset whose output directory holds the predictions_{member}.npz
        files.
    member_names : list of str
        Members to load.

    Returns
    -------
    dict
        Arrays "val_subject", "val_y", "subject" and "y" shared by all
        members, plus "val_probs" and "probs", each an ndarray of shape
        (n_members, n_rows, n_classes) in the order of `member_names`.

    Raises
    ------
    ValueError
        If any member's rows are not aligned with the first member's.
    """
    dumps = [np.load(OUTPUT_DIR / dataset / f"predictions_{m}.npz") for m in member_names]
    shared = {k: dumps[0][k] for k in ("val_subject", "val_y", "subject", "y")}
    for m, dump in zip(member_names[1:], dumps[1:]):
        for k, v in shared.items():
            if not np.array_equal(v, dump[k]):
                raise ValueError(f"Column {k} of {m} is not aligned with {member_names[0]}.")
    shared["val_probs"] = np.stack([d["val_probs"] for d in dumps])
    shared["probs"] = np.stack([d["probs"] for d in dumps])
    return shared


def _balanced_accuracy(y, pred, n_classes):
    recalls = [(pred[y == c] == c).mean() for c in range(n_classes) if (y == c).any()]
    return float(np.mean(recalls))


def subset_score(probs, y, criterion):
    """Scores the uniform average of the given member probabilities.

    Parameters
    ----------
    probs : ndarray, shape (n_members, n_rows, n_classes)
        Member probabilities of the subset to average.
    y : ndarray, shape (n_rows,)
        Integer labels.
    criterion : str
        "nll" (negative mean log-likelihood, sign-flipped), "acc" or
        "bal_acc". Higher is always better.

    Returns
    -------
    float
    """
    avg = probs.mean(axis=0)
    if criterion == "nll":
        return float(np.log(avg[np.arange(len(y)), y] + 1e-12).mean())
    pred = avg.argmax(axis=1)
    if criterion == "acc":
        return float((pred == y).mean())
    if criterion == "bal_acc":
        return _balanced_accuracy(y, pred, avg.shape[1])
    raise ValueError(f"Unknown criterion '{criterion}'. Choices: nll, acc, bal_acc")


def all_subsets(n_members):
    """Every non-empty subset of member indices, as tuples.

    Parameters
    ----------
    n_members : int
        Number of members.

    Returns
    -------
    list of tuple of int
    """
    return [c for size in range(1, n_members + 1) for c in combinations(range(n_members), size)]


def validation_scores(data, criterion):
    """Scores every subset on each validation group.

    Parameters
    ----------
    data : dict
        As returned by load_member_dumps.
    criterion : str
        See subset_score.

    Returns
    -------
    groups : ndarray
        Sorted unique validation group ids.
    subsets : list of tuple of int
        Candidate subsets.
    scores : ndarray, shape (n_groups, n_subsets)
        Validation score of each subset in each group.
    counts : ndarray, shape (n_groups,)
        Validation rows per group.
    """
    groups = np.unique(data["val_subject"])
    subsets = all_subsets(len(data["val_probs"]))
    scores = np.zeros((len(groups), len(subsets)))
    counts = np.zeros(len(groups))
    for g, group in enumerate(groups):
        mask = data["val_subject"] == group
        counts[g] = mask.sum()
        for s, subset in enumerate(subsets):
            scores[g, s] = subset_score(data["val_probs"][list(subset)][:, mask], data["val_y"][mask], criterion)
    return groups, subsets, scores, counts


def select_subsets(scores, counts, scheme, shrink_k):
    """Chooses one subset per validation group under a selection scheme.

    Parameters
    ----------
    scores : ndarray, shape (n_groups, n_subsets)
        Validation scores, from validation_scores.
    counts : ndarray, shape (n_groups,)
        Validation rows per group.
    scheme : str
        "global" picks the subset with the best mean score across groups
        for every group. "per_group" picks each group's own best subset.
        "shrunk" picks each group's best subset of a pseudo-count blend of
        its own score and the global mean score.
    shrink_k : float
        Pseudo-count of global evidence in the "shrunk" blend. A group with
        n validation rows weighs its own score n / (n + shrink_k).

    Returns
    -------
    ndarray, shape (n_groups,)
        Index into the subset list chosen for each group.
    """
    global_scores = scores.mean(axis=0)
    if scheme == "global":
        return np.full(len(scores), global_scores.argmax())
    if scheme == "per_group":
        return scores.argmax(axis=1)
    if scheme == "shrunk":
        w = (counts / (counts + shrink_k))[:, None]
        return (w * scores + (1 - w) * global_scores).argmax(axis=1)
    raise ValueError(f"Unknown scheme '{scheme}'. Choices: global, per_group, shrunk")


def test_group_scores(data, chosen, metric):
    """Per-group test score of per-group chosen subsets.

    Parameters
    ----------
    data : dict
        As returned by load_member_dumps.
    chosen : dict
        Group id mapped to a tuple of member indices to average.
    metric : str
        "acc" or "bal_acc".

    Returns
    -------
    groups : ndarray
        Sorted unique test group ids.
    scores : ndarray
        Test score per group.
    """
    groups = np.unique(data["subject"])
    scores = []
    for group in groups:
        mask = data["subject"] == group
        probs = data["probs"][list(chosen[group])][:, mask]
        scores.append(subset_score(probs, data["y"][mask], metric))
    return groups, np.array(scores)


def run_harness(data, member_names, criterion="nll", metric="acc", shrink_k=50.0, references=()):
    """Scores every selection scheme and fixed baseline on the test rows.

    Parameters
    ----------
    data : dict
        As returned by load_member_dumps.
    member_names : list of str
        Member names, in the order of the data's first axis.
    criterion : str
        Validation criterion for selection, see subset_score.
    metric : str
        Test metric, "acc" or "bal_acc", averaged over test groups.
    shrink_k : float
        See select_subsets.
    references : iterable of tuple of str
        Extra fixed subsets to report as baselines, such as a pair found
        post hoc.

    Returns
    -------
    dict
        Name mapped to per-group test scores, in sorted test group order.
    """
    val_groups, subsets, scores, counts = validation_scores(data, criterion)
    test_groups = np.unique(data["subject"])
    n = len(member_names)
    index = {s: i for i, s in enumerate(subsets)}

    fixed = {"uniform_all": tuple(range(n))}
    best_single = max(range(n), key=lambda i: scores[:, index[(i,)]].mean())
    fixed[f"best_single_val({member_names[best_single]})"] = (best_single,)
    for ref in references:
        fixed[f"reference({'+'.join(ref)})"] = tuple(sorted(member_names.index(m) for m in ref))

    results, chosen_report = {}, {}
    for name, subset in fixed.items():
        results[name] = test_group_scores(data, {g: subset for g in test_groups}, metric)[1]
    for scheme in ("global", "per_group", "shrunk"):
        picks = select_subsets(scores, counts, scheme, shrink_k)
        by_group = {g: subsets[p] for g, p in zip(val_groups, picks)}
        if scheme == "global":
            by_group = {g: subsets[picks[0]] for g in test_groups}
        elif not set(test_groups) <= set(val_groups):
            continue
        results[f"select_{scheme}"] = test_group_scores(data, by_group, metric)[1]
        chosen_report[scheme] = by_group

    for name, s in results.items():
        print(f"{name:42s} mean {metric} = {s.mean():.4f} +/- {s.std(ddof=1):.4f} (n={len(s)})")
    chosen_global = next(iter(chosen_report["global"].values()))
    print(f"global subset chosen on validation ({criterion}): {'+'.join(member_names[i] for i in chosen_global)}")

    baselines = [k for k in results if not k.startswith("select_")]
    for scheme_name in (k for k in results if k.startswith("select_")):
        for base in baselines:
            diff, se = paired_difference(results[scheme_name], results[base])
            wins = int((results[scheme_name] > results[base]).sum())
            ties = int((results[scheme_name] == results[base]).sum())
            print(f"{scheme_name} - {base}: {diff:+.4f} +/- {se:.4f} (SE), better on {wins}, tied on {ties}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members", nargs="+")
    parser.add_argument("--dataset", default="dreyer2023")
    parser.add_argument("--criterion", default="nll", choices=["nll", "acc", "bal_acc"])
    parser.add_argument("--metric", default="acc", choices=["acc", "bal_acc"])
    parser.add_argument("--shrink-k", type=float, default=50.0)
    parser.add_argument("--reference", action="append", default=[], help="Fixed subset to report, members joined by '+'.")
    args = parser.parse_args()

    dump = load_member_dumps(args.dataset, args.members)
    run_harness(
        dump, args.members, criterion=args.criterion, metric=args.metric,
        shrink_k=args.shrink_k, references=[r.split("+") for r in args.reference],
    )
