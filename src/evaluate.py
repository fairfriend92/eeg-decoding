"""Aggregates per-subject results into summary statistics across subjects.

Must be run with this file's own directory (src/) as the script's
directory, e.g. `python evaluate.py` from within src/, or
`python src/evaluate.py` from the repository root — see train.py for why.
"""

import argparse
import json

import numpy as np

from config import OUTPUT_DIR


def load_results(model_name):
    """Loads per-subject results for a given model.

    Parameters
    ----------
    model_name : str
        Either "eegnet" or "conformer".

    Returns
    -------
    list of dict
        Per-subject result dictionaries, as written by train.run_all_subjects.
    """
    path = OUTPUT_DIR / f"{model_name}_results.json"
    with open(path) as f:
        return json.load(f)


def summarize(results):
    """Computes mean and std of test accuracy and kappa across subjects.

    Standard deviation uses ddof=1 (sample std), appropriate given the
    9 subjects are treated as a sample rather than the full population of
    interest.

    Parameters
    ----------
    results : list of dict
        Per-subject results, each containing "test_accuracy" and "test_kappa".

    Returns
    -------
    dict
        Mean and std for both metrics, plus subject count.
    """
    accuracies = np.array([r["test_accuracy"] for r in results])
    kappas     = np.array([r["test_kappa"] for r in results])

    return {
        "n_subjects":    len(results),
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std":  float(accuracies.std(ddof=1)),
        "kappa_mean":    float(kappas.mean()),
        "kappa_std":     float(kappas.std(ddof=1)),
    }


def compare_models(model_names):
    """Loads, summarizes, and reports results for one or more models.

    Parameters
    ----------
    model_names : list of str
        Models to summarize, e.g. ["eegnet", "conformer"].

    Returns
    -------
    dict
        Model name mapped to its summary dict.
    """
    summaries = {}
    for name in model_names:
        results = load_results(name)
        summary = summarize(results)
        summaries[name] = summary
        print(
            f"{name}: accuracy = {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f}, "
            f"kappa = {summary['kappa_mean']:.3f} +/- {summary['kappa_std']:.3f} "
            f"(n={summary['n_subjects']})"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=["eegnet", "conformer"])
    args = parser.parse_args()

    compare_models(args.models)
