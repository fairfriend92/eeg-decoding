"""Riemannian tangent-space plus logistic regression member for ensembling.

Per subject, fits OAS covariance matrices, a tangent-space projection and an
L2 logistic regression on the training split, then writes test softmax
probabilities in the same format as dump_predictions.py, to
output/{dataset}/predictions_riemann.npz. Uses the same split and
normalization as the neural models, so rows align with their dumps.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import get_subject_data
from train import output_dir_for


def riemann_subject(subject_id, cfg: Config):
    """Fits and scores the tangent-space classifier for one subject.

    Parameters
    ----------
    subject_id : int
        Subject identifier, valid for cfg.data.dataset.
    cfg : Config
        Full project configuration.

    Returns
    -------
    probs : ndarray, shape (n_test_trials, n_classes)
        Class probabilities.
    y_test : ndarray, shape (n_test_trials,)
        Integer test labels.
    """
    X_train, y_train, X_test, y_test = get_subject_data(subject_id, cfg.data)
    clf = make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann"), LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test), y_test


def dump_riemann(cfg: Config, subjects=None):
    """Dumps tangent-space test probabilities for every subject.

    Parameters
    ----------
    cfg : Config
        Full project configuration.
    subjects : list of int, optional
        Subset of subject ids; defaults to cfg.data.subject_ids. A subset
        is not written to disk.

    Returns
    -------
    dict
        Arrays "subject", "y" and "probs", one row per test trial.
    """
    ids, labels, probs = [], [], []
    for subject_id in subjects or cfg.data.subject_ids:
        p, y = riemann_subject(subject_id, cfg)
        print(f"[riemann] subject {subject_id}: acc={(p.argmax(axis=1) == y).mean():.3f}", flush=True)
        ids.append(np.full(len(y), subject_id))
        labels.append(y)
        probs.append(p)

    dump = {"subject": np.concatenate(ids), "y": np.concatenate(labels), "probs": np.concatenate(probs)}
    if subjects is None:
        np.savez(output_dir_for(cfg) / "predictions_riemann.npz", **dump)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subset to score without writing files.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_riemann(resolve_config(args), subjects=args.subjects)
