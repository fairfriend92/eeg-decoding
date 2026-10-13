"""Riemannian tangent-space plus logistic regression member for ensembling.

Per subject, fits OAS covariance matrices, a tangent-space projection and an
L2 logistic regression on the training split, then writes test softmax
probabilities in the same format as dump_predictions.py, to
output/{dataset}/predictions_riemann.npz. Validation probabilities come from
a second fit on the training split minus its trailing validation slice. Uses the same split and
normalization as the neural models, so rows align with their dumps.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_covariance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import alignment_subset, get_subject_data, get_subject_validation_data
from train import output_dir_for


def _recenter(covs, subjects, n_trials):
    """Moves each participant's covariance matrices to be centered at the identity."""
    out = np.empty_like(covs)
    for sid in np.unique(subjects):
        idx = np.flatnonzero(subjects == sid)
        center = mean_covariance(covs[idx[alignment_subset(len(idx), n_trials)]], metric="riemann")
        whitener = invsqrtm(center)
        out[idx] = whitener @ covs[idx] @ whitener.T
    return out


def fit_predict_riemann(X_train, y_train, X_eval, subj_train=None, subj_eval=None, recenter_n_trials=0):
    """Fits the tangent-space classifier and predicts class probabilities.

    Parameters
    ----------
    X_train, X_eval : ndarray, shape (n_trials, n_channels, n_times)
        Fitting and evaluation trials.
    y_train : ndarray, shape (n_trials,)
        Fitting labels.
    subj_train, subj_eval : ndarray, shape (n_trials,), optional
        Participant id of every trial. When both are given, each
        participant's covariance matrices are recentered at that
        participant's own Riemannian mean before the tangent-space
        projection, using no labels.
    recenter_n_trials : int
        Trials per participant the Riemannian mean is estimated from. 0
        uses every trial.

    Returns
    -------
    ndarray, shape (len(X_eval), n_classes)
        Class probabilities.
    """
    cov = Covariances(estimator="oas")
    covs_train, covs_eval = cov.transform(X_train), cov.transform(X_eval)
    if subj_train is not None and subj_eval is not None:
        covs_train = _recenter(covs_train, subj_train, recenter_n_trials)
        covs_eval = _recenter(covs_eval, subj_eval, recenter_n_trials)
    clf = make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(max_iter=1000))
    clf.fit(covs_train, y_train)
    return clf.predict_proba(covs_eval)


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
    val_probs : ndarray, shape (n_val_trials, n_classes)
        Class probabilities on the validation slice, from a fit that never
        sees it.
    y_val : ndarray, shape (n_val_trials,)
        Integer validation labels.
    probs : ndarray, shape (n_test_trials, n_classes)
        Class probabilities on the test split, from a fit on the full
        training split.
    y_test : ndarray, shape (n_test_trials,)
        Integer test labels.
    """
    X_fit, y_fit, X_val, y_val = get_subject_validation_data(subject_id, cfg.data)
    X_train, y_train, X_test, y_test = get_subject_data(subject_id, cfg.data)
    return fit_predict_riemann(X_fit, y_fit, X_val), y_val, fit_predict_riemann(X_train, y_train, X_test), y_test


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
        Arrays "subject", "y" and "probs", one row per test trial, and
        "val_subject", "val_y" and "val_probs", one row per validation trial.
    """
    val, test = {"subject": [], "y": [], "probs": []}, {"subject": [], "y": [], "probs": []}
    for subject_id in subjects or cfg.data.subject_ids:
        pv, yv, p, y = riemann_subject(subject_id, cfg)
        print(f"[riemann] subject {subject_id}: acc={(p.argmax(axis=1) == y).mean():.3f}", flush=True)
        for part, labels, probs in ((val, yv, pv), (test, y, p)):
            part["subject"].append(np.full(len(labels), subject_id))
            part["y"].append(labels)
            part["probs"].append(probs)

    dump = {k: np.concatenate(v) for k, v in test.items()}
    dump.update({f"val_{k}": np.concatenate(v) for k, v in val.items()})
    if subjects is None:
        np.savez(output_dir_for(cfg) / "predictions_riemann.npz", **dump)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subset to score without writing files.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_riemann(resolve_config(args), subjects=args.subjects)
