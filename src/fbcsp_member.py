"""Filter-bank CSP plus logistic regression member for ensembling.

Per subject, splits the epoched trials into equal-width sub-bands spanning
the pipeline's own passband, fits CSP in each band on the training split,
and classifies the concatenated log-variance features with an L2 logistic
regression. Writes test softmax probabilities in the same format as
dump_predictions.py, to output/{dataset}/predictions_fbcsp.npz. Validation
probabilities come from a second fit on the training split minus its
trailing validation slice. Uses the
same split and normalization as the neural models, so rows align with their
dumps.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse

import numpy as np
from mne.decoding import CSP
from mne.utils import use_log_level
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import get_subject_data, get_subject_validation_data
from train import output_dir_for

N_BANDS = 8
N_CSP_COMPONENTS = 4


def fbcsp_features(X_train, X_test, y_train, cfg: Config):
    """Fits per-band CSP on the training split and extracts features.

    Parameters
    ----------
    X_train, X_test : ndarray, shape (n_trials, n_channels, n_times)
        Epoched trials, already bandpassed to the pipeline passband.
    y_train : ndarray, shape (n_trials,)
        Training labels.
    cfg : Config
        Supplies the passband and sampling rate that define the bands.

    Returns
    -------
    F_train, F_test : ndarray, shape (n_trials, N_BANDS * N_CSP_COMPONENTS)
        Concatenated log-variance CSP features.
    """
    edges = np.linspace(cfg.data.l_freq, cfg.data.h_freq, N_BANDS + 1)
    F_train, F_test = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sos = butter(4, [lo, hi], btype="bandpass", fs=cfg.data.resample_freq, output="sos")
        Xb_train = sosfiltfilt(sos, X_train, axis=-1)
        Xb_test  = sosfiltfilt(sos, X_test, axis=-1)
        csp = CSP(n_components=N_CSP_COMPONENTS, log=True)
        with use_log_level("ERROR"):
            F_train.append(csp.fit_transform(Xb_train, y_train))
            F_test.append(csp.transform(Xb_test))
    return np.concatenate(F_train, axis=1), np.concatenate(F_test, axis=1)


def _fit_predict(X_train, y_train, X_eval, cfg: Config):
    F_train, F_eval = fbcsp_features(X_train, X_eval, y_train, cfg)

    # Features are log-variances on very different scales across bands.
    mean, std = F_train.mean(axis=0), F_train.std(axis=0) + 1e-8
    clf = LogisticRegression(max_iter=1000).fit((F_train - mean) / std, y_train)
    return clf.predict_proba((F_eval - mean) / std)


def fbcsp_subject(subject_id, cfg: Config):
    """Fits and scores the filter-bank CSP classifier for one subject.

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
    return _fit_predict(X_fit, y_fit, X_val, cfg), y_val, _fit_predict(X_train, y_train, X_test, cfg), y_test


def dump_fbcsp(cfg: Config, subjects=None):
    """Dumps filter-bank CSP test probabilities for every subject.

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
        pv, yv, p, y = fbcsp_subject(subject_id, cfg)
        print(f"[fbcsp] subject {subject_id}: acc={(p.argmax(axis=1) == y).mean():.3f}", flush=True)
        for part, labels, probs in ((val, yv, pv), (test, y, p)):
            part["subject"].append(np.full(len(labels), subject_id))
            part["y"].append(labels)
            part["probs"].append(probs)

    dump = {k: np.concatenate(v) for k, v in test.items()}
    dump.update({f"val_{k}": np.concatenate(v) for k, v in val.items()})
    if subjects is None:
        np.savez(output_dir_for(cfg) / "predictions_fbcsp.npz", **dump)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subset to score without writing files.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_fbcsp(resolve_config(args), subjects=args.subjects)
