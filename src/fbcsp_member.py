"""Filter-bank CSP plus logistic regression member for ensembling.

Per subject, splits the epoched trials into equal-width sub-bands spanning
the pipeline's own passband, fits CSP in each band on the training split,
and classifies the concatenated log-variance features with an L2 logistic
regression. Writes test softmax probabilities in the same format as
dump_predictions.py, to output/{dataset}/predictions_fbcsp.npz. Uses the
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
from data_loader import get_subject_data
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
    probs : ndarray, shape (n_test_trials, n_classes)
        Class probabilities.
    y_test : ndarray, shape (n_test_trials,)
        Integer test labels.
    """
    X_train, y_train, X_test, y_test = get_subject_data(subject_id, cfg.data)
    F_train, F_test = fbcsp_features(X_train, X_test, y_train, cfg)

    # Features are log-variances on very different scales across bands.
    mean, std = F_train.mean(axis=0), F_train.std(axis=0) + 1e-8
    clf = LogisticRegression(max_iter=1000).fit((F_train - mean) / std, y_train)
    return clf.predict_proba((F_test - mean) / std), y_test


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
        Arrays "subject", "y" and "probs", one row per test trial.
    """
    ids, labels, probs = [], [], []
    for subject_id in subjects or cfg.data.subject_ids:
        p, y = fbcsp_subject(subject_id, cfg)
        print(f"[fbcsp] subject {subject_id}: acc={(p.argmax(axis=1) == y).mean():.3f}", flush=True)
        ids.append(np.full(len(y), subject_id))
        labels.append(y)
        probs.append(p)

    dump = {"subject": np.concatenate(ids), "y": np.concatenate(labels), "probs": np.concatenate(probs)}
    if subjects is None:
        np.savez(output_dir_for(cfg) / "predictions_fbcsp.npz", **dump)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subset to score without writing files.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_fbcsp(resolve_config(args), subjects=args.subjects)
