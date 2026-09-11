"""Loading and preprocessing for BCI Competition IV, Dataset 2a, via MOABB.

Each subject's data is split using the dataset's official session labels
(session_T for training, session_E for testing) rather than a random split,
so results remain directly comparable to published baselines.
"""

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from config import DataConfig


def load_subject(subject_id, cfg: DataConfig):
    """Loads and epochs all trials for a single subject.

    Parameters
    ----------
    subject_id : int
        Subject identifier (1-9).
    cfg : DataConfig
        Filtering and epoching parameters.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Epoched, bandpass-filtered EEG signal.
    y : ndarray, shape (n_trials,)
        Integer class labels.
    sessions : ndarray, shape (n_trials,)
        Session label per trial ("session_T" or "session_E").
    """
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4,
        fmin=cfg.l_freq,
        fmax=cfg.h_freq,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        resample=cfg.resample_freq,
    )
    X, y_labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    classes    = sorted(np.unique(y_labels))    # fixed label order across subjects
    label_map  = {label: idx for idx, label in enumerate(classes)}
    y          = np.array([label_map[label] for label in y_labels])
    sessions   = metadata["session"].to_numpy()

    return X, y, sessions


def split_subject(X, y, sessions, cfg: DataConfig):
    """Splits a single subject's trials by official session.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Epoched EEG signal, as returned by `load_subject`.
    y : ndarray, shape (n_trials,)
        Integer class labels.
    sessions : ndarray, shape (n_trials,)
        Session label per trial.
    cfg : DataConfig
        Provides the train/test session labels.

    Returns
    -------
    X_train, y_train, X_test, y_test : ndarray
        Trials partitioned by session, with no overlap.
    """
    train_mask = sessions == cfg.train_session
    test_mask  = sessions == cfg.test_session

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    return X_train, y_train, X_test, y_test


def get_subject_data(subject_id, cfg: DataConfig):
    """Loads and splits a single subject's data in one call.

    Parameters
    ----------
    subject_id : int
        Subject identifier (1-9).
    cfg : DataConfig
        Filtering, epoching, and split parameters.

    Returns
    -------
    X_train, y_train, X_test, y_test : ndarray
        Session-split trials for the subject.
    """
    X, y, sessions = load_subject(subject_id, cfg)
    return split_subject(X, y, sessions, cfg)
