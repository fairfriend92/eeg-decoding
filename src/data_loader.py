"""Loading and preprocessing for MOABB-backed EEG motor imagery datasets.

Supports multiple datasets (see datasets.py's registry) through a common
DataConfig. Which dataset, its class/channel/subject-list identity, and
its session-split strategy are all data-driven rather than hardcoded here.
Each dataset's split strategy is chosen to make the strongest available
train/test separation for that dataset's structure: BCI2a and Stieger2021
both use held-out sessions the model never sees during training (matching
published baselines and the "predict a later session" competition task
respectively); Dreyer2023 has only one session per subject, so it instead
holds out a trailing, per-class-stratified fraction of that session's
trials.
"""

import logging
import os
import re
import time
import warnings

import moabb
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Montage name 'standard_1005' is deprecated.*",
    category=FutureWarning,
)

# Forces the classic direct-download source rather than MOABB's newer NEMAR
# backend, which has an unresolved stall on the manifest-transfer step.
moabb.set_download_provider("upstream")

from moabb.paradigms import MotorImagery

from config import DATA_DIR, DataConfig
from datasets import MOABB_DATASET_CLASSES
import moabb_downloads  # noqa: F401 (applies the download configuration at import)

# Redirects MOABB's download location from its default (~/mne_data) into
# this project's data/ directory, matching the raw-input-only convention.
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MNE_DATA", str(DATA_DIR))


def load_subject(subject_id, cfg: DataConfig):
    """Loads and epochs all trials for a single subject.

    Parameters
    ----------
    subject_id : int
        Subject identifier, valid for cfg.dataset.
    cfg : DataConfig
        Dataset identity, filtering, and epoching parameters.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Epoched, bandpass-filtered EEG signal.
    y : ndarray, shape (n_trials,)
        Integer class labels.
    sessions : ndarray, shape (n_trials,)
        Session label per trial.
    """
    dataset = MOABB_DATASET_CLASSES[cfg.dataset]()
    paradigm = MotorImagery(
        n_classes=cfg.n_classes,
        fmin=cfg.l_freq,
        fmax=cfg.h_freq,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        resample=cfg.resample_freq,
    )
    # get_data() prints MOABB's own internal progress messages (dataset
    # download, event/channel selection); not something this project controls.
    X, y_labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    classes    = sorted(np.unique(y_labels))    # fixed label order across subjects
    label_map  = {label: idx for idx, label in enumerate(classes)}
    y          = np.array([label_map[label] for label in y_labels])
    sessions   = metadata["session"].to_numpy()

    return X, y, sessions


def _session_sort_key(label):
    """Sorts session labels chronologically where possible.

    Extracts the first integer found in the label (e.g. "session_2" -> 2);
    falls back to lexicographic order for labels with no digits, and
    breaks ties on the original string so the sort is always well-defined.
    """
    match = re.search(r"\d+", str(label))
    return (int(match.group()) if match else -1, str(label))


def _split_fixed_sessions(X, y, sessions, cfg: DataConfig):
    """Splits by two explicitly named sessions (e.g. BCI2a's "0train"/"1test")."""
    train_mask = sessions == cfg.train_session
    test_mask  = sessions == cfg.test_session

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def _split_by_session_index(X, y, sessions, cfg: DataConfig):
    """Trains on the earliest cfg.n_train_sessions sessions, tests on the rest.

    Sessions are sorted chronologically (see _session_sort_key) rather than
    relying on label equality, since session labels/counts vary per subject
    for datasets like Stieger2021.
    """
    unique_sessions = sorted(np.unique(sessions), key=_session_sort_key)

    if cfg.n_train_sessions is None or not (0 < cfg.n_train_sessions < len(unique_sessions)):
        raise ValueError(
            f"split_strategy='session_index' requires 0 < n_train_sessions < "
            f"n_available_sessions; got n_train_sessions={cfg.n_train_sessions}, "
            f"n_available_sessions={len(unique_sessions)} ({unique_sessions})"
        )

    train_sessions = set(unique_sessions[:cfg.n_train_sessions])
    test_sessions  = set(unique_sessions[cfg.n_train_sessions:])

    train_mask = np.isin(sessions, list(train_sessions))
    test_mask  = np.isin(sessions, list(test_sessions))

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def _split_trailing(X, y, fraction):
    """Holds out a trailing, per-class fraction of trials.

    Trailing rather than random, so later-recorded trials (more likely to
    reflect fatigue/adaptation than earlier ones) are what gets evaluated
    on, rather than an i.i.d. shuffle across the whole session. Stratified
    per class so the held-out set isn't dominated by whichever class
    happened to be recorded last.
    """
    train_idx, test_idx = [], []
    for label in np.unique(y):
        class_idx = np.where(y == label)[0]
        n_test    = round(len(class_idx) * fraction)
        test_idx.extend(class_idx[len(class_idx) - n_test:])
        train_idx.extend(class_idx[:len(class_idx) - n_test])

    train_idx = np.sort(train_idx)
    test_idx  = np.sort(test_idx)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _split_within_session_holdout(X, y, sessions, cfg: DataConfig):
    """Holds out a trailing, per-class fraction of a single session's trials.

    Used when a dataset has no second session to hold out (e.g.
    Dreyer2023).
    """
    return _split_trailing(X, y, cfg.holdout_fraction)


_SPLIT_STRATEGIES = {
    "fixed_sessions":        _split_fixed_sessions,
    "session_index":         _split_by_session_index,
    "within_session_holdout": _split_within_session_holdout,
}


def split_subject(X, y, sessions, cfg: DataConfig):
    """Splits a single subject's trials into train/test, per cfg.split_strategy.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Epoched EEG signal, as returned by `load_subject`.
    y : ndarray, shape (n_trials,)
        Integer class labels.
    sessions : ndarray, shape (n_trials,)
        Session label per trial.
    cfg : DataConfig
        Provides split_strategy and its strategy-specific fields.

    Returns
    -------
    X_train, y_train, X_test, y_test : ndarray
        Trials partitioned into train/test, with no overlap.
    """
    try:
        split_fn = _SPLIT_STRATEGIES[cfg.split_strategy]
    except KeyError:
        raise ValueError(
            f"Unknown split_strategy '{cfg.split_strategy}'. "
            f"Choices: {sorted(_SPLIT_STRATEGIES)}"
        ) from None

    return split_fn(X, y, sessions, cfg)


def split_validation(X, y, cfg: DataConfig):
    """Carves a trailing, per-class validation slice off a training split.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Training trials in recording order.
    y : ndarray, shape (n_trials,)
        Integer class labels.
    cfg : DataConfig
        Provides val_fraction.

    Returns
    -------
    X_fit, y_fit, X_val, y_val : ndarray
        Training trials split into a fitting part and a later validation
        part, with no overlap.
    """
    return _split_trailing(X, y, cfg.val_fraction)


def normalize_subject(X_train, X_test, eps=1e-8):
    """Z-scores each channel using statistics from the training trials only.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_channels, n_times)
        Training trials. Per-channel mean and std are computed from these
        only, so no test-set information leaks into the normalization.
    X_test : ndarray, shape (n_test, n_channels, n_times)
        Test trials, normalized with the training statistics.
    eps : float
        Added to the standard deviation to avoid division by zero for a
        flat channel.

    Returns
    -------
    X_train, X_test : ndarray
        Normalized trials, same shapes as the inputs.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2), keepdims=True)

    X_train = (X_train - mean) / (std + eps)
    X_test  = (X_test - mean) / (std + eps)
    return X_train, X_test


def crop_trials(X, y, crop_size, stride):
    """Slices each trial into overlapping fixed-length crops.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Trials to crop.
    y : ndarray, shape (n_trials,)
        Trial labels.
    crop_size : int
        Crop length, in samples.
    stride : int
        Step between consecutive crop start positions, in samples.

    Returns
    -------
    X_cropped : ndarray, shape (n_trials * n_crops, n_channels, crop_size)
        Cropped trials.
    y_cropped : ndarray, shape (n_trials * n_crops,)
        Label for each crop (same as its source trial's label).
    trial_index : ndarray, shape (n_trials * n_crops,)
        Index of the source trial for each crop, so crops from the same
        trial can be grouped back together (e.g. for prediction averaging).
    """
    n_times = X.shape[-1]
    starts  = list(range(0, n_times - crop_size + 1, stride))

    X_cropped, y_cropped, trial_index = [], [], []
    for trial_idx, (trial, label) in enumerate(zip(X, y)):
        for start in starts:
            X_cropped.append(trial[:, start:start + crop_size])
            y_cropped.append(label)
            trial_index.append(trial_idx)

    return np.stack(X_cropped), np.array(y_cropped), np.array(trial_index)


def get_subject_data(subject_id, cfg: DataConfig):
    """Loads, splits, and normalizes a single subject's data in one call.

    Parameters
    ----------
    subject_id : int
        Subject identifier (1-9).
    cfg : DataConfig
        Filtering, epoching, and split parameters.

    Returns
    -------
    X_train, y_train, X_test, y_test : ndarray
        Session-split, per-channel-normalized trials for the subject.
    """
    X, y, sessions = load_subject(subject_id, cfg)
    X_train, y_train, X_test, y_test = split_subject(X, y, sessions, cfg)
    X_train, X_test = normalize_subject(X_train, X_test)
    return X_train, y_train, X_test, y_test


def get_subject_validation_data(subject_id, cfg: DataConfig):
    """Loads a subject's data with a validation slice carved from training.

    Parameters
    ----------
    subject_id : int
        Subject identifier, valid for cfg.dataset.
    cfg : DataConfig
        Filtering, epoching, and split parameters.

    Returns
    -------
    X_fit, y_fit, X_val, y_val : ndarray
        The training split minus its trailing validation slice, normalized
        with statistics of the fitting part only. The test split is never
        returned or used.
    """
    X, y, sessions = load_subject(subject_id, cfg)
    X_train, y_train, _, _ = split_subject(X, y, sessions, cfg)
    X_fit, y_fit, X_val, y_val = split_validation(X_train, y_train, cfg)
    X_fit, X_val = normalize_subject(X_fit, X_val)
    return X_fit, y_fit, X_val, y_val
