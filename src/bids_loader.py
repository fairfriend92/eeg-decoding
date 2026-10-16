"""Loading for BIDS EEG datasets distributed through NEMAR rather than MOABB.

A dataset takes this path when its DataConfig sets nemar_id. The signal is
notch filtered, bandpass filtered and resampled as one continuous recording,
then cut into one window per cue, so the windows carry the same filtering a
streaming scorer applies to its recordings.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from mne_bids import BIDSPath, get_entities_from_fname, read_raw_bids

from config import DATA_DIR, DataConfig


def _number(label):
    """Integer value of the digits in a BIDS label (e.g. "P01" -> 1)."""
    return int(re.sub(r"\D", "", label))


def _subject_runs(subject_id, cfg: DataConfig):
    """BIDS entities of a participant's recordings, ordered by session then run number."""
    root = bids_root(cfg)
    entities = [get_entities_from_fname(path.name) for path in root.glob("sub-*/ses-*/eeg/*_eeg.edf")]
    mine = sorted(
        (e for e in entities if _number(e["subject"]) == subject_id),
        key=lambda e: (_number(e["session"]), int(e["run"])),
    )
    if not mine:
        raise ValueError(f"No recordings for subject {subject_id} in {root}. Found: {sorted({e['subject'] for e in entities})}")
    return mine


def _run_cues(root, e, cfg: DataConfig):
    """BIDSPath of one run and its cue rows from events.tsv (empty for a baseline)."""
    path = BIDSPath(
        root=root, subject=e["subject"], session=e["session"], task=e["task"], run=e["run"],
        datatype="eeg", suffix="eeg", extension=".edf",
    )
    events = pd.read_csv(path.copy().update(suffix="events", extension=".tsv").fpath, sep="\t")
    return path, events[events.trial_type.isin(cfg.cue_labels)]


def bids_session_labels(subject_id, cfg: DataConfig):
    """Session label of every window load_bids_subject returns, without reading any EEG.

    Parameters
    ----------
    subject_id : int
        Participant number.
    cfg : DataConfig
        Provides the BIDS location and cue labels.

    Returns
    -------
    ndarray, shape (n_trials,)
        Session label per window, in the order load_bids_subject returns them.
    """
    root = bids_root(cfg)
    labels = []
    for e in _subject_runs(subject_id, cfg):
        _, cues = _run_cues(root, e, cfg)
        labels.extend([e["session"]] * len(cues))
    return np.array(labels)


def bids_root(cfg: DataConfig):
    """Directory of a dataset's downloaded BIDS tree, under MNE_DATA when set.

    Parameters
    ----------
    cfg : DataConfig
        Provides dataset and nemar_id.

    Returns
    -------
    Path
    """
    return Path(os.environ.get("MNE_DATA", DATA_DIR)) / f"NEMAR-{cfg.nemar_id}"


def download_bids(cfg: DataConfig):
    """Downloads a dataset's pinned NEMAR release into bids_root(cfg).

    Parameters
    ----------
    cfg : DataConfig
        Provides nemar_id and nemar_version.
    """
    import nemar

    nemar.download(dataset=cfg.nemar_id, tag=cfg.nemar_version, target_dir=bids_root(cfg))


def load_bids_subject(subject_id, cfg: DataConfig):
    """Loads and epochs all cue windows of one participant of a BIDS dataset.

    Runs are read in recording order within each session. A run without cue
    events (a baseline) contributes no windows.

    Parameters
    ----------
    subject_id : int
        Participant number, matching the digits of the BIDS sub- label (P01 is 1).
    cfg : DataConfig
        Provides the filtering, resampling, window and cue fields.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Filtered windows from cfg.tmin to cfg.tmax after each cue, scaled by cfg.scale.
    y : ndarray, shape (n_trials,)
        Class index of each cue, its position in cfg.cue_labels.
    sessions : ndarray, shape (n_trials,)
        Session label per trial.

    Raises
    ------
    ValueError
        If the participant has no recordings, or a recording's EEG channel
        count differs from cfg.n_channels.
    """
    root = bids_root(cfg)
    mine = _subject_runs(subject_id, cfg)

    fs = cfg.resample_freq
    offset, n_times = round(cfg.tmin * fs), round((cfg.tmax - cfg.tmin) * fs)
    X, y, sessions = [], [], []
    for e in mine:
        path, cues = _run_cues(root, e, cfg)
        if cues.empty:
            continue

        raw = read_raw_bids(path, verbose="ERROR").load_data().pick("eeg")
        if len(raw.ch_names) != cfg.n_channels:
            raise ValueError(f"{path.basename} has {len(raw.ch_names)} EEG channels, expected {cfg.n_channels}")
        if cfg.notch_freqs:
            raw.notch_filter(cfg.notch_freqs, verbose="ERROR")
        raw.filter(cfg.l_freq, cfg.h_freq, verbose="ERROR")
        raw.resample(fs, verbose="ERROR")
        data = raw.get_data() * cfg.scale

        for onset, label in zip(cues.onset, cues.trial_type):
            first = round(onset * fs) + offset
            X.append(data[:, first:first + n_times])
            y.append(cfg.cue_labels.index(label))
            sessions.append(e["session"])

    return np.stack(X), np.array(y), np.array(sessions)
