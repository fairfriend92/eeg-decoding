"""Registry of supported EEG datasets.

Adding a new dataset means adding one entry to each dict below; nothing
else in the pipeline (config.make_config, data_loader.load_subject/
split_subject, train.py/evaluate.py/sweep.py) needs to change, as long as
the dataset is exposed as a MOABB dataset class compatible with
moabb.paradigms.MotorImagery. A BIDS dataset on NEMAR that MOABB does not
provide needs only a DATASET_PRESETS entry that sets nemar_id (see
bids_loader), and carries its own filtering, rate, window, scale and
model_overrides there.

Subject counts/ids and n_train_sessions for stieger2021 are provisional
defaults based on the dataset's published documentation, not yet
confirmed against the actual downloaded data. That dataset is ~399GB, so
confirming firsthand is deferred until it's actually needed. See the
project's plan notes. graz_brainhero is not yet released; its entry
exists so it already appears as a --dataset choice and its preset can be
filled in without touching any other file once real data/class ids are
known.

scherer2015's train_session/test_session values ("0"/"1") are confirmed
against real downloaded data: a live run loaded both sessions for all 9
subjects with no split error.
"""

from moabb.datasets import BNCI2014_001, BNCI2015_004, Dreyer2023, Stieger2021

from config import DataConfig


def _graz_brainhero_not_released(*args, **kwargs):
    raise NotImplementedError(
        "Graz+BrainHero data has not been released yet (NeurIPS EEG/EMG "
        "Foundation Challenge 2026, Track 2). Once released, register its "
        "MOABB dataset class here, or a direct mne-bids loader if MOABB "
        "does not add one, and fill in DATASET_PRESETS['graz_brainhero']."
    )


MOABB_DATASET_CLASSES = {
    "bci2a":          BNCI2014_001,
    "dreyer2023":     Dreyer2023,
    "stieger2021":    Stieger2021,
    "scherer2015":    BNCI2015_004,
    "graz_brainhero": _graz_brainhero_not_released,
}


DATASET_PRESETS = {
    "bci2a": DataConfig(
        dataset="bci2a",
        n_classes=4,
        n_channels=22,
        subject_ids=list(range(1, 10)),
        resample_freq=250.0,
        split_strategy="fixed_sessions",
        train_session="0train",
        test_session="1test",
    ),
    "dreyer2023": DataConfig(
        dataset="dreyer2023",
        n_classes=2,
        n_channels=27,
        subject_ids=list(range(1, 88)),
        resample_freq=250.0,
        split_strategy="within_session_holdout",
        holdout_fraction=0.2,
        test_subject_ids=list(range(61, 82)),
    ),
    "stieger2021": DataConfig(
        dataset="stieger2021",
        n_classes=4,
        n_channels=60,
        subject_ids=list(range(1, 63)),
        resample_freq=250.0,
        split_strategy="session_index",
        n_train_sessions=8,
    ),
    "scherer2015": DataConfig(
        dataset="scherer2015",
        n_classes=5,
        n_channels=30,
        subject_ids=list(range(1, 10)),
        resample_freq=250.0,
        # Within-session cross-validation on the training session peaks for the 4-8 s window.
        tmin=4.0,
        tmax=8.0,
        split_strategy="fixed_sessions",
        train_session="0",    # unverified placeholder, see module docstring
        test_session="1",     # unverified placeholder, see module docstring
    ),
    # Input format of the Codabench Track 02 warm-up: 0.1-75 Hz after a 50/60 Hz
    # notch, 120 Hz, microvolts, 4 s starting 1 s after the cue, 41 EEG channels.
    "proteus": DataConfig(
        dataset="proteus",
        n_classes=3,
        n_channels=41,
        subject_ids=list(range(1, 11)),
        l_freq=0.1,
        h_freq=75.0,
        notch_freqs=[50.0, 60.0],
        tmin=1.0,
        tmax=5.0,
        resample_freq=120.0,
        scale=1e6,
        nemar_id="nm000290",
        nemar_version="1.0.0",
        cue_labels=["mi", "sub", "word"],
        # Sample counts that are 0.5 s (EEGNet kernel) and 100/300/60 ms (Conformer, patches) at 120 Hz.
        model_overrides={
            "eegnet.kernel_length": 60,
            "conformer.kernel_length_ms": 100.0,
            "conformer.pool_size_ms": 300.0,
            "conformer.pool_stride_ms": 60.0,
            "patch_transformer.patch_size_ms": 100.0,
        },
        split_strategy="within_session_holdout",
        holdout_fraction=0.2,
        # Seeded 20% participant draw (random_state 33) as the cross-subject test set.
        test_subject_ids=[4, 9],
    ),
    "graz_brainhero": DataConfig(
        dataset="graz_brainhero",
        n_classes=3,
        n_channels=43,
        subject_ids=[],
        resample_freq=250.0,
        split_strategy="session_index",
        n_train_sessions=3,
    ),
}
