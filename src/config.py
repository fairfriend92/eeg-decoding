"""Hyperparameters and paths shared across data loading, training, and evaluation.

Dataset identity (class count, channel count, sampling rate, subject list,
session-split strategy) lives on DataConfig and is chosen via a dataset
preset in datasets.py (see make_config below) rather than hardcoded here --
this module only defines the schema and default values.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data"
OUTPUT_DIR  = REPO_ROOT / "output"
CONFIGS_DIR = REPO_ROOT / "configs"


@dataclass
class DataConfig:
    """Parameters for loading and epoching an EEG dataset via MOABB/MNE.

    Fields fall into two groups: dataset identity (dataset, n_classes,
    n_channels, subject_ids, split_strategy and its strategy-specific
    fields) which come from a preset in datasets.DATASET_PRESETS and
    normally should not be hand-edited per run, and signal-processing
    knobs (l_freq..resample_freq) which are reasonable to override via a
    CLI --set flag or config file for a given run.

    Parameters
    ----------
    dataset : str
        Dataset name, matching a key in datasets.MOABB_DATASET_CLASSES.
    n_classes : int
        Number of classes the paradigm should decode.
    n_channels : int
        Number of EEG channels the dataset provides (fixes model input shape).
    subject_ids : list of int
        Subject identifiers to iterate over for this dataset.
    l_freq : float
        Bandpass filter low cutoff, in Hz.
    h_freq : float
        Bandpass filter high cutoff, in Hz.
    tmin : float
        Epoch start time relative to cue onset, in seconds.
    tmax : float
        Epoch end time relative to cue onset, in seconds.
    resample_freq : float
        Target sampling rate after resampling, in Hz. Kept at a common
        value (250 Hz) across datasets by default for cross-dataset
        hyperparameter comparability; CroppedConfig's sample counts are
        derived from this at config-build time (see make_cropped_config),
        so changing it does not silently desync crop sizing.
    split_strategy : str
        One of "fixed_sessions" (two explicitly named sessions, used by
        BCI2a), "session_index" (train on the first n_train_sessions
        sessions, test on the rest, used by Stieger2021 to match
        cross-session decoding), or "within_session_holdout"
        (single-session datasets with no second session to hold out, used
        by Dreyer2023; splits a trailing, per-class-stratified fraction of
        trials instead).
    train_session, test_session : str, optional
        Session labels used by the "fixed_sessions" strategy.
    n_train_sessions : int, optional
        Number of earliest sessions used for training by the
        "session_index" strategy; the rest are held out for testing.
    holdout_fraction : float, optional
        Fraction of each class's trailing trials held out by the
        "within_session_holdout" strategy.
    val_fraction : float
        Fraction of each class's trailing training trials carved out as the
        ensemble-selection validation slice (see data_loader.split_validation).
        Disjoint from the test split by construction, since it is taken from
        the training portion only.
    """

    dataset: str = "bci2a"
    n_classes: int = 4
    n_channels: int = 22
    subject_ids: list = field(default_factory=lambda: list(range(1, 10)))

    l_freq: float = 4.0
    h_freq: float = 38.0
    tmin: float = 0.0
    tmax: float = 4.0
    resample_freq: float = 250.0

    split_strategy: str = "fixed_sessions"
    train_session: str = "0train"
    test_session: str = "1test"
    n_train_sessions: int = None
    holdout_fraction: float = None
    val_fraction: float = 0.2


@dataclass
class CroppedConfig:
    """Parameters for optional crop-based training and evaluation.

    Only used when explicitly enabled (see train.py --cropped). Training
    trials are sliced into overlapping crops as extra training examples,
    using crop_stride (coarser, to keep the crop-multiplied training set
    a manageable size). Validation/test trials are scored by cropping
    with eval_crop_stride instead. It is denser because evaluation is done
    far fewer times than training, so a smoother, more accurate per-trial
    average from more overlapping crops is cheap to afford. Predictions
    are then averaged per trial.

    crop_size must match between training and evaluation, since it fixes
    the model's input length; only the crop density (stride) may differ.

    Expressed in raw samples (not seconds) so that a checkpoint manifest's
    recorded crop_size/crop_stride (see evaluate.py) is unambiguous without
    also recording the sampling rate. Use make_cropped_config to build one
    of these from durations in seconds, correctly scaled to a dataset's
    resample_freq. See make_config, which does this automatically.

    Parameters
    ----------
    crop_size : int
        Crop length, in samples. Fixes the model's input length.
    crop_stride : int
        Step between crop start positions during training, in samples.
    eval_crop_stride : int
        Step between crop start positions during validation/test scoring,
        in samples. Denser (smaller) than crop_stride by design.
    """

    crop_size: int = 500
    crop_stride: int = 125
    eval_crop_stride: int = 125


def make_cropped_config(resample_freq, crop_duration_s=2.0, crop_stride_s=0.5, eval_crop_stride_s=0.5):
    """Builds a CroppedConfig sized correctly for a given sampling rate.

    Durations in seconds are dataset-independent; the resulting sample
    counts are not (500 samples means something different at 250 Hz than
    at 512 Hz). Defaults reproduce this project's original BCI2a defaults
    exactly at resample_freq=250.0 (500/125/125 samples).

    Parameters
    ----------
    resample_freq : float
        Sampling rate, in Hz, to scale the durations against.
    crop_duration_s, crop_stride_s, eval_crop_stride_s : float
        Crop length and strides, in seconds.

    Returns
    -------
    CroppedConfig
    """
    return CroppedConfig(
        crop_size=round(crop_duration_s * resample_freq),
        crop_stride=round(crop_stride_s * resample_freq),
        eval_crop_stride=round(eval_crop_stride_s * resample_freq),
    )


@dataclass
class ConformerConfig:
    """Architecture and optimization parameters for the EEG Conformer.

    Defaults follow the original paper's BCI Competition IV 2a setup at
    250 Hz, with sample counts scaled by sampling rate like EEGNetConfig's
    kernel_length.

    Parameters
    ----------
    n_filters : int
        Number of temporal and spatial convolution filters.
    kernel_length : int
        Length of the temporal convolution kernel, in samples.
    pool_size : int
        Average pooling window after the convolutions, in samples. Sets
        each token's temporal extent.
    pool_stride : int
        Average pooling stride, in samples. Sets the token spacing.
    embed_dim : int
        Token embedding dimension after the 1x1 projection.
    n_heads : int
        Number of attention heads per encoder block. Must divide embed_dim.
    n_layers : int
        Number of stacked transformer encoder blocks.
    ff_expansion : int
        Feed-forward hidden size as a multiple of embed_dim.
    dropout : float
        Dropout probability in the convolution module and the encoder.
    lr : float
        Initial learning rate for Adam.
    """

    n_filters: int = 40
    kernel_length: int = 25
    pool_size: int = 75
    pool_stride: int = 15
    embed_dim: int = 40
    n_heads: int = 10
    n_layers: int = 6
    ff_expansion: int = 4
    dropout: float = 0.5
    lr: float = 2e-4


@dataclass
class PatchTransformerConfig:
    """Architecture and optimization parameters for the patch-token transformer.

    Parameters
    ----------
    patch_size : int
        Number of time samples covered by each convolutional patch token.
    embed_dim : int
        Token embedding dimension.
    n_heads : int
        Number of attention heads per encoder block.
    n_layers : int
        Number of stacked transformer encoder blocks.
    ff_dim : int
        Hidden dimension of the feed-forward sublayer.
    dropout : float
        Dropout probability applied throughout the encoder and head.
    lr : float
        Initial learning rate for Adam. Lower than EEGNet's, selected via
        hyperparameter sweep. The larger, less-biased architecture
        overfits at higher learning rates on this dataset's size.
    """

    patch_size: int = 25
    embed_dim: int = 64
    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.5
    lr: float = 3e-4


@dataclass
class EEGNetConfig:
    """Architecture and optimization parameters for the EEGNet CNN baseline.

    Parameters
    ----------
    f1 : int
        Number of temporal filters in the first convolution.
    depth_multiplier : int
        Depth multiplier for the depthwise spatial convolution.
    f2 : int
        Number of pointwise filters in the separable convolution.
    kernel_length : int
        Length of the temporal convolution kernel, in samples. Set to roughly
        half the sampling rate, so the kernel spans ~0.5s of signal.
    dropout : float
        Dropout probability applied after each pooling stage. Sweep found
        dropout in {0.25, 0.5} statistically indistinguishable at this
        learning rate; kept at the original paper's value.
    lr : float
        Initial learning rate for Adam. Confirmed near-optimal by
        hyperparameter sweep. A lower rate (3e-4) underperformed clearly.
    """

    f1: int = 8
    depth_multiplier: int = 2
    f2: int = 16
    kernel_length: int = 125    # ~0.5s at 250 Hz
    dropout: float = 0.25
    lr: float = 1e-3


@dataclass
class TrainConfig:
    """Parameters controlling the per-subject training loop.

    Learning rate is not here. It is model-specific (see EEGNetConfig,
    PatchTransformerConfig, ConformerConfig), since the architectures need
    different values.

    Parameters
    ----------
    batch_size : int
        Training and evaluation batch size.
    weight_decay : float
        L2 regularization coefficient.
    n_epochs : int
        Maximum number of training epochs.
    patience : int
        Epochs without validation improvement before early stopping.
    seed : int
        Seed for model initialization and data shuffling.
    device : str
        Torch device identifier ("cuda" or "cpu").
    """

    batch_size: int = 32
    weight_decay: float = 1e-4
    n_epochs: int = 100
    patience: int = 15
    seed: int = 42
    device: str = "cuda"


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    data:      DataConfig       = field(default_factory=DataConfig)
    patch_transformer: PatchTransformerConfig = field(default_factory=PatchTransformerConfig)
    conformer: ConformerConfig  = field(default_factory=ConformerConfig)
    eegnet:    EEGNetConfig     = field(default_factory=EEGNetConfig)
    train:     TrainConfig      = field(default_factory=TrainConfig)
    cropped:   CroppedConfig    = field(default_factory=CroppedConfig)


def make_config(dataset="bci2a", model=None, overrides=None):
    """Builds a Config for a chosen dataset, with optional field overrides.

    This is the single entry point train.py/evaluate.py/sweep.py should use
    to build their Config, instead of calling Config() directly, so that
    dataset selection and CLI/file overrides are applied consistently
    everywhere.

    Parameters
    ----------
    dataset : str
        Dataset name, matching a key in datasets.DATASET_PRESETS.
    model : str, optional
        "eegnet", "patch_transformer", or "conformer". Only used to resolve bare (non-dotted)
        override keys against the right model sub-config. See
        apply_overrides.
    overrides : dict, optional
        Field name (dotted, e.g. "train.lr", "data.n_train_sessions", or
        bare, e.g. "lr") -> value. Values from a CLI --set flag arrive as
        strings and are coerced to match the target field's current type;
        values from a parsed YAML file arrive already typed.

    Returns
    -------
    Config
    """
    # Local import: datasets.py imports DataConfig from this module, so a
    # module-level import here would be circular.
    from datasets import DATASET_PRESETS

    if dataset not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choices: {sorted(DATASET_PRESETS)}")

    cfg      = Config()
    cfg.data = copy.deepcopy(DATASET_PRESETS[dataset])
    cfg.cropped = make_cropped_config(cfg.data.resample_freq)

    if overrides:
        cfg = apply_overrides(cfg, overrides, model_name=model)

    return cfg


def _coerce(raw, current_value):
    """Coerces a raw string override value to match current_value's type.

    Values already typed (e.g. from a parsed YAML file) pass through
    unchanged. Only plain strings (e.g. from a CLI --set flag) are
    coerced.
    """
    if not isinstance(raw, str) or isinstance(current_value, str):
        return raw
    if isinstance(current_value, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current_value, int):
        return int(raw)
    if isinstance(current_value, float):
        return float(raw)
    return raw


def apply_overrides(cfg: Config, overrides: dict, model_name: str = None) -> Config:
    """Applies a dict of field overrides onto a Config, in place.

    Parameters
    ----------
    cfg : Config
        Config to mutate.
    overrides : dict
        Field name -> value. A dotted key ("section.field", e.g.
        "data.n_train_sessions", "eegnet.dropout", "cropped.crop_size")
        targets that section explicitly. A bare key (e.g. "lr") is
        resolved against cfg.train first, then against the model_name
        sub-config if given. This matches this project's original
        sweep.py override behavior for train/model hyperparameters.
    model_name : str, optional
        "eegnet", "patch_transformer", or "conformer"; enables bare-key resolution against that
        model's sub-config.

    Returns
    -------
    Config
        The same cfg instance, mutated.

    Raises
    ------
    ValueError
        If a key names an unknown section, or a field not present on the
        resolved section.
    """
    for key, value in overrides.items():
        if "." in key:
            section, field_name = key.split(".", 1)
            if not hasattr(cfg, section):
                raise ValueError(f"Unknown config section '{section}' in override '{key}'")
            target = getattr(cfg, section)
        else:
            field_name = key
            if hasattr(cfg.train, field_name):
                target = cfg.train
            elif model_name is not None and hasattr(getattr(cfg, model_name), field_name):
                target = getattr(cfg, model_name)
            else:
                raise ValueError(
                    f"Unknown hyperparameter '{key}'. Use 'section.field' "
                    "to target data/cropped/eegnet/patch_transformer/conformer explicitly."
                )

        if not hasattr(target, field_name):
            raise ValueError(f"Unknown field '{field_name}' for override '{key}'")

        setattr(target, field_name, _coerce(value, getattr(target, field_name)))

    return cfg


def flatten_file_overrides(nested: dict) -> dict:
    """Flattens a one-level-nested overrides dict (as loaded from YAML)
    into the dotted-key form apply_overrides expects.

    Parameters
    ----------
    nested : dict
        E.g. {"train": {"lr": 0.001}, "eegnet": {"dropout": 0.3}}.

    Returns
    -------
    dict
        E.g. {"train.lr": 0.001, "eegnet.dropout": 0.3}. A top-level value
        that is not itself a dict is passed through as a bare key.
    """
    flat = {}
    for section, fields in nested.items():
        if isinstance(fields, dict):
            for field_name, value in fields.items():
                flat[f"{section}.{field_name}"] = value
        else:
            flat[section] = fields
    return flat


def parse_cli_overrides(pairs) -> dict:
    """Parses repeated --set key=value CLI arguments into an overrides dict.

    Parameters
    ----------
    pairs : list of str, optional
        Each element like "train.lr=0.001".

    Returns
    -------
    dict
    """
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        overrides[key] = value
    return overrides


def add_dataset_cli_args(parser):
    """Adds --dataset/--config/--set to an argparse parser.

    Shared by train.py/evaluate.py/sweep.py so all three scripts expose the
    same dataset-selection and override surface. See resolve_config to turn
    the resulting args namespace into a Config.
    """
    parser.add_argument(
        "--dataset", default="bci2a",
        help="Dataset to use (default: bci2a). See datasets.DATASET_PRESETS for choices "
             "(currently bci2a, dreyer2023, stieger2021, scherer2015; graz_brainhero is "
             "registered but not yet released).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML file of config overrides, nested by section, "
             "e.g. 'train:\\n  lr: 0.001'. Applied before --set, so --set wins on conflicts. "
             "A bare filename (e.g. 'ten_subjects.yaml') is looked up in configs/ if it "
             "doesn't resolve relative to the current directory.",
    )
    parser.add_argument(
        "--set", dest="set_overrides", action="append", default=None, metavar="KEY=VALUE",
        help="Override a single config field, e.g. --set train.lr=0.001 or "
             "--set data.n_train_sessions=6. Repeatable.",
    )


def resolve_config(args, model=None) -> Config:
    """Builds a Config from parsed CLI args (see add_dataset_cli_args).

    Parameters
    ----------
    args : argparse.Namespace
        Must have .dataset, .config, .set_overrides attributes.
    model : str, optional
        Passed through to make_config for bare-key override resolution.

    Returns
    -------
    Config
    """
    overrides = {}
    if args.config is not None:
        config_path = args.config if args.config.exists() else CONFIGS_DIR / args.config
        with open(config_path) as f:
            file_overrides = yaml.safe_load(f) or {}
        overrides.update(flatten_file_overrides(file_overrides))
    overrides.update(parse_cli_overrides(args.set_overrides))

    return make_config(dataset=args.dataset, model=model, overrides=overrides)
