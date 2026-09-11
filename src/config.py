"""Hyperparameters and paths shared across data loading, training, and evaluation."""

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data"
OUTPUT_DIR  = REPO_ROOT / "output"

N_CLASSES   = 4    # left hand, right hand, feet, tongue
N_CHANNELS  = 22    # EEG electrodes, BCI-IV-2a montage
SUBJECT_IDS = list(range(1, 10))    # 9 subjects


@dataclass
class DataConfig:
    """Parameters for loading and epoching BCI-IV-2a via MOABB/MNE.

    Parameters
    ----------
    l_freq : float
        Bandpass filter low cutoff, in Hz.
    h_freq : float
        Bandpass filter high cutoff, in Hz.
    tmin : float
        Epoch start time relative to cue onset, in seconds.
    tmax : float
        Epoch end time relative to cue onset, in seconds.
    resample_freq : float
        Target sampling rate after resampling, in Hz.
    train_session : str
        Session label used as the training set (official competition split).
    test_session : str
        Session label used as the test set (official competition split).
    """

    l_freq: float = 4.0
    h_freq: float = 38.0
    tmin: float = 0.0
    tmax: float = 4.0
    resample_freq: float = 250.0
    train_session: str = "session_T"
    test_session: str = "session_E"


@dataclass
class ConformerConfig:
    """Architecture parameters for the transformer encoder model.

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
    """

    patch_size: int = 25
    embed_dim: int = 64
    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.3


@dataclass
class EEGNetConfig:
    """Architecture parameters for the EEGNet CNN baseline.

    Parameters
    ----------
    f1 : int
        Number of temporal filters in the first convolution.
    depth_multiplier : int
        Depth multiplier for the depthwise spatial convolution.
    f2 : int
        Number of pointwise filters in the separable convolution.
    dropout : float
        Dropout probability applied after each pooling stage.
    """

    f1: int = 8
    depth_multiplier: int = 2
    f2: int = 16
    dropout: float = 0.25


@dataclass
class TrainConfig:
    """Parameters controlling the per-subject training loop.

    Parameters
    ----------
    batch_size : int
        Training and evaluation batch size.
    lr : float
        Initial learning rate for Adam.
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
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 100
    patience: int = 15
    seed: int = 42
    device: str = "cuda"


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    data:      DataConfig       = field(default_factory=DataConfig)
    conformer: ConformerConfig  = field(default_factory=ConformerConfig)
    eegnet:    EEGNetConfig     = field(default_factory=EEGNetConfig)
    train:     TrainConfig      = field(default_factory=TrainConfig)
