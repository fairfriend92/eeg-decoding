"""Extracts frozen REVE features for every trial of every subject.

Runs the pretrained REVE encoder once over each subject's trials, in both
views the ensemble members need: the training split with its test split
(final fit), and the training split minus its trailing validation slice with
that slice (validation fit). Each view is normalized with its own fitting
statistics, matching the neural members' dumps. Writes plain arrays to
output/{dataset}/reve_embeddings.npz for reve_member.py, which fits the
per-subject classifiers. Features keep the channel axis, averaged over time
patches only.

Each subject's loaded trials and features are cached in a directory, so an
interrupted run resumes and a change of pooling skips the slow data loading.

REVE needs 200 Hz input, so the dataset's sampling rate is overridden here.
Pretrained weights and the electrode position bank are gated on the
Hugging Face Hub and require an authenticated session.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from braindecode.models import REVE
from moabb.paradigms import MotorImagery

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import get_subject_data, get_subject_validation_data
from datasets import MOABB_DATASET_CLASSES
from train import output_dir_for

REVE_SAMPLING_RATE = 200.0
REVE_CHECKPOINT = "brain-bzh/reve-base"
REVE_CLIP_STD = 15.0


def channel_names(cfg: Config):
    """Reads the dataset's EEG channel names in the order of the trial arrays.

    Parameters
    ----------
    cfg : Config
        Full project configuration.

    Returns
    -------
    list of str
    """
    data = cfg.data
    paradigm = MotorImagery(n_classes=data.n_classes, fmin=data.l_freq, fmax=data.h_freq, tmin=data.tmin, tmax=data.tmax, resample=data.resample_freq)
    epochs, _, _ = paradigm.get_data(dataset=MOABB_DATASET_CLASSES[data.dataset](), subjects=[data.subject_ids[0]], return_epochs=True)
    return list(epochs.ch_names)


def check_pretrained_loaded(model):
    """Confirms the pretrained checkpoint's tensors are present in the model.

    Parameters
    ----------
    model : REVE
        Model built by from_pretrained.

    Raises
    ------
    ValueError
        If no checkpoint tensor matches the model's weights.
    """
    files = [f for f in list_repo_files(REVE_CHECKPOINT) if f.endswith(".safetensors")]
    checkpoint = load_file(hf_hub_download(REVE_CHECKPOINT, files[0]))
    state = model.state_dict()
    matched = sum(k in state and torch.equal(state[k].cpu().float(), v.float()) for k, v in checkpoint.items())
    print(f"[reve] {matched} of {len(checkpoint)} checkpoint tensors match the model", flush=True)
    if matched == 0:
        raise ValueError("No pretrained REVE tensor was loaded into the model.")


def load_reve(cfg: Config, device):
    """Builds the pretrained REVE encoder and the dataset's electrode positions.

    Parameters
    ----------
    cfg : Config
        Full project configuration. Provides the class count for the
        attention-pooling head, which is newly initialized.
    device : torch.device
        Device to move the model to.

    Returns
    -------
    model : REVE
    pos : Tensor, shape (n_channels, 3)
        Electrode positions from the REVE position bank.

    Raises
    ------
    ValueError
        If no pretrained tensor was loaded into the model.
    """
    model = REVE.from_pretrained(REVE_CHECKPOINT, n_outputs=cfg.data.n_classes, attention_pooling=True, strict=False).to(device)
    check_pretrained_loaded(model)
    return model, model.get_positions(channel_names(cfg))


def embed_trials(model, X, pos, device, batch_size):
    """Encodes trials and averages the REVE tokens over time patches.

    Parameters
    ----------
    model : REVE
        Pretrained encoder; switched to eval mode internally.
    X : ndarray, shape (n_trials, n_channels, n_times)
        Trials at 200 Hz, z-scored per channel.
    pos : Tensor, shape (n_channels, 3)
        Electrode positions from the REVE position bank.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    ndarray, shape (n_trials, n_channels, embed_dim)
        Token features averaged over time patches, as float16 to halve the
        stored size.
    """
    X = np.clip(X, -REVE_CLIP_STD, REVE_CLIP_STD)
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            features = model(xb, pos=pos.to(device).expand(len(xb), -1, -1), return_features=True)["features"]
            out.append(features.float().mean(dim=2).cpu())
    return torch.cat(out).numpy().astype(np.float16)


def dump_embeddings(cfg: Config, subjects=None, batch_size=32, cache_dir=None):
    """Embeds every subject's fit, validation, train and test trials.

    Each subject's trials and features are cached in cache_dir as they
    finish, and cached ones are loaded instead of recomputed, so an
    interrupted run resumes where it stopped.

    Parameters
    ----------
    cfg : Config
        Full project configuration. Its sampling rate is replaced by 200 Hz.
    subjects : list of int, optional
        Subset of subject ids; defaults to cfg.data.subject_ids. A subset is
        not written to disk.
    batch_size : int
        Trials per forward pass.
    cache_dir : Path, optional
        Per-subject cache directory; defaults to
        output/{dataset}/reve_embeddings/.

    Returns
    -------
    dict
        For each part in "fit", "val", "train", "test": arrays "{part}_subject",
        "{part}_y" and "{part}_emb" (n_trials, n_channels, embed_dim), one row per
        trial, in subject order.
    """
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, resample_freq=REVE_SAMPLING_RATE))
    cache_dir = Path(cache_dir or output_dir_for(cfg) / "reve_embeddings")
    (cache_dir / "trials").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pos = None

    parts = {name: {"subject": [], "y": [], "emb": []} for name in ("fit", "val", "train", "test")}
    for subject_id in subjects or cfg.data.subject_ids:
        trials_path = cache_dir / "trials" / f"subject{subject_id}.npz"
        if not trials_path.exists():
            t0 = time.time()
            X_fit, y_fit, X_val, y_val = get_subject_validation_data(subject_id, cfg.data)
            X_train, y_train, X_test, y_test = get_subject_data(subject_id, cfg.data)
            np.savez(trials_path, fit_X=X_fit, fit_y=y_fit, val_X=X_val, val_y=y_val, train_X=X_train, train_y=y_train, test_X=X_test, test_y=y_test)
            print(f"[reve] subject {subject_id}: loaded trials in {time.time() - t0:.0f}s", flush=True)
        trials = np.load(trials_path)

        cache_path = cache_dir / f"subject{subject_id}_channels.npz"
        if not cache_path.exists():
            if model is None:
                model, pos = load_reve(cfg, device)
            t0 = time.time()
            np.savez(cache_path, **{f"{name}_emb": embed_trials(model, trials[f"{name}_X"], pos, device, batch_size) for name in parts})
            print(f"[reve] subject {subject_id}: embedded in {time.time() - t0:.0f}s", flush=True)
        cached = np.load(cache_path)
        for name, part in parts.items():
            part["subject"].append(np.full(len(trials[f"{name}_y"]), subject_id))
            part["y"].append(trials[f"{name}_y"])
            part["emb"].append(cached[f"{name}_emb"])

    dump = {f"{name}_{key}": np.concatenate(values) for name, part in parts.items() for key, values in part.items()}
    if subjects is None:
        np.savez(output_dir_for(cfg) / "reve_embeddings.npz", **dump)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subset to embed without writing files.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Per-subject cache directory, for resuming across sessions.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_embeddings(resolve_config(args), subjects=args.subjects, cache_dir=args.cache_dir)
