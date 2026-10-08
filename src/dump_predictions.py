"""Dumps per-trial test softmax probabilities from saved per-subject checkpoints.

Loads each subject's final-fit checkpoint for every requested model, predicts
on the subject's held-out test split, and writes one .npz of plain arrays to
output/{dataset}/predictions_{model}.npz per model. The checkpoints are
never modified, and no training happens here.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
import json

import numpy as np
import torch

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import get_subject_data
from evaluate import MODEL_REGISTRY, load_checkpoint
from train import checkpoints_dir_for, output_dir_for, predict_probs


def dump_subject(entry, model_name, cfg: Config, device):
    """Predicts one subject's test split with the final-fit checkpoint.

    Parameters
    ----------
    entry : dict
        One entry from output/{dataset}/{model}_checkpoints.json.
    model_name : str
        One of "eegnet", "patch_transformer", "conformer".
    cfg : Config
        Full project configuration. Must match what was used to train
        the checkpoint.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    probs : ndarray, shape (n_test_trials, n_classes)
        Softmax probabilities.
    y_test : ndarray, shape (n_test_trials,)
        Integer test labels.
    """
    _, _, X_test, y_test = get_subject_data(entry["subject"], cfg.data)

    # The manifest records paths on the machine that trained the checkpoint.
    checkpoint_path = checkpoints_dir_for(cfg) / f"{model_name}_subject{entry['subject']}_final.pt"
    model = load_checkpoint(checkpoint_path, model_name, X_test.shape[-1], cfg, device)
    return predict_probs(model, X_test, device, cfg.train.batch_size), y_test


def dump_all(model_names, cfg: Config, subjects=None):
    """Dumps final-fit test probabilities for every subject and model.

    Parameters
    ----------
    model_names : list of str
        Models to dump, each with a checkpoint manifest under
        output/{dataset}/.
    cfg : Config
        Full project configuration.
    subjects : list of int, optional
        Subset of subject ids; defaults to every subject in the manifest.

    Returns
    -------
    dict
        Model name mapped to a dict with arrays "subject", "y" and "probs",
        one row per test trial, concatenated in subject order.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    out_dir = output_dir_for(cfg)

    dumps = {}
    for model_name in model_names:
        with open(out_dir / f"{model_name}_checkpoints.json") as f:
            manifest = json.load(f)
        if subjects is not None:
            manifest = [e for e in manifest if e["subject"] in subjects]

        ids, labels, probs = [], [], []
        for entry in manifest:
            p, y = dump_subject(entry, model_name, cfg, device)
            print(f"[{model_name}] subject {entry['subject']}: acc={(p.argmax(axis=1) == y).mean():.3f}", flush=True)
            ids.append(np.full(len(y), entry["subject"]))
            labels.append(y)
            probs.append(p)

        dumps[model_name] = {
            "subject": np.concatenate(ids),
            "y":       np.concatenate(labels),
            "probs":   np.concatenate(probs),
        }
        if subjects is None:
            np.savez(out_dir / f"predictions_{model_name}.npz", **dumps[model_name])

    return dumps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subset to dump without writing files.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_all(args.models, resolve_config(args), subjects=args.subjects)
