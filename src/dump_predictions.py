"""Dumps per-trial softmax probabilities from saved per-subject checkpoints.

Loads each subject's final-fit checkpoint for every requested model, predicts
on the subject's held-out test split, and its validation-fit checkpoint
(see train.py --val-fit) to predict the validation slice. Writes one .npz of
plain arrays to
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
from data_loader import get_subject_data, get_subject_validation_data
from evaluate import MODEL_REGISTRY, load_checkpoint
from train import checkpoints_dir_for, output_dir_for, predict_probs


def dump_subject(entry, model_name, cfg: Config, device):
    """Predicts one subject's test split and validation slice.

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
    val_probs : ndarray, shape (n_val_trials, n_classes)
        Softmax probabilities of the validation-fit checkpoint.
    y_val : ndarray, shape (n_val_trials,)
        Integer validation labels.
    probs : ndarray, shape (n_test_trials, n_classes)
        Softmax probabilities of the final-fit checkpoint.
    y_test : ndarray, shape (n_test_trials,)
        Integer test labels.
    """
    _, _, X_test, y_test = get_subject_data(entry["subject"], cfg.data)

    # The manifest records paths on the machine that trained the checkpoint.
    checkpoints_dir = checkpoints_dir_for(cfg)
    final = load_checkpoint(checkpoints_dir / f"{model_name}_subject{entry['subject']}_final.pt", model_name, X_test.shape[-1], cfg, device)

    _, _, X_val, y_val = get_subject_validation_data(entry["subject"], cfg.data)
    valfit = load_checkpoint(checkpoints_dir / f"{model_name}_subject{entry['subject']}_valfit.pt", model_name, X_val.shape[-1], cfg, device)
    batch_size = cfg.train.batch_size
    return predict_probs(valfit, X_val, device, batch_size), y_val, predict_probs(final, X_test, device, batch_size), y_test


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
        one row per test trial, and "val_subject", "val_y" and "val_probs",
        one row per validation trial, concatenated in subject order.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    out_dir = output_dir_for(cfg)

    dumps = {}
    for model_name in model_names:
        with open(out_dir / f"{model_name}_checkpoints.json") as f:
            manifest = json.load(f)
        if subjects is not None:
            manifest = [e for e in manifest if e["subject"] in subjects]

        val, test = {"subject": [], "y": [], "probs": []}, {"subject": [], "y": [], "probs": []}
        for entry in manifest:
            pv, yv, p, y = dump_subject(entry, model_name, cfg, device)
            print(f"[{model_name}] subject {entry['subject']}: acc={(p.argmax(axis=1) == y).mean():.3f}", flush=True)
            for part, labels, probs in ((val, yv, pv), (test, y, p)):
                part["subject"].append(np.full(len(labels), entry["subject"]))
                part["y"].append(labels)
                part["probs"].append(probs)

        dumps[model_name] = {k: np.concatenate(v) for k, v in test.items()}
        dumps[model_name].update({f"val_{k}": np.concatenate(v) for k, v in val.items()})
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
