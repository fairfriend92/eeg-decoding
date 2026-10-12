"""Cross-subject fine-tuned REVE member for ensembling.

Fine-tunes the whole pretrained REVE encoder and a new linear head on pooled
participants with NeuralBench's fine-tuning recipe (see
reve_finetune.finetune_predict), and writes the probability dump the ensemble
harness reads, to output/{dataset}/cross_subject/predictions_reve_finetuned.npz.
Validation probabilities come from a fit on the fitting participants only,
test probabilities from a fit on every training participant. Each fit stops
early on a fifth of its own participants and never sees the data it predicts.

The validation-fit probabilities are cached, so an interrupted run resumes at
the final fit.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
import copy
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import get_cross_subject_data
from reve_embeddings import REVE_SAMPLING_RATE, load_reve
from reve_finetune import finetune_predict
from train import output_dir_for


def dump_reve_cross_subject(cfg: Config, cache_dir=None, n_jobs=2):
    """Fine-tunes REVE on pooled participants and dumps probabilities.

    Parameters
    ----------
    cfg : Config
        Full project configuration. Its sampling rate is replaced by 200 Hz.
    cache_dir : Path, optional
        Directory for the epoch caches and the validation-fit probabilities;
        defaults to output/{dataset}/. Persistent storage lets a fresh
        machine resume.
    n_jobs : int
        Worker processes used to epoch participants missing from the cache.

    Returns
    -------
    dict
        Arrays "subject", "y" and "probs" for the test rows, and
        "val_subject", "val_y" and "val_probs" for the validation rows.
    """
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, resample_freq=REVE_SAMPLING_RATE))
    out_dir = output_dir_for(cfg) / "cross_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir or output_dir_for(cfg))
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = get_cross_subject_data(cfg.data, n_jobs=n_jobs, cache_dir=cache_dir / "epochs_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, pos = load_reve(cfg, device)
    pos = pos.unsqueeze(0).to(device)
    head = torch.nn.Linear(model.embed_dim, cfg.data.n_classes).to(device)
    init_state = {"model": copy.deepcopy(model.state_dict()), "head": copy.deepcopy(head.state_dict())}

    val_path = cache_dir / "reve_cross_subject_val_probs.npy"
    if val_path.exists():
        val_probs = np.load(val_path)
    else:
        t0 = time.time()
        val_probs = finetune_predict(
            model, head, init_state, data["X_fit"], data["y_fit"], data["X_val"], pos, device, seed=0, groups=data["subject_fit"],
        )
        np.save(val_path, val_probs)
        print(f"[reve_finetuned] validation fit in {time.time() - t0:.0f}s", flush=True)

    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    subject_all = np.concatenate([data["subject_fit"], data["subject_val"]])
    t0 = time.time()
    probs = finetune_predict(model, head, init_state, X_all, y_all, data["X_test"], pos, device, seed=0, groups=subject_all)
    print(f"[reve_finetuned] final fit in {time.time() - t0:.0f}s", flush=True)

    dump = {
        "subject": data["subject_test"], "y": data["y_test"], "probs": probs,
        "val_subject": data["subject_val"], "val_y": data["y_val"], "val_probs": val_probs,
    }
    np.savez(out_dir / "predictions_reve_finetuned.npz", **dump)
    print(f"[reve_finetuned] test acc={(probs.argmax(axis=1) == data['y_test']).mean():.3f}, "
          f"val acc={(val_probs.argmax(axis=1) == data['y_val']).mean():.3f}", flush=True)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Persistent directory for caches, shared across sessions.")
    parser.add_argument("--n-jobs", type=int, default=2, help="Workers epoching participants missing from the cache.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_reve_cross_subject(resolve_config(args), cache_dir=args.cache_dir, n_jobs=args.n_jobs)
