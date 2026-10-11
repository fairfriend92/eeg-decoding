"""Per-subject fine-tuned REVE member for ensembling.

Fine-tunes the whole pretrained REVE encoder and a newly initialized linear
head on the mean-pooled tokens, using NeuralBench's own fine-tuning recipe,
on each subject's trials, then writes softmax
probabilities in the same format as dump_predictions.py, to
output/{dataset}/predictions_reve_finetuned.npz. Test probabilities come from
a fit on the training split, validation probabilities from a fit on the
training split minus its trailing validation slice. Both fits start from the
same pretrained weights. Each fit holds out a stratified fifth of its own
trials for early stopping and never sees the data it predicts.

Reads the per-subject trial arrays cached by reve_embeddings.py, and caches
each subject's probabilities so an interrupted run resumes. A run over a
subset of subjects also writes the npz, containing only those subjects.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
import copy
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from config import Config, add_dataset_cli_args, resolve_config
from reve_embeddings import REVE_CLIP_STD, REVE_SAMPLING_RATE, load_reve
from train import output_dir_for

FINETUNE_MAX_EPOCHS = 40
FINETUNE_PATIENCE = 5
FINETUNE_BATCH_SIZE = 64
FINETUNE_LR = 1e-4
FINETUNE_WEIGHT_DECAY = 0.05
FINETUNE_WARMUP_FRACTION = 0.1
FINETUNE_GRAD_CLIP = 1.0
FINETUNE_INNER_VAL_FRACTION = 0.2


def _logits(model, head, xb, pos):
    features = model(xb, pos=pos.expand(len(xb), -1, -1), return_features=True)["features"]
    return head(features.mean(dim=(1, 2)))


def _predict(model, head, X, pos, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), FINETUNE_BATCH_SIZE):
            xb = torch.tensor(X[start:start + FINETUNE_BATCH_SIZE], dtype=torch.float32, device=device)
            probs.append(torch.softmax(_logits(model, head, xb, pos), dim=1).cpu())
    return torch.cat(probs).numpy()


def finetune_predict(model, head, init_state, X_fit, y_fit, X_eval, pos, device, seed):
    """Fine-tunes REVE from its initial weights, then predicts X_eval.

    Follows NeuralBench's full fine-tune recipe: AdamW with a OneCycle
    schedule over the maximum epoch count, gradient clipping, up to 40
    epochs, and early stopping on the held-out fifth's balanced accuracy.
    The best epoch's weights are used for prediction.

    Parameters
    ----------
    model : REVE
        Encoder; its weights are overwritten with init_state's before
        training.
    head : nn.Linear
        Classifier on the mean-pooled tokens; overwritten likewise.
    init_state : dict
        Pretrained encoder and one fixed head initialization, as
        {"model": state_dict, "head": state_dict}.
    X_fit, X_eval : ndarray, shape (n_trials, n_channels, n_times)
        Trials at 200 Hz, z-scored per channel.
    y_fit : ndarray, shape (n_fit,)
        Integer labels of X_fit.
    pos : Tensor, shape (1, n_channels, 3)
        Electrode positions, broadcast over the batch.
    device : torch.device
        Device to train and predict on.
    seed : int
        Seeds the inner split and the shuffling.

    Returns
    -------
    ndarray, shape (len(X_eval), n_classes)
        Softmax probabilities.
    """
    model.load_state_dict(init_state["model"])
    head.load_state_dict(init_state["head"])
    X_fit = np.clip(X_fit, -REVE_CLIP_STD, REVE_CLIP_STD)
    X_tr, X_in, y_tr, y_in = train_test_split(X_fit, y_fit, test_size=FINETUNE_INNER_VAL_FRACTION, stratify=y_fit, random_state=seed)
    X_tr, y_tr = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)

    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=FINETUNE_LR, weight_decay=FINETUNE_WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(X_tr) / FINETUNE_BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=FINETUNE_LR, total_steps=FINETUNE_MAX_EPOCHS * steps_per_epoch,
        pct_start=FINETUNE_WARMUP_FRACTION, anneal_strategy="cos",
    )
    generator = torch.Generator().manual_seed(seed)

    best_score, best_state, stale = -1.0, None, 0
    for _ in range(FINETUNE_MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(X_tr), generator=generator)
        for start in range(0, len(order), FINETUNE_BATCH_SIZE):
            idx = order[start:start + FINETUNE_BATCH_SIZE]
            loss = F.cross_entropy(_logits(model, head, X_tr[idx].to(device), pos), y_tr[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, FINETUNE_GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        pred = _predict(model, head, X_in, pos, device).argmax(axis=1)
        score = balanced_accuracy_score(y_in, pred)
        if score > best_score:
            best_score, stale = score, 0
            best_state = {"model": copy.deepcopy(model.state_dict()), "head": copy.deepcopy(head.state_dict())}
        else:
            stale += 1
            if stale >= FINETUNE_PATIENCE:
                break

    model.load_state_dict(best_state["model"])
    head.load_state_dict(best_state["head"])
    return _predict(model, head, np.clip(X_eval, -REVE_CLIP_STD, REVE_CLIP_STD), pos, device)


def dump_finetuned(cfg: Config, subjects=None, cache_dir=None):
    """Fine-tunes and scores REVE for every subject, then dumps probabilities.

    Parameters
    ----------
    cfg : Config
        Full project configuration. Its sampling rate is replaced by 200 Hz.
    subjects : list of int, optional
        Subset of subject ids; defaults to cfg.data.subject_ids.
    cache_dir : Path, optional
        Directory of reve_embeddings.py's per-subject caches; defaults to
        output/{dataset}/reve_embeddings/. Its "trials" subfolder must hold
        the subjects' trial arrays.

    Returns
    -------
    dict
        Arrays "subject", "y" and "probs", one row per test trial, and
        "val_subject", "val_y" and "val_probs", one row per validation trial.
    """
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, resample_freq=REVE_SAMPLING_RATE))
    cache_dir = Path(cache_dir or output_dir_for(cfg) / "reve_embeddings")
    (cache_dir / "finetuned_neuralbench").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = head = pos = init_state = None

    val, test = {"subject": [], "y": [], "probs": []}, {"subject": [], "y": [], "probs": []}
    for subject_id in subjects or cfg.data.subject_ids:
        result_path = cache_dir / "finetuned_neuralbench" / f"subject{subject_id}.npz"
        trials = np.load(cache_dir / "trials" / f"subject{subject_id}.npz")
        if not result_path.exists():
            if model is None:
                torch.manual_seed(0)
                model, pos = load_reve(cfg, device)
                pos = pos.unsqueeze(0).to(device)
                head = torch.nn.Linear(model.embed_dim, cfg.data.n_classes).to(device)
                # Every fit restarts from this copy, so subjects share one head initialization.
                init_state = {"model": copy.deepcopy(model.state_dict()), "head": copy.deepcopy(head.state_dict())}
            t0 = time.time()
            val_probs = finetune_predict(model, head, init_state, trials["fit_X"], trials["fit_y"], trials["val_X"], pos, device, seed=subject_id)
            test_probs = finetune_predict(model, head, init_state, trials["train_X"], trials["train_y"], trials["test_X"], pos, device, seed=subject_id)
            np.savez(result_path, val_probs=val_probs, probs=test_probs)
            print(f"[reve_finetuned] subject {subject_id}: two fits in {time.time() - t0:.0f}s", flush=True)
        result = np.load(result_path)
        y, yv = trials["test_y"], trials["val_y"]
        print(f"[reve_finetuned] subject {subject_id}: acc={(result['probs'].argmax(axis=1) == y).mean():.3f}", flush=True)
        for part, labels, probs in ((val, yv, result["val_probs"]), (test, y, result["probs"])):
            part["subject"].append(np.full(len(labels), subject_id))
            part["y"].append(labels)
            part["probs"].append(probs)

    dump = {k: np.concatenate(v) for k, v in test.items()}
    dump.update({f"val_{k}": np.concatenate(v) for k, v in val.items()})
    np.savez(output_dir_for(cfg) / "predictions_reve_finetuned.npz", **dump)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subset of subjects to fine-tune.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory of per-subject caches, shared across sessions.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_finetuned(resolve_config(args), subjects=args.subjects, cache_dir=args.cache_dir)
