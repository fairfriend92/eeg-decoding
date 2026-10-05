"""Joint multi-dataset pretraining of a shared EEGNet/Conformer/PatchTransformer trunk.

EEGNet and PatchTransformer each have exactly one layer whose shape is tied to
a dataset's channel count (EEGNet.spatial_conv, PatchTransformer.PatchEmbedding
.spatial_conv). Every other layer (EEGNet's temporal_conv/separable_conv;
PatchTransformer's patch_embed.temporal_conv/.norm/.activation and its transformer
encoder) is channel-count-independent, so it can be trained jointly across
datasets with different channel counts and reused as a warm-start for a
new target dataset's own model (see train.py's --init-trunk).

Only the training-split trials of each source dataset ever contribute to
pretraining (via data_loader.get_subject_data), so a held-out test session
never leaks into the trunk.
"""

import argparse
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import OUTPUT_DIR, ConformerConfig, EEGNetConfig, PatchTransformerConfig, TrainConfig
from data_loader import get_subject_data
from datasets import DATASET_PRESETS
from models.conformer import EEGConformer
from models.patch_transformer import PatchTransformer
from models.eegnet import EEGNet

MODEL_REGISTRY = {
    "eegnet": EEGNet,
    "patch_transformer": PatchTransformer,
    "conformer": EEGConformer,
}

TRUNK_KEY_PREFIXES = {
    "eegnet": ("temporal_conv.", "separable_conv."),
    "patch_transformer": ("patch_embed.temporal_conv.", "patch_embed.norm.", "encoder."),
    "conformer": (
        "patch_embed.temporal_conv.", "patch_embed.norm.", "patch_embed.projection.", "encoder.",
    ),
}


def filter_trunk_state_dict(state_dict, model_name):
    """Filters a full model state_dict down to its shared-trunk keys.

    Parameters
    ----------
    state_dict : dict
        Full state_dict of an EEGNet or PatchTransformer instance.
    model_name : str
        One of "eegnet", "patch_transformer", "conformer".

    Returns
    -------
    dict
        Only the entries whose key starts with one of
        TRUNK_KEY_PREFIXES[model_name].
    """
    prefixes = TRUNK_KEY_PREFIXES[model_name]
    return {k: v for k, v in state_dict.items() if k.startswith(prefixes)}


def share_trunk_modules(models, model_name):
    """Rewires every model after the first to reuse the first model's own
    trunk submodules in place, so gradients from every dataset's loss
    update the same shared parameter tensors.

    Parameters
    ----------
    models : list of nn.Module
        One freshly constructed EEGNet or PatchTransformer per source dataset,
        each already sized to its own dataset's n_channels/n_classes.
    model_name : str
        One of "eegnet", "patch_transformer", "conformer".

    Returns
    -------
    nn.Module
        models[0], now the sole owner of the shared trunk submodules.
    """
    trunk_owner = models[0]
    for model in models[1:]:
        if model_name == "eegnet":
            model.temporal_conv = trunk_owner.temporal_conv
            model.separable_conv = trunk_owner.separable_conv
        else:
            model.patch_embed.temporal_conv = trunk_owner.patch_embed.temporal_conv
            model.patch_embed.norm = trunk_owner.patch_embed.norm
            model.patch_embed.activation = trunk_owner.patch_embed.activation
            if model_name == "conformer":
                model.patch_embed.projection = trunk_owner.patch_embed.projection
            model.encoder = trunk_owner.encoder
    return trunk_owner


def trunk_and_head_parameters(models, trunk_owner, model_name):
    """Builds one optimizer's worth of parameters: the shared trunk taken
    once, plus every model's own per-dataset head.

    Taking the trunk's parameters once (not once per model, since every
    model after the first shares the same trunk tensors after
    share_trunk_modules) avoids double-stepping those tensors and
    corrupting Adam's per-tensor momentum state.

    Parameters
    ----------
    models : list of nn.Module
        Models after share_trunk_modules has rewired their trunk.
    trunk_owner : nn.Module
        The model whose trunk submodules are the shared instances.
    model_name : str
        One of "eegnet", "patch_transformer", "conformer".

    Returns
    -------
    list of Parameter
    """
    if model_name == "eegnet":
        trunk_params = list(trunk_owner.temporal_conv.parameters()) + list(trunk_owner.separable_conv.parameters())
        head_params = [
            p for m in models for p in list(m.spatial_conv.parameters()) + list(m.classifier.parameters())
        ]
    else:
        trunk_params = (
            list(trunk_owner.patch_embed.temporal_conv.parameters())
            + list(trunk_owner.patch_embed.norm.parameters())
            + list(trunk_owner.encoder.parameters())
        )
        if model_name == "conformer":
            trunk_params += list(trunk_owner.patch_embed.projection.parameters())
        head_params = [
            p for m in models
            for p in list(m.patch_embed.spatial_conv.parameters()) + list(m.classifier.parameters())
        ]
    return trunk_params + head_params


def _with_heartbeat(fn, interval_s=30):
    """Calls fn() in the current thread, printing a heartbeat while it runs.

    A dataset download plus MNE epoching can run silently for minutes per
    subject; a long silent stretch is suspected to kill the Colab tunnel.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to run.
    interval_s : float
        Seconds between heartbeat lines while fn runs.

    Returns
    -------
    object
        fn's return value.
    """
    start = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(interval_s):
            print(f"[heartbeat] still running, {int(time.monotonic() - start)}s elapsed", flush=True)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        return fn()
    finally:
        stop.set()
        heartbeat_thread.join()


def load_pretrain_corpus(dataset_name):
    """Pools every subject's training-split trials for one source dataset.

    Reuses data_loader.get_subject_data as-is; only the training split it
    returns is used, so a dataset's held-out test session never enters
    pretraining. Each subject's load is heartbeat-wrapped, since a fresh
    download plus MNE epoching can run silently for minutes.

    Parameters
    ----------
    dataset_name : str
        Key into datasets.DATASET_PRESETS.

    Returns
    -------
    X, y : ndarray
        Pooled, per-subject-normalized training trials and labels across
        every subject in the dataset's preset.
    """
    preset = DATASET_PRESETS[dataset_name]
    X_list, y_list = [], []
    for subject_id in preset.subject_ids:
        X_subject, y_subject, _, _ = _with_heartbeat(lambda: get_subject_data(subject_id, preset))
        X_list.append(X_subject)
        y_list.append(y_subject)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def _eval_accuracy(model, X, y, device, batch_size):
    """Computes plain accuracy for one model on one array-backed split.

    A small private duplicate of train.py's evaluate_arrays/evaluate_loader:
    train.py imports from this module (see build_target_init_state), so
    this module cannot import back from train.py.
    """
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device)).argmax(dim=1).cpu()
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total


def pretrain_joint_trunk(source_datasets, model_name, model_cfg, train_cfg, exclude_dataset=None):
    """Jointly pretrains a shared trunk across multiple source datasets.

    Builds one model per source dataset, rewires them to share trunk
    submodules (see share_trunk_modules), then trains all of them through
    one shared optimizer via round-robin per-dataset mini-batch steps
    (batches from different datasets cannot be concatenated, since their
    channel counts differ). Early-stops on the mean of each dataset's own
    held-out validation accuracy, so a single large dataset does not
    dominate the stopping decision.

    Parameters
    ----------
    source_datasets : list of str
        Dataset names (datasets.DATASET_PRESETS keys) to pool.
    model_name : str
        One of "eegnet", "patch_transformer", "conformer".
    model_cfg : EEGNetConfig, PatchTransformerConfig, or ConformerConfig
        Architecture hyperparameters, shared by every source dataset's
        model in this run.
    train_cfg : TrainConfig
        n_epochs/patience/batch_size/seed/device for the joint loop.
    exclude_dataset : str, optional
        A dataset name to leave out of source_datasets, for a
        leave-one-dataset-out local validation run (see train.py's
        --init-trunk).

    Returns
    -------
    dict
        Filtered shared-trunk state_dict from the epoch with the best mean
        per-dataset validation accuracy.

    Raises
    ------
    ValueError
        If fewer than 2 source datasets remain after exclude_dataset.
    """
    datasets_to_use = [name for name in source_datasets if name != exclude_dataset]
    if len(datasets_to_use) < 2:
        raise ValueError(
            f"pretrain_joint_trunk needs at least 2 source datasets after excluding "
            f"{exclude_dataset!r}, got {datasets_to_use}"
        )

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_cfg.seed)

    splits = {}
    for name in datasets_to_use:
        X, y = load_pretrain_corpus(name)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=train_cfg.seed
        )
        splits[name] = (X_train, y_train, X_val, y_val)

    model_cls = MODEL_REGISTRY[model_name]
    models = [
        model_cls(
            DATASET_PRESETS[name].n_channels, splits[name][0].shape[-1],
            DATASET_PRESETS[name].n_classes, model_cfg,
        ).to(device)
        for name in datasets_to_use
    ]
    trunk_owner = share_trunk_modules(models, model_name)
    optimizer = torch.optim.Adam(
        trunk_and_head_parameters(models, trunk_owner, model_name),
        lr=model_cfg.lr, weight_decay=train_cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    loaders = {
        name: DataLoader(
            TensorDataset(
                torch.tensor(splits[name][0], dtype=torch.float32),
                torch.tensor(splits[name][1], dtype=torch.long),
            ),
            batch_size=train_cfg.batch_size, shuffle=True,
        )
        for name in datasets_to_use
    }

    best_mean_val_acc = -1.0
    best_trunk_state = None
    epochs_no_improve = 0

    for epoch in range(train_cfg.n_epochs):
        for model in models:
            model.train()

        iters = {name: iter(loader) for name, loader in loaders.items()}
        n_steps = max(len(loader) for loader in loaders.values())
        for step in range(n_steps):
            for idx, name in enumerate(datasets_to_use):
                try:
                    xb, yb = next(iters[name])
                except StopIteration:
                    iters[name] = iter(loaders[name])
                    xb, yb = next(iters[name])
                optimizer.zero_grad()
                loss = criterion(models[idx](xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()
            if step % 50 == 0:
                print(f"[pretrain_trunk:{model_name}] epoch {epoch} step {step}/{n_steps}", flush=True)

        val_accs = [
            _eval_accuracy(models[idx], *splits[name][2:], device, train_cfg.batch_size)
            for idx, name in enumerate(datasets_to_use)
        ]
        mean_val_acc = sum(val_accs) / len(val_accs)

        if mean_val_acc > best_mean_val_acc:
            best_mean_val_acc = mean_val_acc
            best_trunk_state = {
                k: v.cpu().clone()
                for k, v in filter_trunk_state_dict(trunk_owner.state_dict(), model_name).items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"[pretrain_trunk:{model_name}] epoch {epoch:>3} mean_val_acc={mean_val_acc:.3f} "
            f"(best={best_mean_val_acc:.3f}) per_dataset={dict(zip(datasets_to_use, val_accs))}"
        )

        if epochs_no_improve >= train_cfg.patience:
            print(f"[pretrain_trunk:{model_name}] early stopping at epoch {epoch}")
            break

    return best_trunk_state


def build_target_init_state(trunk_state, model_name, n_channels, n_times, n_classes, model_cfg):
    """Merges a pretrained shared-trunk state_dict into a freshly
    constructed target model's own state_dict.

    The result is a complete, correctly-shaped state_dict, so train.py's
    existing strict-by-default load_state_dict calls stay strict: a real
    shape/key mismatch (e.g. model_cfg drifted between pretraining and
    fine-tuning) still fails loudly instead of silently loading nothing.

    Parameters
    ----------
    trunk_state : dict
        Output of pretrain_joint_trunk (or filter_trunk_state_dict).
    model_name : str
        One of "eegnet", "patch_transformer", "conformer".
    n_channels, n_times, n_classes : int
        Target dataset's own dimensions.
    model_cfg : EEGNetConfig, PatchTransformerConfig, or ConformerConfig
        Must match what pretrain_joint_trunk used to build the source
        models, or the trunk tensors have the wrong shape and
        load_state_dict raises a size-mismatch error.

    Returns
    -------
    dict
        Complete state_dict: target's own random init, with trunk keys
        overwritten by trunk_state.
    """
    target = MODEL_REGISTRY[model_name](n_channels, n_times, n_classes, model_cfg)
    merged = {k: v.clone() for k, v in target.state_dict().items()}
    merged.update(trunk_state)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument(
        "--source-datasets", nargs="+", required=True,
        help="Dataset names (datasets.DATASET_PRESETS keys) to pool for joint trunk pretraining.",
    )
    parser.add_argument(
        "--exclude-dataset", default=None,
        help="Leave one dataset out of --source-datasets, for a local leave-one-dataset-out "
             "validation run against that dataset via train.py --init-trunk.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_cfg = {
        "eegnet": EEGNetConfig, "patch_transformer": PatchTransformerConfig, "conformer": ConformerConfig,
    }[args.model]()
    train_cfg = TrainConfig(n_epochs=args.epochs, patience=args.patience, seed=args.seed)

    trunk_state = pretrain_joint_trunk(
        args.source_datasets, args.model, model_cfg, train_cfg, exclude_dataset=args.exclude_dataset
    )

    output_dir = OUTPUT_DIR / "trunk_pretrained"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_excl_{args.exclude_dataset}" if args.exclude_dataset is not None else ""
    output_path = output_dir / f"{args.model}_trunk{suffix}.pt"
    torch.save(trunk_state, output_path)
    print(f"[pretrain_trunk] wrote {output_path}")
