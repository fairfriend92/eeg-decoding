"""Per-subject training loop for EEGNet and EEG Conformer models.

Trains and checkpoints only -- never touches the test session (session_E).

n_folds=1 (default): single 80/20 split per subject, one checkpoint.
n_folds>=2: stratified k-fold cross-validation per subject, one checkpoint
per fold. Both also retrain once on 100% of session_T (no held-out
validation slice), using a step-matched epoch count derived from the
split(s) above (see compute_matched_epochs), saved as a separate "final"
checkpoint. Writes output/{model}_checkpoints.json, a manifest of
checkpoint paths and validation accuracies.

Must be run with this file's own directory (src/) as the script's
directory, e.g. `python train.py` from within src/, or `python src/train.py`
from the repository root. Do not invoke via `python -m src.train`, which
changes the Python path and breaks the flat `from config import ...` style
imports used throughout src/.
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import N_CHANNELS, N_CLASSES, OUTPUT_DIR, SUBJECT_IDS, Config
from csp_init import csp_initialize
from data_loader import crop_trials, get_subject_data
from models.conformer import EEGConformer
from models.eegnet import EEGNet

MODEL_REGISTRY = {
    "eegnet": EEGNet,
    "conformer": EEGConformer,
}


def make_loaders(X_train, y_train, X_val, y_val, batch_size, generator=None):
    """Wraps train/val arrays into DataLoaders.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : ndarray
        Split arrays for a single subject.
    batch_size : int
        Batch size for both loaders.
    generator : torch.Generator, optional
        Governs the training loader's shuffle order. Passed explicitly
        (rather than relying on global torch RNG state) so shuffle order
        is reproducible independent of what else has consumed random
        numbers beforehand.

    Returns
    -------
    DataLoader, DataLoader
        Training and validation loaders.
    """
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate_loader(model, loader, device):
    """Computes accuracy and Cohen's kappa over a DataLoader.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate; switched to eval mode internally.
    loader : DataLoader
        Data to evaluate on.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    float, float
        Accuracy and Cohen's kappa.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.extend(model(xb).argmax(dim=1).cpu().numpy())
            targets.extend(yb.numpy())
    return accuracy_score(targets, preds), cohen_kappa_score(targets, preds)


def evaluate_arrays(model, X, y, device, batch_size):
    """Computes accuracy and Cohen's kappa directly from arrays.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    X, y : ndarray
        Evaluation data and labels.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float, float
        Accuracy and Cohen's kappa.
    """
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return evaluate_loader(model, loader, device)


def predict_probs(model, X, device, batch_size):
    """Computes softmax class probabilities for X via batched inference.

    Parameters
    ----------
    model : nn.Module
        Trained model; switched to eval mode internally.
    X : ndarray
        Data to predict on.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    ndarray, shape (len(X), n_classes)
        Class probabilities.
    """
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            all_probs.append(torch.softmax(model(xb), dim=1).cpu())
    return torch.cat(all_probs, dim=0).numpy()


def predict_probs_cropped(model, X, cropped_cfg, device, batch_size):
    """Computes per-trial softmax probabilities via crop-averaging.

    Slices each trial into crops (matching how the model was trained),
    predicts on each crop individually, then averages a trial's crops'
    probabilities back into one probability vector per trial.

    Parameters
    ----------
    model : nn.Module
        Model trained on crops of this same crop_size.
    X : ndarray
        Whole (uncropped) trials.
    cropped_cfg : CroppedConfig
        crop_size (must match training) and eval_crop_stride (denser
        evaluation-time stride) to slice X with.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    ndarray, shape (len(X), n_classes)
        Crop-averaged class probabilities, one row per trial.
    """
    # Labels are required by crop_trials' signature but unused here.
    dummy_labels = np.zeros(len(X), dtype=int)
    X_crops, _, trial_index = crop_trials(X, dummy_labels, cropped_cfg.crop_size, cropped_cfg.eval_crop_stride)
    crop_probs = predict_probs(model, X_crops, device, batch_size)

    n_trials  = len(X)
    n_classes = crop_probs.shape[1]
    avg_probs = np.zeros((n_trials, n_classes))
    for trial_idx in range(n_trials):
        avg_probs[trial_idx] = crop_probs[trial_index == trial_idx].mean(axis=0)

    return avg_probs


def evaluate_ensemble(models, X, y, device, batch_size):
    """Scores the average of several models' softmax predictions.

    Parameters
    ----------
    models : list of nn.Module
        Trained models (e.g. one per cross-validation fold) to ensemble.
    X, y : ndarray
        Evaluation data and labels.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float, float
        Accuracy and Cohen's kappa of the ensembled predictions.
    """
    avg_probs = sum(predict_probs(model, X, device, batch_size) for model in models) / len(models)
    preds     = avg_probs.argmax(axis=1)
    return accuracy_score(y, preds), cohen_kappa_score(y, preds)


def evaluate_cropped(model, X, y, cropped_cfg, device, batch_size):
    """Scores a model on whole trials via crop-averaged predictions.

    Same averaging principle as evaluate_ensemble, but over a trial's own
    crops rather than over several models.

    Parameters
    ----------
    model : nn.Module
        Model trained on crops of this same crop_size.
    X, y : ndarray
        Whole (uncropped) evaluation trials and their labels.
    cropped_cfg : CroppedConfig
        crop_size/crop_stride to slice X with.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float, float
        Accuracy and Cohen's kappa of the crop-averaged predictions.
    """
    avg_probs = predict_probs_cropped(model, X, cropped_cfg, device, batch_size)
    preds     = avg_probs.argmax(axis=1)
    return accuracy_score(y, preds), cohen_kappa_score(y, preds)


def evaluate_ensemble_cropped(models, X, y, cropped_cfg, device, batch_size):
    """Scores the average of several crop-trained models' crop-averaged
    predictions.

    Combines both averaging steps: each model's prediction on a trial is
    itself a crop-average (see predict_probs_cropped), and those per-model
    per-trial probabilities are then averaged across models.

    Parameters
    ----------
    models : list of nn.Module
        Trained models (e.g. one per cross-validation fold) to ensemble,
        each trained on crops of this same crop_size.
    X, y : ndarray
        Whole (uncropped) evaluation trials and their labels.
    cropped_cfg : CroppedConfig
        crop_size/crop_stride to slice X with.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float, float
        Accuracy and Cohen's kappa of the ensembled, crop-averaged
        predictions.
    """
    avg_probs = sum(
        predict_probs_cropped(model, X, cropped_cfg, device, batch_size) for model in models
    ) / len(models)
    preds = avg_probs.argmax(axis=1)
    return accuracy_score(y, preds), cohen_kappa_score(y, preds)



def fit_on_split(X_train, y_train, X_val, y_val, model_name, cfg: Config, subject_id=None, seed=None, cropped_cfg=None, csp_init=False):
    """Trains a single model on an explicit train/val split, with early
    stopping.

    Core training loop, reused by both `fit` (single 80/20 split) and any
    cross-validation caller that supplies its own fold split.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : ndarray
        Already-split training and validation data (whole trials).
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.
    subject_id : int, optional
        Used only to label progress messages.
    seed : int, optional
        Seeds model init and the training loader's shuffle order. Defaults
        to `cfg.train.seed` (or `cfg.train.seed + subject_id` if given),
        matching `fit`'s behavior; a cross-validation caller should pass a
        distinct seed per fold.
    cropped_cfg : CroppedConfig, optional
        If given, X_train/y_train are sliced into crops before training
        (the model is built with crop_size as its input length), and
        validation is scored via crop-averaged predictions on the whole
        X_val trials (see evaluate_cropped) rather than evaluate_loader.
    csp_init : bool
        EEGNet only. If True, initializes the depthwise spatial_conv
        weights via per-temporal-band CSP (see csp_init.py) instead of
        the default random init, before training starts.

    Returns
    -------
    model : nn.Module
        Trained model, loaded with its best-validation-accuracy weights.
    best_val_acc : float
        Best validation accuracy reached during training.
    best_epoch : int
        0-indexed epoch at which best_val_acc was reached. The number of
        completed epochs contributing to the retained weights is
        best_epoch + 1.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    label  = f"subject {subject_id}" if subject_id is not None else "subject"

    if seed is None:
        seed = cfg.train.seed if subject_id is None else cfg.train.seed + subject_id
    torch.manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed)

    if cropped_cfg is not None:
        X_train_input, y_train_input, _ = crop_trials(X_train, y_train, cropped_cfg.crop_size, cropped_cfg.crop_stride)
    else:
        X_train_input, y_train_input = X_train, y_train

    train_loader, val_loader = make_loaders(
        X_train_input, y_train_input, X_val, y_val, cfg.train.batch_size, generator=loader_generator
    )

    n_times   = X_train_input.shape[-1]
    model_cls = MODEL_REGISTRY[model_name]
    model_cfg = getattr(cfg, model_name)
    model     = model_cls(N_CHANNELS, n_times, N_CLASSES, model_cfg).to(device)

    if csp_init:
        if model_name != "eegnet":
            raise ValueError(f"csp_init is only supported for eegnet, got model_name={model_name!r}")
        print(f"[{model_name}] {label}: initializing spatial_conv via per-band CSP")
        csp_initialize(model, X_train_input, y_train_input, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(
        f"[{model_name}] {label}: training on {len(X_train_input)} "
        f"{'crops' if cropped_cfg is not None else 'trials'}, "
        f"validating on {len(X_val)} trials, device={device}"
    )

    best_val_acc       = -1.0
    best_state         = None
    best_epoch         = 0
    epochs_no_improve  = 0

    for epoch in range(cfg.train.n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if cropped_cfg is not None:
            val_acc, _ = evaluate_cropped(model, X_val, y_val, cropped_cfg, device, cfg.train.batch_size)
        else:
            val_acc, _ = evaluate_loader(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch        = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epochs_no_improve >= cfg.train.patience:
            print(
                f"[{model_name}] {label}: epoch {epoch:>3} - "
                f"val_acc={val_acc:.3f} (best={best_val_acc:.3f}, "
                f"no_improve={epochs_no_improve}/{cfg.train.patience})"
            )

        if epochs_no_improve >= cfg.train.patience:
            print(f"[{model_name}] {label}: early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc, best_epoch


def fit(X_train_full, y_train_full, model_name, cfg: Config, subject_id=None, cropped_cfg=None, csp_init=False):
    """Trains a single model on already-loaded data, with early stopping.

    Splits the given training data into a single 80/20 train/validation
    split for early stopping only; takes no test data, so callers
    (including a hyperparameter sweep) cannot leak test-set information
    into model selection even indirectly.

    Parameters
    ----------
    X_train_full, y_train_full : ndarray
        A subject's full session_T trials and labels, as returned by
        `get_subject_data`.
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.
    subject_id : int, optional
        Used only to label progress messages.
    cropped_cfg : CroppedConfig, optional
        If given, enables crop-based training/validation -- see
        fit_on_split.
    csp_init : bool
        EEGNet only. See fit_on_split.

    Returns
    -------
    model : nn.Module
        Trained model, loaded with its best-validation-accuracy weights.
    best_val_acc : float
        Best validation accuracy reached during training.
    best_epoch : int
        0-indexed epoch at which best_val_acc was reached.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=cfg.train.seed,
    )
    return fit_on_split(
        X_train, y_train, X_val, y_val, model_name, cfg,
        subject_id=subject_id, cropped_cfg=cropped_cfg, csp_init=csp_init,
    )


def compute_matched_epochs(epoch_train_pairs, n_train_full, batch_size):
    """Converts one or more (best_epoch, train_size) observations into a
    step-matched epoch count for training on n_train_full examples.

    An epoch means a different amount of training depending on dataset
    size (more examples -> more gradient steps per epoch), so a raw
    epoch count from a smaller split does not transfer directly to a
    larger one. This matches on total gradient steps instead.

    Parameters
    ----------
    epoch_train_pairs : list of (int, int)
        (best_epoch, train_size) pairs from one or more validation runs.
        best_epoch is 0-indexed; the number of completed epochs
        contributing to the retained weights is best_epoch + 1.
    n_train_full : int
        Number of training examples the final model will be trained on.
    batch_size : int
        Training batch size (assumed constant across runs).

    Returns
    -------
    int
        Epoch count for the full-data run, at least 1.
    """
    total_steps = []
    for best_epoch, train_size in epoch_train_pairs:
        steps_per_epoch     = math.ceil(train_size / batch_size)
        n_completed_epochs  = best_epoch + 1
        total_steps.append(n_completed_epochs * steps_per_epoch)

    mean_steps            = sum(total_steps) / len(total_steps)
    steps_per_epoch_full  = math.ceil(n_train_full / batch_size)
    return max(1, round(mean_steps / steps_per_epoch_full))


def fit_full(X, y, model_name, cfg: Config, n_epochs, subject_id=None, seed=None, cropped_cfg=None, csp_init=False):
    """Trains a model on the full dataset for a fixed number of epochs,
    with no validation split and no early stopping.

    Intended for a final retrain once hyperparameters and an epoch budget
    have already been chosen via cross-validation (see
    compute_matched_epochs) -- uses every available training example
    instead of holding a slice out for early stopping.

    Parameters
    ----------
    X, y : ndarray
        Full training data (e.g. all of a subject's session_T).
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.
    n_epochs : int
        Fixed number of epochs to train for.
    subject_id : int, optional
        Used only to label progress messages.
    seed : int, optional
        Seeds model init and the training loader's shuffle order. Defaults
        to `cfg.train.seed` (or `cfg.train.seed + subject_id` if given).
    cropped_cfg : CroppedConfig, optional
        If given, X/y are sliced into crops before training (the model is
        built with crop_size as its input length).
    csp_init : bool
        EEGNet only. See fit_on_split.

    Returns
    -------
    nn.Module
        The trained model, after exactly n_epochs of training.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    label  = f"subject {subject_id}" if subject_id is not None else "subject"

    if seed is None:
        seed = cfg.train.seed if subject_id is None else cfg.train.seed + subject_id
    torch.manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed)

    if cropped_cfg is not None:
        X_input, y_input, _ = crop_trials(X, y, cropped_cfg.crop_size, cropped_cfg.crop_stride)
    else:
        X_input, y_input = X, y

    ds     = TensorDataset(torch.tensor(X_input, dtype=torch.float32), torch.tensor(y_input, dtype=torch.long))
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, generator=loader_generator)

    n_times   = X_input.shape[-1]
    model_cls = MODEL_REGISTRY[model_name]
    model_cfg = getattr(cfg, model_name)
    model     = model_cls(N_CHANNELS, n_times, N_CLASSES, model_cfg).to(device)

    if csp_init:
        if model_name != "eegnet":
            raise ValueError(f"csp_init is only supported for eegnet, got model_name={model_name!r}")
        print(f"[{model_name}] {label}: initializing spatial_conv via per-band CSP")
        csp_initialize(model, X_input, y_input, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss()

    unit = "crops" if cropped_cfg is not None else "trials"
    print(f"[{model_name}] {label}: final fit on {len(X_input)} {unit} for {n_epochs} epochs (no validation split)")

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"[{model_name}] {label}: final fit epoch {epoch:>3}/{n_epochs}")

    return model


def train_one_subject(subject_id, model_name, cfg: Config, n_folds=1, cropped_cfg=None, csp_init=False):
    """Loads a subject's training data and fits model(s), saving checkpoints.

    Does not touch test data (session_E) -- this function only trains and
    checkpoints; see evaluate.py for scoring saved checkpoints on the test
    set, kept as a separate step.

    n_folds=1: single 80/20 split, one model, one checkpoint -- unchanged
    from the original behavior.

    n_folds>=2: stratified k-fold cross-validation over session_T, one
    model per fold, each saved separately.

    Both branches also retrain once on the full session_T (no held-out
    validation slice) for a step-matched epoch count derived from the
    split(s) above -- see compute_matched_epochs -- saved as a separate
    "final" checkpoint.

    Parameters
    ----------
    subject_id : int
        Subject identifier (1-9).
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.
    n_folds : int
        1 for a single split; >=2 for that many cross-validation folds.
    cropped_cfg : CroppedConfig, optional
        If given, enables crop-based training/validation throughout (see
        fit_on_split) -- recorded in the returned manifest entry so
        evaluate.py knows to score these checkpoints via crop-averaging.
    csp_init : bool
        EEGNet only. See fit_on_split.

    Returns
    -------
    dict
        n_folds=1: subject id, val accuracy, checkpoint path, final-fit
        epoch count, final checkpoint path, cropped config (or None).
        n_folds>=2: subject id, n_folds, per-fold val accuracy +
        checkpoint path, final-fit epoch count, final checkpoint path,
        cropped config (or None).
    """
    print(f"[{model_name}] subject {subject_id}: loading and epoching EEG data...")
    X_train_full, y_train_full, _, _ = get_subject_data(subject_id, cfg.data)
    n_train_full = len(X_train_full)

    cropped_info = (
        {"crop_size": cropped_cfg.crop_size, "crop_stride": cropped_cfg.crop_stride}
        if cropped_cfg is not None else None
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if n_folds == 1:
        model, best_val_acc, best_epoch = fit(
            X_train_full, y_train_full, model_name, cfg, subject_id=subject_id,
            cropped_cfg=cropped_cfg, csp_init=csp_init,
        )

        ckpt_path = OUTPUT_DIR / f"{model_name}_subject{subject_id}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[{model_name}] subject {subject_id}: done - val_acc={best_val_acc:.3f}")

        # fit()'s internal split holds out 20% for validation -- the split
        # size actually used, for step-matching against the full dataset.
        split_train_size = n_train_full - round(n_train_full * 0.2)
        final_n_epochs = compute_matched_epochs(
            [(best_epoch, split_train_size)], n_train_full, cfg.train.batch_size
        )
        final_model = fit_full(
            X_train_full, y_train_full, model_name, cfg, final_n_epochs,
            subject_id=subject_id, cropped_cfg=cropped_cfg, csp_init=csp_init,
        )
        final_ckpt_path = OUTPUT_DIR / f"{model_name}_subject{subject_id}_final.pt"
        torch.save(final_model.state_dict(), final_ckpt_path)

        return {
            "subject":          subject_id,
            "model":            model_name,
            "n_folds":          1,
            "val_accuracy":     best_val_acc,
            "checkpoint":       str(ckpt_path),
            "final_n_epochs":   final_n_epochs,
            "final_checkpoint": str(final_ckpt_path),
            "cropped":          cropped_info,
        }

    base_seed = cfg.train.seed + subject_id
    splitter  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)

    fold_results      = []
    epoch_train_pairs = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        model, val_acc, best_epoch = fit_on_split(
            X_train_full[train_idx], y_train_full[train_idx],
            X_train_full[val_idx], y_train_full[val_idx],
            model_name, cfg, subject_id=subject_id, seed=base_seed + fold_idx,
            cropped_cfg=cropped_cfg, csp_init=csp_init,
        )

        ckpt_path = OUTPUT_DIR / f"{model_name}_subject{subject_id}_fold{fold_idx}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[{model_name}] subject {subject_id} fold {fold_idx}: val_acc={val_acc:.3f}")

        fold_results.append({
            "fold":         fold_idx,
            "val_accuracy": val_acc,
            "checkpoint":   str(ckpt_path),
        })
        epoch_train_pairs.append((best_epoch, len(train_idx)))

    print(f"[{model_name}] subject {subject_id}: done training {n_folds} folds")

    final_n_epochs = compute_matched_epochs(epoch_train_pairs, n_train_full, cfg.train.batch_size)
    final_model = fit_full(
        X_train_full, y_train_full, model_name, cfg, final_n_epochs,
        subject_id=subject_id, cropped_cfg=cropped_cfg, csp_init=csp_init,
    )
    final_ckpt_path = OUTPUT_DIR / f"{model_name}_subject{subject_id}_final.pt"
    torch.save(final_model.state_dict(), final_ckpt_path)

    return {
        "subject":          subject_id,
        "model":            model_name,
        "n_folds":          n_folds,
        "fold_results":     fold_results,
        "final_n_epochs":   final_n_epochs,
        "final_checkpoint": str(final_ckpt_path),
        "cropped":          cropped_info,
    }


def run_all_subjects(model_name, cfg: Config = None, n_folds=1, cropped_cfg=None, csp_init=False):
    """Trains one model (or one per fold) per subject; writes a checkpoint
    manifest to disk.

    Parameters
    ----------
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config, optional
        Full project configuration; defaults to `Config()`.
    n_folds : int
        1 for a single 80/20 split per subject (default, matches prior
        behavior); >=2 for that many cross-validation folds per subject.
    cropped_cfg : CroppedConfig, optional
        If given, enables crop-based training/validation for every
        subject (see fit_on_split).
    csp_init : bool
        EEGNet only. See fit_on_split.

    Returns
    -------
    list of dict
        Per-subject checkpoint manifest entries, in subject order. See
        evaluate.py to score these checkpoints on the test set.
    """
    cfg = cfg or Config()
    print(
        f"Training {model_name} for {len(SUBJECT_IDS)} subjects "
        f"(n_folds={n_folds}, cropped={cropped_cfg is not None}, csp_init={csp_init}, "
        f"within-subject, official session_T/session_E split)"
    )
    results = [
        train_one_subject(sid, model_name, cfg, n_folds=n_folds, cropped_cfg=cropped_cfg, csp_init=csp_init)
        for sid in SUBJECT_IDS
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / f"{model_name}_checkpoints.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{model_name}] checkpoint manifest written to {manifest_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument(
        "--n-folds", type=int, default=1,
        help="1 for a single 80/20 split per subject (default); >=2 for that many "
             "stratified CV folds per subject. Writes checkpoints only -- run evaluate.py next.",
    )
    parser.add_argument(
        "--cropped", action="store_true",
        help="Enable crop-based training/validation (see CroppedConfig in config.py). Off by default.",
    )
    parser.add_argument(
        "--csp-init", action="store_true",
        help="EEGNet only: initialize the depthwise spatial_conv via per-temporal-band CSP "
             "(see csp_init.py) instead of random init. Off by default.",
    )
    args = parser.parse_args()

    if args.csp_init and args.model != "eegnet":
        parser.error("--csp-init is only supported for the eegnet model")

    cropped_cfg = Config().cropped if args.cropped else None
    run_all_subjects(args.model, n_folds=args.n_folds, cropped_cfg=cropped_cfg, csp_init=args.csp_init)
