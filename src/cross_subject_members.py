"""Cross-subject ensemble members for datasets with a whole-participant split.

Fits each member on pooled participants and writes the per-trial probability
dump the ensemble harness reads, to output/{dataset}/cross_subject/
predictions_{member}.npz. Test probabilities come from a fit on every
training participant (fitting plus validation). Validation probabilities
come from a fit on the fitting participants only, so ensemble selection never
sees validation labels. That fit also scores the test participants
("probs_fit"), for a paired comparison against the refit. The split is
data_loader.get_cross_subject_data.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import get_cross_subject_data, participant_folds
from fbcsp_member import fit_predict_fbcsp
from riemann_member import fit_predict_riemann
from train import compute_matched_epochs, fit, fit_full, fit_on_split, output_dir_for, predict_probs


def _riemann_probs(data, cfg: Config):
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    subject_all = np.concatenate([data["subject_fit"], data["subject_val"]])
    recenter = cfg.data.alignment == "riemannian"
    n_trials = cfg.data.alignment_n_trials
    subject_fit = data["subject_fit"] if recenter else None
    return (
        fit_predict_riemann(
            data["X_fit"], data["y_fit"], data["X_val"],
            subject_fit, data["subject_val"] if recenter else None, n_trials),
        fit_predict_riemann(
            X_all, y_all, data["X_test"],
            subject_all if recenter else None, data["subject_test"] if recenter else None, n_trials),
        fit_predict_riemann(
            data["X_fit"], data["y_fit"], data["X_test"],
            subject_fit, data["subject_test"] if recenter else None, n_trials),
    )


def _fbcsp_probs(data, cfg: Config):
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    return (
        fit_predict_fbcsp(data["X_fit"], data["y_fit"], data["X_val"], cfg),
        fit_predict_fbcsp(X_all, y_all, data["X_test"], cfg),
        fit_predict_fbcsp(data["X_fit"], data["y_fit"], data["X_test"], cfg),
    )


def _save_final_weights(model, cfg: Config, model_name, n_epochs, weights_path):
    """Writes the final refit's state dict in the layout of the k-fold weight files."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dicts": [state], "best_epochs": [n_epochs], "model_cfg": dataclasses.asdict(getattr(cfg, model_name))}, weights_path)


def _neural_probs(data, cfg: Config, model_name, ensemble, n_epochs=None, weights_path=None):
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    batch_size = cfg.train.batch_size

    if not ensemble:
        if n_epochs is None:
            # The epoch count comes from every training participant's windows, so no
            # participant is set aside. fit() holds out a fifth of them for stopping.
            _, _, best_epoch = fit(X_all, y_all, model_name, cfg)
            n_train = len(X_all) - round(len(X_all) * 0.2)
            n_epochs = compute_matched_epochs([(best_epoch, n_train)], len(X_all), batch_size)
        final = fit_full(X_all, y_all, model_name, cfg, n_epochs)
        _save_final_weights(final, cfg, model_name, n_epochs, weights_path)
        return None, predict_probs(final, data["X_test"], device, batch_size), None

    # fit() holds out a fifth of the fitting trials for early stopping, so the
    # epoch budget is matched on the trials it actually trained on.
    model, _, best_epoch = fit(data["X_fit"], data["y_fit"], model_name, cfg)
    val_probs = predict_probs(model, data["X_val"], device, batch_size)
    probs_fit = predict_probs(model, data["X_test"], device, batch_size)
    n_fit_train = len(data["X_fit"]) - round(len(data["X_fit"]) * 0.2)
    n_epochs = compute_matched_epochs([(best_epoch, n_fit_train)], len(X_all), batch_size)

    final = fit_full(X_all, y_all, model_name, cfg, n_epochs)
    _save_final_weights(final, cfg, model_name, n_epochs, weights_path)
    return val_probs, predict_probs(final, data["X_test"], device, batch_size), probs_fit


def dump_early_stopped_all(member, data, cfg: Config, fold=None, tag=""):
    """Fits one neural member on fitting plus validation participants with early stopping and writes its dump.

    The counterpart of the refit that keeps early stopping: fit() holds out a
    fifth of the pooled trials, stratified by class, to pick the stopping epoch,
    and the model at that epoch predicts the test participants.

    Parameters
    ----------
    member : str
        "eegnet", "patch_transformer" or "conformer".
    data : dict
        As returned by data_loader.get_cross_subject_data.
    cfg : Config
        Full project configuration.
    fold : int, optional
        Index of the test fold, appended to the dump's file name as "_f{fold}".
    tag : str
        Extra file-name text placed before the fold index.

    Returns
    -------
    dict
        Arrays "subject", "y", "session", "probs" for the test rows and
        "best_epoch" (0-indexed).

    Raises
    ------
    ValueError
        If member is not a neural member.
    """
    if member not in ("eegnet", "patch_transformer", "conformer"):
        raise ValueError(f"Early stopping on pooled participants applies to neural members only, not '{member}'.")
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model, _, best_epoch = fit(X_all, y_all, member, cfg)
    dump = {
        "subject": data["subject_test"], "y": data["y_test"], "session": data["session_test"],
        "probs": predict_probs(model, data["X_test"], device, cfg.train.batch_size), "best_epoch": np.array(best_epoch),
    }
    out_dir = output_dir_for(cfg) / "cross_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"predictions_{member}{alignment_suffix(cfg)}_esall{tag}{'' if fold is None else f'_f{fold}'}.npz", **dump)
    print(f"[{member}] early-stopped on pooled participants: test acc={(dump['probs'].argmax(axis=1) == data['y_test']).mean():.3f}, "
          f"best epoch={best_epoch}", flush=True)
    return dump


def dump_kfold_ensemble(member, data, cfg: Config, n_inner, fold=None, tag=""):
    """Fits a participant-grouped k-fold ensemble of one neural member and writes its dump.

    The fitting and validation participants are shuffled with
    cfg.data.val_subject_seed and split into n_inner groups. Model j trains on
    the other groups and early stops on group j, so every participant is held
    out once and no epoch count is fixed in advance. The test probabilities are
    the mean over the n_inner models.

    Parameters
    ----------
    member : str
        "eegnet", "patch_transformer" or "conformer".
    data : dict
        As returned by data_loader.get_cross_subject_data.
    cfg : Config
        Full project configuration.
    n_inner : int
        Number of inner participant groups, hence models.
    fold : int, optional
        Index of the test fold, appended to the dump's file name as "_f{fold}".
    tag : str
        Extra file-name text placed before the fold index.

    Returns
    -------
    dict
        Arrays "subject", "y", "session" and "probs" (ensemble mean) for the
        test rows, "probs_models" of shape (n_inner, n_rows, n_classes) and
        "best_epochs" and "held_out_acc" of shape (n_inner,).

    Notes
    -----
    The n_inner models' state dicts are saved next to the dump as
    "weights_{member}{suffix}_kfold{n_inner}{tag}{fold}.pt", a dict with
    "state_dicts" (CPU tensors), "best_epochs" and "model_cfg" (the member's
    architecture config as a dict).

    Raises
    ------
    ValueError
        If member is not a neural member.
    """
    if member not in ("eegnet", "patch_transformer", "conformer"):
        raise ValueError(f"The k-fold ensemble applies to neural members only, not '{member}'.")
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    subject_all = np.concatenate([data["subject_fit"], data["subject_val"]])
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    batch_size = cfg.train.batch_size

    shuffled = np.random.default_rng(cfg.data.val_subject_seed).permutation(np.unique(subject_all))
    probs_models, best_epochs, held_out_acc, state_dicts = [], [], [], []
    for j, held_out in enumerate(np.array_split(shuffled, n_inner)):
        mask = np.isin(subject_all, held_out)
        model, acc, best_epoch = fit_on_split(
            X_all[~mask], y_all[~mask], X_all[mask], y_all[mask], member, cfg, seed=cfg.train.seed + j)
        probs_models.append(predict_probs(model, data["X_test"], device, batch_size))
        state_dicts.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        best_epochs.append(best_epoch)
        held_out_acc.append(acc)
        print(f"[{member}] inner fold {j}: held out {sorted(int(s) for s in held_out)}, "
              f"held-out acc={acc:.3f}, best epoch={best_epoch}", flush=True)

    probs_models = np.stack(probs_models)
    dump = {
        "subject": data["subject_test"], "y": data["y_test"], "session": data["session_test"],
        "probs": probs_models.mean(axis=0), "probs_models": probs_models,
        "best_epochs": np.array(best_epochs), "held_out_acc": np.array(held_out_acc),
    }
    out_dir = output_dir_for(cfg) / "cross_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{member}{alignment_suffix(cfg)}_kfold{n_inner}{tag}{'' if fold is None else f'_f{fold}'}"
    np.savez(out_dir / f"predictions_{stem}.npz", **dump)
    torch.save({"state_dicts": state_dicts, "best_epochs": best_epochs, "model_cfg": dataclasses.asdict(getattr(cfg, member))}, out_dir / f"weights_{stem}.pt")
    print(f"[{member}] k-fold ensemble test acc={(dump['probs'].argmax(axis=1) == data['y_test']).mean():.3f}", flush=True)
    return dump


MEMBERS = {
    "riemann": lambda data, cfg, ensemble, n_epochs, weights_path: _riemann_probs(data, cfg),
    "fbcsp": lambda data, cfg, ensemble, n_epochs, weights_path: _fbcsp_probs(data, cfg),
    "eegnet": lambda data, cfg, ensemble, n_epochs, weights_path: _neural_probs(data, cfg, "eegnet", ensemble, n_epochs, weights_path),
    "patch_transformer": lambda data, cfg, ensemble, n_epochs, weights_path: _neural_probs(data, cfg, "patch_transformer", ensemble, n_epochs, weights_path),
    "conformer": lambda data, cfg, ensemble, n_epochs, weights_path: _neural_probs(data, cfg, "conformer", ensemble, n_epochs, weights_path),
}


def alignment_suffix(cfg: Config):
    """File-name suffix of a dump, empty for the unaligned setting."""
    suffix = {"none": "", "euclidean": "_ea", "riemannian": "_ra", "causal": "_ca"}[cfg.data.alignment]
    if suffix and cfg.data.alignment_n_trials:
        suffix += f"_n{cfg.data.alignment_n_trials}"
    if cfg.data.test_alignment:
        suffix += f"_test{cfg.data.test_alignment}"
    return suffix


def dump_cross_subject_member(member, data, cfg: Config, fold=None, ensemble=False, tag="", n_epochs=None):
    """Fits one member and writes its probability dump.

    Parameters
    ----------
    member : str
        Key of MEMBERS.
    data : dict
        As returned by data_loader.get_cross_subject_data.
    cfg : Config
        Full project configuration.
    fold : int, optional
        Index of the test fold (see data_loader.participant_folds), appended to
        the dump's file name as "_f{fold}". None for the configured split.
    ensemble : bool
        True keeps the ensemble harness layout: the epoch count of a neural
        member comes from a fit on the fitting participants only, and the dump
        carries validation probabilities and "probs_fit". False takes the epoch
        count from every training participant's windows and writes the test
        arrays only.
    tag : str
        Extra file-name text placed before the fold index.
    n_epochs : int, optional
        Fixed epoch count of a neural member's final fit, skipping the
        early-stopped fit that otherwise sets it. Ignored with ensemble.

    Returns
    -------
    dict
        Arrays "subject", "y", "probs" (trained on fitting plus validation) and
        "session" for the test rows. With ensemble, also "probs_fit" (trained on
        the fitting participants only) and "val_subject", "val_y", "val_probs"
        and "val_session" for the validation rows.

    Notes
    -----
    A neural member's final refit is saved next to the dump as
    "weights_{member}{suffix}{tag}{fold}.pt", in the layout of the k-fold
    weight files with a single state dict.
    """
    if cfg.data.alignment == "riemannian" and member != "riemann":
        raise ValueError("Riemannian alignment applies to the riemann member only.")
    out_dir = output_dir_for(cfg) / "cross_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{member}{alignment_suffix(cfg)}{tag}{'' if fold is None else f'_f{fold}'}"
    val_probs, probs, probs_fit = MEMBERS[member](data, cfg, ensemble, n_epochs, out_dir / f"weights_{stem}.pt")
    dump = {"subject": data["subject_test"], "y": data["y_test"], "probs": probs, "session": data["session_test"]}
    if val_probs is not None:
        dump.update({
            "probs_fit": probs_fit,
            "val_subject": data["subject_val"], "val_y": data["y_val"], "val_probs": val_probs, "val_session": data["session_val"],
        })
    np.savez(out_dir / f"predictions_{stem}.npz", **dump)
    val_report = "" if val_probs is None else f", val acc={(val_probs.argmax(axis=1) == data['y_val']).mean():.3f}"
    print(f"[{member}] test acc={(probs.argmax(axis=1) == data['y_test']).mean():.3f}{val_report}", flush=True)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members", nargs="+", choices=list(MEMBERS))
    parser.add_argument("--n-folds", type=int, default=0, help="Cross-validate over this many disjoint test-participant folds instead of the preset's test_subject_ids. 0 runs the preset split.")
    parser.add_argument("--n-jobs", type=int, default=6, help="Workers epoching participants missing from the cache.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Persistent directory for the output and epoch caches, shared across sessions.")
    parser.add_argument("--kfold", type=int, default=0, help="Write a participant-grouped k-fold ensemble dump with this many inner folds instead of the single refit dump. Neural members only.")
    parser.add_argument("--early-stop-all", action="store_true", help="Write an early-stopped fit on fitting plus validation participants instead of the single refit dump. Neural members only.")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble harness layout of the single dump: neural epoch counts come from the fitting participants only and the dump carries validation probabilities.")
    parser.add_argument("--exclude-subjects", type=int, nargs="*", default=[], help="Participants left out of the pool entirely, neither trained on nor tested.")
    parser.add_argument("--epochs", type=int, default=None, help="Fixed epoch count of the neural members' final fit, skipping the early-stopped fit that sets it. Ignored with --ensemble.")
    parser.add_argument("--tag", default="", help="Extra dump file-name text placed before the fold index.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    cfg = resolve_config(args)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, subject_ids=[s for s in cfg.data.subject_ids if s not in args.exclude_subjects]))
    cache_dir = args.cache_dir / "epochs_cache" if args.cache_dir else None
    folds = participant_folds(cfg.data, args.n_folds) if args.n_folds else [cfg.data.test_subject_ids]
    for k, test_ids in enumerate(folds):
        fold_cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, test_subject_ids=test_ids))
        data = get_cross_subject_data(fold_cfg.data, n_jobs=args.n_jobs, cache_dir=cache_dir)
        # A fixed epoch count in the single refit trains on fit plus validation together, with no stopping.
        fixed_refit = args.epochs is not None and not (args.ensemble or args.kfold or args.early_stop_all)
        parts = (("train (fit + val, fixed epochs)", ("fit", "val")), ("test", ("test",))) if fixed_refit else (("fit", ("fit",)), ("val", ("val",)), ("test", ("test",)))
        print(" ".join(
            f"{label} participants {sorted(int(s) for p in keys for s in np.unique(data['subject_' + p]))} ({sum(len(data['y_' + p]) for p in keys)} windows)"
            for label, keys in parts), flush=True)
        for name in args.members:
            if args.early_stop_all:
                dump_early_stopped_all(name, data, fold_cfg, fold=k if args.n_folds else None, tag=args.tag)
            elif args.kfold:
                dump_kfold_ensemble(name, data, fold_cfg, args.kfold, fold=k if args.n_folds else None, tag=args.tag)
            else:
                dump_cross_subject_member(name, data, fold_cfg, fold=k if args.n_folds else None, ensemble=args.ensemble, tag=args.tag, n_epochs=args.epochs)
