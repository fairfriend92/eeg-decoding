"""Runs one of our models, or a NeuralBench-native baseline, through
NeuralBench's own harness.

NeuralBench (https://github.com/facebookresearch/neuroai/tree/main/neuralbench-repo)
takes an already-built torch.nn.Module via neuralbench.check_model /
neuralbench.evaluate_model, no repo-side registration needed. This exists
because our own dreyer2023 accuracy numbers use different preprocessing and
aren't comparable to what the NeurIPS 2026 EEG/EMG Foundation Challenge will
actually score.

Builds a freshly initialized model, not one of our trained checkpoints: a
checkpoint's classifier layer is shape-locked to our own preprocessing's
(n_channels, n_times), which NeuralBench's own preprocessing does not
reproduce (see the Lazy*Adapter classes below). `--pretrained-trunk`
partially relaxes this: it warm-starts only the channel-count-independent
trunk layers (see src/pretrain_trunk.py) from a checkpoint pretrained
across other datasets, leaving spatial_conv/classifier randomly
initialized for whatever shape NeuralBench's own preprocessing reveals.

`--model reve` (or any other NeuralBench-registered baseline name) skips
check_model/evaluate_model, which only accept an already-built nn.Module,
and instead calls build_experiment_configs/BenchmarkAggregator directly.
NeuralBench's own CLI couples one --debug flag to both config content
(reduced epochs, subsampled data) and execution mode (local vs an
exca job-array submission that returns without running anything), so its
plain non-debug invocation never executes locally; see
_run_native_baseline.

Must be run with src/ importable; this file inserts its repo's sibling
src/ onto sys.path itself, so it works both from a local checkout and
after scripts/colab_neuralbench.sh uploads it alongside src/ on Colab.
"""

import argparse
import hashlib
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# globals().get fallback: colab_neuralbench.sh's bootstrap runs this file's
# source through exec(), where __file__ is not defined; cwd is REMOTE_ROOT/
# scripts there (see the bootstrap's chdir), so 'src' still resolves as a
# sibling of it.
_here = Path(globals().get("__file__", Path.cwd() / "neuralbench_eval.py")).resolve()
sys.path.insert(0, str(_here.parent.parent / "src"))

import torch
import torch.nn as nn

from config import ConformerConfig, EEGNetConfig, PatchTransformerConfig
from models.eegnet import EEGNet
from models.conformer import EEGConformer
from models.patch_transformer import PatchTransformer


def _apply_pretrained_trunk(net, model_name, trunk_path):
    """Loads a shared-trunk checkpoint's matching keys into an already-built
    net, in place.

    Called from a Lazy*Adapter's forward() on every rebuild (NeuralBench's
    montage-width probing can rebuild self.net more than once; each fresh
    build is random-init again, so the trunk must be reapplied each time,
    not only on the first build).

    Parameters
    ----------
    net : EEGNet, PatchTransformer, or EEGConformer
        Freshly constructed backbone to load trunk weights into.
    model_name : str
        One of "eegnet", "patch_transformer", "conformer".
    trunk_path : Path
        Path to a checkpoint written by src/pretrain_trunk.py.

    Raises
    ------
    RuntimeError
        If none of the checkpoint's keys matched net's own state_dict, a
        real misconfiguration (e.g. mismatched EEGNetConfig/PatchTransformerConfig/ConformerConfig
        between pretrain_trunk.py and this script) rather than something to
        silently tolerate.
    """
    from pretrain_trunk import filter_trunk_state_dict    # sys.path has src/, see module docstring

    trunk_state = torch.load(trunk_path, map_location="cpu")
    result = net.load_state_dict(trunk_state, strict=False)
    if len(result.missing_keys) == len(net.state_dict()):
        raise RuntimeError(f"pretrained trunk at {trunk_path} shared no keys with the built {model_name} net")


def _register_trunk_tag(module, pretrained_trunk_path):
    """Registers a buffer holding the SHA-256 of the trunk file's bytes, so
    the trunk's contents participate in neuralbench.external.model_digest.

    Hashing contents, not the path, keeps two different checkpoints saved
    at the same path from sharing one cached experiment.

    model_digest hashes str(model) + state_dict() before the lazy net is
    ever built (neuralbench's check_forward only inspects forward's
    signature, it never calls it), so two Lazy*Adapter instances that will
    end up loading different trunks, or no trunk at all, are otherwise
    byte-identical to NeuralBench and get treated as the same cached
    experiment.

    Parameters
    ----------
    module : nn.Module
        The Lazy*Adapter instance being constructed.
    pretrained_trunk_path : Path or None
        Value of the adapter's own pretrained_trunk_path argument. The file
        must exist when the adapter is constructed.

    Returns
    -------
    None
    """
    if pretrained_trunk_path is not None:
        digest = hashlib.sha256(Path(pretrained_trunk_path).read_bytes()).digest()
        module.register_buffer("_trunk_tag", torch.frombuffer(bytearray(digest), dtype=torch.uint8))


class LazyEEGNetAdapter(nn.Module):
    """Builds a real EEGNet sized to the first batch it sees, then delegates.

    EEGNet's constructor fixes n_channels/n_times into its classifier layer
    (see EEGNet._infer_flat_dim), but NeuralBench probes several montage
    widths in check_model and only reveals a task's real shape at the first
    forward call. Rebuilding whenever the observed shape changes mirrors
    NeuralBench's own model-building contract: neuralbench.model_factory.
    init_lazy_layers already runs one dummy forward pass before real
    training starts specifically to materialize shape-dependent layers.
    """

    def __init__(self, n_classes, cfg: EEGNetConfig, pretrained_trunk_path=None):
        super().__init__()
        self.n_classes             = n_classes
        self.cfg                   = cfg
        self.pretrained_trunk_path = pretrained_trunk_path
        self.net                   = None
        self._built_shape          = None
        _register_trunk_tag(self, pretrained_trunk_path)

    def forward(self, x, channel_positions=None, ch_names=None):
        shape = tuple(x.shape[1:])
        if shape != self._built_shape:
            self.net          = EEGNet(shape[0], shape[1], self.n_classes, self.cfg).to(x.device)
            self._built_shape = shape
            if self.pretrained_trunk_path is not None:
                _apply_pretrained_trunk(self.net, "eegnet", self.pretrained_trunk_path)
        return self.net(x, channel_positions=channel_positions, ch_names=ch_names)


class LazyPatchTransformerAdapter(nn.Module):
    """Builds a real PatchTransformer sized to the first batch it sees.

    Same rebuild-on-shape-change contract as LazyEEGNetAdapter, since
    PatchTransformer's PatchEmbedding also fixes n_channels at construction.
    """

    def __init__(self, n_classes, cfg: PatchTransformerConfig, pretrained_trunk_path=None):
        super().__init__()
        self.n_classes             = n_classes
        self.cfg                   = cfg
        self.pretrained_trunk_path = pretrained_trunk_path
        self.net                   = None
        self._built_shape          = None
        _register_trunk_tag(self, pretrained_trunk_path)

    def forward(self, x, channel_positions=None, ch_names=None):
        shape = tuple(x.shape[1:])
        if shape != self._built_shape:
            self.net          = PatchTransformer(shape[0], shape[1], self.n_classes, self.cfg).to(x.device)
            self._built_shape = shape
            if self.pretrained_trunk_path is not None:
                _apply_pretrained_trunk(self.net, "patch_transformer", self.pretrained_trunk_path)
        return self.net(x, channel_positions=channel_positions, ch_names=ch_names)


class LazyConformerAdapter(nn.Module):
    """Builds a real EEGConformer sized to the first batch it sees.

    Same rebuild-on-shape-change contract as LazyEEGNetAdapter, since
    EEGConformer's ConvModule also fixes n_channels at construction.
    """

    def __init__(self, n_classes, cfg: ConformerConfig, pretrained_trunk_path=None):
        super().__init__()
        self.n_classes             = n_classes
        self.cfg                   = cfg
        self.pretrained_trunk_path = pretrained_trunk_path
        self.net                   = None
        self._built_shape          = None
        _register_trunk_tag(self, pretrained_trunk_path)

    def forward(self, x, channel_positions=None, ch_names=None):
        shape = tuple(x.shape[1:])
        if shape != self._built_shape:
            self.net          = EEGConformer(shape[0], shape[1], self.n_classes, self.cfg).to(x.device)
            self._built_shape = shape
            if self.pretrained_trunk_path is not None:
                _apply_pretrained_trunk(self.net, "conformer", self.pretrained_trunk_path)
        return self.net(x, channel_positions=channel_positions, ch_names=ch_names)


MODEL_ADAPTERS = {
    "eegnet": (LazyEEGNetAdapter, EEGNetConfig),
    "patch_transformer": (LazyPatchTransformerAdapter, PatchTransformerConfig),
    "conformer": (LazyConformerAdapter, ConformerConfig),
}


def _run_with_heartbeat(cmd, interval_s=60):
    """Runs a command, printing a heartbeat line at a fixed interval.

    A silent subprocess run inside a Colab session (an OSF download
    stalled on rate-limit retries, for instance) can otherwise starve
    the kernel's own output for many minutes, which is what the
    disconnect this wraps around is suspected to be triggered by.

    Parameters
    ----------
    cmd : list of str
        Argv to execute.
    interval_s : float
        Seconds between heartbeat lines while the command runs.

    Returns
    -------
    int
        The command's exit code.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    start = time.monotonic()

    def heartbeat():
        while proc.poll() is None:
            time.sleep(interval_s)
            if proc.poll() is None:
                print(f"[heartbeat] still running, {int(time.monotonic() - start)}s elapsed", flush=True)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    heartbeat_thread.join()
    return proc.returncode


def _resource_snapshot():
    """Returns cumulative CPU seconds and syscall read/write bytes for this
    process and its children.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        (cpu_seconds, read_bytes, write_bytes), summed over the process tree.
    """
    import psutil

    me = psutil.Process()
    cpu = read = write = 0
    for proc in [me] + me.children(recursive=True):
        try:
            times = proc.cpu_times()
            io = proc.io_counters()
        except psutil.NoSuchProcess:
            continue
        cpu += times.user + times.system
        # Syscall-level counters, so writes through the Drive FUSE mount count.
        read += io.read_chars
        write += io.write_chars
    return cpu, read, write


def _with_heartbeat(fn, interval_s=60):
    """Calls fn() in the current thread, printing a heartbeat while it runs.

    For in-process work (no subprocess to read stdout from), unlike
    _run_with_heartbeat above. A long silent stretch here is suspected to
    kill the Colab tunnel the same way a silent subprocess does, see
    CLAUDE.md.

    Each line also reports CPU use and read/write volume over the last
    interval, to tell compute-bound work from I/O-bound work, followed by
    the innermost frames of the thread running fn.

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
    worker_id = threading.get_ident()

    def heartbeat():
        last_time, last = time.monotonic(), _resource_snapshot()
        while not stop.wait(interval_s):
            now_time, now = time.monotonic(), _resource_snapshot()
            span = now_time - last_time
            print(
                f"[heartbeat] still running, {int(now_time - start)}s elapsed, "
                f"cpu {(now[0] - last[0]) / span * 100:.0f}% of one core, "
                f"read {(now[1] - last[1]) / span / 1e6:.1f} MB/s, "
                f"write {(now[2] - last[2]) / span / 1e6:.1f} MB/s",
                flush=True,
            )
            last_time, last = now_time, now
            frame = sys._current_frames().get(worker_id)
            for entry in traceback.extract_stack(frame)[-4:]:
                print(f"    at {entry.filename.split('/')[-1]}:{entry.lineno} {entry.name}", flush=True)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        return fn()
    finally:
        stop.set()
        heartbeat_thread.join()


def _stage_moabb_data_locally(local_root=Path("/content/local_neuralbench_data")):
    """Copies NeuralBench's Drive-backed MOABB study downloads to local VM
    disk and redirects NeuralBench's own config to read from there.

    Reading NeuralBench's per-subject raw files straight off the Drive
    mount hits Google Drive's API rate limit after roughly 25-30 small
    file-open requests: each subsequent open then costs 1.3-2.0s instead
    of 0.04-0.07s. Copying only each subject's single zip archive instead
    (skipping moabb's own already-extracted per-subject BIDS folders, the
    exact access pattern that triggers the limit) sustains ~20MB/s with no
    throttling; extracting each zip locally afterward needs no further
    Drive access for the rest of this process's run.

    NeuralBench's default config takes each study's source path from the
    module attribute neuralbench.config_manager.DATA_DIR, which is fixed
    the first time any config is built. Mutating get_config()'s cached
    dict alone does not change it, so both are set here. A second call in
    the same process is a no-op.

    Each dataset folder's top-level metadata files (manifest, README,
    participants and channels tables) are copied along with the zips.
    Without them moabb downloads each one again from OSF, over an
    unverified connection and with no rate-limit protection.

    Parameters
    ----------
    local_root : Path
        Local VM directory to stage the copy under.

    Returns
    -------
    None
    """
    import shutil
    import zipfile
    import neuralbench.config_manager as nb_config_manager
    from neuralbench.evaluate import get_config

    drive_data_dir = Path(get_config()["DATA_DIR"])
    if drive_data_dir == local_root:
        return
    drive_moabb_dir = drive_data_dir / "moabb"
    if not drive_moabb_dir.is_dir():
        return

    for study_dir in drive_moabb_dir.iterdir():
        if not study_dir.is_dir():
            continue
        local_study_dir = local_root / "moabb" / study_dir.name
        local_download_dir = local_study_dir / "download"
        local_download_dir.mkdir(parents=True, exist_ok=True)

        for name in ("timelines.csv", "timelines.csv.bak"):
            src = study_dir / name
            if src.exists():
                shutil.copy(src, local_study_dir / name)

        drive_download_dir = study_dir / "download"
        if not drive_download_dir.is_dir():
            continue
        # moabb nests its own DATASET_FOLDER under "download" (e.g.
        # "download/MNE-dreyer2023-data/sub-01.zip"), so zips are found by
        # recursive search, not a flat listing; each is copied to the same
        # relative path and extracted into its own parent, reproducing
        # moabb's already-extracted per-subject folders next to it without
        # ever copying those many-small-file folders themselves.
        for zip_path in drive_download_dir.rglob("*.zip"):
            rel = zip_path.relative_to(drive_download_dir)
            dst = local_download_dir / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"[stage] {zip_path} -> {dst}", flush=True)
            shutil.copy(zip_path, dst)
            with zipfile.ZipFile(dst) as zf:
                zf.extractall(dst.parent)

        for dataset_dir in {zip_path.parent for zip_path in drive_download_dir.rglob("*.zip")}:
            local_dataset_dir = local_download_dir / dataset_dir.relative_to(drive_download_dir)
            for meta_path in dataset_dir.iterdir():
                if meta_path.is_file() and meta_path.suffix != ".zip":
                    shutil.copy(meta_path, local_dataset_dir / meta_path.name)

    nb_config_manager._config["DATA_DIR"] = str(local_root)
    # Initialize first, or the lazy init would later overwrite DATA_DIR.
    nb_config_manager._ensure_initialized()
    nb_config_manager.DATA_DIR = str(local_root)
    print(f"[stage] NeuralBench DATA_DIR redirected to local copy at {local_root}", flush=True)


def _run_and_save_checkpoint(args):
    """Runs a real NeuralBench training experiment for --model and saves its
    trained checkpoint, not just its scores, for scripts/submission/.

    evaluate_model (the path used elsewhere in this file) never exposes the
    trained model it builds internally: it discards the BenchmarkAggregator
    once it has read back the score frame. This mirrors evaluate_model's own
    call sequence (_external_config -> download -> prepare-warm -> real run,
    all through neuralbench.evaluate's own private helpers, since there is no
    public one that returns the aggregator) but keeps that aggregator, so the
    Lightning checkpoint main.py writes per experiment
    (experiment.infra.uid_folder() / "best.ckpt") can be found afterward.
    cluster=None (both here and in evaluate_model's own default) is what
    makes this run in-process and block rather than submit an async cluster
    job: see neuralbench.experiment_config.apply_cluster's docstring.

    The checkpoint holds the full BrainModule, prefixed "model.<...>": the
    backbone at "model.wrapped_model.net." (LazyEEGNetAdapter/
    LazyPatchTransformerAdapter's lazily-built real network) and, since
    --downstream-wrapper's finetune_* presets train a
    nn.Linear(n_classes, n_classes) probe on top of the backbone's own
    output, that probe at "model.probe.". Both must ship in weights.pt:
    scripts/submission/submission.py's model applies them in the same order.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments; args.save_checkpoint is the output path.

    Returns
    -------
    None
    """
    from neuralbench.evaluate import _external_config, _experiment_configs, _run
    from neuralbench.experiment_config import _download_dataset

    adapter_cls, cfg_cls = MODEL_ADAPTERS[args.model]
    model = adapter_cls(args.n_classes, cfg_cls(), pretrained_trunk_path=args.pretrained_trunk)

    config = _external_config(model)
    overlay = {
        "brain_model_name": f"ours-{args.model}",
        "brain_model_config": {"=replace=": True, **config.model_dump()},
    }
    # delete_checkpoints_on_exit defaults to True (main.py's Experiment):
    # without this override, the checkpoint we need is unlinked the moment
    # the run finishes.
    phase = dict(
        device="eeg", task=args.task,
        overrides={"delete_checkpoints_on_exit": False},
        downstream_wrapper=args.downstream_wrapper,
        cluster=None, debug=args.debug, dataset=args.dataset,
    )
    configs = _experiment_configs(overlay, **phase)

    for cfg in {c["data.study.source"]["name"]: c for c in configs}.values():
        _download_dataset(cfg)
    _with_heartbeat(_stage_moabb_data_locally)
    _with_heartbeat(lambda: _run(_experiment_configs(overlay, prepare=True, **phase), args.debug))
    agg = _with_heartbeat(lambda: _run(configs, args.debug))

    net_prefix, probe_prefix = "model.wrapped_model.net.", "model.probe."
    for experiment in agg.experiments:
        ckpt_path = Path(experiment.infra.uid_folder()) / "best.ckpt"
        if not ckpt_path.exists():
            print(f"[warn] no checkpoint at {ckpt_path} (seed {experiment.seed}), skipping")
            continue
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = raw["state_dict"]
        net_state = {k[len(net_prefix):]: v for k, v in state.items() if k.startswith(net_prefix)}
        probe_state = {k[len(probe_prefix):]: v for k, v in state.items() if k.startswith(probe_prefix)}
        if not net_state or not probe_state:
            raise RuntimeError(
                f"expected keys prefixed {net_prefix!r} and {probe_prefix!r} in {ckpt_path}, "
                f"got {sorted(state)[:10]}... (NeuralBench's internal wrapping may have "
                "changed; update the prefixes above to match)"
            )
        torch.save({"net": net_state, "probe": probe_state}, args.save_checkpoint)
        print(f"wrote {args.save_checkpoint} from seed {experiment.seed}'s checkpoint "
              f"({len(net_state)} backbone tensors, {len(probe_state)} probe tensors)")
        return
    raise RuntimeError("no experiment produced a checkpoint; see [warn] lines above")


def _run_native_baseline(args):
    """Runs a NeuralBench-registered baseline (e.g. REVE) for real.

    Bypasses check_model/evaluate_model, which only wrap an already-built
    nn.Module, and NeuralBench's own CLI, whose non-debug path submits an
    exca job array and returns without executing anything locally, verified
    against real dreyer2023 data. build_experiment_configs(debug=False)
    builds the real, full-epoch configs; BenchmarkAggregator(debug=True) is
    the separate flag that forces those configs to run synchronously in
    this process instead of going through a job array. Reads results back
    with collect(), not run(): run() unconditionally calls
    plot_all_results, which raises for a single-model result set like this
    one (it compares models against each other).

    args.collect_only skips prepare() (which calls each experiment's
    run(), recomputing anything not cleanly cached) and only calls
    collect(cached_only=True), a pickle-load with no recomputation. Use it
    to read back results from an already-completed run without risking a
    partially-failed job's cache entry triggering a real rerun.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments; args.model names the NeuralBench baseline.

    Returns
    -------
    None
    """
    from neuralbench.experiment_config import build_experiment_configs
    # Not neuralbench.aggregator directly: importing BenchmarkAggregator
    # from there skips main.py's module-level BenchmarkAggregator.
    # model_rebuild(_types_namespace={"Experiment": Experiment}), leaving
    # its "experiments: list[Experiment]" forward-ref unresolved.
    from neuralbench.main import BenchmarkAggregator

    configs = build_experiment_configs(
        "eeg", args.task, model=args.model, dataset=args.dataset, debug=args.debug,
    )
    agg = BenchmarkAggregator(experiments=configs, debug=True)
    if not args.collect_only:
        _with_heartbeat(agg.prepare)
    results = agg.collect(cached_only=args.collect_only)
    for result in results:
        print(result)

    import json
    with open("reve_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("wrote reve_results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="eegnet",
        help=f"{sorted(MODEL_ADAPTERS)} run through check_model/evaluate_model as one "
             "of our own models. Any other name (e.g. 'reve') is run for real as a "
             "NeuralBench-native baseline via _run_native_baseline instead.",
    )
    parser.add_argument("--task", default="motor_imagery")
    parser.add_argument("--dataset", default="dreyer2023")
    parser.add_argument(
        "--n-classes", type=int, default=2,
        help="NeuralBench's own label count for --task/--dataset, not necessarily "
             "src/datasets.py's n_classes preset for the same dataset name.",
    )
    parser.add_argument(
        "--downstream-wrapper", default="finetune_flatten",
        help="finetune_* unfreezes the whole model, matching how train.py trains "
             "our models end to end (the default linear_probe_mean instead freezes "
             "it, correct only for a pretrained foundation model). '_flatten' "
             "aggregation is a no-op reshape for a (batch, n_classes) output; '_mean' "
             "would incorrectly average across classes, so avoid it here.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run check_model's fast synthetic-shape probe and exit, skip evaluate_model.",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="For a NeuralBench-native --model (e.g. reve): skip prepare() and read "
             "back already-cached results with a pickle-load only, no recomputation. "
             "Ignored for --model eegnet/patch_transformer/conformer.",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download --dataset's corpus via neuralbench's own CLI, heartbeat-wrapped, "
             "then exit without building a model or running check_model/evaluate_model.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Reduced epochs/batches, forced local execution. A wiring check, not a "
             "real score; see evaluate_model's own docstring.",
    )
    parser.add_argument(
        "--save-checkpoint", type=Path, default=None,
        help="Instead of check_model/evaluate_model, run --model for real through "
             "_run_and_save_checkpoint and save its trained backbone+probe weights "
             "to this path, for scripts/submission/build_submission.py. Only valid "
             "for --model eegnet/patch_transformer/conformer.",
    )
    parser.add_argument(
        "--pretrained-trunk", type=Path, default=None,
        help="Path to a shared-trunk checkpoint from src/pretrain_trunk.py. Applied to "
             "the lazily-built backbone's trunk layers on every NeuralBench shape-probe "
             "rebuild. Only valid for --model eegnet/patch_transformer/conformer.",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    if args.download:
        returncode = _run_with_heartbeat(
            ["neuralbench", "eeg", args.task, "--dataset", args.dataset, "--download"]
        )
        if returncode != 0:
            raise RuntimeError(f"neuralbench download exited with code {returncode}")
        return

    if args.model not in MODEL_ADAPTERS:
        _run_native_baseline(args)
        return

    if args.save_checkpoint is not None:
        _run_and_save_checkpoint(args)
        return

    from neuralbench import check_model, evaluate_model

    adapter_cls, cfg_cls = MODEL_ADAPTERS[args.model]
    model = adapter_cls(args.n_classes, cfg_cls(), pretrained_trunk_path=args.pretrained_trunk)

    print("== check_model ==")
    print(check_model(model, "eeg", args.task, dataset=args.dataset).to_string())

    if args.check_only:
        return

    # Downloaded (Drive-backed, durable) and staged locally ourselves here so
    # evaluate_model's own real run reads the fast local copy: see
    # _stage_moabb_data_locally's docstring. download=False below skips its
    # own redundant internal download of the same, now-cached corpus.
    from neuralbench.evaluate import _external_config, _experiment_configs
    from neuralbench.experiment_config import _download_dataset

    config = _external_config(model)
    overlay = {
        "brain_model_name": f"ours-{args.model}",
        "brain_model_config": {"=replace=": True, **config.model_dump()},
    }
    phase = dict(
        device="eeg", task=args.task, overrides=None,
        downstream_wrapper=args.downstream_wrapper,
        cluster=None, debug=args.debug, dataset=args.dataset,
    )
    for cfg in {c["data.study.source"]["name"]: c for c in _experiment_configs(overlay, **phase)}.values():
        _download_dataset(cfg)
    _with_heartbeat(_stage_moabb_data_locally)

    print("== evaluate_model ==")
    scores = _with_heartbeat(lambda: evaluate_model(
        model, "eeg", args.task,
        name=f"ours-{args.model}",
        dataset=args.dataset,
        downstream_wrapper=args.downstream_wrapper,
        debug=args.debug,
        download=False,
    ))
    print(scores.to_string())
    scores.to_json(f"{args.model}_results.json", orient="records", indent=2)
    print(f"wrote {args.model}_results.json")


if __name__ == "__main__":
    main()
