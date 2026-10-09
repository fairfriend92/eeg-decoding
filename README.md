# EEG Motor Imagery Decoding: EEGNet and EEG Conformer

Motor imagery decoding across two public EEG datasets: BCI Competition IV
Dataset 2a [3] and Dreyer 2023 [5]. Compares EEGNet [1], a standard lightweight
CNN baseline for this task, against the EEG Conformer [2], a convolution module
followed by a transformer encoder.

EEGNet reaches 0.727 accuracy on all 87 Dreyer2023 subjects (2 classes, chance
0.5) and 0.685 on BCI2a (4 classes, chance 0.25), within the range published
for this architecture on BCI2a. The Conformer trails EEGNet on Dreyer2023 by
0.061 ± 0.019 accuracy (final fit, paired per subject).

This project's benchmark submission to the NeurIPS 2026 EEG/EMG Foundation
Challenge is documented separately, in
[README_NEURIPS2026.md](README_NEURIPS2026.md).

## Task

Each dataset poses a different generalization shift (cross-session or
within-session holdout); see `datasets.py` for the exact preset (class count,
channel count, sampling rate, subject list, split strategy) backing each one below.

### BCI2a

Subjects imagine a movement (left hand, right hand, feet, or tongue) without actually
moving it; the task is decoding which one from EEG alone. 9 subjects, 22 channels,
288 trials in each of two recorded sessions (`0train`, `1test`),
recorded by the Institute for Knowledge Discovery (Laboratory of Brain-Computer
Interfaces), Graz University of Technology, for BCI Competition IV [3]. Models are
trained and evaluated per subject, using the dataset's original session split rather
than a random one, so results are directly comparable to published baselines rather
than to an arbitrary held-out slice.

### Dreyer2023

2-class right/left hand motor imagery, 87 subjects, 27 EEG channels (plus EMG/EOG,
unused here) at 512 Hz native, single session per subject (6 runs: 2 calibration + 4
online-feedback) [5]. With no second session to hold out, the split instead takes a
trailing, per-class-stratified 20% of that session's trials as the test set (see
`within_session_holdout` in `data_loader.py`).

## Results

![Results summary](output/figures/results_summary.png)

### BCI2a

| | Accuracy | Kappa |
|---|---|---|
| EEGNet, ensemble (5-fold, cropped training) | 0.636 ± 0.168 | 0.525 ± 0.222 |
| EEGNet, final-fit (100% of train session, cropped) | 0.632 ± 0.177 | 0.506 ± 0.224 |
| EEGNet, ensemble (5-fold, cropped + CSP-init) | 0.685 ± 0.115 | 0.580 ± 0.154 |
| EEGNet, final-fit (cropped + CSP-init) | 0.643 ± 0.119 | 0.524 ± 0.159 |

Mean ± std across the 9 subjects (sample std). Both EEGNet numbers sit in the range
reported for this exact architecture and dataset in the published literature [1]
(roughly 63-72% accuracy, κ≈0.5-0.6). The Conformer comparison is on Dreyer2023.

### Dreyer2023

| | Accuracy | Kappa |
|---|---|---|
| EEGNet, best checkpoint (single 80/20 split) | 0.721 ± 0.164 | 0.443 ± 0.328 |
| EEGNet, final-fit (100% of session, held-out tail excluded) | 0.727 ± 0.170 | 0.454 ± 0.339 |
| Conformer, best checkpoint (single 80/20 split) | 0.688 ± 0.149 | 0.377 ± 0.299 |
| Conformer, final-fit (100% of session, held-out tail excluded) | 0.666 ± 0.145 | 0.333 ± 0.290 |

Mean ± std across all 87 subjects (sample std), default hyperparameters, no cropped
training. The paired per-subject accuracy difference, EEGNet minus Conformer, is
+0.033 ± 0.020 for the best checkpoint (EEGNet ahead on 44 subjects, Conformer on
37, 6 ties) and +0.061 ± 0.019 for the final fit (55, 26, 6), mean ± standard
error. EEGNet leads on both, clearly for the final fit. With roughly 100 to 200
training trials per subject, the Conformer's larger parameter count and weaker
inductive bias for EEG do not pay off against EEGNet's compact design.

## Methodology

- **Split**: BCI2a uses the official session split, Dreyer2023 a trailing,
  per-class-stratified 20% holdout of its single session. Never a random split.
  Train/val are drawn only from the training portion; the test portion is held out
  entirely until final scoring, never touched during model or hyperparameter
  selection.
- **Preprocessing**: 4-38 Hz bandpass, resampled to 250 Hz, per-channel z-score
  normalization fit on the training session only.
- **Validation**: single 80/20 split or stratified k-fold, selectable per run.
  Hyperparameters are selected via k-fold rather than a single split.
- **Cropped training**: trials are sliced into overlapping windows as training
  augmentation, loosely following [4]; evaluation crops test trials the same way and
  averages predictions per trial.
- **Reporting**: accuracy and Cohen's kappa, mean ± std across subjects, not a single
  headline number. For k-fold runs, three numbers are computed: the plain average of
  each fold model's own score, an ensemble score (averaging the fold models'
  predictions, not their scores; these are not the same thing, and the ensemble is
  consistently the stronger of the two), and a final model retrained on the full
  training session using a step-matched epoch budget from cross-validation.
- **Reproducibility**: model initialization and data shuffling are seeded per
  subject, so repeat runs with identical config are exactly reproducible.

## Repository structure

```
data/                   raw input only (MOABB/MNE cache)
output/                 results and figures, one subdirectory per dataset
  {dataset}/            manifests, results, and summary JSON
    checkpoints/        .pt model checkpoints
src/
  config.py             hyperparameter dataclasses, Config-building/override logic
  datasets.py           per-dataset presets (class/channel count, sampling rate,
                        subject list, session-split strategy) and MOABB dataset classes
  data_loader.py        MOABB/MNE loading, train/test split, normalization, cropping
  moabb_downloads.py    MOABB download configuration (verified HTTPS, OSF redirect, retries)
  download_dataset.py   download-only fetch of a dataset's per-subject files
  models/
    eegnet.py             CNN baseline
    conformer.py          EEG Conformer: convolution module plus transformer encoder
  train.py                trains and checkpoints; never touches test data
  evaluate.py             scores saved checkpoints on the held-out test data
  sweep.py                hyperparameter search, single-split or k-fold
  pretrain_trunk.py       shared-trunk pretraining (see README_NEURIPS2026.md)
scripts/
  make_results_figure.py  regenerates output/figures/results_summary.png
```

`train.py` and `evaluate.py` are deliberately separate: training only ever produces
checkpoints, scoring is a distinct step that loads them back. This means evaluation
logic can change (a new metric, a different ensembling scheme) without retraining,
and it keeps the "never touch the test set during training" guarantee mechanical
rather than a matter of remembering not to.

## Usage

Run everything from inside `src/`. Requires `pip install -r requirements.txt`; a GPU
runtime (Colab or similar) is assumed for anything beyond a quick smoke test.

All three scripts below accept the same dataset-selection flags:
```
--dataset {bci2a,dreyer2023}      (default: bci2a)
--config path/to/overrides.yaml   (optional)
--set section.field=value         (repeatable, optional)
```
`--dataset` picks a preset from `datasets.py` (class count, channel count, sampling
rate, subject list, session-split strategy). `bci2a` is the dataset used for the
BCI2a Results above, and `dreyer2023` provides the Dreyer2023 Results.
`--config` loads a YAML file of overrides, nested by section, e.g.:
```yaml
train:
  lr: 0.001
eegnet:
  dropout: 0.3
data:
  n_train_sessions: 6
```
`--set` overrides a single field the same way from the command line, e.g.
`--set eegnet.dropout=0.3`; `--set` wins over `--config` on conflicts. A bare key with
no section (e.g. `--set lr=0.001`) resolves against `train`, then the model being run.

Train:
```
python train.py {eegnet,conformer} [--n-folds N] [--cropped] [--csp-init] [--pretrain] [dataset flags]
```
`--n-folds` defaults to 1 (single 80/20 split); any value ≥2 runs stratified k-fold,
producing one checkpoint per fold plus a final model retrained on the full training
data. `--cropped` enables crop-based training/evaluation, sized from `resample_freq`
so it scales correctly across datasets. `--csp-init` initializes EEGNet's spatial
filters from per-subject CSP components instead of random weights (EEGNet only).
Writes `output/{dataset}/{model}_checkpoints.json`, including which dataset produced
it.

Evaluate:
```
python evaluate.py {eegnet,conformer} [...] [dataset flags]
```
Loads the checkpoint manifest written by `train.py` for the selected `--dataset`,
scores every checkpoint on the held-out test data, prints per-subject and aggregate
metrics, and writes `output/{dataset}/{model}_results.json` and
`output/{dataset}/summary.json`. Passing a `--dataset`/`--config`/`--set` that doesn't
match what a checkpoint was trained with (per its manifest entry) raises immediately
rather than silently scoring against the wrong setup.

Sweep:
```
python sweep.py {eegnet,conformer} [--n-folds N] [dataset flags]
```
Grid search over a small `lr`/`dropout` grid (edit `DEFAULT_GRIDS` in `sweep.py` to
change it), evaluated via k-fold by default, on top of the chosen dataset/config
baseline. Writes `output/{dataset}/{model}_sweep.json`, sorted best first. Does not
modify any file. Applying a sweep result as the new default means adding it to
`datasets.py`'s preset or passing it via `--set`/`--config` on subsequent runs.

### Running on Colab

`scripts/` wraps the [Colab CLI](https://github.com/googlecolab/google-colab-cli) so
training runs on a Colab GPU while editing files as usual, without syncing to Drive
or copying into a notebook.

One-time setup:
```
scripts/setup_colab_cli.sh
```
Installs the CLI into `.venv`, pinning `jupyter_kernel_client==0.15.0`.

Authenticate once:
```
.venv/bin/colab sessions
```

Train on a GPU session (`trainer` is an example session name; any name works):
```
scripts/colab_train.sh -s trainer --gpu T4 -- eegnet --n-folds 5
```
Creates the session if it does not already exist (`--gpu` accepts `T4`, `L4`, `G4`,
`H100`, `A100`, or is omitted for a CPU session), uploads `src/`, installs
`requirements.txt`, runs `train.py` with everything after `--` forwarded as its
arguments, and downloads `output/` back on completion. `data/` is not uploaded;
MOABB downloads into Drive, so a dataset fetched once is reused by every later
session. `--script NAME` runs another `src/` entry point (e.g. `sweep.py`) instead
of `train.py`. Pass `--skip-sync` to skip the upload and install steps on a
session already set up this way.

Download a dataset on a CPU-only session before a GPU run:
```
scripts/colab_download.sh -s downloader -- dreyer2023
```
Runs `download_dataset.py` without a GPU, so no accelerator sits idle during the
transfer, and a failed download cannot abort a training run. Downloads retry on
timeouts and rate limits.

`scripts/colab_sync.sh -s trainer` uploads `src/` on its own; `--data` also uploads the
local `data/` cache to the Drive data directory that the pipeline scripts read
from, so a dataset present only locally is a cache hit on the next run.

Stop the session when done:
```
.venv/bin/colab stop -s trainer
```

## References

1. V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J.
   Lance, "EEGNet: A compact convolutional neural network for EEG-based
   brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5,
   p. 056013, 2018.
2. Y. Song, Q. Zheng, B. Liu, and X. Gao, "EEG Conformer: Convolutional transformer
   for EEG decoding and visualization," *IEEE Transactions on Neural Systems and
   Rehabilitation Engineering*, vol. 31, pp. 710-719, 2023.
3. C. Brunner, R. Leeb, G. Müller-Putz, A. Schlögl, and G. Pfurtscheller, "BCI
   Competition 2008 - Graz data set A," 2008.
   [Dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf) ·
   [competition page](https://www.bbci.de/competition/iv/).
4. R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter,
   K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, "Deep
   learning with convolutional neural networks for EEG decoding and
   visualization," *Human Brain Mapping*, vol. 38, no. 11, pp. 5391-5420, 2017.
5. P. Dreyer, A. Roc, L. Pillette, S. Rimbert, and F. Lotte, "A large EEG database
   with users' profile information for motor imagery brain-computer interface
   research," *Scientific Data*, vol. 10, p. 580, 2023.
   [DOI: 10.1038/s41597-023-02445-z](https://doi.org/10.1038/s41597-023-02445-z).

## Limitations

The BCI2a results are within-subject only (no cross-subject generalization
claim), 9 subjects. The Dreyer2023 results use a within-session holdout, so
they measure generalization to later trials of the same session, not to a new
day. Conformer hyperparameters are the original paper's defaults. A learning-rate
and dropout grid on subjects 1-10 left validation accuracy within one point across
all four settings.
