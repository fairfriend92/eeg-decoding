# EEG Motor Imagery Decoding with a Transformer Encoder

Motor imagery and related mental-task decoding across multiple EEG datasets:
BCI Competition IV Dataset 2a [3], Dreyer 2023 [5], and Stieger 2021 [6]. A
not-yet-released Graz+BrainHero dataset [7] is registered for future support.
Compares a transformer encoder based on the EEG Conformer architecture [2]
against EEGNet [1], a standard lightweight CNN baseline for this task.

## Task

Each dataset poses a different generalization shift (within-session, single-session
holdout, or cross-session); see `datasets.py` for the exact preset (class count,
channel count, sampling rate, subject list, split strategy) backing each one below.

### BCI2a

Subjects imagine a movement (left hand, right hand, feet, or tongue) without actually
moving it; the task is decoding which one from EEG alone. 9 subjects, 22 channels,
288 trials per subject split evenly across two recorded sessions (`0train`, `1test`),
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

### Stieger2021

4-class cursor-control BCI task (right hand, left hand, both hands, rest) via
lateralized/total alpha-power modulation, 62 subjects, 60 channels at 1000 Hz native,
up to 11 sessions per subject recorded over several weeks [6]. Trained on the
earliest `n_train_sessions` (8 by default) and tested on the remaining, later
sessions (`session_index` split). This is a genuine cross-session generalization
test, unlike BCI2a's or Dreyer2023's single-day splits.

### Graz+BrainHero (not yet released)

Registered as a placeholder dataset (`graz_brainhero`, raises immediately if
selected) for the NeurIPS 2026 EEG/EMG Foundation Challenge's Track 2 longitudinal
corpus [7]: 3 cued mental commands (kinesthetic motor imagery, mental calculation,
word association) across 6 sessions per participant, evaluating whether a model
calibrated on early sessions still decodes reliably on later, unseen-day sessions.

## Results

![Results summary](output/figures/results_summary.png)

### BCI2a

| | Accuracy | Kappa |
|---|---|---|
| EEGNet, ensemble (5-fold, cropped training) | 0.636 ± 0.168 | 0.525 ± 0.222 |
| EEGNet, final-fit (100% of train session, cropped) | 0.632 ± 0.177 | 0.506 ± 0.224 |
| EEGNet, ensemble (5-fold, cropped + CSP-init) | 0.685 ± 0.115 | 0.580 ± 0.154 |
| EEGNet, final-fit (cropped + CSP-init) | 0.643 ± 0.119 | 0.524 ± 0.159 |
| Conformer, ensemble (5-fold, cropped training) | 0.459 ± 0.160 | 0.278 ± 0.213 |
| Conformer, final-fit (100% of train session, cropped) | 0.439 ± 0.139 | 0.253 ± 0.186 |

Mean ± std across the 9 subjects (sample std). Both EEGNet numbers sit in the range
reported for this exact architecture and dataset in the published literature [1]
(roughly 63-72% accuracy, κ≈0.5-0.6).

## Methodology

- **Split**: session-based (official competition split), not random. Train/val
  are drawn only from the training session; the test session is held out entirely
  until final scoring, never touched during model or hyperparameter selection.
- **Preprocessing**: 4-38 Hz bandpass, resampled to 250 Hz, per-channel z-score
  normalization fit on the training session only.
- **Validation**: single 80/20 split or stratified k-fold, selectable per run.
  Hyperparameters are selected via k-fold rather than a single split.
- **Cropped training**: trials are sliced into overlapping windows as training
  augmentation, loosely following [4]; evaluation crops test trials the same way and
  averages predictions per trial. This was the single largest lever tried, worth
  roughly 6-9 accuracy points on its own.
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
data/                 raw input only (MOABB/MNE cache)
output/                checkpoints, results, figures, one subdirectory per dataset
src/
  config.py            hyperparameter dataclasses, Config-building/override logic
  datasets.py            per-dataset presets (class/channel count, sampling rate,
                        subject list, session-split strategy) and MOABB dataset classes
  data_loader.py        MOABB/MNE loading, train/test split, normalization, cropping
  models/
    eegnet.py            CNN baseline
    conformer.py          transformer encoder
  train.py              trains and checkpoints; never touches test data
  evaluate.py            scores saved checkpoints on the held-out test data
  sweep.py               hyperparameter search, single-split or k-fold
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
--dataset {bci2a,dreyer2023,stieger2021,graz_brainhero}   (default: bci2a)
--config path/to/overrides.yaml                             (optional)
--set section.field=value                                    (repeatable, optional)
```
`--dataset` picks a preset from `datasets.py` (class count, channel count, sampling
rate, subject list, session-split strategy). `bci2a` is the dataset used for the
Results above; `dreyer2023`/`stieger2021` are implemented against MOABB's own dataset
classes but not yet run end-to-end here (see Limitations). `graz_brainhero` is
registered as a placeholder for the not-yet-released Graz+BrainHero dataset and
raises immediately if selected. `--config` loads a YAML file of overrides, nested by
section, e.g.:
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
python evaluate.py {eegnet,conformer} [{eegnet,conformer} ...] [dataset flags]
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
`data_loader.py` downloads the dataset directly on the VM via MOABB. Pass
`--skip-sync` to skip the upload and install steps on a session already set up
this way.

`scripts/colab_sync.sh -s trainer` uploads `src/` on its own; `--data` uploads the
local `data/` cache instead.

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
6. J. R. Stieger, S. A. Engel, and B. He, "Continuous sensorimotor rhythm based
   brain computer interface learning in a large population," *Scientific Data*,
   vol. 8, p. 98, 2021.
   [DOI: 10.1038/s41597-021-00883-1](https://doi.org/10.1038/s41597-021-00883-1).
7. "EEG/EMG Foundation Challenge 2026," Brain and Body Workshop at NeurIPS 2026,
   Track 2 (BCI decoding).
   [Tracks & data](https://neural-interfaces26.github.io/tracks.html) ·
   [rules & FAQ](https://neural-interfaces26.github.io/rules.html).

## Limitations

Results above are for BCI2a only, within-subject (no cross-subject generalization
claim), 9 subjects. `dreyer2023` and `stieger2021` support (see Usage) is implemented
against MOABB's dataset classes and covered by unit-level checks on synthetic data,
but not yet validated against the real downloaded data (Dreyer2023 ~19GB,
Stieger2021 ~399GB). In particular, MOABB's `MotorImagery` paradigm's class/event
selection has not been hands-on confirmed for Stieger2021's `rest` class or
Dreyer2023's 2-class-only event set. `stieger2021`'s `n_train_sessions=8` default
session split is a provisional guess pending inspection of real per-subject session
counts (documented as 7-11, not exactly 11 for everyone).
