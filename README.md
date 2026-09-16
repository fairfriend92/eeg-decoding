# EEG Motor Imagery Decoding with a Transformer Encoder

Within-subject 4-class motor imagery classification on BCI Competition IV Dataset 2a,
comparing a custom transformer encoder against EEGNet, a standard lightweight CNN
baseline for this task.

## Task

Subjects imagine a movement (left hand, right hand, feet, or tongue) without actually
moving it; the task is decoding which one from EEG alone. 9 subjects, 22 channels,
288 trials per subject split evenly across two recorded sessions (`0train`, `1test`).
Models are trained and evaluated per subject, using the dataset's original session
split rather than a random one, so results are directly comparable to published
baselines rather than to an arbitrary held-out slice.

**Source**: recorded by the Institute for Knowledge Discovery (Laboratory of
Brain-Computer Interfaces), Graz University of Technology, for BCI Competition IV.
[Dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf) ·
[competition page](https://www.bbci.de/competition/iv/).

## Results

![Results summary](output/figures/results_summary.png)

| | Accuracy | Kappa |
|---|---|---|
| EEGNet, ensemble (5-fold, cropped training) | 0.636 ± 0.168 | 0.525 ± 0.222 |
| EEGNet, final-fit (100% of train session, cropped) | 0.632 ± 0.177 | 0.506 ± 0.224 |
| Conformer, ensemble (5-fold, pre-cropping) | 0.407 ± 0.109 | 0.209 ± 0.145 |

Mean ± std across the 9 subjects (sample std). Both EEGNet numbers sit in the range
reported for this exact architecture and dataset in the published literature
(roughly 63-72% accuracy, κ≈0.5-0.6). The spread is wide because it's real: one
subject (S2) is consistently near chance across every model and configuration tried,
which is a known characteristic of this dataset, not a bug — subject-wise numbers are
in `output/*_results.json` for anyone who wants to check.

Conformer underperforms EEGNet throughout, by a wide margin. This is read as the
expected outcome, not a failure: EEGNet encodes strong priors for this signal
(frequency content, then spatial filtering across electrodes) into its architecture,
while the Conformer has to learn everything from ~230 training trials per subject with
far more parameters and no equivalent prior. It's the accuracy/complexity/data-
efficiency tradeoff the comparison was set up to demonstrate. Conformer hasn't yet
been trained with the cropping setup that gave EEGNet its biggest jump; that's the
next obvious thing to try before drawing a final conclusion on the gap.

## Methodology

- **Split**: session-based (official competition split), not random — train/val
  drawn only from the training session, test session held out entirely until final
  scoring, never touched during model or hyperparameter selection.
- **Preprocessing**: 4-38 Hz bandpass, resampled to 250 Hz, per-channel z-score
  normalization fit on the training session only.
- **Validation**: single 80/20 split or stratified k-fold, selectable per run.
  Hyperparameters were selected via k-fold rather than a single split after an
  earlier single-split sweep picked a Conformer configuration that looked better on
  validation but scored worse on the test set — a single noisy split is enough to
  produce that kind of false signal.
- **Cropped training**: EEGNet trials are sliced into overlapping windows as training
  augmentation (loosely following Schirrmeister et al., 2017); evaluation crops test
  trials the same way and averages predictions per trial. This was the single largest
  lever tried, worth roughly 6-9 accuracy points on its own.
- **Reporting**: accuracy and Cohen's kappa, mean ± std across subjects, not a single
  headline number. For k-fold runs, three numbers are computed: the plain average of
  each fold model's own score, an ensemble score (averaging the fold models'
  predictions, not their scores — these are not the same thing and the ensemble is
  consistently the stronger of the two), and a final model retrained on the full
  training session using a step-matched epoch budget from cross-validation.
- **Reproducibility**: model initialization and data shuffling are seeded per
  subject, so repeat runs with identical config are exactly reproducible.

## Repository structure

```
data/                 raw input only (MOABB/MNE cache)
output/                checkpoints, results, figures
src/
  config.py            all hyperparameters and paths, as dataclasses
  data_loader.py        MOABB/MNE loading, session split, normalization, cropping
  models/
    eegnet.py            CNN baseline
    conformer.py          transformer encoder
  train.py              trains and checkpoints; never touches the test session
  evaluate.py            scores saved checkpoints on the test session
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

Train:
```
python train.py {eegnet,conformer} [--n-folds N] [--cropped]
```
`--n-folds` defaults to 1 (single 80/20 split); any value ≥2 runs stratified k-fold,
producing one checkpoint per fold plus a final model retrained on the full training
session. `--cropped` enables crop-based training/evaluation (EEGNet only, so far).
Writes `output/{model}_checkpoints.json`.

Evaluate:
```
python evaluate.py {eegnet,conformer} [{eegnet,conformer} ...]
```
Loads the checkpoint manifest written by `train.py`, scores every checkpoint on the
test session, prints per-subject and aggregate metrics, and writes
`output/{model}_results.json` and `output/summary.json`. `config.py` must match
whatever was in effect when the checkpoints were trained — nothing syncs this
automatically.

Sweep:
```
python sweep.py {eegnet,conformer} [--n-folds N]
```
Grid search over a small `lr`/`dropout` grid (edit `DEFAULT_GRIDS` in `sweep.py` to
change it), evaluated via k-fold by default. Writes `output/{model}_sweep.json`,
sorted best first. Does not modify `config.py` — applying a sweep result as the new
default is a manual edit.

## Limitations

Single dataset, within-subject only (no cross-subject generalization claim), 9
subjects. Conformer's architecture hasn't been given the same tuning attention as
EEGNet at this point (cropping, in particular). Numbers here should be read as a
snapshot of an ongoing comparison, not a final result — see `HANDOFF.md` for
open items.
