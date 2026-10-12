# NeurIPS 2026 EEG/EMG Foundation Challenge Benchmark

This document covers this project's benchmark work for the NeurIPS 2026
EEG/EMG Foundation Challenge [1] (Track 2, BCI decoding): comparing the
patch transformer and EEGNet against REVE [3], the Challenge's pretrained
foundation-model baseline, through NeuralBench, the Challenge's official
benchmark harness. Assumes the project layout, architectures, and pipeline
described in [README.md](README.md).

## Task

### NeurIPS 2026 Challenge benchmark (Dreyer2023, cross-subject)

Benchmarks the patch transformer and EEGNet against REVE [3] via NeuralBench. Uses the
same Dreyer2023 [4] corpus as this project's own `dreyer2023` preset, but
NeuralBench's own preprocessing and cross-subject split (a fixed set of 21
subjects, 61-81, held out as test; the remaining subjects for train/val),
not this project's own within-session split, so these results are not
directly comparable to the Dreyer2023 numbers in [README.md](README.md).

### Cross-subject ensemble on the warm-up split (this project's pipeline)

The same 21 test participants (61 to 81) scored with this project's own
pipeline instead of NeuralBench's: EEGNet [5], the patch transformer,
Riemannian tangent-space [6] and filter-bank CSP [7] classifiers, and REVE [3]
fine-tuned on pooled participants, combined by a selection harness that picks
a member subset on held-out validation participants.

### Graz+BrainHero (not yet released)

Registered as a placeholder dataset (`graz_brainhero`, raises immediately if
selected) for the Challenge's Track 2 longitudinal corpus [1]: 3 cued mental
commands (kinesthetic motor imagery, mental calculation, word association)
across 6 sessions per participant, evaluating whether a model calibrated on
early sessions still decodes reliably on later, unseen-day sessions.

## Results

### NeurIPS 2026 Challenge benchmark (Dreyer2023, cross-subject via NeuralBench)

| | Balanced accuracy | AUROC |
|---|---|---|
| REVE [3] (pretrained foundation model, 69.4M params) | 0.782 ± 0.013 | 0.858 ± 0.020 |
| EEGNet, ours | 0.742 (1 seed) | 0.821 (1 seed) |
| Patch transformer, ours (0.31M params) | 0.735 ± 0.007 | 0.810 ± 0.012 |
| EEG Conformer [2], ours (0.44M params) | 0.716 ± 0.014 | 0.789 ± 0.019 |

REVE, the patch transformer, and the EEG Conformer report mean ± std across 3 seeds (sample std);
EEGNet reports a single seed (33). REVE outperforms this project's own
architectures by roughly 4-5 balanced accuracy points. EEGNet matches the patch
transformer within one point while using roughly an order of magnitude fewer
parameters.

### Cross-subject ensemble members (this project's pipeline)

Balanced accuracy on the 21 test participants, mean ± std across participants:

| | Balanced accuracy |
|---|---|
| EEGNet | 0.813 ± 0.107 |
| REVE [3], fine-tuned on pooled participants | 0.794 ± 0.089 |
| Patch transformer | 0.736 ± 0.094 |
| Riemannian tangent space [6] | 0.641 ± 0.137 |
| FBCSP [7] | 0.625 ± 0.123 |
| EEGNet + REVE (subset chosen on validation participants) | 0.812 ± 0.098 |
| Uniform average of all five members | 0.806 ± 0.101 |

EEGNet is the best single member and no ensemble exceeds it. The paired
difference of the selected EEGNet + REVE pair against EEGNet is
-0.001 ± 0.007 (standard error over test participants), and EEGNet leads
Riemannian and FBCSP by +0.172 ± 0.030 and +0.188 ± 0.032. Adding the patch
transformer, Riemannian or FBCSP to a uniform average lowers it, and the
validation-based selection avoids that cost (+0.010 ± 0.003 over the uniform
average of EEGNet, REVE and the patch transformer). Fine-tuned REVE
(single seed) sits within one standard deviation of the 3-seed NeuralBench REVE
score in the table above (0.782 ± 0.013).

## Methodology

Evaluated through NeuralBench's own harness rather than this project's
`evaluate.py`. REVE, the patch transformer, the EEG Conformer, and EEGNet are all fine-tuned end-to-end
(not linear-probed) on the same cross-subject Dreyer2023 split described
above, up to 40 epochs, batch size 64, early stopping on validation
balanced accuracy (patience 5). REVE's, the patch transformer's, and the Conformer's reported numbers are
mean ± std across 3 seeds; EEGNet's is a single seed (33).

The ensemble members above use this project's pipeline, not NeuralBench's:
MOABB epochs from 0 to 4 s after the cue, 4 to 38 Hz, 250 Hz (200 Hz for
REVE), each participant z-scored per channel with its own statistics. The
66 non-test participants are split whole into 52 fitting and 14 validation
participants (`random_state=33`). Test
probabilities come from a fit on all 66 training participants, validation
probabilities from a fit on the fitting participants only. Member subsets
are averaged uniformly, and the subset with the best validation negative
log-likelihood, shared by all test participants, is scored on the test
participants.

Warm-starting either model's shared trunk (`temporal_conv`/`separable_conv` for
EEGNet, the patch embedding and encoder for the patch transformer) with weights
pretrained jointly on BCI2a and Scherer2015 (`src/pretrain_trunk.py`, applied
through `--pretrained-trunk`) gives no measurable change: EEGNet about 0.73
against 0.742 from random initialization, patch transformer 0.734 against
0.735 balanced accuracy.

## Repository structure

Adds to the layout in [README.md](README.md):

```
scripts/
  neuralbench_eval.py      runs one of our models, or a NeuralBench-native
                           baseline (e.g. REVE), through NeuralBench's harness
  colab_neuralbench.sh     Colab wrapper for neuralbench_eval.py
  submission/
    eegnet_submission.py            Codabench Track 02 entry: this project's EEGNet,
                                    inlined per the submission contract
    patch_transformer_submission.py same contract, this project's patch
                                    transformer inlined
    build_submission.py             packages a checkpoint + the selected model's
                                    submission file into an upload-ready ZIP
  colab_cross_subject_members.sh    Colab wrapper for cross_subject_members.py
  colab_reve_cross_subject.sh       Colab wrapper for reve_cross_subject.py
src/
  cross_subject_members.py   EEGNet, patch transformer, Riemannian and FBCSP
                             members fitted on pooled participants, writing
                             probability dumps
  reve_cross_subject.py      REVE fine-tuned on pooled participants
  ensemble_harness.py        member subset selection on validation
                             participants, scored on test participants
```

## Usage

Run from a Colab session; see [README.md](README.md)'s "Running on Colab" for
one-time setup and authentication. `scripts/colab_neuralbench.sh` installs
`neuralbench==0.3.1 moabb==1.7.2` (kept out of `requirements.txt`, a heavy
dependency set only needed here, pinned to exact versions so the Drive-backed
cache keys stay stable across sessions), syncs `src/` and
`scripts/neuralbench_eval.py`, then runs the latter on the session:

```
scripts/colab_neuralbench.sh [-s SESSION] [--gpu TYPE] [--skip-sync] -- [neuralbench_eval.py args]
```

`NEURALBENCH_DATA_DIR`/`SAVE_DIR`/`CACHE_DIR` all point directly at the
session's mounted Drive, so a download or checkpoint is durable the moment
it is written.

Download a dataset's corpus (no GPU needed):
```
scripts/colab_neuralbench.sh -s downloader -- --download --dataset dreyer2023
```

Check that one of our models satisfies NeuralBench's forward-signature
contract, without running a full evaluation:
```
scripts/colab_neuralbench.sh -s trainer --gpu A100 -- --check-only
```

Evaluate one of our models (`eegnet`, `patch_transformer`, or `conformer`) through
`check_model`/`evaluate_model`:
```
scripts/colab_neuralbench.sh -s trainer --gpu A100 -- --model patch_transformer --downstream-wrapper finetune_flatten
```
`--debug` reduces epochs/batches and forces local execution: a wiring check,
not a real score.

Run a NeuralBench-native baseline (e.g. REVE) for real, bypassing
`check_model`/`evaluate_model`:
```
scripts/colab_neuralbench.sh -s trainer --gpu A100 -- --model reve --dataset dreyer2023
```
`--collect-only` skips recomputation and reads back an already-completed
run's cached results, for any `--model` including `eegnet` and `conformer`.
The other arguments must match the original run, and a run of one of our
models made with `--save-checkpoint` is read back by passing that flag too.

Train one of our models through NeuralBench's own harness and keep the
trained checkpoint (backbone plus the linear probe the `finetune_*`
downstream wrapper trains on top of it), rather than just its scores:
```
scripts/colab_neuralbench.sh -s trainer --gpu A100 -- \
    --model eegnet --save-checkpoint ckpt.pt
```
Package it into a Codabench-ready ZIP (`--model` selects which of
`eegnet_submission.py`/`patch_transformer_submission.py` gets zipped as
`submission.py`; matches whichever `--model` trained the checkpoint):
```
python scripts/submission/build_submission.py ckpt.pt --model eegnet
```
Upload the resulting ZIP through Codabench's My Submissions tab (Track 02,
warm-up or sealed phase).

Fit the cross-subject ensemble members locally (the Riemannian, FBCSP and
EEGNet members; the first run epochs every participant and caches the result)
and score the ensembles:
```
python src/cross_subject_members.py riemann fbcsp eegnet --dataset dreyer2023
python src/ensemble_harness.py eegnet riemann fbcsp --dataset dreyer2023/cross_subject --metric bal_acc
```
The pooled fits need about 8 GB of memory. Run the patch transformer and REVE
members on a Colab A100:
```
scripts/colab_cross_subject_members.sh -s SESSION --gpu A100 --dataset dreyer2023 --members patch_transformer
scripts/colab_reve_cross_subject.sh -s SESSION --gpu A100 --dataset dreyer2023
```
Each script downloads its dump to `output/dreyer2023/cross_subject/` and
stops the session.

## References

1. "EEG/EMG Foundation Challenge 2026," Brain and Body Workshop at NeurIPS 2026,
   Track 2 (BCI decoding).
   [Tracks & data](https://neural-interfaces26.github.io/tracks.html) ·
   [rules & FAQ](https://neural-interfaces26.github.io/rules.html).
2. Y. Song, Q. Zheng, B. Liu, and X. Gao, "EEG Conformer: Convolutional transformer
   for EEG decoding and visualization," *IEEE Transactions on Neural Systems and
   Rehabilitation Engineering*, vol. 31, pp. 710-719, 2023.
3. Y. El Ouahidi, J. Lys, P. Thölke, N. Farrugia, B. Pasdeloup, V. Gripon,
   K. Jerbi, and G. Lioi, "REVE: A foundation model for EEG, adapting to any
   setup with large-scale pretraining on 25,000 subjects," in *Advances in
   Neural Information Processing Systems 38 (NeurIPS 2025)*, 2025.
4. P. Dreyer, A. Roc, L. Pillette, S. Rimbert, and F. Lotte, "A large EEG database
   with users' profile information for motor imagery brain-computer interface
   research," *Scientific Data*, vol. 10, p. 580, 2023.
   [DOI: 10.1038/s41597-023-02445-z](https://doi.org/10.1038/s41597-023-02445-z).
5. V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and
   B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based
   brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5,
   p. 056013, 2018.
6. A. Barachant, S. Bonnet, M. Congedo, and C. Jutten, "Multiclass
   brain-computer interface classification by Riemannian geometry," *IEEE
   Transactions on Biomedical Engineering*, vol. 59, no. 4, pp. 920-928, 2012.
7. K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan, "Filter bank common spatial
   pattern (FBCSP) in brain-computer interface," in *Proc. IEEE International
   Joint Conference on Neural Networks (IJCNN)*, pp. 2390-2397, 2008.

## Limitations

The patch transformer's headline benchmark numbers above are a 3-seed mean; EEGNet's is
a single seed, run once to validate the submission pipeline end to end
rather than to produce a final number, so it is not yet directly
comparable on variance. The Graz+BrainHero dataset is not yet released;
its preset is registered but raises immediately if selected. The EEG Conformer
[2] has no Codabench submission file yet. Both
`scripts/submission/eegnet_submission.py` and `patch_transformer_submission.py`,
plus the checkpoint-saving and packaging path behind them, have been
verified end to end on Colab (real, non-debug runs, checkpoints packaged
into valid ZIPs), but not yet against a real Codabench upload.

The ensemble members use this project's preprocessing and one seed per member,
so their numbers are not directly comparable to the NeuralBench rows above.
EEGNet scores 0.813 here against 0.742 in NeuralBench on the same test
participants, and the cause is not isolated: per-participant normalization,
the training recipe, and the epoching are candidates. The EEG Conformer is not
run as an ensemble member. Only the 21 warm-up test participants are scored,
and the warm-up scores are indicative only.
