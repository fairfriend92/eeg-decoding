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

## Methodology

Evaluated through NeuralBench's own harness rather than this project's
`evaluate.py`. REVE, the patch transformer, the EEG Conformer, and EEGNet are all fine-tuned end-to-end
(not linear-probed) on the same cross-subject Dreyer2023 split described
above, up to 40 epochs, batch size 64, early stopping on validation
balanced accuracy (patience 5). REVE's, the patch transformer's, and the Conformer's reported numbers are
mean ± std across 3 seeds; EEGNet's is a single seed (33).

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
run's cached results.

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
