# EEG decoding project

Read `docs/STYLE_GUIDE.md` in full before writing or editing any Python or
`README.md` in this repo. It is the source of truth; what follows is a condensed
checklist, not a replacement.

## Mechanical rules

- NumPy-style docstrings (`Parameters`, `Returns`, `Raises`) on every public
  function, module, and class.
- No type hints, except a parameter that receives a project config dataclass
  (`cfg: DataConfig`, `cfg: Config`), annotated for navigation, nothing else.
- No ` -- ` or em dash, in code or in Markdown. Start a new sentence instead.
- Comments are 1-2 lines, tied to one specific line, never restating what a
  docstring already covers and never narrating the debugging journey.

## Structural rules

- Config and dataset identity are `@dataclass` only (`config.py`, `datasets.py`).
  Everything else (EEG trials, labels, sessions, checkpoint manifests) is a
  plain numpy array, list, or dict, never a wrapper class.
- `snake_case`, single responsibility. `train.py` only trains and checkpoints;
  `evaluate.py` only scores saved checkpoints against held-out data. Never merge
  the two.
- Every path derives from `config.REPO_ROOT`, never a hardcoded absolute path or
  a `"../data"`-style relative string.
- `ALL_CAPS` constants are defined at their point of first use, not hoisted to
  the top of the file.
- Minimal error handling: a `raise ValueError(...)` for a real invariant is
  fine. Don't add defensive handling for a failure mode that hasn't been hit.
- Adding a new dataset means adding one entry to each of `datasets.py`'s two
  registries; it should never require touching `train.py`/`evaluate.py`/`sweep.py`.

## README rules

- Fixed section order: Task, Results, Methodology, Repository structure, Usage,
  References, Limitations.
- State conclusions, not process. No hedging language. Cite every architecture
  or dataset on first mention (`EEGNet [1]`), resolved in a numbered, IEEE-style
  `## References` list.
- No authorship/tooling mentions, no cross-references to internal working docs.

## Project layout

`src/` is flat aside from `models/`: `config.py` (hyperparameters, config
building), `datasets.py` (per-dataset registry), `data_loader.py` (loading,
splitting, normalization, cropping), `csp_init.py` (CSP-based weight init),
`models/{eegnet,conformer}.py` (architectures), and the three pipeline entry
points `train.py` (trains and checkpoints), `evaluate.py` (scores checkpoints),
`sweep.py` (hyperparameter search). `scripts/` holds the Colab CLI wrappers,
outside `src/`.
