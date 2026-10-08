# EEG decoding project

Read `docs/STYLE_GUIDE.md` in full before writing or editing any Python or
top-level README file (`README.md`, `README_NEURIPS2026.md`) in this repo. It
is the source of truth; what follows is a condensed checklist, not a
replacement.

## Colab sessions: always stream output, always heartbeat

**Every** `colab exec` call, whether it runs `scripts/colab_*.sh` or a raw
ad hoc snippet, must satisfy both of these, no exceptions:

- Wire its output into a visible terminal (e.g. `mcp__terminal__run_in_terminal`),
  never a silent background call whose output only appears after it returns.
- Wrap any command or code that can run silently for more than a few seconds
  with a periodic heartbeat print (every 15-60s), whether that's
  `scripts/neuralbench_eval.py`'s `_run_with_heartbeat` for a subprocess, or an
  inline `threading.Thread` heartbeat for in-process code (e.g. `ds.get_data(...)`).

A long silent stretch is suspected to kill the Colab tunnel while the
kernel/VM stays alive (`colab status` reports `IDLE`, the session looks
alive, but every subsequent `colab exec` fails). This has now been observed
directly: an ad hoc, non-heartbeat-wrapped `colab exec` call died with
`RuntimeError: Connection was lost` at its own bootstrap step. See
[ANALYSIS_NOTES.md#colab-session-stability-silent-tunnel-hypothesis](docs/ANALYSIS_NOTES.md#colab-session-stability-silent-tunnel-hypothesis).

## Colab exec timeout: never a tuned guess

`colab exec --timeout` is a hard client-side ceiling on that one call, not
a bound on the remote kernel and not reset by the heartbeat above. Never
size it to how long a run is expected to take: guessing low loses the live
connection, with no way to reconnect to that same output, while the kernel
keeps running unaffected. Pass one large, effectively unbounded value
instead, and check on a long run from a second, separately Drive-mounted
session (list `NEURALBENCH_SAVE_DIR`'s checkpoint timestamps, for
instance) rather than waiting on the busy call itself. To actually stop
something, use `colab restart-kernel` or `colab stop` on the busy session
directly. See
[ANALYSIS_NOTES.md#colab-exec---timeout-is-a-client-side-ceiling-not-a-kernel-guard](docs/ANALYSIS_NOTES.md#colab-exec---timeout-is-a-client-side-ceiling-not-a-kernel-guard).

## Colab sessions: every run gets a fresh session

Create a new session for every training or evaluation run, and stop it when
the run is done. Never reuse a session that has already run something. The
same NeuralBench prepare pass took about 8 s per recording on a long-lived
session and was much faster on a freshly created one, with no code change in
between. If a run is unexpectedly slow, stop it, stop the session, and start
over on a new one before investigating. See
[ANALYSIS_NOTES.md#a-reused-colab-session-ran-the-same-prepare-pass-several-times-slower](docs/ANALYSIS_NOTES.md#a-reused-colab-session-ran-the-same-prepare-pass-several-times-slower).

## Don't open a terminal tab for a command that doesn't need one

`mcp__terminal__run_in_terminal` always opens a brand new tab, with no way
to target or reuse an existing one. Reserve it for commands that actually
need a visible, watchable, possibly long-running terminal: a
heartbeat-wrapped `colab exec` (per the rule below), a dev server, anything
the user might want to inspect mid-run or stop with Ctrl-C. For a short
command that returns within the turn (`colab new`, `colab drivemount`,
`colab stop`, `colab sessions`, a quick `colab exec` finishing in seconds),
use the `Bash` tool instead: it reuses one persistent shell and opens no
tab at all. Calling `run_in_terminal` once per trivial command scatters one
logical sequence of work across many tabs for no reason.

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
- `README.md` covers the project generally and this project's own research;
  `README_NEURIPS2026.md` covers everything specific to the NeurIPS 2026
  Challenge submission and assumes `README.md` as prerequisite reading. Content
  common to both (e.g. repository structure, pipeline usage) lives in
  `README.md` only, linked from the other rather than duplicated.

## Docs

- `docs/HISTORY.md` is an append-only log of completed work: what was done
  and its results, most recent last. Never open items, next steps, or
  reasoning.
- `docs/TO_DO.md` holds what remains: open items, blockers, next actions,
  and gotchas relevant to future work. Never a record of what's already
  done.
- `docs/ANALYSIS_NOTES.md` holds bug documentation, root-cause analysis, and
  technique/gotcha explanations, referenced from `HISTORY.md`/`TO_DO.md`
  rather than inlined there.
- When a `TO_DO.md` item is finished, move its outcome into `HISTORY.md`
  (trimmed to the result, not the journey it took) and delete it from
  `TO_DO.md`. This is what keeps either file from regrowing into one long,
  hard-to-navigate log.

## Project layout

`src/` is flat aside from `models/`: `config.py` (hyperparameters, config
building), `datasets.py` (per-dataset registry), `data_loader.py` (loading,
splitting, normalization, cropping), `csp_init.py` (CSP-based weight init),
`models/{eegnet,conformer,patch_transformer}.py` (architectures), and the three pipeline entry
points `train.py` (trains and checkpoints), `evaluate.py` (scores checkpoints),
`sweep.py` (hyperparameter search). `scripts/` holds the Colab CLI wrappers,
outside `src/`.
