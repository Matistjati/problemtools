# CLAUDE.md

Guidance for working in this repository.

## What this is

This is a **fork of Kattis' [`problemtools`](https://github.com/Kattis/problemtools)** — tools for
managing competitive-programming problem packages in the [Kattis problem package
format](https://www.kattis.com/problem-package-format/). The fork (`llm-additions` branch off
`master`) focuses on improving the workflow for **larger, multi-subtask problems**.

Three command-line programs are provided (entry points in `pyproject.toml`):

- `verifyproblem` (`problemtools/verifyproblem.py`) — run a full check on a problem package, compiling
  and running every submission on every test case. This is where almost all the fork's changes live.
- `problem2pdf` (`problemtools/problem2pdf.py`) — render a problem statement to PDF.
- `problem2html` (`problemtools/problem2html.py`) — render a problem statement to HTML.

Run any with `-h` for its arguments. Two package format versions are supported: `legacy` (production)
and `2023-07-draft` (partial — scoring still behaves like legacy). Use legacy for real problems.

## Running and building

Method 3 from the README (run from the repo) is the dev workflow:

```sh
python3 -m venv venv            # Python 3.11+ required
venv/bin/pip install -r requirements.txt
make                            # builds support programs (checktestdata via git submodule + support/)
bin/verifyproblem.sh examples/hello
```

`make` runs `git submodule update --init` (for `support/checktestdata`) and `make -C support`. Tests
live in `tests/` (pytest): `test_verify_hello.py`, `test_default_validator.py`, `test_metadata.py`,
etc. Example problem packages are in `examples/` and `tests/problems/`.

## Architecture

- `problemtools/verifyproblem.py` — the driver. `model.load_problem()` parses the package, then
  `ProblemVerifier.check(context)` runs a list of `_CheckStep`s (one per `checks.check_*` function,
  tagged with the `-p/--parts` group it belongs to) against it. Also holds the `main()` argparser.
- `problemtools/model/` — the parsed package, as plain frozen dataclasses with no checking logic:
  `Problem`, `TestCase`/`TestDataGroup`, `Submission`/`Submissions`, `InputValidators`/
  `OutputValidators`, `Graders`, `Statements`, `Attachments`, `Includes`.
- `problemtools/checks/` — one module per part, each a `check_<part>(model..., diag)` function that
  reports through `Diagnostics` rather than owning any state.
- `problemtools/results_table.py` — the fork's live `rich` results table (see below).
- `problemtools/context.py` — `Context` carries per-run flags and the thread-pool executors through the
  check tree.
- `problemtools/judge/` — the judging core:
  - `submission_judge.py` (`SubmissionJudge`) — runs a submission across test cases / groups, grades.
  - `execute.py`, `grade.py`, `validate.py` — execute a test case, grade a group, run output validators.
  - `result.py` (`SubmissionResult`) — verdict, runtime, score, and (fork-added) precision/first-failure.
  - `cache.py` — `ResultStore` (in-process, thread-safe) plus the fork-added `ResultCache` (cross-run).
- `problemtools/diagnostics.py` — `LoggingDiagnostics` collects errors/warnings.
- `problemtools/run/` — language detection, compilation, sandboxed execution, resource limits.
- `problemtools/metadata.py`, `config.py`, `languages.py`, `formatversion.py` — package/config parsing.
- `problemtools/ProblemPlasTeX/`, `tex2html.py`, `md2html.py`, `statement_util.py` — statement rendering.

## What this fork adds over `master`

All additions are oriented around verifying multi-subtask problems faster and reading results more
easily. New runtime dependency: **`rich>=13.0`** (`requirements.txt`).

New `verifyproblem` CLI flags (`argparser_basic_arguments` in `verifyproblem.py`):

- `--score` — show per-subtask **score** instead of runtime in the results table.
- `--cache` — cache per-testcase results across runs in `/tmp/problemtools/cache` to speed up repeated
  verification.

The features:

1. **Live rich results table** (`SubtaskResultsTable` in `results_table.py`, driven from
   `checks/submissions.py`). Replaces upstream's `_print_results_table` with a live-updating `rich`
   table: one row per submission, one column per
   subtask group (sample + secret subgroups). Rows are grouped/sorted by verdict category
   (`AC, PAC, WA, RTE, TLE`). AC cells show runtime (or score with `--score`). It installs itself as a
   `rich.live.Live` and reroutes logging stream handlers so log output coexists with the live table.
   Does **not** support `EXPECTED_GRADE`.

2. **Cross-run result caching** (`ResultCache` in `judge/cache.py`, wired in `submission_judge.py`).
   Keys on a SHA-256 of language + source contents + input + answer + validator source + validator
   flags + memory limit, stored as JSON under `/tmp/problemtools/cache`. Disabled for interactive
   problems. Cached results are reclassified against the current time limit (a TLE cached at a higher
   limit can be reused; a faster limit cannot be served upward). The in-process `ResultStore` (thread
   coordination via `claim`/`complete`/`get`, with `Future`s for in-flight runs) is from master; the
   persistent `ResultCache` is the fork addition.

3. **Float-precision column(s)** (`parse_float_tolerances` / `_compute_precision` in
   `judge/validate.py`; `max_abs_err` / `max_rel_err` / `max_best_err` on `SubmissionResult`). For
   problems using the default validator with `float_tolerance` / `float_absolute_tolerance` /
   `float_relative_tolerance`, computes per-testcase the max absolute and relative errors over float
   tokens (and, when both tolerances are set, the per-token `min(abs_err, rel_err)`), propagated up
   groups as a max. The table shows `Abs precision`, `Rel precision`, a combined `Precision`, or both,
   depending on which tolerances are configured (see `_precision_mode` in `SubtaskResultsTable`). The
   parsing and comparison mirror `default_validator.cc`.

4. **Thread-safe diagnostics** (`diagnostics.py`). `_Counts` now holds a `threading.Lock` so
   error/warning counters are safe under the new concurrency.

## Conventions

- Python 3.11+; `from __future__ import annotations` is used throughout. Type hints expected
  (`mypy.ini`, `py.typed`). `pre-commit` is configured (`.pre-commit-config.yaml`).
- Keep the model/checks split: `model/` parses and holds data, `checks/` inspects it and reports via
  `diag.msg`/`diag.warning`/`diag.error` (plain `print` for non-diagnostic progress output), with
  per-run flags threaded through `Context`.
- `ruff` (pinned in `.pre-commit-config.yaml`) and `mypy` (`disallow_untyped_defs`) both must pass.
- `todo.md` tracks the fork's open items (e.g. show per-group worth in subtask tables).
