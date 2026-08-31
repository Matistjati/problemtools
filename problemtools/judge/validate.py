from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..diagnostics import Diagnostics
from ..metadata import Metadata
from ..model import DEFAULT_VALIDATOR, TestCase
from ..run import Program
from .result import SubmissionResult

_PRECISION_MAX_TOKENS = 1_000_000


def parse_float_tolerances(flags: list[str]) -> tuple[float | None, float | None]:
    """Parse abs and rel tolerance values from default-validator flags. Mirrors default_validator.cc:135-150."""
    abs_tol: float | None = None
    rel_tol: float | None = None
    i = 0
    while i < len(flags):
        f = flags[i]
        if f in ('float_absolute_tolerance', 'float_relative_tolerance', 'float_tolerance') and i + 1 < len(flags):
            try:
                v = float(flags[i + 1])
            except ValueError:
                i += 1
                continue
            if f == 'float_absolute_tolerance':
                abs_tol = v
            elif f == 'float_relative_tolerance':
                rel_tol = v
            else:
                abs_tol = v
                rel_tol = v
            i += 2
        else:
            i += 1
    return abs_tol, rel_tol


def _read_tokens(path: Path) -> list[str]:
    try:
        with open(path, errors='replace') as f:
            return f.read().split()
    except OSError:
        return []


def _compute_precision(
    team_path: Path,
    judge_path: Path,
    both_tols_set: bool,
) -> tuple[float, float, float] | None:
    """Compute (max_abs_err, max_rel_err, max_best_err) over float-token pairs.

    Returns None if the team output is structurally incompatible (different token count, or any
    non-float token pair that doesn't string-match) — in those cases the failure isn't a
    float-tolerance issue and a precision number would be misleading. max_best_err is 0 when
    both_tols_set is False (it's only meaningful when both tolerances apply).
    """
    judge_tokens = _read_tokens(judge_path)
    team_tokens = _read_tokens(team_path)
    if not judge_tokens or len(judge_tokens) != len(team_tokens):
        return None
    max_abs = 0.0
    max_rel = 0.0
    max_best = 0.0
    saw_float_pair = False
    n = min(len(judge_tokens), _PRECISION_MAX_TOKENS)
    for i in range(n):
        j_str = judge_tokens[i]
        t_str = team_tokens[i]
        try:
            jval = float(j_str)
            tval = float(t_str)
        except ValueError:
            # Default validator falls back to (case-insensitive) string compare for non-float tokens.
            if j_str.lower() != t_str.lower():
                return None
            continue
        saw_float_pair = True
        abs_err = abs(jval - tval)
        if jval == 0:
            rel_err = 0.0 if abs_err == 0 else float('inf')
        else:
            rel_err = abs_err / abs(jval)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        if both_tols_set:
            best = min(rel_err, abs_err)
            max_best = max(max_best, best)
    if not saw_float_pair:
        return None
    return max_abs, max_rel, max_best


def _get_feedback(feedback_dir: Path) -> str | None:
    all_feedback = []
    for path in feedback_dir.iterdir():
        if path.stat().st_size == 0:
            continue
        all_feedback.append(f'=== {path.name}: ===')
        # Note: The file could contain non-unicode characters, "replace" to be on the safe side
        with open(path, errors='replace') as f:
            # Cap amount of feedback per file at some high-ish
            # size, so that a buggy validator spewing out lots of
            # data doesn't kill us.
            all_feedback.append(f.read(128 * 1024))
    return '\n'.join(all_feedback) if all_feedback else None


def _parse_validator_result(
    val: Program,
    status: int,
    feedback_dir: Path,
    metadata: Metadata,
) -> SubmissionResult:
    if not os.WIFEXITED(status):
        return SubmissionResult(
            'JE',
            reason=f'output validator {val} crashed, status {status}',
            additional_info=_get_feedback(feedback_dir),
        )

    ret = os.WEXITSTATUS(status)
    if ret not in [42, 43]:
        return SubmissionResult(
            'JE',
            reason=f'output validator {val} exited with status {ret}',
            additional_info=_get_feedback(feedback_dir),
        )

    score_file = feedback_dir / 'score.txt'

    if ret == 43:
        if score_file.is_file():
            return SubmissionResult('JE', reason='validator produced "score.txt" on a WA verdict')
        return SubmissionResult('WA', additional_info=_get_feedback(feedback_dir))

    if not metadata.is_custom_score_allowed() and score_file.is_file():
        return SubmissionResult('JE', reason='validator produced "score.txt" but problem does not have custom scoring activated')

    score: float | None = None
    if metadata.is_custom_score_allowed():
        if score_file.is_file():
            try:
                score = float(score_file.read_text())
            except Exception as e:
                return SubmissionResult('JE', reason=f'failed to parse validator score: {e}')
        elif metadata.is_multi_pass() and (feedback_dir / 'nextpass.in').is_file():
            score = 0.0
        elif metadata.is_custom_score_mandatory():
            return SubmissionResult('JE', reason='problem has custom scoring but validator did not produce "score.txt"')

    return SubmissionResult('AC', score=score)


def _validate_output(
    testcase: TestCase,
    submission_output: Path,
    output_validator: Program,
    metadata: Metadata,
    execution_dir: Path,
    base_dir: Path,
    diag: Diagnostics,
    infile: Path | None = None,
) -> SubmissionResult:
    feedback_dir = execution_dir / 'feedback'
    effective_infile = infile if infile is not None else testcase.infile
    flags = testcase.output_validator_flags
    val_timelim = metadata.limits.validation_time
    val_memlim = metadata.limits.validation_memory

    output_size = os.path.getsize(submission_output) / 1024.0 / 1024.0
    if output_size > metadata.limits.output:
        return SubmissionResult(
            'OLE', reason=f'output ({output_size:.1f} MiB) exceeds output limit ({metadata.limits.output} MiB)'
        )

    if not output_validator.compile(base_dir).success:
        return SubmissionResult('JE', reason=f'output validator {output_validator} failed to compile')
    val_stdout = execution_dir / 'val_stdout'
    val_stderr = execution_dir / 'val_stderr'
    status, _ = output_validator.run(
        infile=str(submission_output),
        args=[str(effective_infile), str(testcase.ansfile), str(feedback_dir) + os.sep] + flags,
        timelim=val_timelim,
        memlim=val_memlim,
        outfile=str(val_stdout),
        errfile=str(val_stderr),
    )
    for label, path in [('stdout', val_stdout), ('stderr', val_stderr)]:
        try:
            if content := path.read_text(errors='replace'):
                diag.debug(f'Validator {label}: {content}')
        except OSError as e:
            diag.info(f'Failed to read validator output: {e}')
    result = _parse_validator_result(output_validator, status, feedback_dir, metadata)
    if result.verdict in ('AC', 'WA') and output_validator is DEFAULT_VALIDATOR:
        abs_tol, rel_tol = parse_float_tolerances(flags)
        if abs_tol is not None or rel_tol is not None:
            both_set = abs_tol is not None and rel_tol is not None
            triplet = _compute_precision(submission_output, testcase.ansfile, both_set)
            if triplet is not None:
                result.max_abs_err = triplet[0]
                result.max_rel_err = triplet[1]
                if both_set:
                    result.max_best_err = triplet[2]
    return result


def validate_output(
    testcase: TestCase,
    submission_output: Path,
    output_validator: Program,
    metadata: Metadata,
    base_dir: Path,
    diag: Diagnostics,
) -> SubmissionResult:
    with tempfile.TemporaryDirectory(dir=base_dir) as exec_dir:
        execution_dir = Path(exec_dir)
        (execution_dir / 'feedback').mkdir()
        return _validate_output(testcase, submission_output, output_validator, metadata, execution_dir, base_dir, diag)
