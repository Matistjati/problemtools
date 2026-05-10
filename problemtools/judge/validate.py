from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..diagnostics import Diagnostics
from ..metadata import Metadata
from ..run import Program
from .result import SubmissionResult

if TYPE_CHECKING:
    from ..verifyproblem import TestCase


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


def _compute_precision_ratio(team_path: Path, judge_path: Path, abs_tol: float | None, rel_tol: float | None) -> float | None:
    """Per token, take the smaller of (abs_err / abs_tol, rel_err / rel_tol) using only the tolerances that
    are set — the validator passes that token if either is <= 1, so the smaller of the two reflects the
    metric the token "passed under". Return the max of those ratios over all tokens (and -∞ if no float
    tokens were comparable). Result is dimensionless: <1 means within tolerance, =1 is exactly at the edge.
    """
    if abs_tol is None and rel_tol is None:
        return None
    judge_tokens = _read_tokens(judge_path)
    team_tokens = _read_tokens(team_path)
    if not judge_tokens or not team_tokens:
        return None
    max_ratio: float | None = None
    n = min(len(judge_tokens), len(team_tokens), _PRECISION_MAX_TOKENS)
    for i in range(n):
        try:
            jval = float(judge_tokens[i])
            tval = float(team_tokens[i])
        except ValueError:
            continue
        abs_err = abs(jval - tval)
        candidates: list[float] = []
        if abs_tol is not None:
            if abs_tol > 0:
                candidates.append(abs_err / abs_tol)
            else:
                candidates.append(0.0 if abs_err == 0 else float('inf'))
        if rel_tol is not None:
            if jval == 0:
                candidates.append(0.0 if abs_err == 0 else float('inf'))
            elif rel_tol > 0:
                candidates.append((abs_err / abs(jval)) / rel_tol)
            else:
                candidates.append(0.0 if abs_err == 0 else float('inf'))
        if not candidates:
            continue
        ratio = min(candidates)
        if max_ratio is None or ratio > max_ratio:
            max_ratio = ratio
    return max_ratio


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

    if ret == 43:
        return SubmissionResult('WA', additional_info=_get_feedback(feedback_dir))

    # ret == 42 (AC); check score handling
    score_file = feedback_dir / 'score.txt'

    if not metadata.is_custom_score_allowed() and score_file.is_file():
        return SubmissionResult('JE', reason='validator produced "score.txt" but problem does not have custom scoring activated')

    score: float | None = None
    if metadata.is_custom_score_mandatory():
        if score_file.is_file():
            try:
                score = float(score_file.read_text())
            except Exception as e:
                return SubmissionResult('JE', reason=f'failed to parse validator score: {e}')
        elif metadata.is_multi_pass() and (feedback_dir / 'nextpass.in').is_file():
            score = 0.0
        else:
            return SubmissionResult('JE', reason='problem has custom scoring but validator did not produce "score.txt"')

    return SubmissionResult('AC', score=score)


def _validate_output(
    testcase: TestCase,
    submission_output: Path,
    output_validator: Program,
    metadata: Metadata,
    execution_dir: Path,
    diag: Diagnostics,
    infile: Path | None = None,
) -> SubmissionResult:
    feedback_dir = execution_dir / 'feedback'
    effective_infile = infile if infile is not None else testcase.infile_path
    flags = testcase.output_validator_flags
    val_timelim = metadata.limits.validation_time
    val_memlim = metadata.limits.validation_memory

    if not output_validator.compile()[0]:
        return SubmissionResult('JE', reason=f'output validator {output_validator} failed to compile')
    val_stdout = execution_dir / 'val_stdout'
    val_stderr = execution_dir / 'val_stderr'
    status, _ = output_validator.run(
        infile=str(submission_output),
        args=[str(effective_infile), str(testcase.ansfile_path), str(feedback_dir) + os.sep] + flags,
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
    if result.verdict == 'AC' and testcase._problem.output_validators.uses_default_validator():
        abs_tol, rel_tol = parse_float_tolerances(flags)
        if abs_tol is not None or rel_tol is not None:
            result.precision = _compute_precision_ratio(submission_output, testcase.ansfile_path, abs_tol, rel_tol)
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
        return _validate_output(testcase, submission_output, output_validator, metadata, execution_dir, diag)
