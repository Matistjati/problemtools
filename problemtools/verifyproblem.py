#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import queue
import glob
import string
import hashlib
import collections
import os
import signal
import re
import shutil
import logging
import tempfile
import sys
import copy
import random
import traceback
import uuid
import difflib
from pathlib import Path

import colorlog
import yaml

from . import config
from . import languages
from . import metadata
from . import problem2html
from . import problem2pdf
from . import run
from . import statement_util
from .formatversion import FormatVersion, get_format_version
from .version import add_version_arg

from abc import ABC
from typing import Any, Callable, ClassVar, Literal, Pattern, Match, ParamSpec, TypeVar, cast
from pydantic import ValidationError

random.seed(42)

log = logging.getLogger(__name__)

Verdict = Literal['AC', 'TLE', 'OLE', 'MLE', 'RTE', 'WA', 'PAC', 'JE']


def is_TLE(status: int, may_signal_with_usr1: bool = False) -> bool:
    return os.WIFSIGNALED(status) and (
        os.WTERMSIG(status) == signal.SIGXCPU or (may_signal_with_usr1 and os.WTERMSIG(status) == signal.SIGUSR1)
    )


def is_RTE(status: int) -> bool:
    return not os.WIFEXITED(status) or bool(os.WEXITSTATUS(status))


class SubmissionResult:
    def __init__(self, verdict: str, score: float | None = None, reason: str | None = None, additional_info: str | None = None):
        self.verdict = verdict
        self.score = score
        self.reason = reason
        self.additional_info = additional_info
        self.testcase: TestCase | None = None
        self.runtime_testcase: TestCase | None = None
        self.runtime = -1.0
        self.ac_runtime = -1.0
        self.ac_runtime_testcase: TestCase | None = None
        self.validator_first = False
        self.sample_failures: list[SubmissionResult] = []

    def set_ac_runtime(self) -> None:
        if self.verdict == 'AC':
            self.ac_runtime = self.runtime
            self.ac_runtime_testcase = self.runtime_testcase

    def __str__(self) -> str:
        verdict = self.verdict
        details = []

        if verdict == 'AC' and self.score is not None:
            verdict += f' ({self.score:.0f})'

        if self.reason is not None:
            details.append(self.reason)
        if self.testcase is not None:
            details.append(f'testcase: {self.testcase}')
        if self.runtime != -1:
            details.append(f'CPU: {self.runtime:.2f}s @ {self.runtime_testcase}')

        if len(details) == 0:
            return verdict
        return f'{verdict} [{", ".join(details)}]'


class VerifyError(Exception):
    pass


_T = TypeVar('_T')
_P = ParamSpec('_P')


class Context:
    def __init__(self, args: argparse.Namespace, executor: ThreadPoolExecutor | None) -> None:
        self.data_filter: Pattern[str] = args.data_filter
        self.submission_filter: Pattern[str] = args.submission_filter
        self.fixed_timelim: float | None = args.fixed_timelim
        self.executor = executor
        self._background_work: list[concurrent.futures.Future[object]] = []

    def submit_background_work(self, job: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> None:
        assert self.executor
        self._background_work.append(self.executor.submit(job, *args, **kwargs))

    def wait_for_background_work(self) -> None:
        concurrent.futures.wait(self._background_work)


class ProblemAspect(ABC):
    errors: int = 0
    warnings: int = 0
    _check_res: bool | None = None
    problem: Problem

    def __append_additional_info(self, msg: str, additional_info: str | None) -> str:
        max_additional_info = self.problem.max_additional_info()
        if additional_info is None or max_additional_info <= 0:
            return msg
        additional_info = additional_info.rstrip()
        if not additional_info:
            return msg
        lines = additional_info.split('\n')
        if len(lines) == 1:
            return f'{msg} ({lines[0]})'
        if len(lines) > max_additional_info:
            lines = lines[:max_additional_info] + [f'[.....truncated to {max_additional_info} lines.....]']

        return f'{msg}:\n' + '\n'.join(' ' * 8 + line for line in lines)

    def __init__(self, name: str, problem: Problem) -> None:
        self.log = log.getChild(name)
        self.problem = problem

    def fatal(self, msg: str, additional_info: str | None = None, *args) -> None:
        self._check_res = False
        self._add_error()
        self.log.critical(self.__append_additional_info(msg, additional_info), *args)
        raise VerifyError(msg)

    def error(self, msg: str, additional_info: str | None = None, *args) -> None:
        self._check_res = False
        self._add_error()
        self.log.error(self.__append_additional_info(msg, additional_info), *args)
        if self.problem.bail_on_error():
            raise VerifyError(msg)

    def warning(self, msg: str, additional_info: str | None = None, *args) -> None:
        if self.problem.consider_warnings_errors():
            self.error(msg, additional_info, *args)
            return
        self._add_warning()
        self.log.warning(self.__append_additional_info(msg, additional_info), *args)

    def error_in_2025_09(self, msg: str, additional_info: str | None = None, *args) -> None:
        if self.problem.format is FormatVersion.LEGACY:
            self.warning(msg, additional_info, *args)
        else:
            self.error(msg, additional_info, *args)

    def info(self, msg: str, *args) -> None:
        self.log.info(msg, *args)

    def debug(self, msg: str, *args) -> None:
        self.log.debug(msg, *args)

    def msg(self, msg):
        print(msg)

    def warn_directory(self, name: str, prop: str) -> None:
        """Warns if a directory meant for a different problem format version exists"""
        good_dir = getattr(self.problem.format, prop)
        bad_dirs = {getattr(version, prop) for version in FormatVersion} - {good_dir}
        problem_root = Path(self.problem.probdir)
        for directory in bad_dirs:
            if (problem_root / directory).exists():
                self.warning(f'Found directory "{directory}". Version {self.problem.format} looks for {name} in "{good_dir}"')

    def _add_error(self) -> None:
        self.errors += 1
        if self.problem is not self:
            self.problem._add_error()

    def _add_warning(self) -> None:
        self.warnings += 1
        if self.problem is not self:
            self.problem._add_warning()


class ProblemPart(ProblemAspect):
    """Baseclass for all parts that can be included in a problem-format."""

    """Should always be overridden by the subclass. Specifies the name that will be used to refer
    to the part e.g for logs.
    """
    PART_NAME: ClassVar[str]

    def __init__(self, problem: Problem) -> None:
        if self.PART_NAME is None:
            raise NotImplementedError('Every problem-part must override PART_NAME')
        super().__init__(f'{problem.shortname}.{self.PART_NAME}', problem)
        self.setup()

    def setup(self) -> None:
        pass

    def start_background_work(self, context: Context) -> None:
        pass

    def check(self, context: Context) -> bool:
        return True


class TestCase(ProblemAspect):
    Result = tuple[SubmissionResult, SubmissionResult, SubmissionResult]

    # Keys allowed in per-test-case YAML (2025-09)
    _ALLOWED_TESTCASE_YAML_KEYS = frozenset({
        'args', 'input_validator_args', 'output_validator_args',
        'input_visualizer_args', 'output_visualizer_args',
        'full_feedback', 'hint', 'description',
    })

    def __init__(self, problem: Problem, base: str, testcasegroup: TestCaseGroup) -> None:
        super().__init__(f'{problem.shortname}.test.{testcasegroup.name}.{os.path.basename(base)}', problem)
        self._base = base
        self.infile = f'{base}.in'
        self.ansfile = f'{base}.ans'
        self.outfile = f'{base}.out' if os.path.isfile(f'{base}.out') else None
        self.filesdir = f'{base}.files' if os.path.isdir(f'{base}.files') else None
        self._problem = problem
        self.testcasegroup = testcasegroup
        self.reuse_result_from: TestCase | None = None
        self.counter = len(problem.testcase_by_infile)
        problem.testcase_by_infile[self.infile] = self

        # Per-test-case configuration (2025-09)
        self.tc_config: dict[str, Any] = {}
        self._load_testcase_yaml()

    def _load_testcase_yaml(self) -> None:
        """Load per-test-case YAML configuration if it exists (2025-09)."""
        if self._problem.format is not FormatVersion.V_2025_09:
            return
        yaml_file = f'{self._base}.yaml'
        if os.path.isfile(yaml_file):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data is not None:
                    if not isinstance(data, dict):
                        self.error(f'Per-test-case YAML {yaml_file} must be a mapping')
                    else:
                        for key in data:
                            if key not in TestCase._ALLOWED_TESTCASE_YAML_KEYS:
                                self.warning(f"Unknown key '{key}' in per-test-case YAML {yaml_file}")
                        self.tc_config = data
            except Exception as e:
                self.error(f'Failed to parse per-test-case YAML {yaml_file}: {e}')

    def check_newlines(self, filename: str) -> None:
        with open(filename, 'rb') as f:
            rawdata = f.read()
            try:
                data = rawdata.decode('utf-8', 'strict')
            except UnicodeDecodeError:
                self.warning(f'The file {filename} could not be decoded as utf-8')
                return
        if data.find('\r') != -1:
            self.warning(f'The file {filename} contains non-standard line breaks.')
        if len(data) > 0 and data[-1] != '\n':
            self.warning(f"The file {filename} does not end with '\\n'.")

    def check_size_limits(self, filename: str) -> None:
        filesize = os.path.getsize(filename) / 1024.0 / 1024.0
        if filesize > 1000:
            self.error(f'The file {filename} ({filesize:.1f} Mb) is larger than 1000 Mb and can not be installed.')
        elif filesize > 100:
            self.warning(
                f'The file {filename} ({filesize:.1f} Mb) is larger than 100 Mb. This may cause performance issues and is not recommended.'
            )

    def strip_path_prefix(self, path: str) -> str:
        return os.path.relpath(path, os.path.join(self._problem.probdir, 'data'))

    def is_in_sample_group(self) -> bool:
        return self.strip_path_prefix(self.infile).startswith('sample')

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True
        self.check_newlines(self.infile)
        self.check_newlines(self.ansfile)
        self.check_size_limits(self.infile)
        self.check_size_limits(self.ansfile)
        self._problem.input_validators.validate(self)
        anssize = os.path.getsize(self.ansfile) / 1024.0 / 1024.0
        outputlim = self._problem.metadata.limits.output
        if anssize > outputlim:
            self.error(
                f'Answer file ({anssize:.1f} Mb) is larger than output limit ({outputlim} Mb), you need to increase output limit'
            )
        elif 2 * anssize > outputlim:
            self.warning(
                f'Answer file ({anssize:.1f} Mb) is within 50% of output limit ({outputlim} Mb), you might want to increase output limit'
            )
        if not self._problem.is_interactive() and not self._problem.is_multi_pass():
            val_res = self._problem.output_validators.validate(self, self.ansfile)
            if val_res.verdict != 'AC':
                if self.is_in_sample_group():
                    self.error(f'judge answer file got {val_res} on testcase {self.strip_path_prefix(self.ansfile)}')
                else:
                    self.warning(f'judge answer file got {val_res} on testcase {self.strip_path_prefix(self.ansfile)}')

        # 2025-09: Validate .out file (display override) must pass output validation
        if self.outfile is not None and self._problem.format is FormatVersion.V_2025_09:
            self.check_newlines(self.outfile)
            if not self._problem.is_interactive():
                out_res = self._problem.output_validators.validate(self, self.outfile)
                if out_res.verdict != 'AC':
                    self.error(f'.out file {os.path.basename(self.outfile)} failed output validation: {out_res}')

        self._check_symlinks()
        return self._check_res

    def __str__(self) -> str:
        return f'testcase {self.strip_path_prefix(self._base)}'

    def matches_filter(self, filter_re: Pattern[str]) -> bool:
        return filter_re.search(self.strip_path_prefix(self._base)) is not None

    def set_symlinks(self) -> None:
        if not os.path.islink(self.infile):
            return
        target = os.path.realpath(self.infile)
        if target in self._problem.testcase_by_infile:
            self.reuse_result_from = self._problem.testcase_by_infile[target]

    def _check_symlinks(self) -> bool:
        if not os.path.islink(self.infile):
            return True
        nicepath = os.path.relpath(self.infile, self._problem.probdir)
        in_target = os.path.realpath(self.infile)
        ans_target = os.path.realpath(self.ansfile)
        if not in_target.endswith('.in'):
            self.error(f"Symbolic link does not point to a .in file for input '{nicepath}'")
            return False
        if ans_target != f'{in_target[:-3]}.ans':
            self.error(f"Symbolic link '{nicepath}' must have a corresponding link for answer file")
            return False
        if self.reuse_result_from is None:
            self.error(f"Symbolic link points outside data/ directory for file '{nicepath}'")
            return False
        if (
            self.testcasegroup.config['output_validator_flags']
            != self.reuse_result_from.testcasegroup.config['output_validator_flags']
        ):
            self.error(f"Symbolic link '{nicepath}' points to testcase with different output validator flags")
            return False
        return True

    def run_submission(self, sub, runner: Runner, context: Context) -> Result:
        (res, res_low, res_high), reused = runner.run(self)
        res = self._init_result_for_testcase(res)
        res_low = self._init_result_for_testcase(res_low)
        res_high = self._init_result_for_testcase(res_high)
        msg = 'Reused test file result' if reused else 'Test file result'
        self.info(f'{msg}: {res}')
        if res.verdict != 'AC' and self.is_in_sample_group():
            res.sample_failures.append(res)

        # Track per-testcase verdict for submissions.yaml (2025-09)
        if self._problem.format is FormatVersion.V_2025_09:
            data_dir = os.path.join(self._problem.probdir, 'data')
            tc_path = os.path.relpath(self.infile, data_dir)
            if tc_path.endswith('.in'):
                tc_path = tc_path[:-3]
            self._problem._testcase_verdicts[tc_path] = res

        return (res, res_low, res_high)

    def run_normal(self, sub, infile: Path, time_limit: float, feedback_dir: Path) -> SubmissionResult:
        """
        Run a submission batch-style (non-interactive)
        """
        outfile = Path(self._problem.tmpdir) / f'output-{self.counter}'
        errfile = Path(self._problem.tmpdir) / f'error-{self.counter}'

        status, runtime = sub.run(
            infile=str(infile),
            outfile=str(outfile),
            errfile=str(errfile),
            timelim=math.ceil(time_limit) + 1,
            memlim=self._problem.metadata.limits.memory,
            work_dir=sub.path,
        )
        if is_TLE(status) or runtime > time_limit:
            res_high = SubmissionResult('TLE')
        elif is_RTE(status):
            try:
                with open(errfile, mode='rt') as f:
                    info = f.read()
            except IOError:
                self.info('Failed to read error file %s', errfile)
                info = None
            res_high = SubmissionResult('RTE', additional_info=info)
        else:
            res_high = self._problem.output_validators.validate(
                self, submission_output=str(outfile), infile=str(infile), feedback_dir_path=str(feedback_dir)
            )

        res_high.runtime = runtime
        return res_high

    def run_submission_multipass(self, feedback_dir: Path, run_sub_fn) -> SubmissionResult:
        # This may be called off-main thread.

        infile = Path(self.infile)
        validation_passes = self._problem.metadata.limits.validation_passes

        input_dir = Path(tempfile.mkdtemp(prefix=f'input-{self.counter}-', dir=self.problem.tmpdir))

        slowest_pass = 0
        for curr_pass in range(validation_passes):
            res = run_sub_fn(infile)

            slowest_pass = max(slowest_pass, res.runtime)
            res.runtime = slowest_pass

            nextpass_file = feedback_dir / 'nextpass.in'

            if res.verdict != 'AC':
                if nextpass_file.is_file():
                    return SubmissionResult('JE', reason='Output validator produced nextpass.in despite non-42 exit code')
                return res

            # Done with passes
            if not nextpass_file.is_file():
                return res

            infile = input_dir / 'input.in'
            # Remove nextpass from feedback
            nextpass_file.rename(infile)

        return SubmissionResult('JE', reason=f'Multipass validator did not give verdict in {validation_passes=} passes')

    def run_submission_real(self, sub, context: Context, timelim: float, timelim_low: float, timelim_high: float) -> Result:
        # This may be called off-main thread.
        # Get per-test-case args (2025-09): from test case YAML, falling back to group config
        tc_args = None
        if self._problem.format is FormatVersion.V_2025_09:
            tc_args = self.tc_config.get('args', self.testcasegroup.config.get('args', None)) or None

        if self._problem.is_interactive():
            res_high = self._problem.output_validators.validate_interactive(self, sub, timelim_high, self._problem.submissions)
        elif self._problem.is_multi_pass():
            res_high = self._run_multi_pass(sub, tc_args, timelim_high)
        else:
            res_high = self._run_single_pass(sub, tc_args, timelim_high)

        if res_high.runtime <= timelim_low:
            res_low = res_high
            res = res_high
        elif res_high.runtime <= timelim:
            res_low = SubmissionResult('TLE')
            res = res_high
        elif res_high.validator_first and res_high.verdict == 'WA':
            # WA can override TLE for interactive problems (see comment in validate_interactive).
            res = SubmissionResult('WA')
            res.validator_first = True
            res_low = res
            res_high.runtime = timelim_low
        else:
            res_low = SubmissionResult('TLE')
            res = res_low

        res.runtime = res_high.runtime
        res_low.runtime = res_high.runtime
        res.set_ac_runtime()
        res_low.set_ac_runtime()
        res_high.set_ac_runtime()
        return (res, res_low, res_high)

    def _run_single_pass(self, sub, tc_args: list[str] | None, timelim_high: int) -> SubmissionResult:
        """Run a non-interactive, non-multi-pass submission on this test case."""
        outfile = os.path.join(self._problem.tmpdir, f'output-{self.counter}')
        errfile = os.path.join(self._problem.tmpdir, f'error-{self.counter}')
        status, runtime = sub.run(
            infile=self.infile,
            outfile=outfile,
            errfile=errfile,
            args=tc_args,
            timelim=timelim_high + 1,
            memlim=self._problem.metadata.limits.memory,
            work_dir=sub.path,
        )
        if is_TLE(status) or runtime > timelim_high:
            res = SubmissionResult('TLE')
        elif is_RTE(status):
            try:
                with open(errfile, mode='rt') as f:
                    info = f.read()
            except IOError:
                self.info('Failed to read error file %s', errfile)
                info = None
            res = SubmissionResult('RTE', additional_info=info)
        else:
            res = self._problem.output_validators.validate(self, outfile)
        res.runtime = runtime
        return res

    def _run_multi_pass(self, sub, tc_args: list[str] | None, timelim_high: int) -> SubmissionResult:
        """Run a multi-pass submission, looping through passes.

        Per the 2025-09 spec:
        - The submission is executed multiple times.
        - After each pass, the output validator is invoked.
        - If the validator exits with 42 and creates nextpass.in, another pass runs.
        - Time/memory limits apply per invocation separately.
        - Feedback directory persists between passes.
        - It's a judge error if nextpass.in is created but exit code != 42.
        - Max passes is validation_passes from problem.yaml.
        """
        max_passes = self._problem.metadata.limits.validation_passes
        total_runtime = 0.0
        feedbackdir = tempfile.mkdtemp(prefix='feedback_mp', dir=self._problem.tmpdir)
        current_infile = self.infile

        try:
            for pass_num in range(1, max_passes + 1):
                # Remove nextpass.in from previous pass (spec: "nextpass.in will be removed before the next pass")
                nextpass_file = os.path.join(feedbackdir, 'nextpass.in')
                if os.path.isfile(nextpass_file):
                    os.unlink(nextpass_file)

                outfile = os.path.join(self._problem.tmpdir, f'output-{self.counter}-pass{pass_num}')
                errfile = os.path.join(self._problem.tmpdir, f'error-{self.counter}-pass{pass_num}')

                status, runtime = sub.run(
                    infile=current_infile,
                    outfile=outfile,
                    errfile=errfile,
                    args=tc_args,
                    timelim=timelim_high + 1,
                    memlim=self._problem.metadata.limits.memory,
                    work_dir=sub.path,
                )
                total_runtime = max(total_runtime, runtime)  # Per spec: limits apply per invocation

                if is_TLE(status) or runtime > timelim_high:
                    res = SubmissionResult('TLE')
                    res.runtime = total_runtime
                    return res
                elif is_RTE(status):
                    try:
                        with open(errfile, mode='rt') as f:
                            info = f.read()
                    except IOError:
                        self.info('Failed to read error file %s', errfile)
                        info = None
                    res = SubmissionResult('RTE', additional_info=info)
                    res.runtime = total_runtime
                    return res

                # Validate this pass's output. Pass feedbackdir so it persists between passes.
                res = self._problem.output_validators.validate(self, outfile, feedbackdir=feedbackdir)
                res.runtime = total_runtime

                nextpass_file = os.path.join(feedbackdir, 'nextpass.in')
                has_nextpass = os.path.isfile(nextpass_file)

                if res.verdict != 'AC':
                    # Validator rejected: judging stops
                    if has_nextpass:
                        # Judge error: nextpass.in created but exit code was not 42 (WA = 43)
                        return SubmissionResult(
                            'JE',
                            reason=f'Pass {pass_num}: nextpass.in created but validator did not accept (verdict: {res.verdict})',
                        )
                    return res

                # Validator accepted (exit 42)
                if not has_nextpass:
                    # No more passes needed, submission is accepted
                    return res

                # More passes to come: use nextpass.in as input for next pass
                # Copy to a stable temp file so the feedback dir can be cleaned per pass
                next_infile = os.path.join(self._problem.tmpdir, f'nextpass-{self.counter}-pass{pass_num}.in')
                shutil.copy2(nextpass_file, next_infile)
                current_infile = next_infile

            # Exceeded max passes: judge error
            return SubmissionResult(
                'JE',
                reason=f'Multi-pass: exceeded maximum {max_passes} passes (still producing nextpass.in)',
            )
        finally:
            shutil.rmtree(feedbackdir, ignore_errors=True)

    def _init_result_for_testcase(self, res: SubmissionResult) -> SubmissionResult:
        res = copy.copy(res)
        res.testcase = self
        res.runtime_testcase = self
        if self._problem.format is FormatVersion.V_2025_09:
            self._init_result_2025_09(res)
        else:
            self._init_result_legacy(res)
        return res

    def _init_result_legacy(self, res: SubmissionResult) -> None:
        """Set score for legacy format using accept_score/reject_score."""
        if res.score is None:
            if res.verdict == 'AC':
                res.score = self.testcasegroup.config['accept_score']
            else:
                res.score = self.testcasegroup.config['reject_score']

    def _init_result_2025_09(self, res: SubmissionResult) -> None:
        """Set score for 2025-09 format using score_aggregation logic."""
        if not self._problem.is_scoring():
            # Pass-fail: no scores
            res.score = None
            return

        aggregation = self.testcasegroup.config.get('score_aggregation', 'pass-fail')
        if aggregation == 'pass-fail':
            # In pass-fail aggregation groups, individual test case scores aren't used
            res.score = None
            return

        if res.verdict != 'AC':
            res.score = 0.0
            return

        # For AC in sum/min groups: determine score
        max_score_group = self.testcasegroup.config.get('max_score', 'unbounded')

        if res.score is not None:
            # Score was set by the validator (score.txt or score_multiplier.txt)
            # If it came from score_multiplier.txt, it's a 0-1 multiplier and we need to multiply by per-case max
            # For now, scores from validators are already computed correctly
            return

        # No score from validator
        if max_score_group == 'unbounded':
            # For unbounded max_score, validator MUST produce score.txt
            # This will be caught as a JE elsewhere
            res.score = 0.0
            return

        # Bounded max_score with no validator score: score = max per-case score
        # Per-case max = (max_score - static_validation_score) / N
        num_cases = len(self.testcasegroup.get_testcases())
        static_score = self.testcasegroup.config.get('static_validation_score', 0)
        if isinstance(static_score, str) and static_score == 'pass-fail':
            static_score = 0  # pass-fail static validation doesn't reduce the score pool
        if num_cases > 0:
            per_case = (float(max_score_group) - float(static_score)) / num_cases
            if aggregation == 'min':
                per_case = float(max_score_group)  # For min, each case has max_score as its max
            res.score = per_case
        else:
            res.score = 0.0

    def get_all_testcases(self) -> list[TestCase]:
        return [self]

    def all_datasets(self) -> list[str]:
        return [self._base]


class TestCaseGroup(ProblemAspect):
    name: str
    _DEFAULT_CONFIG_LEGACY = config.load_config('testdata.yaml')
    _DEFAULT_CONFIG_2025_09 = config.load_config('testdata_2025_09.yaml')
    _SCORING_ONLY_KEYS_LEGACY = ['accept_score', 'reject_score', 'range']
    # 2025-09 scoring keys that are not valid in sample groups
    _SCORING_KEYS_2025_09 = ['max_score', 'score_aggregation', 'static_validation_score', 'require_pass']

    def _find_config_file(self) -> str | None:
        """Find the config file for this test data group (test_group.yaml for 2025-09, testdata.yaml for legacy)."""
        new_config = os.path.join(self._datadir, 'test_group.yaml')
        legacy_config = os.path.join(self._datadir, 'testdata.yaml')
        has_new = os.path.isfile(new_config)
        has_legacy = os.path.isfile(legacy_config)

        if has_new and has_legacy:
            self.error(f'Both test_group.yaml and testdata.yaml found in {self._datadir}. Only one is allowed.')
            return new_config  # Prefer the new one

        if self._problem.format is FormatVersion.LEGACY:
            if has_new:
                self.warning(f'Found test_group.yaml in {self._datadir}, but problem uses legacy format. Use testdata.yaml instead.')
            return legacy_config if has_legacy else None
        else:
            if has_legacy:
                self.warning(f'Found testdata.yaml in {self._datadir}, but problem uses format {self._problem.format}. Use test_group.yaml instead.')
                return legacy_config  # Still use it, just warn
            return new_config if has_new else None

    def __init__(self, problem: Problem, datadir: str | None = None, parent: TestCaseGroup | None = None):
        self._parent = parent
        self._problem = problem
        datadir = datadir or os.path.join(problem.probdir, 'data')
        self._datadir = datadir
        self.name = os.path.relpath(os.path.abspath(self._datadir), os.path.abspath(self._problem.probdir)).replace('/', '.')

        super().__init__(f'{problem.shortname}.test.{self.name}', problem)

        self._seen_oob_scores = False
        self.debug('Loading test data group %s', datadir)
        configfile = self._find_config_file()
        self.config: dict[str, Any] = {}
        if configfile is not None and os.path.isfile(configfile):
            try:
                with open(configfile) as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                self.error(str(e))
            if self.config is None:
                self.config = {}

        # Substitute constants in test_group.yaml values (2025-09 only)
        if problem.format is FormatVersion.V_2025_09 and self.config and problem.metadata.constants:
            self.config, undef = statement_util.substitute_constants_in_yaml(self.config, problem.metadata.constants)
            for ref in undef:
                self.warning("Undefined constant '{{%s}}' in %s" % (ref, configfile))

        # For non-root groups, missing properties are inherited from the parent group
        if parent:
            for field, parent_value in parent.config.items():
                if field not in self.config:
                    self.config[field] = parent_value

        # Apply version-specific defaults
        if problem.format is FormatVersion.LEGACY:
            self._apply_legacy_defaults(problem)
        else:
            self._apply_2025_09_defaults()

        # Directories under data/ that are for validation testing, not for running submissions
        _VALIDATION_DIRS = frozenset({'invalid_input', 'invalid_output', 'valid_output'})

        self._items: list[TestCaseGroup | TestCase] = []
        if os.path.isdir(datadir):
            for filename in sorted(os.listdir(datadir)):
                filepath = os.path.join(datadir, filename)
                if os.path.isdir(filepath):
                    # Skip validation directories at the root level - they're not test case groups
                    if not parent and os.path.basename(filepath) in _VALIDATION_DIRS:
                        continue
                    self._items.append(TestCaseGroup(problem, filepath, self))
                else:
                    base, ext = os.path.splitext(filepath)
                    if ext == '.ans' and os.path.isfile(f'{base}.in'):
                        self._items.append(TestCase(problem, base, self))

        if not parent:
            self.set_symlinks()

    def _apply_legacy_defaults(self, problem: Problem) -> None:
        """Apply legacy testdata defaults and inherit from problem config."""
        legacy_grading = problem.metadata.legacy_grading
        for key in ['accept_score', 'reject_score', 'range']:
            if getattr(legacy_grading, key) is not None:
                self.config[key] = getattr(legacy_grading, key)

        problem_on_reject = legacy_grading.on_reject
        if problem_on_reject == 'first_error':
            self.config['on_reject'] = 'break'
        if problem_on_reject == 'grade':
            self.config['on_reject'] = 'continue'

        if self._problem.is_pass_fail():
            for key in TestCaseGroup._SCORING_ONLY_KEYS_LEGACY:
                if key not in self.config:
                    self.config[key] = None

        for field, default in TestCaseGroup._DEFAULT_CONFIG_LEGACY.items():
            if field not in self.config:
                self.config[field] = default

    def _apply_2025_09_defaults(self) -> None:
        """Apply 2025-09 test_group.yaml defaults."""
        # Determine context-dependent defaults based on group name
        basename = os.path.basename(self._datadir)
        is_sample = basename == 'sample' or (self._parent is not None and 'sample' in self.name)
        is_secret_root = basename == 'secret' and self._parent is not None and self._parent._parent is None

        # max_score and score_aggregation have context-dependent defaults
        if 'max_score' not in self.config:
            if is_secret_root:
                self.config['max_score'] = 100
            else:
                self.config['max_score'] = 'unbounded'

        if 'score_aggregation' not in self.config:
            if is_secret_root:
                self.config['score_aggregation'] = 'sum'
            else:
                self.config['score_aggregation'] = 'pass-fail'

        if 'full_feedback' not in self.config:
            self.config['full_feedback'] = is_sample

        if 'require_pass' not in self.config:
            self.config['require_pass'] = []

        for field, default in TestCaseGroup._DEFAULT_CONFIG_2025_09.items():
            if field not in self.config:
                self.config[field] = default

        # Derive on_reject from score_aggregation for runtime compatibility
        # pass-fail → break on first failure; sum/min → continue (run all cases)
        if 'on_reject' not in self.config:
            if self.config.get('score_aggregation') == 'pass-fail':
                self.config['on_reject'] = 'break'
            else:
                self.config['on_reject'] = 'continue'

        # Legacy grading key needed by grader code path
        if 'grading' not in self.config:
            self.config['grading'] = 'default'

        # Bridge 2025-09 args to legacy flags format for runtime compatibility
        # These are used by validators and other code that checks config['output_validator_flags'] etc.
        if 'output_validator_flags' not in self.config:
            args = self.config.get('output_validator_args', [])
            self.config['output_validator_flags'] = ' '.join(args) if isinstance(args, list) else str(args)
        if 'input_validator_flags' not in self.config:
            args = self.config.get('input_validator_args', [])
            if isinstance(args, list):
                self.config['input_validator_flags'] = ' '.join(args)
            elif isinstance(args, dict):
                # Map format: join all values; the validators check this differently
                self.config['input_validator_flags'] = ''
            else:
                self.config['input_validator_flags'] = str(args)
        if 'grader_flags' not in self.config:
            self.config['grader_flags'] = ''
        # Bridge scoring defaults for runtime compatibility
        if 'accept_score' not in self.config:
            self.config['accept_score'] = 1.0
        if 'reject_score' not in self.config:
            self.config['reject_score'] = 0.0
        if 'range' not in self.config:
            self.config['range'] = '-inf +inf'

    def start_background_work(self, context: Context) -> None:
        pass

    def __str__(self) -> str:
        return f'testcase group {self.name}'

    def set_symlinks(self) -> None:
        for sub in self._items:
            sub.set_symlinks()

    def matches_filter(self, filter_re: Pattern[str]) -> bool:
        return True

    def get_all_testcases(self) -> list:
        res: list = []
        for child in self._items:
            res += child.get_all_testcases()
        return res

    def get_testcases(self) -> list[TestCase]:
        return [child for child in self._items if isinstance(child, TestCase)]

    def get_subgroups(self) -> list[TestCaseGroup]:
        return [child for child in self._items if isinstance(child, TestCaseGroup)]

    def get_subgroup(self, name: str) -> TestCaseGroup | None:
        return next(
            (child for child in self._items if isinstance(child, TestCaseGroup) and os.path.basename(child._datadir) == name),
            None,
        )

    def has_custom_groups(self) -> bool:
        return any(group.get_subgroups() for group in self.get_subgroups())

    def get_score_range(self) -> tuple[float, float]:
        try:
            score_range = self.config['range']
            min_score, max_score = list(map(float, score_range.split()))
            return (min_score, max_score)
        except Exception:
            return (float('-inf'), float('inf'))

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        if self._problem.format is FormatVersion.LEGACY:
            self._check_legacy(context)
        else:
            self._check_2025_09(context)

        # Common checks for both formats
        self._check_files_and_ordering(context)
        if self._parent is None:
            self._check_top_level()

        # Recursively check children
        for child in self._items:
            if child.matches_filter(context.data_filter):
                child.check(context)

        return self._check_res

    def _check_legacy(self, context: Context) -> None:
        """Legacy-format specific testdata validation."""
        if self.config['grading'] not in ['default', 'custom']:
            self.error('Invalid grading policy in testdata.yaml')

        if self.config['grading'] == 'custom' and self._problem.graders._grader is None:
            self._problem.graders.fatal(f'{self} has custom grading but no custom graders provided')
        if self.config['grading'] == 'default' and Graders._default_grader is None:
            self._problem.graders.fatal(f'{self} has default grading but I could not find default grader')

        if self.config['grading'] == 'default' and 'ignore_sample' in self.config['grader_flags'].split():
            if self._parent is not None:
                self.error("'grader_flags: ignore_sample' is specified, but that flag is only allowed at top level")
            elif self.config['on_reject'] == 'break':
                self.error(
                    "'grader_flags: ignore_sample' is specified, but 'on_reject: break' may cause secret data not to be judged"
                )

        for field in self.config.keys():
            if field not in TestCaseGroup._DEFAULT_CONFIG_LEGACY.keys():
                self.warning(f"Unknown key '{field}' in '{os.path.join(self._datadir, 'testdata.yaml')}'")

        if not self._problem.is_scoring():
            for key in TestCaseGroup._SCORING_ONLY_KEYS_LEGACY:
                if self.config.get(key) is not None:
                    self.error(f"Key '{key}' is only applicable for scoring problems, this is a pass-fail problem")

        if self.config['on_reject'] not in ['break', 'continue']:
            self.error(f"Invalid value '{self.config['on_reject']}' for on_reject policy")

        if self._problem.is_scoring():
            # Check grading
            try:
                score_range = self.config['range']
                min_score, max_score = list(map(float, score_range.split()))
                if min_score > max_score:
                    self.error(f"Invalid score range '{score_range}': minimum score cannot be greater than maximum score")
            except VerifyError:
                raise
            except Exception:
                self.error(f"Invalid format '{score_range}' for range: must be exactly two floats")

    def _check_2025_09(self, context: Context) -> None:
        """2025-09 format specific test_group.yaml validation."""
        basename = os.path.basename(self._datadir)
        is_sample = basename == 'sample' or (self._parent is not None and 'sample' in self.name)

        # Validate known keys
        known_keys = set(TestCaseGroup._DEFAULT_CONFIG_2025_09.keys()) | set(TestCaseGroup._SCORING_KEYS_2025_09)
        # Also allow legacy-derived keys that we compute internally
        known_keys |= {'on_reject', 'grading', 'accept_score', 'reject_score', 'range',
                        'grader_flags', 'input_validator_flags', 'output_validator_flags'}
        for field in self.config.keys():
            if field not in known_keys:
                self.warning(f"Unknown key '{field}' in '{os.path.join(self._datadir, 'test_group.yaml')}'")

        # Validate score_aggregation value
        if self.config.get('score_aggregation') not in ['pass-fail', 'sum', 'min', None]:
            self.error(f"Invalid value '{self.config.get('score_aggregation')}' for score_aggregation (must be 'pass-fail', 'sum', or 'min')")

        # Scoring keys not permitted in sample
        if is_sample:
            _SAMPLE_FORBIDDEN_KEYS = {'max_score', 'score_aggregation', 'require_pass', 'static_validation_score'}
            configfile = os.path.join(self._datadir, 'test_group.yaml')
            if os.path.isfile(configfile):
                try:
                    with open(configfile) as f:
                        raw_config = yaml.safe_load(f) or {}
                except Exception:
                    raw_config = {}
                for key in _SAMPLE_FORBIDDEN_KEYS:
                    if key in raw_config:
                        self.error(f"Key '{key}' is not permitted in sample test_group.yaml")

        # Pass-fail problems must not use scoring keys
        if not self._problem.is_scoring():
            configfile = os.path.join(self._datadir, 'test_group.yaml')
            if os.path.isfile(configfile):
                try:
                    with open(configfile) as f:
                        raw_config = yaml.safe_load(f) or {}
                except Exception:
                    raw_config = {}
                for key in ('max_score', 'score_aggregation', 'require_pass'):
                    if key in raw_config:
                        self.error(f"Key '{key}' is for scoring problems only, but this is a pass-fail problem")

        # Validate max_score value
        max_score = self.config.get('max_score')
        if max_score is not None and max_score != 'unbounded':
            if not isinstance(max_score, int) or max_score < 0:
                self.error(f"Invalid value '{max_score}' for max_score (must be a non-negative integer or 'unbounded')")

        # Validate max_score consistency: groups may only be unbounded if secret is unbounded
        if max_score == 'unbounded' and self._parent is not None:
            # Walk up to find the secret root's max_score
            parent = self._parent
            while parent._parent is not None:
                parent = parent._parent
            secret_group = parent.get_subgroup('secret')
            if secret_group and secret_group.config.get('max_score') != 'unbounded':
                self.error("Test data group has unbounded max_score but 'secret' has bounded max_score")

        # Validate require_pass contains valid group references
        require_pass = self.config.get('require_pass', [])
        if isinstance(require_pass, str):
            require_pass = [require_pass]
            self.config['require_pass'] = require_pass

        # Validate static_validation_score
        static_val_score = self.config.get('static_validation_score')
        if static_val_score is not None:
            if not hasattr(self._problem, 'static_validator') or not self._problem.static_validator.has_static_validator():
                self.error("static_validation_score specified but no static_validator/ directory found")
            if static_val_score == 'pass-fail':
                aggregation = self.config.get('score_aggregation', 'pass-fail')
                if aggregation != 'pass-fail' and self._problem.is_scoring():
                    self.error("static_validation_score: 'pass-fail' requires score_aggregation to be 'pass-fail'")
            elif isinstance(static_val_score, int):
                if static_val_score < 0:
                    self.error(f"static_validation_score must be non-negative, got {static_val_score}")
                if not self._problem.is_scoring():
                    self.error("static_validation_score (as integer) is only for scoring problems")
            else:
                self.error(f"Invalid static_validation_score value: {static_val_score} (must be non-negative int or 'pass-fail')")

            # Cannot use static_validator_args without static_validation_score
            # (already validated by having static_validation_score)

        # Validate static_validator_args without static_validation_score
        if self.config.get('static_validator_args') and static_val_score is None:
            self.error("static_validator_args specified without static_validation_score")

        for ref in require_pass:
            if ref == 'sample':
                continue  # 'sample' is always valid
            # Validate that referenced groups exist and use pass-fail aggregation
            # Referenced groups must be sibling groups that come lexicographically earlier
            if self._parent is not None:
                ref_group = self._parent.get_subgroup(ref)
                if ref_group is None:
                    self.error(f"require_pass references group '{ref}' which does not exist")
                elif ref_group.config.get('score_aggregation', 'pass-fail') != 'pass-fail':
                    self.error(f"require_pass references group '{ref}' which does not use pass-fail aggregation")
                elif os.path.basename(ref_group._datadir) >= basename:
                    self.warning(f"require_pass references group '{ref}' which is not lexicographically earlier than '{basename}'")

    def _check_top_level(self) -> None:
        """Check top-level data directory structure."""
        seen_secret = False
        seen_sample = False
        _VALIDATION_DIRS = frozenset({'invalid_input', 'invalid_output', 'valid_output'})
        for item in self._items:
            if not isinstance(item, TestCaseGroup):
                self.error("Can't have individual test data files at top level")
            else:
                name = os.path.basename(item._datadir)
                if name == 'secret':
                    seen_secret = True
                elif name == 'sample':
                    seen_sample = True
                elif name in _VALIDATION_DIRS:
                    if self._problem.format is FormatVersion.LEGACY:
                        self.error(f'Validation directory "{name}" is not allowed in legacy format')
                else:
                    if self._problem.format is FormatVersion.LEGACY:
                        self.error('Test data at top level can only have the groups sample and secret')
                    else:
                        self.error(f'Test data at top level has unknown group "{name}"; expected sample, secret, invalid_input, invalid_output, or valid_output')
                    self.debug(str(self._items))
        if not seen_secret:
            self.error('No secret data provided')
        if not seen_sample:
            self.warning('No sample data provided')

        hashes = collections.defaultdict(list)
        for root, dirs, files in os.walk(self._datadir):
            for filename in files:
                filepath = os.path.join(root, filename)
                if filepath.endswith('.in') and not os.path.islink(filepath):
                    md5 = hashlib.md5()
                    with open(filepath, 'rb') as f:
                        for buf in iter(lambda: f.read(1024), b''):
                            md5.update(buf)
                    filehash = md5.digest()
                    hashes[filehash].append(os.path.relpath(filepath, self._problem.probdir))
        for _, files in hashes.items():
            if len(files) > 1:
                self.warning(f"Identical input files: '{str(files)}'")

    def _check_files_and_ordering(self, context: Context) -> None:
        """Check input/answer file pairs and group ordering."""
        infiles = glob.glob(os.path.join(self._datadir, '*.in'))
        ansfiles = glob.glob(os.path.join(self._datadir, '*.ans'))

        for infile in infiles:
            if os.path.isdir(infile):
                continue
            if f'{infile[:-3]}.ans' not in ansfiles:
                self.error(f"No matching answer file for input '{infile}'")
        for ansfile in ansfiles:
            if os.path.isdir(ansfile):
                continue
            if f'{ansfile[:-4]}.in' not in infiles:
                self.error(f"No matching input file for answer '{ansfile}'")

        # 2025-09: Validate .out files (display override - must pass output validation)
        if self._problem.format is FormatVersion.V_2025_09:
            outfiles = glob.glob(os.path.join(self._datadir, '*.out'))
            for outfile in outfiles:
                if os.path.isdir(outfile):
                    continue
                base = outfile[:-4]
                if f'{base}.in' not in infiles:
                    self.error(f"No matching input file for .out file '{outfile}'")
                if f'{base}.ans' not in ansfiles:
                    self.error(f"No matching answer file for .out file '{outfile}'")

            # Validate .files directories
            for entry in os.scandir(self._datadir):
                if entry.is_dir() and entry.name.endswith('.files'):
                    base = os.path.join(self._datadir, entry.name[:-6])
                    if f'{base}.in' not in infiles:
                        self.error(f"No matching input file for .files directory '{entry.name}'")

            # Check .in.statement, .ans.statement, .in.download, .ans.download files
            for ext_pair in [('.in.statement', '.in'), ('.ans.statement', '.ans'),
                             ('.in.download', '.in'), ('.ans.download', '.ans')]:
                pattern = os.path.join(self._datadir, f'*{ext_pair[0]}')
                for override_file in glob.glob(pattern):
                    if os.path.isdir(override_file):
                        continue
                    # Derive base name
                    base_name = override_file[:-(len(ext_pair[0]))]
                    expected_file = f'{base_name}{ext_pair[1]}'
                    if not os.path.isfile(expected_file):
                        self.error(f"No matching '{ext_pair[1]}' file for override '{os.path.basename(override_file)}'")

        if not self.get_subgroups() and not self.get_testcases():
            if os.path.basename(self._datadir) != 'sample':
                self.error(f'Testcase group {self._datadir} exists, but does not contain any testcases')
            else:
                if not (
                    (self._problem.is_interactive() or self._problem.is_multi_pass())
                    and glob.glob(os.path.join(self._datadir, '*.interaction'))
                ):
                    self.warning(f'Sample testcase group {self._datadir} exists, but does not contain any testcases')

        # Check whether a <= b according to a natural sorting where numeric components
        # are compactified, so that e.g. "a" < "a1" < "a2" < "a10" = "a010" < "a10a".
        def natural_sort_le(a: str, b: str) -> bool:
            a += '\0'
            b += '\0'
            i = j = 0

            def parse_num(s: str, i: int) -> tuple[int, int]:
                ret = 0
                while ord('0') <= ord(s[i]) <= ord('9'):
                    ret = ret * 10 + ord(s[i]) - ord('0')
                    i += 1
                return ret, i

            while i < len(a) and j < len(b):
                if ord('0') <= ord(a[i]) <= ord('9') and ord('0') <= ord(b[i]) <= ord('9'):
                    anum, i = parse_num(a, i)
                    bnum, j = parse_num(b, j)
                    if anum == bnum:
                        continue
                    return anum < bnum
                if a[i] == b[j]:
                    i += 1
                    j += 1
                    continue
                return a[i] < b[j]
            return True

        last_testgroup_name = ''
        for group in self.get_subgroups():
            name = os.path.relpath(group._datadir, self._problem.probdir)
            if natural_sort_le(name, last_testgroup_name):
                self.warning(f"Test data group '{last_testgroup_name}' will be ordered before '{name}'; consider zero-padding")
            last_testgroup_name = name

    def run_submission(self, sub, runner: Runner, context: Context) -> TestCase.Result:
        self.info(f'Running on {self}')

        # For 2025-09: check require_pass dependencies
        if self._problem.format is FormatVersion.V_2025_09:
            require_pass = self.config.get('require_pass', [])
            if isinstance(require_pass, str):
                require_pass = [require_pass]
            for dep_name in require_pass:
                dep_result = self._problem._group_results.get(dep_name)
                if dep_result is None:
                    self.debug(f'require_pass: dependency "{dep_name}" has no result yet, skipping group')
                    skip_res = SubmissionResult('AC')
                    skip_res.score = 0.0
                    skip_res.verdict = 'AC'  # Not accepted but scored 0
                    return (skip_res, skip_res, skip_res)
                if dep_result.verdict != 'AC':
                    self.info(f'require_pass: dependency "{dep_name}" did not pass (verdict: {dep_result.verdict}), skipping group')
                    skip_res = SubmissionResult('AC')
                    skip_res.score = 0.0
                    return (skip_res, skip_res, skip_res)

        subres: list[SubmissionResult] = []
        subres_low: list[SubmissionResult] = []
        subres_high: list[SubmissionResult] = []
        active_low, active = True, True
        on_reject = self.config['on_reject']
        broken = False
        for child in self._items:
            if not child.matches_filter(context.data_filter):
                continue
            res, res_low, res_high = child.run_submission(sub, runner, context)
            subres_high.append(res_high)
            if active:
                subres.append(res)
            if active_low:
                subres_low.append(res_low)
            if on_reject == 'break':
                active_low &= res_low.verdict == 'AC'
                active &= res.verdict == 'AC'
                if res_high.verdict != 'AC':
                    broken = True
                    break

        runner.mark_group_done(self, broken)

        result = self.aggregate_results(sub, subres)
        result_low = self.aggregate_results(sub, subres_low, shadow_result=True)
        result_high = self.aggregate_results(sub, subres_high, shadow_result=True)

        # For 2025-09: store group result for require_pass checking by later groups
        if self._problem.format is FormatVersion.V_2025_09 and self._parent is not None:
            group_name = os.path.basename(self._datadir)
            self._problem._group_results[group_name] = result

        return (result, result_low, result_high)

    def aggregate_results(self, sub, sub_results: list[SubmissionResult], shadow_result: bool = False) -> SubmissionResult:
        res = SubmissionResult('JE')

        for r in sub_results:
            if r.runtime > res.runtime:
                res.runtime = r.runtime
                res.runtime_testcase = r.runtime_testcase
            if r.ac_runtime > res.ac_runtime:
                res.ac_runtime = r.ac_runtime
                res.ac_runtime_testcase = r.ac_runtime_testcase
            res.sample_failures.extend(r.sample_failures)

        judge_error = next((r for r in sub_results if r.verdict == 'JE'), None)
        if judge_error:
            res.verdict = judge_error.verdict
            res.reason = judge_error.reason
            res.additional_info = judge_error.additional_info
            res.testcase = judge_error.testcase
        elif self._problem.format is FormatVersion.V_2025_09:
            res.verdict, score = self._aggregate_2025_09(sub_results, shadow_result)
            if sub_results:
                res.testcase = sub_results[-1].testcase
                res.additional_info = sub_results[-1].additional_info
            if self._problem.is_scoring():
                res.score = score
                max_score = self.config.get('max_score')
                if max_score is not None and max_score != 'unbounded':
                    if score is not None and score > max_score and not self._seen_oob_scores:
                        self._seen_oob_scores = True
                        groupname = os.path.relpath(self._datadir, self._problem.probdir)
                        self.error(
                            f'submission {sub} got score {score} on group {groupname}, which exceeds max_score {max_score}'
                        )
        else:
            res.verdict, score = self._problem.graders.grade(sub_results, self, shadow_result)
            if sub_results:
                res.testcase = sub_results[-1].testcase
                res.additional_info = sub_results[-1].additional_info
            if self._problem.is_scoring():
                res.score = score
                min_score, max_score = self.get_score_range()
                if score is not None and not (min_score <= score <= max_score) and not self._seen_oob_scores:
                    self._seen_oob_scores = True
                    groupname = os.path.relpath(self._datadir, self._problem.probdir)
                    self.error(
                        f'submission {sub} got {res} on group {groupname}, which is outside of expected score range [{min_score}, {max_score}]'
                    )
        return res

    def _aggregate_2025_09(self, sub_results: list[SubmissionResult], shadow_result: bool = False) -> tuple[Verdict, float | None]:
        """Aggregate results using 2025-09 score_aggregation logic."""
        aggregation = self.config.get('score_aggregation', 'pass-fail')

        if not sub_results:
            if not shadow_result:
                self.info(f'No results on {self}')
            if aggregation == 'pass-fail':
                return ('AC', self._get_max_score_value())
            return ('AC', 0.0)

        # Check if all accepted
        all_ac = all(r.verdict == 'AC' for r in sub_results)

        if not self._problem.is_scoring():
            # Pass-fail problem: simple AC/non-AC
            if all_ac:
                return ('AC', None)
            # Return the first non-AC verdict
            first_fail = next(r for r in sub_results if r.verdict != 'AC')
            return (first_fail.verdict, None)

        # Scoring problem
        if aggregation == 'pass-fail':
            max_score = self._get_max_score_value()
            if all_ac:
                return ('AC', max_score)
            first_fail = next(r for r in sub_results if r.verdict != 'AC')
            return (first_fail.verdict, 0.0)

        elif aggregation == 'sum':
            scores = [r.score if r.score is not None else 0.0 for r in sub_results]
            total = sum(scores)
            if all_ac:
                verdict: Verdict = 'AC'
            else:
                # Report the "worst" non-AC verdict
                verdict = self._worst_verdict(sub_results)
            if not shadow_result:
                self.debug(f'Sum aggregation on {self}: {total} (from {scores})')
            return (verdict, total)

        elif aggregation == 'min':
            scores = [r.score if r.score is not None else 0.0 for r in sub_results]
            min_score = min(scores)
            if all_ac:
                verdict = 'AC'
            else:
                verdict = self._worst_verdict(sub_results)
            if not shadow_result:
                self.debug(f'Min aggregation on {self}: {min_score} (from {scores})')
            return (verdict, min_score)

        else:
            self.error(f"Unknown score_aggregation '{aggregation}'")
            return ('JE', None)

    def _get_max_score_value(self) -> float:
        """Get the numeric max_score value, returning 0.0 if unbounded."""
        max_score = self.config.get('max_score')
        if max_score is None or max_score == 'unbounded':
            return 0.0
        return float(max_score)

    @staticmethod
    def _worst_verdict(sub_results: list[SubmissionResult]) -> Verdict:
        """Return the 'worst' non-AC verdict from a list of results."""
        priority: dict[str, int] = {'JE': 0, 'WA': 1, 'RTE': 2, 'TLE': 3, 'AC': 4}
        worst: Verdict = 'AC'
        for r in sub_results:
            if priority.get(r.verdict, 0) < priority.get(worst, 4):
                worst = r.verdict  # type: ignore
        return worst

    def all_datasets(self) -> list:
        res: list = []
        for child in self._items:
            res += child.all_datasets()
        return res


class ProblemStatement(ProblemPart):
    statements: dict[str, list[Path]]  # Maps language code -> statement(s)
    PART_NAME = 'statement'

    def setup(self):
        self.debug('  Loading problem statement')
        self.statements = statement_util.find_statements(Path(self.problem.probdir), self.problem.format)

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        self.warn_directory('problem statements', 'statement_directory')

        for ifilename in glob.glob(os.path.join(self.problem.probdir, 'data/sample/*.interaction')):
            if not self.problem.is_interactive() and not self.problem.is_multi_pass():
                self.error(f'Problem is not interactive or multi-pass, but there is an interaction sample {ifilename}')
            with open(ifilename, 'r') as interaction:
                for i, line in enumerate(interaction):
                    if len(line) == 0:
                        continue
                    if line.rstrip() == '---':
                        # Multi-pass pass separator
                        if not self.problem.is_multi_pass():
                            self.error(f'Interaction {ifilename}: line {i + 1} has --- separator but problem is not multi-pass')
                        continue
                    if line[0] != '<' and line[0] != '>':
                        self.error(f'Interaction {ifilename}: line {i + 1} does not start with < or >')
                        break

        if not self.statements:
            if self.problem.format is FormatVersion.LEGACY:
                allowed_statements = ', '.join(
                    f'problem.{ext}, problem.<language>.{ext}' for ext in self.problem.format.statement_extensions
                )
            else:
                allowed_statements = ', '.join(f'problem.<language>.{ext}' for ext in self.problem.format.statement_extensions)

            self.error(
                f'No problem statements found (expected file of one of following forms in directory {self.problem.format.statement_directory}/: {allowed_statements})'
            )

        def _latex_heuristic(name: str) -> bool:
            return '\\' in name or '$' in name

        for lang, files in self.statements.items():
            if len(files) > 1:
                self.error(f'Found multiple statements in the same language {lang}: {", ".join((file.name for file in files))}')

            if lang not in self.problem.metadata.name:
                self.error(f'No problem name given in language {lang}')
            elif not self.problem.metadata.name[lang]:
                self.error(f'Problem name in language {lang} is empty')
            elif not self.problem.metadata.name[lang].strip():
                self.error(f'Problem name in language {lang} contains only whitespace')
            elif self.problem.format is FormatVersion.LEGACY and _latex_heuristic(self.problem.metadata.name[lang]):
                self.warning(f'Problem name in language {lang} looks like LaTeX. Consider using plainproblemname.')

            for file in files:
                try:
                    options = problem2pdf.get_parser().parse_args([''])
                    options.problem = self.problem.probdir
                    options.language = lang
                    options.nopdf = True
                    options.quiet = True
                    if not problem2pdf.convert(options, file):
                        self.error(
                            f'Could not compile problem statement for language "{lang}".  Run problem2pdf --language {lang} on the problem to diagnose.'
                        )
                except Exception as e:
                    self.error(
                        f'Error raised when checking problem statement for language {lang}:\n{e}\n{traceback.format_exc()}'
                    )

                try:
                    options = problem2html.get_parser().parse_args([''])
                    options.problem = self.problem.probdir
                    options.destdir = os.path.join(self.problem.tmpdir, 'html')
                    options.language = lang
                    options.quiet = True
                    problem2html.convert(options, file)
                except Exception as e:
                    self.error(
                        f'Could not convert problem statement to html for language "{lang}".  Run problem2html --language {lang} on the problem to diagnose.\n{e}\n{traceback.format_exc()}'
                    )

        return self._check_res

    def __str__(self) -> str:
        return 'problem statement'


class ProblemConfig(ProblemPart):
    PART_NAME = 'config'

    def setup(self):
        self.debug('  Loading problem config')
        try:
            self._metadata, self._origdata = metadata.load_metadata(Path(self.problem.probdir))
            self.problem._set_metadata(self._metadata)
        except ValidationError as e:
            error_str = '\n'.join([f'    {"->".join((str(loc) for loc in err["loc"]))}: {err["msg"]}' for err in e.errors()])
            self.fatal(f'Failed parsing problem.yaml. Found {len(e.errors())} errors:\n{error_str}')
        except Exception as e:
            self.fatal(f'Failed loading problem configuration: {e}')

    def __str__(self) -> str:
        return 'problem configuration'

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        INCOMPATIBLE_TYPES = [
            (metadata.ProblemType.PASS_FAIL, metadata.ProblemType.SCORING),
            (metadata.ProblemType.SUBMIT_ANSWER, metadata.ProblemType.MULTI_PASS),
            (metadata.ProblemType.SUBMIT_ANSWER, metadata.ProblemType.INTERACTIVE),
        ]
        for t1, t2 in INCOMPATIBLE_TYPES:
            if t1 in self._metadata.type and t2 in self._metadata.type:
                self.error(f'Problem has incompatible types: {t1}, {t2}')

        if self.problem.is_multi_pass():
            if self._metadata.limits.validation_passes < 2:
                self.error(f'Multi-pass problem must have validation_passes >= 2 (got {self._metadata.limits.validation_passes})')
        else:
            if 'validation_passes' in self._origdata.get('limits', {}):
                self.warning('validation_passes is specified but problem is not multi-pass')

        if self.problem.is_submit_answer():
            self.warning('The type submit-answer is not yet fully supported. '
                         'Example submissions are verified as programs, not as answer files.')

        # Check rights_owner
        if self._metadata.license == metadata.License.PUBLIC_DOMAIN:
            if self._metadata.rights_owner:
                self.error('Can not have a rights_owner for a problem in public domain')
        elif self._metadata.license != metadata.License.UNKNOWN:
            if not self._metadata.rights_owner and not self._metadata.source and not self._metadata.credits.authors:
                self.error('No author, source or rights_owner provided')

        # Sanity check that the author name is parsed reasonably
        disallowed_in_name = [',', '&']
        for author in self._metadata.credits.authors:
            for disallowed_character in disallowed_in_name:
                if disallowed_character in author.name:
                    self.warning(f'Author name parsed to "{author.name}", which contains character "{disallowed_character}".')

        # Check license
        if self._metadata.license == metadata.License.UNKNOWN:
            self.warning("License is 'unknown'")

        if self._metadata.uuid is None:
            self.error_in_2025_09(f'Missing uuid from problem.yaml. Add "uuid: {uuid.uuid4()}" to problem.yaml.')

        names_with_no_statement = [lang for lang in self._metadata.name if lang not in self.problem.statement.statements]
        if names_with_no_statement:
            self.error(f'Names exist for languages without problem statements: {", ".join(names_with_no_statement)}')

        if self._metadata.legacy_grading.show_test_data_groups and self.problem.is_pass_fail():
            self.error('Showing test data groups is only supported for scoring problems, this is a pass-fail problem')
        if (
            not self.problem.is_pass_fail()
            and self.problem.testdata.has_custom_groups()
            and 'show_test_data_groups' not in self._origdata.get('grading', {})
            and self.problem.format is FormatVersion.LEGACY
        ):
            self.warning(
                'Problem has custom testcase groups, but does not specify a value for grading.show_test_data_groups; defaulting to false'
            )

        if self._metadata.legacy_grading.on_reject is not None:
            if self.problem.is_pass_fail() and self._metadata.legacy_grading.on_reject == 'grade':
                self.error("Invalid on_reject policy 'grade' for problem type 'pass-fail'")

        for deprecated_grading_key in ['accept_score', 'reject_score', 'range', 'on_reject']:
            if getattr(self._metadata.legacy_grading, deprecated_grading_key) is not None:
                self.warning(
                    f"Grading key '{deprecated_grading_key}' is deprecated in problem.yaml, use '{deprecated_grading_key}' in testdata.yaml instead"
                )

        if self._metadata.legacy_validation:
            val = self._metadata.legacy_validation.split()
            validation_type = val[0]
            validation_params = val[1:]
            if validation_type not in ['default', 'custom']:
                self.error(f"Invalid value '{validation_type}' for validation, first word must be 'default' or 'custom'")

            if validation_type == 'default' and len(validation_params) > 0:
                self.error(f"Invalid value '{self._metadata.legacy_validation}' for validation")

            if validation_type == 'custom':
                for param in validation_params:
                    if param not in ['score', 'interactive']:
                        self.error(f"Invalid parameter '{param}' for custom validation")

        if self._metadata.limits.time_limit is not None and not self._metadata.limits.time_limit.is_integer():
            self.warning(
                'Time limit configured to non-integer value. This can be fragile, and may not be supported by your CCS (Kattis does not).'
            )
        if not self._metadata.limits.time_resolution.is_integer():
            self.warning(
                'Time resolution is not an integer. This can be fragile, and may not be supported by your CCS (Kattis does not).'
            )

        # Validate constant names and values
        constant_name_re = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
        for name, value in self._metadata.constants.items():
            if not constant_name_re.match(name):
                self.error(f"Invalid constant name '{name}' (must match [a-zA-Z_][a-zA-Z0-9_]*)")
            if isinstance(value, dict):
                if 'value' not in value:
                    self.error(f"Constant '{name}' is a map but missing required 'value' key")
                else:
                    # Validate variant keys and check if variants appear to differ from 'value'
                    main_value = value['value']
                    for variant_key, variant_value in value.items():
                        if variant_key == 'value':
                            continue
                        if not constant_name_re.match(variant_key):
                            self.error(f"Invalid variant key '{variant_key}' for constant '{name}' (must match [a-zA-Z_][a-zA-Z0-9_]*)")
                        # Best-effort heuristic: warn if numeric values differ
                        try:
                            if float(main_value) != float(variant_value):
                                pass  # Variants are expected to differ in representation
                        except (ValueError, TypeError):
                            pass

        return self._check_res


class Attachments(ProblemPart):
    """Represents the attachments of a problem.

    Attributes:
        attachments: The absolute paths to the attachment files for this problem.
    """

    attachments: list[Path]

    PART_NAME = 'attachments'

    # Directories allowed inside attachments/ (2025-09)
    _ALLOWED_ATTACHMENT_DIRS = frozenset({'templates'})

    def setup(self):
        attachments_dir = Path(self.problem.probdir) / 'attachments'
        self.attachments = [p for p in attachments_dir.iterdir()] if attachments_dir.is_dir() else []
        self.debug(f'Adding attachments {str(self.attachments)}')

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        if self.problem.format is FormatVersion.V_2025_09:
            self._check_2025_09()
        else:
            self._check_legacy()

        # Check solution/ directory
        solution_dir = Path(self.problem.probdir) / 'solution'
        if solution_dir.is_dir():
            if self.problem.format is FormatVersion.LEGACY:
                self.warning("'solution/' directory is only supported in 2025-09 format")
            else:
                self._check_solution_dir(solution_dir)

        # Check generators/ directory (informational only, just accept it)
        generators_dir = Path(self.problem.probdir) / 'generators'
        if generators_dir.is_dir():
            if self.problem.format is FormatVersion.LEGACY:
                self.warning("'generators/' directory is only supported in 2025-09 format")
            else:
                self.debug("Found generators/ directory (informational only)")

        # Check include/ directory
        include_dir = Path(self.problem.probdir) / 'include'
        if include_dir.is_dir():
            self._check_include_dir(include_dir)

        return self._check_res

    def _check_legacy(self):
        """Legacy format: directories not allowed as attachments."""
        for attachment_path in self.attachments:
            if os.path.isdir(attachment_path):
                self.error(f'Directories are not allowed as attachments ({attachment_path} is a directory)')

    def _check_2025_09(self):
        """2025-09 format: only known subdirectories allowed in attachments/."""
        for attachment_path in self.attachments:
            if attachment_path.is_dir():
                if attachment_path.name not in self._ALLOWED_ATTACHMENT_DIRS:
                    self.error(f"Unknown directory '{attachment_path.name}' in attachments/ (allowed: {', '.join(sorted(self._ALLOWED_ATTACHMENT_DIRS))})")
        # Validate templates/ if present
        templates_dir = Path(self.problem.probdir) / 'attachments' / 'templates'
        if templates_dir.is_dir():
            self._check_templates_dir(templates_dir)

    def _check_templates_dir(self, templates_dir: Path):
        """Validate attachments/templates/ directory structure."""
        for entry in templates_dir.iterdir():
            if entry.is_dir():
                if not any(entry.iterdir()):
                    self.error(f"Template directory '{entry.name}' is empty")
            else:
                self.warning(f"Unexpected file '{entry.name}' in attachments/templates/ (expected language directories)")

    def _check_solution_dir(self, solution_dir: Path):
        """Validate solution/ directory for 2025-09 format."""
        solution_pattern = re.compile(r'^solution(\.[a-z]{2})?\.(?:tex|md|pdf)$')
        has_solution_file = False
        for entry in solution_dir.iterdir():
            if entry.is_file() and solution_pattern.match(entry.name):
                has_solution_file = True
        if not has_solution_file:
            self.warning("solution/ directory exists but contains no solution.<language>.<filetype> files")

    def _check_include_dir(self, include_dir: Path):
        """Validate include/ directory structure.

        Each subdirectory must be named 'default' or a language code, and must be non-empty.
        """
        for entry in include_dir.iterdir():
            if entry.is_dir():
                if not any(entry.iterdir()):
                    self.error(f"Include directory '{entry.name}' is empty (must be non-empty)")
            else:
                self.warning(f"Unexpected file '{entry.name}' directly in include/ (expected language directories)")

    def get_attachment_paths(self):
        return self.attachments

    def __str__(self) -> str:
        return 'attachments'


# Junk data. The validator should reject these cases
_JUNK_CASES = [
    ('an empty file', b''),
    ('a binary file with random bytes', bytearray(random.Random(42).randbytes(1024))),
    ('a text file with the ASCII characters 32 up to 127', bytearray(x for x in range(32, 127))),
    (
        'a random text file with printable ASCII characters',
        (lambda rng: bytearray(rng.choice(string.printable.encode('utf8')) for _ in range(200)))(random.Random(42)),
    ),
]

# Try to crash the output validator, causing a judge error
_JUNK_CASES_CRASH = [
    ('a file with the number -1', b'-1'),
    ('a file with the number 2147483647', b'2147483647'),
    ('a file with the number 2147483648', b'2147483648'),
    ('a file with the number 9223372036854775808', b'9223372036854775808'),
    ('a file with the number 0', b'0'),
    ('a file with the number 1', b'1'),
    ('a file with the number 1.0', b'1.0'),
    ('a file with the string "a"', b'a'),
    ('a file with the contents "2\\n-1 1"', b'2\n-1 1'),
    ('a file with the contents "2\\n1"', b'2\n1'),
    ('a file with the contents "1\\n-1 1"', b'1\n-1 1'),
    ('a file with the contents "1\\na"', b'1\na'),
    ('a file with the contents "(()"', b'(()'),
    ('a file with the contents "1-"', b'1-'),
    ('a file with the contents "1/0"', b'1/0'),
    ('a file with the contents "2\\n<"', b'2\n<'),
    ('a file with the contents "NaN"', b'NaN'),
    ('a file with the contents "inf"', b'inf'),
    ('a file with the contents "\\x00"', b'\x00'),
    ('a file with the contents "\\x80"', b'\x80'),
]


def _build_junk_modifier(
    desc: str, pattern: str, repl: str | Callable[[Match[str]], str]
) -> tuple[str, Callable, Callable[[str], str]]:
    p = re.compile(pattern)
    return (desc, p.search, lambda text: p.sub(repl, text))


_JUNK_MODIFICATIONS = [
    _build_junk_modifier('spaces added where there already is whitespace', r'\s', lambda m: m.group(0) + ' '),
    _build_junk_modifier('spaces added to the end of a line', r'\n', lambda m: m.group(0) + ' '),
    _build_junk_modifier('newlines added where there already are newlines', '\n', lambda m: '\n\n'),
    _build_junk_modifier('leading zeros added to integers', r'(^|[^.]\b)([0-9]+)\b', r'\g<1>0000000000\g<2>'),
    _build_junk_modifier('trailing zeros added to real number decimal portion', r'\.[0-9]+\b', r'\g<0>0000000000'),
    (
        'random junk added to the end of the file',
        lambda f: True,
        lambda f: f + ''.join(random.choice(string.printable) for _ in range(200)),
    ),
]

# 2025-09 spec: validators/visualizers without build/run must be Python 3, C, or C++
_ALLOWED_VALIDATOR_LANG_IDS = frozenset({'python3', 'c', 'cpp'})


def _check_validator_language(part: ProblemPart, program, label: str) -> None:
    """Warn if a validator program uses a language not allowed in 2025-09.

    Only applies to SourceCode programs (BuildRun, Viva, Checktestdata are exempt).
    """
    if part.problem.format is not FormatVersion.V_2025_09:
        return
    if isinstance(program, run.SourceCode) and program.language.lang_id not in _ALLOWED_VALIDATOR_LANG_IDS:
        part.warning(
            '%s %s uses language %s; 2025-09 requires Python 3, C, or C++ for programs without build/run'
            % (label, program.name, program.language.name)
        )


class InputValidators(ProblemPart):
    PART_NAME = 'input_validator'

    def setup(self):
        input_validators_path = os.path.join(self.problem.probdir, 'input_format_validators')
        if os.path.isdir(input_validators_path):
            self._uses_old_path = True
        else:
            self._uses_old_path = False
            new_input_validators_path = os.path.join(self.problem.probdir, 'input_validators')
            if os.path.isdir(new_input_validators_path):
                input_validators_path = new_input_validators_path
        self._validators = run.find_programs(
            input_validators_path,
            language_config=self.problem.language_config,
            allow_validation_script=True,
            work_dir=self.problem.tmpdir,
        )
        # Substitute constants in validator source files (2025-09 only)
        if self.problem.format is FormatVersion.V_2025_09 and self.problem.metadata.constants:
            for val in self._validators:
                if hasattr(val, 'path') and os.path.isdir(val.path):
                    undef = statement_util.substitute_constants_in_directory(val.path, self.problem.metadata.constants)
                    for ref in undef:
                        self.warning("Undefined constant '{{%s}}' in input validator %s" % (ref, val))
        return {}

    def __str__(self) -> str:
        return 'input format validators'

    def start_background_work(self, context: Context) -> None:
        for val in self._validators:
            context.submit_background_work(lambda v: v.compile(), val)

    def check(self, context: Context | None) -> bool:
        if self._check_res is not None:
            return self._check_res
        if self._uses_old_path:
            self.warning('input_format_validators is a deprecated name; please use input_validators instead')
        self._check_res = True
        if len(self._validators) == 0:
            self.error('No input format validators found')

        for val in self._validators[:]:
            _check_validator_language(self, val, 'Input validator')
            try:
                success, msg = val.compile()
                if not success:
                    self.error(f'Compile error for {val}', msg)
                    self._validators.remove(val)
            except run.ProgramError as e:
                self.error(str(e))

        # Only sanity check input validators if they all actually compiled
        if self._check_res:
            all_flags: set[str] = set()

            def collect_flags(group: TestCaseGroup, flags: set[str]) -> None:
                if len(group.get_testcases()) > 0:
                    flags.add(group.config['input_validator_flags'])
                for subgroup in group.get_subgroups():
                    collect_flags(subgroup, flags)

            collect_flags(self.problem.testdata, all_flags)

            fd, file_name = tempfile.mkstemp()
            os.close(fd)
            for desc, case in _JUNK_CASES:
                f = open(file_name, 'wb')
                f.write(case)
                f.close()
                for flags_str in all_flags:
                    flags = flags_str.split()
                    for val in self._validators:
                        status, _ = val.run(file_name, args=flags)
                        if os.WEXITSTATUS(status) != 42:
                            break
                    else:
                        self.warning(f'No validator rejects {desc} with flags "{" ".join(flags)}"')

            def modified_input_validates(applicable, modifier):
                for testcase in self.problem.testdata.get_all_testcases():
                    with open(testcase.infile) as infile:
                        infile_data = infile.read()
                    if not applicable(infile_data):
                        continue

                    with open(file_name, 'wb') as f:
                        f.write(modifier(infile_data).encode('utf8'))

                    for flags_str in all_flags:
                        flags = flags_str.split()
                        for val in self._validators:
                            status, _ = val.run(file_name, args=flags)
                            if os.WEXITSTATUS(status) != 42:
                                # expected behavior; validator rejects modified input
                                return False

                    # we found a file we could modify, and all validators
                    # accepted the modifications
                    return True

                # no files were modifiable
                return False

            for desc, applicable, modifier in _JUNK_MODIFICATIONS:
                if modified_input_validates(applicable, modifier):
                    self.warning(f'No validator rejects {desc}')

            os.unlink(file_name)

        return self._check_res

    def validate(self, testcase: TestCase) -> None:
        flags = testcase.testcasegroup.config['input_validator_flags'].split()

        # Remove input validators that don't compile, even without -p validators
        self.check(None)

        for val in self._validators:
            with tempfile.NamedTemporaryFile() as outfile, tempfile.NamedTemporaryFile() as errfile:
                status, _ = val.run(testcase.infile, outfile.name, errfile.name, args=flags)
                if not os.WIFEXITED(status):
                    emsg = f'Input format validator {val} crashed on input {testcase.infile}'
                elif os.WEXITSTATUS(status) != 42:
                    emsg = f'Input format validator {val} did not accept input {testcase.infile}, exit code: {os.WEXITSTATUS(status)}'
                else:
                    continue
                validator_stdout = outfile.read().decode('utf-8', 'replace')
                validator_stderr = errfile.read().decode('utf-8', 'replace')
                validator_output = '\n'.join(out for out in [validator_stdout, validator_stderr] if out)
                testcase.error(emsg, validator_output)


class Graders(ProblemPart):
    _default_grader = run.get_tool('default_grader')

    PART_NAME = 'grader'

    def setup(self):
        graders: list = run.find_programs(
            os.path.join(self.problem.probdir, 'graders'),
            language_config=self.problem.language_config,
            work_dir=self.problem.tmpdir,
        )
        if len(graders) > 1:
            self.fatal('There is more than one custom grader')
        self._grader = graders[0] if graders else None
        return {}

    def __str__(self) -> str:
        return 'graders'

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        # In 2025-09, graders/ directory is not allowed - scoring is done via score_aggregation
        if self.problem.format is FormatVersion.V_2025_09:
            if len(self._graders) > 0:
                self.error('The graders/ directory is not allowed in 2025-09 format. Use score_aggregation in test_group.yaml instead.')
            return self._check_res

        if self.problem.is_pass_fail() and len(self._graders) > 0:
            self.error('There are grader programs but the problem is pass-fail')

            success, msg = self._grader.compile()
            if not success:
                self.fatal(f'Compile error for {self._grader}', msg)
        return self._check_res

    def grade(
        self, sub_results: list[SubmissionResult], testcasegroup: TestCaseGroup, shadow_result: bool = False
    ) -> tuple[Verdict, float | None]:
        if testcasegroup.config['grading'] == 'default':
            if not self._default_grader:
                self.fatal('Failed to locate default grader')
                return ('JE', None)
            grader = self._default_grader
        else:
            if not self._grader:
                self.fatal('Problem has grading: custom without any custom grader')
                return ('JE', None)
            grader = self._grader

        if not grader.compile()[0]:
            self.fatal(f'Failed to compile grader {grader}', grader.compile()[1])
            return ('JE', None)

        grader_input = ''.join([f'{r.verdict} {0 if r.score is None else r.score}\n' for r in sub_results])
        grader_output_re = r'^((AC)|(WA)|(TLE)|(RTE)|(JE))\s+-?[0-9.]+\s*$'
        verdict: Verdict = 'AC'
        score: float = 0

        if not sub_results:
            self.info(f'No results on {testcasegroup}, so no grader ran')
            return (verdict, score)

        grader_flags = testcasegroup.config['grader_flags'].split()
        self.debug(f'Grading {len(sub_results)} results:\n{grader_input}')
        self.debug(f'Grader flags: {grader_flags}')

        infile_path = outfile_path = errfile_path = None
        try:
            # Create input and output files for grader
            # We do it in this awkward way because the files need to be closed before reading/writing
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as infile:
                infile.write(grader_input)
                infile_path = infile.name

            with tempfile.NamedTemporaryFile(delete=False) as outfile:
                outfile_path = outfile.name

            with tempfile.NamedTemporaryFile(delete=False) as errfile:
                errfile_path = errfile.name

            status, runtime = grader.run(infile_path, outfile_path, errfile_path, args=grader_flags)

            with open(outfile_path, 'r') as fh:
                grader_output = fh.read()

            with open(errfile_path, 'r') as errfile:
                stderr_content = errfile.read()

            if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                if not os.WIFEXITED(status):
                    self.error(f'Judge error: {grader} crashed')
                else:
                    self.error(f'Judge error: exit code {os.WEXITSTATUS(status)} for grader {grader}, expected 0')
                self.error(f'Grader stderr:\n{stderr_content}\n')
                self.debug(f'Grader input:\n{grader_input}')
                return ('JE', None)

            if not re.match(grader_output_re, grader_output):
                self.error('Judge error: invalid format of grader output')
                self.debug(f'Output must match: "{grader_output_re}"')
                self.debug(f'Output was: "{grader_output}"')
                return ('JE', None)

            verdict_str, score_str = grader_output.split()
            # Make mypy happy by explicitly using cast
            verdict = cast(Verdict, verdict_str)
            score = float(score_str)

            if not shadow_result:
                self.debug(f'Grade on {testcasegroup} is {verdict} ({score})')

            return (verdict, score)
        except Exception as e:
            self.error(f'Grader failed with exception {e}')
            return ('JE', None)
        finally:
            for path in [infile_path, outfile_path, errfile_path]:
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass


class OutputValidators(ProblemPart):
    _default_validator = run.get_tool('default_validator')

    PART_NAME = 'output_validator'

    def setup(self):
        self._validators = run.find_programs(
            os.path.join(self.problem.probdir, self.problem.format.output_validator_directory),
            language_config=self.problem.language_config,
            work_dir=self.problem.tmpdir,
        )
        self._has_precompiled = False
        # Substitute constants in validator source files (2025-09 only)
        if self.problem.format is FormatVersion.V_2025_09 and self.problem.metadata.constants:
            for val in self._validators:
                if hasattr(val, 'path') and os.path.isdir(val.path):
                    undef = statement_util.substitute_constants_in_directory(val.path, self.problem.metadata.constants)
                    for ref in undef:
                        self.warning("Undefined constant '{{%s}}' in output validator %s" % (ref, val))

    def uses_default_validator(self) -> bool:
        if self.problem.format is FormatVersion.LEGACY:
            return self.problem.metadata.legacy_validation == 'default'
        return not self._validators

    def __str__(self) -> str:
        return 'output validators'

    def start_background_work(self, context: Context) -> None:
        if not self._has_precompiled:
            for val in self._actual_validators():
                context.submit_background_work(lambda v: v.compile(), val)
            self._has_precompiled = True

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        self.warn_directory('output validators', 'output_validator_directory')

        for v in self._validators:
            if isinstance(v, run.SourceCode) and v.language.lang_id not in _ALLOWED_VALIDATOR_LANG_IDS:
                self.error_in_2025_09(
                    f'Output validator in {v.language.name}. Only Python 3, C, C++ are standardized for programs without build/run.'
                )

        if len(self._validators) > 1:
            self.error_in_2025_09('Found more than one output validator. This was allowed in legacy (but not on Kattis)')

        if self.uses_default_validator() and self._validators:
            self.error('There are validator programs but problem.yaml has validation = "default"')
        elif not self.uses_default_validator() and not self._validators:
            self.fatal('problem.yaml specifies custom validator but no validator programs found')

        if self.uses_default_validator() and self._default_validator is None:
            self.fatal('Unable to locate default validator')

        for val in self._validators[:]:
            try:
                success, msg = val.compile()
                if not success:
                    self.fatal(f'Compile error for output validator {val}', msg)
            except run.ProgramError as e:
                self.error(str(e))

        # Only sanity check output validators if they all actually compiled
        if self._check_res:
            flags = self.problem.metadata.legacy_validator_flags

            # Sanity check cases that should be rejected by the output validator
            def run_junk_case(case_desc: str, junk_content: bytes, testcases: list[TestCase]) -> list[SubmissionResult]:
                results = []
                with tempfile.NamedTemporaryFile(mode='wb') as f:
                    f.write(junk_content)
                    f.flush()
                    for testcase in testcases:
                        result = self.validate(testcase, f.name)
                        results.append(result)
                        if result.verdict == 'JE':
                            self.error(f'{case_desc} as output, and output validator flags "{" ".join(flags)}" gave {result}')
                            break
                return results

            # Junk cases that the output validator should reject
            for desc, junk_case_content in _JUNK_CASES:
                results = run_junk_case(desc, junk_case_content, self.problem.testdata.get_all_testcases())
                rejected = any(result.verdict != 'AC' for result in results)
                if not rejected:
                    self.warning(f'{desc} gets AC')

            # For performance reasons, strongly limit the amount of testcases we run on
            fast_languages = {'c', 'cpp'}
            all_validators_are_fast = True
            for val in self._validators:
                if isinstance(val, run.SourceCode):
                    all_validators_are_fast &= val.language.lang_id in fast_languages
            num_testcases = 3 if all_validators_are_fast else 1
            test_cases = self.problem.testdata.get_all_testcases()[:num_testcases]
            # Malformed cases that a poorly-written output validator might crash on
            # Note that these might be valid output, so we only check if it crashes
            for desc, junk_case_content in _JUNK_CASES_CRASH:
                run_junk_case(desc, junk_case_content, test_cases)

        return self._check_res

    @staticmethod
    def _get_feedback(feedback_dir: str) -> str | None:
        all_feedback = []
        for feedback_file in os.listdir(feedback_dir):
            feedback_path = os.path.join(feedback_dir, feedback_file)
            if os.path.getsize(feedback_path) == 0:
                continue
            all_feedback.append(f'=== {feedback_file}: ===')
            # Note: The file could contain non-unicode characters, "replace" to be on the safe side
            with open(feedback_path, 'r', errors='replace') as feedback:
                # Cap amount of feedback per file at some high-ish
                # size, so that a buggy validator spewing out lots of
                # data doesn't kill us.
                all_feedback.append(feedback.read(128 * 1024))
        if all_feedback:
            return '\n'.join(all_feedback)
        return None

    def _parse_validator_results(self, val, status: int, feedbackdir, testcase) -> SubmissionResult:
        score = None

        if not os.WIFEXITED(status):
            return SubmissionResult(
                'JE',
                reason=f'output validator {val} crashed, status {status}',
                additional_info=OutputValidators._get_feedback(feedbackdir),
            )
        ret = os.WEXITSTATUS(status)
        if ret not in [42, 43]:
            return SubmissionResult(
                'JE',
                reason=f'output validator {val} exited with status {ret}',
                additional_info=OutputValidators._get_feedback(feedbackdir),
            )

        if ret == 43:
            return SubmissionResult('WA', additional_info=OutputValidators._get_feedback(feedbackdir))

        # ret == 42 means AC. Now handle scoring.
        score_file = os.path.join(feedbackdir, 'score.txt')
        score_multiplier_file = os.path.join(feedbackdir, 'score_multiplier.txt')
        has_score = os.path.isfile(score_file)
        has_multiplier = os.path.isfile(score_multiplier_file)

        if self.problem.format is FormatVersion.V_2025_09:
            return self._parse_score_2025_09(has_score, has_multiplier, score_file, score_multiplier_file, feedbackdir, testcase)
        else:
            return self._parse_score_legacy(has_score, score_file, feedbackdir)

    def _parse_score_legacy(self, has_score: bool, score_file: str, feedbackdir: str) -> SubmissionResult:
        """Parse score for legacy format."""
        custom_score = self.problem.metadata.legacy_custom_score
        score = None

        if not custom_score and has_score:
            return SubmissionResult(
                'JE', reason='validator produced "score.txt" but problem does not have custom scoring activated'
            )

        if custom_score:
            if has_score:
                try:
                    score_str = open(score_file).read()
                    score = float(score_str)
                except Exception as e:
                    return SubmissionResult('JE', reason=f'failed to parse validator score: {e}')
            else:
                # If we're running multipass, we do not need to output a score after every pass
                # We accept the small risk of allowing a non-multipass output validator to not output score.txt
                # if it produces a file called nextpass.in
                if (Path(feedbackdir) / 'nextpass.in').exists():
                    score = 0
                else:
                    return SubmissionResult('JE', reason='problem has custom scoring but validator did not produce "score.txt"')

        return SubmissionResult('AC', score=score)

    def _parse_score_2025_09(self, has_score: bool, has_multiplier: bool, score_file: str,
                              score_multiplier_file: str, feedbackdir: str, testcase) -> SubmissionResult:
        """Parse score for 2025-09 format, supporting score.txt and score_multiplier.txt."""
        score = None

        if has_score and has_multiplier:
            return SubmissionResult(
                'JE', reason='validator produced both "score.txt" and "score_multiplier.txt" (only one is allowed)'
            )

        if not self.problem.is_scoring():
            # Pass-fail problem: score files are not expected
            if has_score or has_multiplier:
                # Not a judge error, just ignored for sample test cases on scoring problems.
                # For pass-fail, it IS an error.
                return SubmissionResult(
                    'JE', reason='validator produced score file but problem is not a scoring problem'
                )
            return SubmissionResult('AC')

        # Scoring problem with AC verdict
        if has_score:
            try:
                score_str = open(score_file).read().strip()
                score = float(score_str)
            except Exception as e:
                return SubmissionResult('JE', reason=f'failed to parse score.txt: {e}',
                                       additional_info=OutputValidators._get_feedback(feedbackdir))
            if score < 0:
                return SubmissionResult('JE', reason=f'score.txt contains negative value: {score}')

        elif has_multiplier:
            try:
                mult_str = open(score_multiplier_file).read().strip()
                multiplier = float(mult_str)
            except Exception as e:
                return SubmissionResult('JE', reason=f'failed to parse score_multiplier.txt: {e}',
                                       additional_info=OutputValidators._get_feedback(feedbackdir))
            if not (0.0 <= multiplier <= 1.0):
                return SubmissionResult('JE', reason=f'score_multiplier.txt value {multiplier} is not in [0, 1]')

            # Compute score from multiplier and max score per test case
            # The actual max score per test case depends on the group configuration
            # For now, store the multiplier as the score - the TestCaseGroup will adjust
            score = multiplier  # Will be multiplied by max test case score in _init_result_for_testcase

        # If no score file and problem is scoring, the score will be determined by the test case config
        # (default for bounded: max_score per case; required for unbounded: must produce score.txt)
        return SubmissionResult('AC', score=score)

    def _actual_validators(self) -> list:
        if self.uses_default_validator():
            return [self._default_validator]
        return self._validators

    def validate_interactive(self, testcase, submission, timelim: int, errorhandler: Submissions) -> SubmissionResult:
        # This may be called off-main thread.
        interactive_output_re = r'\d+ \d+\.\d+ \d+ \d+\.\d+ (validator|submission)'
        res = SubmissionResult('JE')
        interactive = run.get_tool('interactive')
        if interactive is None:
            errorhandler.error('Could not locate interactive runner')
            return res
        # file descriptor, wall time lim
        initargs = ['1', str(math.ceil(2 * timelim))]
        validator_args = [infile if infile else testcase.infile, testcase.ansfile, '<feedbackdir>']
        submission_args = submission.get_runcmd(memlim=self.problem.metadata.limits.memory)

        val_memlim = self.problem.metadata.limits.validation_memory
        for i, val in enumerate(self._actual_validators()):
            if val.compile()[0]:
                # If we are running multiple output validators in legacy, make sure to wipe it
                # If we are running multipass, i will always be 0 and we do not accidentally wipe feedback
                if i > 0 and feedback_dir_path:
                    shutil.rmtree(feedback_dir_path)
                    Path(feedback_dir_path).mkdir()

                if feedback_dir_path:
                    feedbackdir = feedback_dir_path
                else:
                    feedbackdir = tempfile.mkdtemp(prefix='feedback', dir=self.problem.tmpdir)

                validator_args[2] = feedbackdir + os.sep
                f = tempfile.NamedTemporaryFile(delete=False)
                interactive_out = f.name
                f.close()
                i_status, _ = interactive.run(
                    outfile=interactive_out,
                    args=initargs + val.get_runcmd(memlim=val_memlim) + validator_args + [';'] + submission_args,
                    work_dir=submission.path,
                )
                if is_RTE(i_status):
                    errorhandler.error(f'Interactive crashed, status {i_status}')
                else:
                    interactive_output = open(interactive_out).read()
                    errorhandler.debug(f'Interactive output: "{interactive_output}"')
                    if not re.match(interactive_output_re, interactive_output):
                        errorhandler.error(
                            f'Output from interactive does not follow expected format, got output "{interactive_output}"'
                        )
                    else:
                        val_status_str, _, sub_status_str, sub_runtime_str, first = interactive_output.split()
                        sub_status = int(sub_status_str)
                        sub_runtime = float(sub_runtime_str)
                        val_status = int(val_status_str)
                        val_JE = not os.WIFEXITED(val_status) or os.WEXITSTATUS(val_status) not in [42, 43]
                        val_WA = os.WIFEXITED(val_status) and os.WEXITSTATUS(val_status) == 43
                        if val_JE or (val_WA and first == 'validator'):
                            # If the validator crashed, or exited first with WA,
                            # always follow validator verdict, even if that early
                            # exit caused the submission to behave erratically and
                            # time out.
                            if sub_runtime > timelim:
                                sub_runtime = timelim
                            res = self._parse_validator_results(val, val_status, feedbackdir, testcase)
                        elif is_TLE(sub_status, True) or sub_runtime > timelim:
                            res = SubmissionResult('TLE')
                        elif is_RTE(sub_status):
                            res = SubmissionResult('RTE')
                        else:
                            res = self._parse_validator_results(val, val_status, feedbackdir, testcase)

                        res.runtime = sub_runtime
                        res.validator_first = first == 'validator'

                os.unlink(interactive_out)
                if feedback_dir_path is None:
                    shutil.rmtree(feedbackdir)
                if res.verdict != 'AC':
                    return res
        return res

    def validate(self, testcase, submission_output: str, feedbackdir: str | None = None) -> SubmissionResult:
        """Validate submission output against expected answer.

        Args:
            testcase: The test case being validated
            submission_output: Path to file with submission output
            feedbackdir: If provided, use this feedback directory (and don't delete it).
                         Used for multi-pass to persist feedback between passes.

        Returns:
            SubmissionResult with verdict
        """
        res = SubmissionResult('JE')
        val_timelim = self.problem.metadata.limits.validation_time
        val_memlim = self.problem.metadata.limits.validation_memory
        flags = (
            self.problem.metadata.legacy_validator_flags.split() + testcase.testcasegroup.config['output_validator_flags'].split()
        )
        caller_owns_feedbackdir = feedbackdir is not None
        for val in self._actual_validators():
            if val.compile()[0]:
                if not caller_owns_feedbackdir:
                    feedbackdir = tempfile.mkdtemp(prefix='feedback', dir=self.problem.tmpdir)
                validator_output = tempfile.mkdtemp(prefix='checker_out', dir=self.problem.tmpdir)
                outfile = validator_output + '/out.txt'
                errfile = validator_output + '/err.txt'
                status, runtime = val.run(
                    infile=submission_output,
                    args=[infile if infile else testcase.infile, testcase.ansfile, feedbackdir] + flags,
                    timelim=val_timelim,
                    memlim=val_memlim,
                    outfile=outfile,
                    errfile=errfile,
                )
                if self.log.isEnabledFor(logging.DEBUG):
                    try:
                        with open(outfile, mode='rt') as f:
                            output = f.read()
                        if output:
                            self.log.debug('Validator output:\n%s', output)
                        with open(errfile, mode='rt') as f:
                            error = f.read()
                        if error:
                            self.log.debug('Validator stderr:\n%s', error)
                    except IOError as e:
                        self.info('Failed to read validator output: %s', e)
                res = self._parse_validator_results(val, status, feedbackdir, testcase)
                if not caller_owns_feedbackdir:
                    shutil.rmtree(feedbackdir)
                shutil.rmtree(validator_output)
                if feedback_dir_path is None:
                    shutil.rmtree(feedbackdir)
                if res.verdict != 'AC':
                    return res

        # TODO: check that all output validators give same result
        return res


class StaticValidator(ProblemPart):
    """Represents the static validator (2025-09 only).

    Static validators analyze submission source code to accept/reject submissions.
    They can optionally assign scores for static validation test cases.
    """

    PART_NAME = 'static_validator'

    def setup(self):
        static_dir = os.path.join(self.problem.probdir, 'static_validator')
        if os.path.isdir(static_dir):
            self._programs = run.find_programs(
                static_dir,
                language_config=self.problem.language_config,
                work_dir=self.problem.tmpdir,
            )
            # Substitute constants in static validator source files (2025-09 only)
            if self.problem.format is FormatVersion.V_2025_09 and self.problem.metadata.constants:
                for prog in self._programs:
                    if hasattr(prog, 'path') and os.path.isdir(prog.path):
                        undef = statement_util.substitute_constants_in_directory(prog.path, self.problem.metadata.constants)
                        for ref in undef:
                            self.warning("Undefined constant '{{%s}}' in static validator %s" % (ref, prog))
        else:
            self._programs = []

    def __str__(self) -> str:
        return 'static validator'

    def has_static_validator(self) -> bool:
        return len(self._programs) > 0

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        if self.problem.format is FormatVersion.LEGACY:
            if self._programs:
                self.error("static_validator/ is not supported in legacy format")
            return self._check_res

        if not self._programs:
            return self._check_res

        if self.problem.is_submit_answer():
            self.error("static_validator/ is not allowed for submit-answer problems")

        if len(self._programs) > 1:
            self.warning(f'Found {len(self._programs)} static validators; expected at most 1')

        for prog in self._programs:
            _check_validator_language(self, prog, 'Static validator')
            success, msg = prog.compile()
            if not success:
                self.error(f'Compile error for static validator {prog}', msg)

        # Check that static_validation_score is only specified if we have a static validator
        # This is checked in TestCaseGroup._check_2025_09 already

        return self._check_res


class OutputVisualizer(ProblemPart):
    """Represents the output visualizer (2025-09 only).

    Output visualizers generate images illustrating submission output.
    They run after the output validator.
    """

    PART_NAME = 'output_visualizer'

    def setup(self):
        visualizer_dir = os.path.join(self.problem.probdir, 'output_visualizer')
        if os.path.isdir(visualizer_dir):
            self._programs = run.find_programs(
                visualizer_dir,
                language_config=self.problem.language_config,
                work_dir=self.problem.tmpdir,
            )
        else:
            self._programs = []

    def __str__(self) -> str:
        return 'output visualizer'

    def has_output_visualizer(self) -> bool:
        return len(self._programs) > 0

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        if self.problem.format is FormatVersion.LEGACY:
            if self._programs:
                self.error("output_visualizer/ is not supported in legacy format")
            return self._check_res

        if not self._programs:
            return self._check_res

        if len(self._programs) > 1:
            self.warning(f'Found {len(self._programs)} output visualizers; expected at most 1')

        for prog in self._programs:
            _check_validator_language(self, prog, 'Output visualizer')
            success, msg = prog.compile()
            if not success:
                # Compile errors in visualizers are not judge errors
                self.warning(f'Compile error for output visualizer {prog}: {msg}')

        return self._check_res


class InputVisualizer(ProblemPart):
    """Represents the input visualizer (2025-09 only).

    Input visualizers generate test case illustrations.
    They are informational only and not invoked by the judging system.
    """

    PART_NAME = 'input_visualizer'

    def setup(self):
        visualizer_dir = os.path.join(self.problem.probdir, 'input_visualizer')
        if os.path.isdir(visualizer_dir):
            self._programs = run.find_programs(
                visualizer_dir,
                language_config=self.problem.language_config,
                work_dir=self.problem.tmpdir,
            )
        else:
            self._programs = []

    def __str__(self) -> str:
        return 'input visualizer'

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        if self.problem.format is FormatVersion.LEGACY:
            if self._programs:
                self.warning("input_visualizer/ is not typically used in legacy format")
            return self._check_res

        if not self._programs:
            return self._check_res

        for prog in self._programs:
            _check_validator_language(self, prog, 'Input visualizer')
            success, msg = prog.compile()
            if not success:
                # Compile errors in visualizers are informational only
                self.warning(f'Compile error for input visualizer {prog}: {msg}')

        return self._check_res


class ValidationTestData(ProblemPart):
    """Validates data/invalid_input/, data/invalid_output/, and data/valid_output/ (2025-09 only)."""

    PART_NAME = 'validation_test_data'

    def setup(self):
        datadir = os.path.join(self.problem.probdir, 'data')
        self._invalid_inputs: list[str] = []  # .in files
        self._invalid_outputs: list[tuple[str, str, str]] = []  # (.in, .out, .ans) files
        self._valid_outputs: list[tuple[str, str, str]] = []  # (.in, .out, .ans) files

        invalid_input_dir = os.path.join(datadir, 'invalid_input')
        if os.path.isdir(invalid_input_dir):
            self._collect_invalid_inputs(invalid_input_dir)

        invalid_output_dir = os.path.join(datadir, 'invalid_output')
        if os.path.isdir(invalid_output_dir):
            self._collect_output_cases(invalid_output_dir, self._invalid_outputs)

        valid_output_dir = os.path.join(datadir, 'valid_output')
        if os.path.isdir(valid_output_dir):
            self._collect_output_cases(valid_output_dir, self._valid_outputs)

    def _collect_invalid_inputs(self, dirpath: str) -> None:
        """Recursively collect .in files from invalid_input directory tree."""
        for root, dirs, files in os.walk(dirpath):
            for f in sorted(files):
                if f.endswith('.in'):
                    self._invalid_inputs.append(os.path.join(root, f))

    def _collect_output_cases(self, dirpath: str, target: list[tuple[str, str, str]]) -> None:
        """Recursively collect (.in, .out, .ans) triples from invalid_output/valid_output directories."""
        for root, dirs, files in os.walk(dirpath):
            in_files = {f for f in files if f.endswith('.in')}
            for inf in sorted(in_files):
                base = inf[:-3]
                inpath = os.path.join(root, inf)
                outpath = os.path.join(root, f'{base}.out')
                anspath = os.path.join(root, f'{base}.ans')
                target.append((inpath, outpath, anspath))

    def __str__(self) -> str:
        return 'validation test data'

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        if self.problem.format is FormatVersion.LEGACY:
            # Legacy format doesn't support validation directories
            datadir = os.path.join(self.problem.probdir, 'data')
            for dirname in ('invalid_input', 'invalid_output', 'valid_output'):
                if os.path.isdir(os.path.join(datadir, dirname)):
                    self.error(f'Validation directory "{dirname}" is not allowed in legacy format')
            return self._check_res

        self._check_invalid_inputs()
        self._check_invalid_outputs()
        self._check_valid_outputs()

        return self._check_res

    def _check_invalid_inputs(self) -> None:
        """Each .in file under invalid_input must be rejected by at least one input validator."""
        if not self._invalid_inputs:
            return

        self.info(f'Checking {len(self._invalid_inputs)} invalid input file(s)')

        # Ensure input validators are compiled
        self.problem.input_validators.check(None)

        for infile in self._invalid_inputs:
            relpath = os.path.relpath(infile, self.problem.probdir)
            rejected = False
            for val in self.problem.input_validators._validators:
                with tempfile.NamedTemporaryFile() as outf, tempfile.NamedTemporaryFile() as errf:
                    status, _ = val.run(infile, outf.name, errf.name, args=[])
                    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 42:
                        rejected = True
                        break
            if not rejected:
                self.error(f'Invalid input file {relpath} was accepted by all input validators (should be rejected by at least one)')

    def _check_invalid_outputs(self) -> None:
        """Each case under invalid_output: .in must be valid input, .out must fail output validation with .ans."""
        if not self._invalid_outputs:
            return

        self.info(f'Checking {len(self._invalid_outputs)} invalid output case(s)')

        for infile, outfile, ansfile in self._invalid_outputs:
            relpath = os.path.relpath(infile, self.problem.probdir)

            # Check that .in file exists and is valid input
            if not os.path.isfile(infile):
                self.error(f'Missing input file {relpath}')
                continue

            # Validate input
            for val in self.problem.input_validators._validators:
                with tempfile.NamedTemporaryFile() as outf, tempfile.NamedTemporaryFile() as errf:
                    status, _ = val.run(infile, outf.name, errf.name, args=[])
                    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 42:
                        self.error(f'Input file {relpath} in invalid_output is not valid input (rejected by input validator)')
                        break

            # Check that .out exists
            if not os.path.isfile(outfile):
                self.error(f'Missing .out file for {relpath}')
                continue

            # Check that .ans exists
            if not os.path.isfile(ansfile):
                self.error(f'Missing .ans file for {relpath}')
                continue

            # Create a temporary TestCase-like object for the output validator
            # The .out must fail output validation with .ans
            tc = _ValidationTestCaseProxy(self.problem, infile, ansfile)
            result = self.problem.output_validators.validate(tc, outfile)
            if result.verdict == 'AC':
                relout = os.path.relpath(outfile, self.problem.probdir)
                self.error(f'Invalid output file {relout} was accepted by output validator (should be rejected)')

    def _check_valid_outputs(self) -> None:
        """Each case under valid_output: .in must be valid input, .out must pass output validation with .ans."""
        if not self._valid_outputs:
            return

        self.info(f'Checking {len(self._valid_outputs)} valid output case(s)')

        for infile, outfile, ansfile in self._valid_outputs:
            relpath = os.path.relpath(infile, self.problem.probdir)

            # Check that .in file exists and is valid input
            if not os.path.isfile(infile):
                self.error(f'Missing input file {relpath}')
                continue

            # Validate input
            for val in self.problem.input_validators._validators:
                with tempfile.NamedTemporaryFile() as outf, tempfile.NamedTemporaryFile() as errf:
                    status, _ = val.run(infile, outf.name, errf.name, args=[])
                    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 42:
                        self.error(f'Input file {relpath} in valid_output is not valid input (rejected by input validator)')
                        break

            # Check that .out exists
            if not os.path.isfile(outfile):
                self.error(f'Missing .out file for {relpath}')
                continue

            # Check that .ans exists
            if not os.path.isfile(ansfile):
                self.error(f'Missing .ans file for {relpath}')
                continue

            # The .out must pass output validation with .ans
            tc = _ValidationTestCaseProxy(self.problem, infile, ansfile)
            result = self.problem.output_validators.validate(tc, outfile)
            if result.verdict != 'AC':
                relout = os.path.relpath(outfile, self.problem.probdir)
                self.error(f'Valid output file {relout} was rejected by output validator: {result}')


class _ValidationTestCaseProxy:
    """Minimal proxy object for OutputValidators.validate() that mimics a TestCase for validation directories."""

    def __init__(self, problem: Problem, infile: str, ansfile: str):
        self.infile = infile
        self.ansfile = ansfile
        self._problem = problem
        # Provide a minimal config-like testcasegroup for output_validator_flags
        self.testcasegroup = _ValidationGroupProxy(problem)

    def is_in_sample_group(self) -> bool:
        return False


class _ValidationGroupProxy:
    """Minimal proxy for TestCaseGroup that provides the config needed by OutputValidators.validate()."""

    def __init__(self, problem: Problem):
        self.config = {
            'output_validator_flags': '',
        }


class Runner:
    def __init__(self, problem: Problem, sub, context: Context, timelim: float, timelim_low: float, timelim_high: float) -> None:
        self._problem = problem
        self._sub = sub
        self._context = context
        self._multithreaded = context.executor is not None
        self._timelim = timelim
        self._timelim_low = timelim_low
        self._timelim_high = timelim_high
        self._cache: dict[TestCase, TestCase.Result] = {}
        if self._multithreaded:
            self._queues: dict[TestCase, queue.Queue[TestCase.Result]] = {}
            self._lock = threading.Lock()
            self._started_jobs: set[TestCase] = set()
            self._done_groups: set[TestCaseGroup] = set()
            self._remaining_jobs: list[TestCase] = []
            self._recompute_jobs()

    def __enter__(self) -> Runner:
        if self._multithreaded:
            for i in range(len(self._remaining_jobs)):
                self._context.submit_background_work(self._work)
        return self

    def __exit__(self, *exc) -> None:
        if self._multithreaded:
            with self._lock:
                self._remaining_jobs = []

    def run(self, testcase: TestCase) -> tuple[TestCase.Result, bool]:
        while testcase.reuse_result_from:
            testcase = testcase.reuse_result_from

        if testcase in self._cache:
            return (self._cache[testcase], True)

        if sys.stdout.isatty():
            msg = f'Running {self._sub} on {testcase}...'
            sys.stdout.write(msg)
            sys.stdout.flush()

        if self._multithreaded:
            result = self._queues[testcase].get()
        else:
            result = self._run_submission_real(testcase)

        if sys.stdout.isatty():
            sys.stdout.write('\b \b' * len(msg))

        self._cache[testcase] = result
        return (result, False)

    def mark_group_done(self, group: TestCaseGroup, broken: bool) -> None:
        if self._multithreaded:
            self._done_groups.add(group)
            if broken:
                # Since a group was broken out of, some test cases may no
                # longer be relevant to run. Recompute the work list.
                self._recompute_jobs()

    def _run_submission_real(self, item: TestCase) -> TestCase.Result:
        return item.run_submission_real(self._sub, self._context, self._timelim, self._timelim_low, self._timelim_high)

    def _work(self) -> None:
        item = self._next_job()
        if item:
            res = self._run_submission_real(item)
            self._queues[item].put(res)

    def _gather_testcases(self, item: TestCase | TestCaseGroup) -> list[TestCase]:
        if not item.matches_filter(self._context.data_filter):
            return []
        if isinstance(item, TestCase):
            if item.reuse_result_from:
                return self._gather_testcases(item.reuse_result_from)
            else:
                return [item]
        elif item not in self._done_groups:
            ret = []
            for child in item.get_testcases() + item.get_subgroups():
                ret.extend(self._gather_testcases(child))
            return ret
        else:
            return []

    def _next_job(self) -> TestCase | None:
        with self._lock:
            if self._remaining_jobs:
                job = self._remaining_jobs.pop()
                self._started_jobs.add(job)
                return job
            else:
                return None

    def _recompute_jobs(self) -> None:
        with self._lock:
            seen = set(self._started_jobs)
            self._remaining_jobs = []
            for testcase in self._gather_testcases(self._problem.testdata):
                if testcase not in seen:
                    seen.add(testcase)
                    self._remaining_jobs.append(testcase)
                    if testcase not in self._queues:
                        self._queues[testcase] = queue.Queue(maxsize=1)
            self._remaining_jobs.reverse()


class _SubmissionDir:
    """Configuration for a submission directory."""
    __slots__ = ('name', 'accepted_verdicts', 'must_exist', 'is_partial',
                 'permitted', 'required')

    def __init__(self, name: str, accepted_verdicts: frozenset[str], must_exist: bool = False, is_partial: bool = False,
                 permitted: frozenset[str] | None = None, required: frozenset[str] | None = None):
        self.name = name
        self.accepted_verdicts = accepted_verdicts
        self.must_exist = must_exist
        # For legacy partially_accepted: run with relaxed time limit, warn if fully accepted
        self.is_partial = is_partial
        # 2025-09: permitted/required sets derived from submissions.yaml or defaults
        self.permitted = permitted
        self.required = required

    def verdict_ok(self, verdict: str) -> bool:
        return verdict in self.accepted_verdicts


def _brace_expand(pattern: str) -> list[str]:
    """Expand brace patterns in a glob string.

    Supports simple brace expansion like {a,b,c} but not nested braces.
    Returns a list of expanded patterns.

    Examples:
        'foo.{py,cpp}' -> ['foo.py', 'foo.cpp']
        '{simple,complex}.py' -> ['simple.py', 'complex.py']
        'noBraces' -> ['noBraces']
    """
    # Find the first top-level brace pair
    depth = 0
    start = -1
    for i, ch in enumerate(pattern):
        if ch == '{' and depth == 0:
            start = i
            depth = 1
        elif ch == '{':
            depth += 1
        elif ch == '}' and depth == 1:
            # Found matching brace
            prefix = pattern[:start]
            suffix = pattern[i + 1:]
            alternatives = pattern[start + 1:i].split(',')
            results = []
            for alt in alternatives:
                results.extend(_brace_expand(prefix + alt + suffix))
            return results
        elif ch == '}':
            depth -= 1
    return [pattern]


def _glob_match(pattern: str, path: str) -> bool:
    """Match a glob pattern against a path.

    * matches anything except /. No ** support. No [xyz] support.
    Brace expansion is handled before calling this.
    """
    import fnmatch
    # fnmatch doesn't handle path separators - we need to ensure * doesn't match /
    # Split both pattern and path by / and match each component
    pattern_parts = pattern.split('/')
    path_parts = path.split('/')
    if len(pattern_parts) != len(path_parts):
        return False
    return all(fnmatch.fnmatch(pp, pat) for pat, pp in zip(pattern_parts, path_parts))


def _submission_matches_glob(glob_pattern: str, submission_path: str) -> bool:
    """Check if a submission path matches a glob pattern.

    submission_path is relative to submissions/ (e.g. 'accepted/hello.py').
    A submission is matched if either itself or a parent directory is matched.
    """
    for expanded in _brace_expand(glob_pattern):
        if _glob_match(expanded, submission_path):
            return True
        # Also check if a parent directory of submission_path matches
        parts = submission_path.split('/')
        for i in range(1, len(parts)):
            parent = '/'.join(parts[:i])
            if _glob_match(expanded, parent):
                return True
    return False


def _testdata_matches_glob(glob_pattern: str, testdata_path: str) -> bool:
    """Check if a test data path matches a glob pattern.

    testdata_path is relative to data/ (e.g. 'secret/group1/case01').
    A test case is matched if either itself or any parent group is matched.
    """
    for expanded in _brace_expand(glob_pattern):
        if _glob_match(expanded, testdata_path):
            return True
        # Check parent groups
        parts = testdata_path.split('/')
        for i in range(1, len(parts)):
            parent = '/'.join(parts[:i])
            if _glob_match(expanded, parent):
                return True
    return False


_VALID_VERDICTS = frozenset({'AC', 'WA', 'TLE', 'RTE'})

_SUBMISSIONS_YAML_TOP_KEYS = frozenset({
    'language', 'entrypoint', 'authors', 'model_solution',
    'permitted', 'required', 'score', 'message', 'use_for_time_limit',
})

# Default submission directory configs for 2025-09 (from spec)
_DEFAULT_SUBMISSION_RULES_2025_09: dict[str, dict[str, Any]] = {
    'accepted': {
        'permitted': ['AC'],
    },
    'rejected': {
        'required': ['RTE', 'TLE', 'WA'],
    },
    'wrong_answer': {
        'permitted': ['AC', 'WA'],
        'required': ['WA'],
    },
    'time_limit_exceeded': {
        'permitted': ['AC', 'TLE'],
        'required': ['TLE'],
    },
    'run_time_error': {
        'permitted': ['AC', 'RTE'],
        'required': ['RTE'],
    },
    'brute_force': {
        'permitted': ['AC', 'RTE', 'TLE'],
        'required': ['RTE', 'TLE'],
    },
}


class _SubmissionConstraint:
    """A parsed constraint from submissions.yaml, applied to a specific glob pattern."""
    __slots__ = ('glob_pattern', 'permitted', 'required', 'score', 'message',
                 'use_for_time_limit', 'language', 'entrypoint', 'authors',
                 'model_solution', 'group_constraints')

    def __init__(self, glob_pattern: str, config: dict[str, Any]):
        self.glob_pattern = glob_pattern
        self.permitted = _parse_verdict_set(config.get('permitted'))
        self.required = _parse_verdict_set(config.get('required'))
        self.score = config.get('score')  # float or [float, float]
        self.message = config.get('message')  # str
        self.use_for_time_limit = config.get('use_for_time_limit')  # bool or 'lower'/'upper'
        self.language = config.get('language')
        self.entrypoint = config.get('entrypoint')
        self.authors = config.get('authors')
        self.model_solution = config.get('model_solution', False)
        # Per-group constraints: group_name -> {permitted, required, score, message, use_for_time_limit}
        self.group_constraints: dict[str, dict[str, Any]] = {}
        for key, value in config.items():
            if key not in _SUBMISSIONS_YAML_TOP_KEYS and isinstance(value, dict):
                self.group_constraints[key] = value


def _parse_verdict_set(verdicts: list[str] | None) -> frozenset[str] | None:
    """Parse a list of verdicts into a frozenset, or None if not specified."""
    if verdicts is None:
        return None
    return frozenset(verdicts)


_SUBMISSION_DIRS_LEGACY: list[_SubmissionDir] = [
    _SubmissionDir('accepted', frozenset({'AC'}), must_exist=True),
    _SubmissionDir('partially_accepted', frozenset({'AC'}), is_partial=True),
    _SubmissionDir('wrong_answer', frozenset({'WA'})),
    _SubmissionDir('run_time_error', frozenset({'RTE'})),
    _SubmissionDir('time_limit_exceeded', frozenset({'TLE'})),
]

_SUBMISSION_DIRS_2025_09: list[_SubmissionDir] = [
    _SubmissionDir('accepted', frozenset({'AC'}), must_exist=True),
    _SubmissionDir('rejected', frozenset({'WA', 'RTE', 'TLE'})),
    _SubmissionDir('wrong_answer', frozenset({'WA'})),
    _SubmissionDir('time_limit_exceeded', frozenset({'TLE'})),
    _SubmissionDir('run_time_error', frozenset({'RTE'})),
    _SubmissionDir('brute_force', frozenset({'RTE', 'TLE'})),
]


class Submissions(ProblemPart):
    _SUB_REGEXP = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*[a-zA-Z0-9](\.c\+\+)?$')

    PART_NAME = 'submission'

    @property
    def _submission_dirs(self) -> list[_SubmissionDir]:
        if self.problem.format is FormatVersion.LEGACY:
            return _SUBMISSION_DIRS_LEGACY
        return _SUBMISSION_DIRS_2025_09

    def setup(self):
        self._submissions: dict[str, list] = {}
        self._constraints: list[_SubmissionConstraint] = []
        # Track per-testcase results for submissions.yaml validation
        # Key: (submission_path, testcase_path) -> SubmissionResult
        self._testcase_results: dict[tuple[str, str], SubmissionResult] = {}
        # Track per-group results for submissions.yaml validation
        # Key: (submission_path, group_path) -> SubmissionResult
        self._submission_group_results: dict[tuple[str, str], SubmissionResult] = {}

        srcdir = os.path.join(self.problem.probdir, 'submissions')
        for sdir in self._submission_dirs:
            self._submissions[sdir.name] = run.find_programs(
                os.path.join(srcdir, sdir.name),
                language_config=self.problem.language_config,
                work_dir=self.problem.tmpdir,
                include_dir=os.path.join(self.problem.probdir, 'include'),
            )

        # Substitute constants in submission source files (2025-09 only)
        if self.problem.format is FormatVersion.V_2025_09 and self.problem.metadata.constants:
            for sdir in self._submission_dirs:
                for sub in self._submissions.get(sdir.name, []):
                    if hasattr(sub, 'path') and os.path.isdir(sub.path):
                        undef = statement_util.substitute_constants_in_directory(sub.path, self.problem.metadata.constants)
                        for ref in undef:
                            self.warning("Undefined constant '{{%s}}' in submission %s" % (ref, sub))

        # Parse submissions.yaml for 2025-09 format
        if self.problem.format is FormatVersion.V_2025_09:
            self._parse_submissions_yaml(srcdir)

        return {}

    def _parse_submissions_yaml(self, srcdir: str) -> None:
        """Parse submissions/submissions.yaml for 2025-09 format."""
        yaml_path = os.path.join(srcdir, 'submissions.yaml')
        if not os.path.isfile(yaml_path):
            return

        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.error(f'Failed to parse submissions.yaml: {e}')
            return

        if data is None:
            return
        if not isinstance(data, dict):
            self.error('submissions.yaml must be a YAML mapping (dict)')
            return

        for glob_pattern, config_val in data.items():
            if not isinstance(glob_pattern, str):
                self.error(f'submissions.yaml key must be a string, got {type(glob_pattern).__name__}')
                continue
            if config_val is None:
                config_val = {}
            if not isinstance(config_val, dict):
                self.error(f'submissions.yaml value for "{glob_pattern}" must be a mapping, got {type(config_val).__name__}')
                continue

            # Validate top-level keys and identify group constraint keys
            for key in config_val:
                if key not in _SUBMISSIONS_YAML_TOP_KEYS and not isinstance(config_val[key], dict):
                    self.warning(f'Unknown key "{key}" in submissions.yaml for pattern "{glob_pattern}"')

            # Validate verdict sets
            self._validate_verdict_list(config_val.get('permitted'), f'{glob_pattern}.permitted')
            self._validate_verdict_list(config_val.get('required'), f'{glob_pattern}.required')

            # Validate score
            score = config_val.get('score')
            if score is not None:
                if isinstance(score, list):
                    if len(score) != 2 or not all(isinstance(s, (int, float)) for s in score):
                        self.error(f'submissions.yaml "{glob_pattern}": score must be a float or [float, float], got {score}')
                elif not isinstance(score, (int, float)):
                    self.error(f'submissions.yaml "{glob_pattern}": score must be a float or [float, float], got {type(score).__name__}')

            # Validate use_for_time_limit
            uftl = config_val.get('use_for_time_limit')
            if uftl is not None and uftl not in (True, False, 'lower', 'upper'):
                self.error(f'submissions.yaml "{glob_pattern}": use_for_time_limit must be bool or "lower"/"upper", got {uftl}')

            # Validate per-group constraints
            for key, value in config_val.items():
                if key not in _SUBMISSIONS_YAML_TOP_KEYS and isinstance(value, dict):
                    self._validate_verdict_list(value.get('permitted'), f'{glob_pattern}.{key}.permitted')
                    self._validate_verdict_list(value.get('required'), f'{glob_pattern}.{key}.required')
                    group_score = value.get('score')
                    if group_score is not None:
                        if isinstance(group_score, list):
                            if len(group_score) != 2 or not all(isinstance(s, (int, float)) for s in group_score):
                                self.error(f'submissions.yaml "{glob_pattern}.{key}": score must be a float or [float, float]')
                        elif not isinstance(group_score, (int, float)):
                            self.error(f'submissions.yaml "{glob_pattern}.{key}": score must be a float or [float, float]')

            constraint = _SubmissionConstraint(glob_pattern, config_val)
            self._constraints.append(constraint)

        self.info(f'Parsed {len(self._constraints)} constraints from submissions.yaml')

    def _validate_verdict_list(self, verdicts: list | None, context: str) -> None:
        """Validate a verdict list from submissions.yaml."""
        if verdicts is None:
            return
        if not isinstance(verdicts, list):
            self.error(f'submissions.yaml {context}: must be a list of verdicts, got {type(verdicts).__name__}')
            return
        for v in verdicts:
            if v not in _VALID_VERDICTS:
                self.error(f'submissions.yaml {context}: unknown verdict "{v}" (valid: {", ".join(sorted(_VALID_VERDICTS))})')

    def _collect_testcase_paths(self, group: TestCaseGroup) -> list[str]:
        """Collect all testcase paths relative to data/ from a test case group."""
        data_dir = os.path.join(self.problem.probdir, 'data')
        paths = []
        for tc in group.get_testcases():
            rel = os.path.relpath(tc.infile, data_dir)
            # Remove .in extension
            if rel.endswith('.in'):
                rel = rel[:-3]
            paths.append(rel)
        for subgroup in group.get_subgroups():
            paths.extend(self._collect_testcase_paths(subgroup))
        return paths

    def _collect_group_paths(self, group: TestCaseGroup) -> list[str]:
        """Collect all group paths relative to data/ from a test case group."""
        data_dir = os.path.join(self.problem.probdir, 'data')
        paths = []
        rel = os.path.relpath(group._datadir, data_dir)
        paths.append(rel)
        for subgroup in group.get_subgroups():
            paths.extend(self._collect_group_paths(subgroup))
        return paths

    def _check_submission_constraints(self, sub_path: str, result: SubmissionResult) -> None:
        """Check submissions.yaml constraints for a specific submission after it has been run.

        Args:
            sub_path: Submission path relative to submissions/, e.g. 'accepted/hello.py'
            result: The overall submission result
        """
        tc_verdicts = dict(self.problem._testcase_verdicts)  # snapshot
        group_results = dict(self.problem._group_results)  # snapshot

        for constraint in self._constraints:
            if not _submission_matches_glob(constraint.glob_pattern, sub_path):
                continue

            # Check top-level permitted/required against all test cases
            all_verdicts = [r.verdict for r in tc_verdicts.values()]

            if constraint.permitted is not None:
                for tc_path, tc_res in tc_verdicts.items():
                    if tc_res.verdict not in constraint.permitted:
                        self.error(
                            f'submissions.yaml: submission {sub_path} got verdict {tc_res.verdict} on {tc_path}, '
                            f'but only {sorted(constraint.permitted)} are permitted (pattern: {constraint.glob_pattern})'
                        )
                        break  # One error per constraint per submission is enough

            if constraint.required is not None:
                has_required = any(v in constraint.required for v in all_verdicts)
                if not has_required:
                    self.error(
                        f'submissions.yaml: submission {sub_path} has no test case with verdict in {sorted(constraint.required)} '
                        f'(pattern: {constraint.glob_pattern}), got verdicts: {sorted(set(all_verdicts))}'
                    )

            # Check score constraint
            if constraint.score is not None and result.score is not None:
                if isinstance(constraint.score, list):
                    lo, hi = constraint.score
                    if not (lo <= result.score <= hi):
                        self.error(
                            f'submissions.yaml: submission {sub_path} got score {result.score}, '
                            f'expected [{lo}, {hi}] (pattern: {constraint.glob_pattern})'
                        )
                else:
                    if result.score != constraint.score:
                        self.error(
                            f'submissions.yaml: submission {sub_path} got score {result.score}, '
                            f'expected {constraint.score} (pattern: {constraint.glob_pattern})'
                        )

            # Check message constraint
            if constraint.message is not None:
                has_message = False
                for tc_path, tc_res in tc_verdicts.items():
                    if tc_res.additional_info and constraint.message in tc_res.additional_info:
                        has_message = True
                        break
                if not has_message:
                    self.error(
                        f'submissions.yaml: submission {sub_path} has no test case with message containing '
                        f'"{constraint.message}" (pattern: {constraint.glob_pattern})'
                    )

            # Check per-group constraints
            for group_pattern, group_config in constraint.group_constraints.items():
                group_tc_verdicts = {
                    tc_path: tc_res
                    for tc_path, tc_res in tc_verdicts.items()
                    if _testdata_matches_glob(group_pattern, tc_path)
                }

                if not group_tc_verdicts:
                    self.debug(
                        f'submissions.yaml: no test cases matched group pattern "{group_pattern}" for {sub_path}'
                    )
                    continue

                group_verdicts = [r.verdict for r in group_tc_verdicts.values()]

                group_permitted = _parse_verdict_set(group_config.get('permitted'))
                if group_permitted is not None:
                    for tc_path, tc_res in group_tc_verdicts.items():
                        if tc_res.verdict not in group_permitted:
                            self.error(
                                f'submissions.yaml: submission {sub_path} got verdict {tc_res.verdict} on {tc_path}, '
                                f'but only {sorted(group_permitted)} are permitted (pattern: {constraint.glob_pattern}, group: {group_pattern})'
                            )
                            break

                group_required = _parse_verdict_set(group_config.get('required'))
                if group_required is not None:
                    has_required = any(v in group_required for v in group_verdicts)
                    if not has_required:
                        self.error(
                            f'submissions.yaml: submission {sub_path} has no test case in group "{group_pattern}" '
                            f'with verdict in {sorted(group_required)} (pattern: {constraint.glob_pattern}), '
                            f'got verdicts: {sorted(set(group_verdicts))}'
                        )

                # Check per-group score constraint
                group_score = group_config.get('score')
                if group_score is not None:
                    # Find the group result for this group pattern
                    matching_group_result = None
                    for gname, gresult in group_results.items():
                        if _testdata_matches_glob(group_pattern, gname) or group_pattern == gname:
                            matching_group_result = gresult
                            break
                    if matching_group_result and matching_group_result.score is not None:
                        if isinstance(group_score, list):
                            lo, hi = group_score
                            if not (lo <= matching_group_result.score <= hi):
                                self.error(
                                    f'submissions.yaml: submission {sub_path} got score {matching_group_result.score} '
                                    f'on group "{group_pattern}", expected [{lo}, {hi}] (pattern: {constraint.glob_pattern})'
                                )
                        elif matching_group_result.score != group_score:
                            self.error(
                                f'submissions.yaml: submission {sub_path} got score {matching_group_result.score} '
                                f'on group "{group_pattern}", expected {group_score} (pattern: {constraint.glob_pattern})'
                            )

                # Check per-group message constraint
                group_message = group_config.get('message')
                if group_message is not None:
                    has_msg = any(
                        tc_res.additional_info and group_message in tc_res.additional_info
                        for tc_res in group_tc_verdicts.values()
                    )
                    if not has_msg:
                        self.error(
                            f'submissions.yaml: submission {sub_path} has no test case in group "{group_pattern}" '
                            f'with message containing "{group_message}" (pattern: {constraint.glob_pattern})'
                        )

    def __str__(self) -> str:
        return 'submissions'

    def check_submission(
        self, sub, context: Context, sdir: _SubmissionDir, timelim: int, timelim_low: int, timelim_high: int
    ) -> SubmissionResult:
        desc = f'{sdir.name} submission {sub}'
        partial = sdir.is_partial
        if partial:
            # For partially accepted solutions, use the low timelim instead of the real one,
            # to make sure we have margin in both directions.
            pass
        else:
            timelim_low = timelim

        with Runner(self.problem, sub, context, timelim, timelim_low, timelim_high) as runner:
            # Clear group results for require_pass tracking (2025-09)
            self.problem._group_results.clear()
            self.problem._testcase_verdicts.clear()
            result, result_low, result_high = self.problem.testdata.run_submission(sub, runner, context)

        if result.verdict == 'AC' and sdir.verdict_ok('AC') and not partial and result.sample_failures:
            res = result.sample_failures[0]
            self.warning(f'{desc} got {res.verdict} on sample: {res}')

        if result_low.verdict != result_high.verdict or result_low.score != result_high.score:
            r1, r2 = (
                (result_low, result_high)
                if result_low.verdict == result_high.verdict
                else (result_low.verdict, result_high.verdict)
            )
            self.warning(
                f'{desc} sensitive to time limit: limit of {timelim_low} secs -> {r1}, limit of {timelim_high} secs -> {r2}'
            )

        if partial and self.fully_accepted(result):
            self.warning(f'{desc} got {result}')
        elif sdir.verdict_ok(result.verdict):
            self.msg(f'   {desc} OK: {result}')
            if result.verdict == 'AC' and not partial and not self.fully_accepted(result) and self.full_score_finite():
                # For some heuristic problems, this is expected. Thus, only warn.
                self.warning(f'{desc} did not attain full score (consider moving it to partially_accepted)')
        elif sdir.verdict_ok(result_high.verdict) and not (partial and self.fully_accepted(result_high)):
            self.msg(f'   {desc} OK with extra time: {result_high}')
        else:
            self.error(f'{desc} got {result}', result_high.additional_info)

        # For 2025-09: check submissions.yaml constraints
        if self.problem.format is FormatVersion.V_2025_09 and self._constraints:
            sub_path = os.path.join(sdir.name, str(sub.name))
            self._check_submission_constraints(sub_path, result)

        return result

    def full_score_finite(self) -> bool:
        min_score, max_score = self.problem.testdata.get_score_range()
        if self.problem.metadata.legacy_grading.objective == 'min':
            return min_score != float('-inf')
        else:
            return max_score != float('inf')

    def fully_accepted(self, result: SubmissionResult) -> bool:
        min_score, max_score = self.problem.testdata.get_score_range()
        best_score = min_score if self.problem.metadata.legacy_grading.objective == 'min' else max_score
        return result.verdict == 'AC' and (not self.problem.is_scoring() or result.score == best_score)

    def start_background_work(self, context: Context) -> None:
        # Send off an early background compile job for each submission and
        # validator, to avoid a bottleneck step at the start of each test run.
        self.problem.output_validators.start_background_work(context)
        for dirname in self._submissions:
            for sub in self._submissions[dirname]:
                context.submit_background_work(lambda s: s.compile(), sub)

    def check(self, context: Context) -> bool:
        if self._check_res is not None:
            return self._check_res
        self._check_res = True

        limits = self.problem.metadata.limits
        ac_to_time_limit = limits.time_multipliers.ac_to_time_limit

        fixed_limit: float | None = context.fixed_timelim if context.fixed_timelim is not None else limits.time_limit
        lower_bound_runtime: float | None = None  # The runtime of the slowest submission used to lower bound the time limit.

        if limits.time_limit is not None and context.fixed_timelim is not None:
            self.warning('There is a fixed time limit in problem.yaml, and you provided one on command line. Using command line.')

        # Track per-directory runtimes for 2025-09 time limit inference
        dir_runtimes: dict[str, list[float]] = {}

        for sdir in self._submission_dirs:
            if sdir.must_exist and not self._submissions[sdir.name]:
                self.error(f'Require at least one "{sdir.name}" submission')

            runtimes = []

            for sub in self._submissions[sdir.name]:
                sub_name = sub.name  # type: ignore
                if context.submission_filter.search(os.path.join(sdir.name, sub_name)):
                    self.info(f'Check {sdir.name} submission {sub}')

                    if sub.code_size() > 1024 * limits.code:
                        self.error(
                            f'{sdir.name} submission {sub} has size {sub.code_size() / 1024.0:.1f} kiB, exceeds code size limit of {limits.code} kiB'
                        )
                        continue

                    success, msg = sub.compile()
                    if not success:
                        self.error(f'Compile error for {sdir.name} submission {sub}', additional_info=msg)
                        continue

                    res = self.check_submission(sub, context, sdir, timelim, timelim_margin_lo, timelim_margin)
                    runtimes.append(res.runtime)

            dir_runtimes[sdir.name] = runtimes

            if sdir.name == 'accepted':
                if len(runtimes) > 0:
                    max_runtime = max(runtimes)
                    exact_timelim = max_runtime * time_multiplier
                    max_runtime_str = f'{max_runtime:.3f}'
                    timelim = max(1, int(0.5 + exact_timelim))
                    timelim_margin_lo = max(1, min(int(0.5 + exact_timelim / safety_margin), timelim - 1))
                    timelim_margin = max(timelim + 1, int(0.5 + exact_timelim * safety_margin))
                else:
                    max_runtime_str = None
                if context.fixed_timelim is not None and context.fixed_timelim != timelim:
                    self.msg(
                        f'   Solutions give timelim of {timelim} seconds, but will use provided fixed limit of {context.fixed_timelim} seconds instead'
                    )
                    timelim = context.fixed_timelim
                    timelim_margin = round(timelim * safety_margin)

                if fixed_limit is not None and lower_bound_runtime is not None:
                    if lower_bound_runtime * ac_to_time_limit > fixed_limit:
                        self.error(
                            f'Time limit fixed to {_f_n(fixed_limit)}, but slowest AC runs in {_f_n(lower_bound_runtime)} which is within a factor {_f_n(ac_to_time_limit)}.'
                        )
                    tl_from_subs, _ = self._compute_time_limit(None, lower_bound_runtime)
                    if not math.isclose(fixed_limit, tl_from_subs):
                        self.msg(
                            f'   Solutions give timelim of {_f_n(tl_from_subs)} seconds, but will use provided fixed limit of {_f_n(fixed_limit)} seconds instead'
                        )

                timelim, timelim_margin = self._compute_time_limit(fixed_limit, lower_bound_runtime)
                self.msg(
                    f'   Slowest AC runtime: {_f_n(lower_bound_runtime)}, setting timelim to {_f_n(timelim)} secs, safety margin to {_f_n(timelim_margin)} secs'
                )
                self.problem._set_timelim(timelim)

        # 2025-09: validate time limit bounds from submissions.yaml constraints
        if self.problem.format is FormatVersion.V_2025_09:
            self._validate_timelim_bounds_2025_09(dir_runtimes, timelim, time_multiplier, safety_margin)

        return self._check_res

    def _validate_timelim_bounds_2025_09(
        self, dir_runtimes: dict[str, list[float]], timelim: int,
        time_multiplier: float, safety_margin: float
    ) -> None:
        """Validate that the computed time limit satisfies 2025-09 bounds.

        Lower bound: from submissions not permitted TLE (accepted/ by default).
        Upper bound: from submissions requiring only TLE (time_limit_exceeded/ by default).
        Submissions.yaml use_for_time_limit can override.
        """
        lower_bound = 0.0
        upper_bound = float('inf')
        has_lower = False

        # Default directory-based bounds
        accepted_runtimes = dir_runtimes.get('accepted', [])
        if accepted_runtimes:
            lower_bound = max(lower_bound, max(accepted_runtimes) * time_multiplier)
            has_lower = True

        tle_runtimes = dir_runtimes.get('time_limit_exceeded', [])
        if tle_runtimes:
            # Note: TLE submission runtimes may be capped at timelim_margin
            # since they were run with the safety margin. The upper bound
            # is T / safety_margin.
            max_tle = max(tle_runtimes)
            upper_bound = min(upper_bound, max_tle / safety_margin)

        # Process submissions.yaml constraints for use_for_time_limit
        for constraint in self._constraints:
            uftl = constraint.use_for_time_limit
            if uftl is False:
                continue

            if uftl == 'lower':
                # Explicitly opted in as lower bound
                # Find runtimes for matching submissions
                for sdir_name, subs in self._submissions.items():
                    for sub in subs:
                        sub_path = os.path.join(sdir_name, str(sub.name))
                        if _submission_matches_glob(sub_path, constraint.glob_pattern):
                            if hasattr(sub, 'runtime') and sub.runtime > 0:
                                lower_bound = max(lower_bound, sub.runtime * time_multiplier)
                                has_lower = True
            elif uftl == 'upper':
                # Explicitly opted in as upper bound
                for sdir_name, subs in self._submissions.items():
                    for sub in subs:
                        sub_path = os.path.join(sdir_name, str(sub.name))
                        if _submission_matches_glob(sub_path, constraint.glob_pattern):
                            if hasattr(sub, 'runtime') and sub.runtime > 0:
                                upper_bound = min(upper_bound, sub.runtime / safety_margin)

        if not has_lower:
            self.warning('No submission provides a lower bound for the time limit (2025-09 requires at least one)')

        if lower_bound > 0 and upper_bound < float('inf') and lower_bound > upper_bound:
            self.warning(
                'Time limit bounds are inconsistent: lower bound %.3f > upper bound %.3f '
                '(from accepted/TLE submissions and submissions.yaml constraints)'
                % (lower_bound, upper_bound)
            )


PROBLEM_PARTS = ['config', 'data', 'graders', 'statement', 'submissions', 'validation_test_data', 'validators', 'visualizers']


class Problem(ProblemAspect):
    """Represents a checkable problem"""

    def __init__(self, probdir: str, args: argparse.Namespace):
        self.probdir = os.path.realpath(probdir)
        self.shortname: str = os.path.basename(self.probdir)
        super().__init__(self.shortname, self)
        self.language_config = languages.load_language_config(Path(self.probdir).parent)
        self.testcase_by_infile: dict[str, TestCase] = {}
        self.loaded = False
        self._metadata: metadata.Metadata | None = None
        self._args = args
        self._timelim: float | None = None
        # Track group results for require_pass checking (2025-09)
        self._group_results: dict[str, SubmissionResult] = {}
        # Track per-testcase results for submissions.yaml validation (2025-09)
        # Key: testcase path relative to data/ (without .in) -> SubmissionResult
        self._testcase_verdicts: dict[str, SubmissionResult] = {}

    # Unfortunately must be before metadata, otherwise mypy gets confused about the type metadata.Metadata (feels like a bug)
    def _set_metadata(self, metadata: metadata.Metadata) -> None:  # Should only be called by ProblemConfig
        assert self._metadata is None, 'Attempted to set metadata twice'
        self._metadata = metadata

    @property
    def metadata(self) -> metadata.Metadata:
        assert self._metadata is not None, 'Attempted to access config before it was set. load() or check() first.'
        return self._metadata

    @property
    def timelim(self) -> float:
        assert self._timelim is not None, 'Attempted to access timelim before it was set. check() first.'
        return self._timelim

    def _set_timelim(self, timelim: float) -> None:  # Should only be called by Submissions
        assert self._timelim is None, 'Attempted to set timelim twice'
        self._timelim = timelim

    def is_pass_fail(self) -> bool:
        return self.metadata.is_pass_fail()

    def is_scoring(self) -> bool:
        return self.metadata.is_scoring()

    def is_interactive(self) -> bool:
        return self.metadata.is_interactive()

    def is_multi_pass(self) -> bool:
        return self.metadata.is_multi_pass()

    def is_submit_answer(self) -> bool:
        return self.metadata.is_submit_answer()

    def load(self) -> None:
        """Parses the problem package statically, loading up information with very little verification.

        Call this if you want to get a usable Problem object without expensive
        steps (such as compiling validators, and testing submissions).

        N.B., This api is EXPERIMENTAL. We eventually want to create a stable
        API from problemtools, this is a first move in that direction.

        Raises:
            VerifyError: if problem package is too broken to parse safely
        """

        if self.loaded:
            return

        if not os.path.isdir(self.probdir):
            self.fatal(f"Problem directory '{self.probdir}' not found")

        try:
            self.format = get_format_version(Path(self.probdir))
        except Exception as e:
            self.fatal(f'Failed loading problem version: {e}')
        self.config = ProblemConfig(self)  # Populates self.metadata as a side effect. Needs to run first.
        self.statement = ProblemStatement(self)
        self.attachments = Attachments(self)
        self.input_validators = InputValidators(self)
        self.output_validators = OutputValidators(self)
        self.graders = Graders(self)
        self.static_validator = StaticValidator(self)
        self.output_visualizer = OutputVisualizer(self)
        self.input_visualizer = InputVisualizer(self)
        self.validation_test_data = ValidationTestData(self)
        self.testdata = TestCaseGroup(self, os.path.join(self.probdir, 'data'))
        self.submissions = Submissions(self)
        self.loaded = True

    def __enter__(self) -> Problem:
        self.tmpdir = tempfile.mkdtemp(prefix=f'verify-{self.shortname}-')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        shutil.rmtree(self.tmpdir)

    def __str__(self) -> str:
        return str(self.shortname)

    def check(self) -> tuple[int, int]:
        """Loads and checks the problem package

        Loads the problem package and runs checks. After this has completed,
        the Problem object is fully populated. You do not need to manually
        run load() first.

        Returns:
            Tuple with the number of errors, warnings found.

        Raises:
            VerifyError: if problem package is too broken to parse safely
        """
        try:
            self.load()
        except VerifyError:
            return self.errors, self.warnings

        executor = ThreadPoolExecutor(self._args.threads) if self._args.threads > 1 else None
        context = Context(self._args, executor)

        try:
            part_mapping: dict[str, list] = {
                'config': [self.config],
                'statement': [self.statement, self.attachments],
                'validators': [self.input_validators, self.output_validators],
                'graders': [self.graders],
                'data': [self.testdata],
                'validation_test_data': [self.validation_test_data],
                'visualizers': [self.static_validator, self.output_visualizer, self.input_visualizer],
                'submissions': [self.submissions],
            }
            assert sorted(part_mapping.keys()) == sorted(PROBLEM_PARTS), 'part_mapping and PROBLEM_PARTS must be kept in sync'

            if not re.match('^[a-z0-9]+$', self.shortname):
                self.error(f"Invalid shortname '{self.shortname}' (must be [a-z0-9]+)")
            if self.format is FormatVersion.V_2025_09:
                self.info('Running in 2025-09 format mode')

            self._check_symlinks()
            self._check_file_and_directory_names()
            self._check_submission_directory_names()

            run.limit.check_limit_capabilities(self)

            parts = [
                part for part in part_mapping if part in self._args.parts
            ]  # Parts from _args in the order they appear in part_mapping
            if executor:
                for part in parts:
                    for item in part_mapping[part]:
                        item.start_background_work(context)

            for part in parts:
                self.msg(f'Checking {part}')
                for item in part_mapping[part]:
                    item.check(context)
        except VerifyError:
            pass
        finally:
            # Wait for background work to finish before performing an rmtree on
            # the directory tree it uses.
            context.wait_for_background_work()
        return self.errors, self.warnings

    def _check_submission_directory_names(self):
        """Heuristically check if submissions contain any directories that will be ignored because of typos or format mismatches"""
        submission_directories = [p.name for p in (Path(self.probdir) / 'submissions').glob('*') if p.is_dir()]
        if len(submission_directories) == 0:
            return

        def most_similar(present_dir: str, format_version: FormatVersion):
            similarities = [
                (spec_dir, difflib.SequenceMatcher(None, present_dir, spec_dir).ratio())
                for spec_dir in format_version.submission_directories
            ]
            return max(similarities, key=lambda x: x[1])

        for present_dir in submission_directories:
            most_similar_dir, max_similarity = most_similar(present_dir, self.format)

            if max_similarity == 1:
                # Exact match, no typo
                continue

            if 0.75 <= max_similarity:
                self.warning(f'Potential typo: directory submissions/{present_dir} is similar to {most_similar_dir}')
            else:
                for other_version in [v for v in FormatVersion if v != self.format]:
                    _, max_similarity = most_similar(present_dir, other_version)
                    if max_similarity == 1:
                        self.warning(
                            f'Directory submissions/{present_dir} is not part of format version {self.format}, but part of {other_version}'
                        )
                        break

    def _check_symlinks(self):
        """Check that all symlinks point to something existing within the problem package"""
        probdir = os.path.realpath(self.probdir)
        for root, dirs, files in os.walk(probdir):
            for file in dirs + files:
                filename = os.path.join(root, file)
                if os.path.islink(filename):
                    target = os.path.realpath(filename)
                    # relfile is the filename of the symlink, relative to the problem root (only used for nicer error messages)
                    relfile = os.path.relpath(filename, self.probdir)
                    # reltarget is what the symlink points to (absolute, or relative to where the symlink is)
                    reltarget = os.readlink(filename)
                    if not os.path.exists(target):
                        self.error(f'Symlink {relfile} links to {reltarget} which does not exist')
                    if os.path.commonpath([probdir, target]) != probdir:
                        self.error(f'Symlink {relfile} links to {reltarget} which is outside of problem package')
                    if os.path.isabs(reltarget):
                        self.error(
                            f'Symlink {relfile} links to {reltarget} which is an absolute path. Symlinks must be relative.'
                        )

    def _check_file_and_directory_names(self):
        regex = re.compile(r'^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,254}$')

        def _special_case_allowed_files(file: str, reldir: str) -> bool:
            return file == '.gitignore' or (file == '.timelimit' and reldir == os.path.basename(self.probdir))

        def _special_case_allowed_dirs(directory: str, reldir: str) -> bool:
            return directory == '.git' and reldir == os.path.basename(self.probdir)

        for root, dirs, files in os.walk(self.probdir):
            # Path of the directory we're in, starting with problem shortname. Only used for nicer error messages.
            reldir = os.path.relpath(root, os.path.dirname(self.probdir))
            for file in files:
                if not regex.match(file) and not _special_case_allowed_files(file, reldir):
                    self.error(f"Invalid file name '{file}' in {reldir}, should match {regex.pattern}")
            for directory in dirs:
                if not directory_regex.match(directory):
                    self.error_in_2025_09(
                        f"Invalid directory name '{directory}' in {reldir} (should match {directory_regex.pattern} ignoring case)"
                    )

    def bail_on_error(self) -> bool:
        return self._args.bail_on_error

    def consider_warnings_errors(self) -> bool:
        return self._args.werror

    def max_additional_info(self) -> int:
        return self._args.max_additional_info


def re_argument(s: str) -> Pattern[str]:
    try:
        r = re.compile(s)
        return r
    except re.error:
        raise argparse.ArgumentTypeError(f'{s} is not a valid regex')


def part_argument(s: str) -> str:
    if s not in PROBLEM_PARTS:
        raise argparse.ArgumentTypeError(f'Invalid problem part specified: {s}')
    return s


def argparser_basic_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-b', '--bail_on_error', action='store_true', help='bail verification on first error')
    parser.add_argument('-l', '--log_level', default='warning', help='set log level (debug, info, warning, error, critical)')
    parser.add_argument('-e', '--werror', action='store_true', help='consider warnings as errors')
    parser.add_argument(
        '--max_additional_info',
        type=int,
        default=15,
        help='maximum number of lines of additional info (e.g. compiler output or validator feedback) to display about an error (set to 0 to disable additional info)',
    )


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate a problem package in the Kattis problem format.')
    parser.add_argument(
        '-s',
        '--submission_filter',
        metavar='SUBMISSIONS',
        type=re_argument,
        default=re.compile('.*'),
        help='run only submissions whose name contains this regex.  The name includes category (accepted, wrong_answer, etc), e.g. "accepted/hello.java" (for a single file submission) or "wrong_answer/hello" (for a directory submission)',
    )
    parser.add_argument(
        '-d',
        '--data_filter',
        metavar='DATA',
        type=re_argument,
        default=re.compile('.*'),
        help='use only data files whose name contains this regex.  The name includes path relative to the data directory but not the extension, e.g. "sample/hello" for a sample data file',
    )
    parser.add_argument(
        '-t',
        '--fixed_timelim',
        type=float,
        help='use this fixed time limit (useful in combination with -d and/or -s when all AC submissions might not be run on all data)',
    )
    parser.add_argument(
        '-p',
        '--parts',
        metavar='PROBLEM_PART',
        type=part_argument,
        nargs='+',
        default=PROBLEM_PARTS,
        help=f'only test the indicated parts of the problem.  Each PROBLEM_PART can be one of {PROBLEM_PARTS}.',
    )
    parser.add_argument(
        '-j',
        '--threads',
        type=int,
        default=1,
        help='run validation using multiple threads. This will make timings less reliable, but can be convenient during development',
    )

    add_version_arg(parser)
    argparser_basic_arguments(parser)

    parser.add_argument('problemdir', nargs='+')
    return parser


def initialize_logging(args: argparse.Namespace) -> None:
    fmt = '%(log_color)s%(levelname)s %(message)s'
    colorlog.basicConfig(stream=sys.stdout, format=fmt, level=getattr(logging, args.log_level.upper()))


def main() -> None:
    args = argparser().parse_args()

    initialize_logging(args)

    total_errors = 0
    try:
        for problemdir in args.problemdir:
            print(f'Loading problem {os.path.basename(os.path.realpath(problemdir))}')
            with Problem(problemdir, args) as prob:
                errors, warnings = prob.check()

                def p(x: int) -> str:
                    return '' if x == 1 else 's'

                print(f'{prob.shortname} tested: {errors} error{p(errors)}, {warnings} warning{p(warnings)}')
                total_errors += errors

    except KeyboardInterrupt:
        print('\naborting...')
    finally:
        if total_errors > 0:
            sys.exit(1)


if __name__ == '__main__':
    main()
