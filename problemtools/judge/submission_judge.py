from __future__ import annotations

import copy
import sys
from concurrent.futures import Future
from pathlib import Path
from threading import Lock

from ..context import Context
from ..diagnostics import Diagnostics
from ..metadata import Metadata
from ..model import DEFAULT_GRADER, Graders, TestCase, TestDataGroup
from ..run import Program
from .cache import ResultCache, ResultStore
from .execute import execute_testcase
from .grade import grade_group
from .result import SubmissionResult


class _Cancelled:
    """Thread-safe set of cancelled testcase identities (by Path to infile)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._ids: set[Path] = set()

    def __contains__(self, testcase: TestCase) -> bool:
        with self._lock:
            return testcase.infile in self._ids

    def add(self, testcase: TestCase) -> None:
        with self._lock:
            self._ids.add(testcase.infile)


class SubmissionJudge:
    """Run a submission against a test case group tree and collect results.

    The typical flow uses two phases:

      1. precompute(timelim) — submits all filtered testcases as background jobs that
         execute the submission and populate a result cache.  Returns immediately.
      2. judge(timelim) — walks the test tree in DFS order, consuming cached results
         (blocking on any still in-flight) or running synchronously if a worker missed
         a testcase.  Returns a flat list of SubmissionResults, one per testcase plus
         one aggregate per group, with the root group's result last.

    This lets the submission run on all testcases in parallel while the consumer
    processes results in order for grading and early-exit logic.

    When an on_reject:break group encounters a non-AC result, pending (not-yet-started)
    background jobs for the remaining testcases in that subtree are skipped.  In-flight
    jobs complete normally; their results are simply not consumed by judge().
    """

    def __init__(
        self,
        sub: Program,
        output_validator: Program,
        metadata: Metadata,
        root: TestDataGroup,
        base_dir: Path,
        context: Context,
        graders: Graders,
        diag: Diagnostics,
    ) -> None:
        self._sub = sub
        self._output_validator = output_validator
        self._metadata = metadata
        self._base_dir = base_dir
        self._context = context
        self._diag = diag
        self._graders = graders
        self._store = ResultStore()
        # Cross-run cache. Interactive problems aren't cacheable: their result depends on the
        # interactor's behaviour, which we don't hash.
        self._result_cache = ResultCache() if context.use_cache and not metadata.is_interactive() else None
        self._root = root
        self._cancelled = _Cancelled()
        self._precompute_started = False

    def precompute(self, timelim: float) -> None:
        """Submit all filtered testcases as background jobs.

        Returns immediately; workers run concurrently and deposit results into the
        cache as they finish.  Call judge() afterwards to consume results in DFS order.
        May be called at most once.
        """
        assert not self._precompute_started, 'precompute() called more than once'
        self._precompute_started = True
        filtered_testcases = (item for item in self._root.get_all_testcases() if item.matches_filter(self._context.data_filter))
        for testcase in filtered_testcases:
            self._context.submit_background_work(self._populate_cache_for_testcase, testcase, timelim)

    def judge(self, timelim: float) -> list[SubmissionResult]:
        """Walk the test tree in DFS order and return results as a flat list.

        Each SubmissionResult has test_node set to the TestCase or TestDataGroup it
        covers.  Group results immediately follow all their descendants; the root
        group's result is the last element.  Returns an empty list if all testcases
        were filtered out.

        Blocks on any cache entry still being computed by a precompute() worker.
        Testcases not yet claimed by a worker are run synchronously.  Safe to call
        multiple times with different timelim values; subsequent calls almost always
        hit the cache without new work.  When querying multiple time limits, call
        with the largest first so that cached results can be reused for smaller limits.
        """
        return self._judge_group(self._root, timelim)

    def _run(self, testcase: TestCase, timelim: float) -> SubmissionResult:
        return execute_testcase(
            testcase,
            self._sub,
            self._output_validator,
            self._metadata,
            timelim,
            self._base_dir,
            self._diag,
        )

    def _run_cached(self, testcase: TestCase, timelim: float) -> SubmissionResult:
        """Run testcase at timelim, going through the cross-run result cache.

        Returns a cached result when the cache holds one that is reusable at timelim,
        otherwise executes the submission and records the result for later runs.  JE
        results are never cached: they say nothing about the submission.

        Every path that executes a testcase goes through here, so that the cache works
        whether or not the run is multithreaded.
        """
        if self._result_cache is not None:
            cached = self._result_cache.lookup(self._sub, testcase, self._output_validator, self._metadata, timelim)
            if cached is not None:
                # lookup() rebuilds the result from JSON and so can't know which testcase it
                # belongs to; execute_testcase() fills these in on the miss path.
                cached.test_node = testcase
                cached.runtime_testcase = testcase
                return cached

        result = self._run(testcase, timelim)

        if self._result_cache is not None and result.verdict != 'JE':
            self._result_cache.store(self._sub, testcase, self._output_validator, self._metadata, timelim, result)
        return result

    def _populate_cache_for_testcase(self, testcase: TestCase, timelim: float) -> None:
        if testcase in self._cancelled:
            return
        if not self._store.claim(testcase):
            return  # duplicate testcase (same reuse_key) or already in store

        try:
            result = self._run_cached(testcase, timelim)
        except Exception as e:
            result = SubmissionResult('JE', reason=f'Internal error: {e}')

        self._store.complete(testcase, result, timelim)

    def _judge_testcase(self, testcase: TestCase, timelim: float) -> SubmissionResult:
        val = self._store.get(testcase, timelim)
        if isinstance(val, Future):
            return val.result()  # block until worker finishes
        if val is not None:
            return val
        # Synchronous fallback: worker hasn't claimed this testcase yet, or second
        # judge() call with a timelim the store can't serve.  Claim so any pending
        # worker for it bails out rather than duplicating work.
        claimed = self._store.claim(testcase)
        result = self._run_cached(testcase, timelim)
        if claimed:
            self._store.complete(testcase, result, timelim)
        return result

    def _cancel_subtree(self, group: TestDataGroup) -> None:
        for testcase in group.get_all_testcases():
            self._cancelled.add(testcase)

    def _grader_for(self, group: TestDataGroup) -> Program | None:
        if group.config.get('grading') == 'custom':
            return self._graders.grader
        return DEFAULT_GRADER

    def _judge_group(self, group: TestDataGroup, timelim: float) -> list[SubmissionResult]:
        all_results: list[SubmissionResult] = []  # Results of all children, groups and test cases, in DFS order. Our return value
        child_results: list[SubmissionResult] = []  # Results of our direct children, what we'll pass to the grader

        filtered_items = (item for item in group.items if item.matches_filter(self._context.data_filter))
        for item in filtered_items:
            if isinstance(item, TestDataGroup):
                sub = self._judge_group(item, timelim)
                if not sub:  # If everything in a group is filtered, it returns an empty list.
                    continue
                all_results.extend(sub)
                result = sub[-1]  # last element is the subgroup's own result
            else:
                # When a live results table is up it owns the terminal, so route progress through
                # its status callback instead of writing (and erasing) directly on stdout.
                status_callback = self._context.status_callback
                msg = ''
                if status_callback is not None:
                    status_callback(f'Running {self._sub} on {item}...')
                elif sys.stdout.isatty():
                    msg = f'Running {self._sub} on {item}...'
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                result = self._judge_testcase(item, timelim)
                if msg:
                    sys.stdout.write('\b \b' * len(msg))

                # Apply default score here - after we've entered it into the cache, as it may also be present in other groups with different defaults
                if result.score is None:
                    result = copy.copy(result)
                    if result.verdict == 'AC':
                        result.score = group.config['accept_score']
                    else:
                        result.score = group.config['reject_score']
                all_results.append(result)

            child_results.append(result)
            if result.verdict != 'AC' and group.config.get('on_reject') == 'break':
                self._cancel_subtree(group)  # Stop starting more precomputations for submissions in this group or below
                break

        if not all_results:  # All our children were filtered
            return []

        group_verdict = self._aggregate_group_result(child_results, group)
        all_results.append(group_verdict)
        return all_results

    def _aggregate_group_result(self, child_results: list[SubmissionResult], group: TestDataGroup) -> SubmissionResult:
        judge_error = next((r for r in child_results if r.verdict == 'JE'), None)
        if judge_error:
            result = copy.copy(judge_error)
        else:
            grader = self._grader_for(group)
            if grader is None:
                result = SubmissionResult('JE', reason='grader not found')
            else:
                grader_flags = group.config.get('grader_flags', '').split()
                verdict, score = grade_group(child_results, grader, grader_flags, self._base_dir, self._diag)
                # The grader spec is a bit funky, forcing graders to return a score even for pass-fail
                result = SubmissionResult(verdict, score=score if self._metadata.is_scoring() else None)
                slowest = max(child_results, key=lambda r: r.runtime)
                result.runtime = slowest.runtime
                result.runtime_testcase = slowest.runtime_testcase
                # Float-precision metrics aggregate as a max over the group's children.
                for field in ('max_abs_err', 'max_rel_err', 'max_best_err'):
                    values = [v for r in child_results if (v := getattr(r, field)) is not None]
                    if values:
                        setattr(result, field, max(values))
                # The grader doesn't tell us why it gave a certain result. We still want to propagate reason
                # and additional_info. As a heuristic, look for the last entry with the same verdict as the
                # group got, and copy from there.
                matching = next((r for r in reversed(child_results) if r.verdict == verdict), None)
                if matching:
                    result.reason = matching.reason
                    result.additional_info = matching.additional_info
        result.test_node = group
        return result
