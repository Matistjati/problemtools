from __future__ import annotations

import copy
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

from .result import SubmissionResult

if TYPE_CHECKING:
    from ..verifyproblem import TestCase


@dataclass(frozen=True)
class CacheKey:
    input_hash: bytes
    ans_hash: bytes
    validator_flags: tuple[str, ...]


@dataclass
class _CacheEntry:
    result: SubmissionResult
    run_timelim: float


def _reclassify(result: SubmissionResult, timelim: float) -> SubmissionResult:
    """Reclassify a cached result against a (possibly lower) time limit."""
    if result.runtime > timelim:
        if result.validator_first and result.verdict == 'WA':
            # Interactive: validator exited first with WA. This can cause the submission to run
            # longer than it should. Cap runtimes at timelim so this doesn't inflate the time limit.
            wa = copy.copy(result)
            wa.runtime = timelim
            return wa
        tle = SubmissionResult('TLE')
        tle.runtime = result.runtime
        return tle
    return result


def _with_test_node(result: SubmissionResult, testcase: TestCase) -> SubmissionResult:
    """Return result with test_node and runtime_testcase set to testcase, copying only if needed."""
    if result.test_node is testcase and result.runtime_testcase is testcase:
        return result
    result = copy.copy(result)
    result.test_node = testcase
    result.runtime_testcase = testcase
    return result


class ResultStore:
    """Thread-safe store mapping testcase reuse keys to execution results.

    Background workers populate the store via claim()/complete(); the consumer
    reads results via get().  A key progresses through three states: absent
    (not yet claimed), in-flight (claimed, Future not yet resolved), and
    completed (_CacheEntry).

    Because results are always run at the high time limit, a completed entry
    can serve any query whose time limit is <= the run limit: a result whose
    runtime exceeds the query limit is reclassified as TLE. A query with a
    higher limit than the run limit cannot be served from cache and returns None.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._store: dict[CacheKey, Future[SubmissionResult] | _CacheEntry] = {}

    def claim(self, testcase: TestCase) -> bool:
        """Atomically claim testcase for execution.

        Returns True if the key was unclaimed; the caller must eventually call
        complete().  Returns False if the key is already in-flight or completed.
        """
        key = testcase.reuse_key
        with self._lock:
            if key in self._store:
                return False
            self._store[key] = Future()
            return True

    def complete(self, testcase: TestCase, result: SubmissionResult, run_timelim: float) -> None:
        """Store the completed result and wake any consumer waiting on the future."""
        key = testcase.reuse_key
        with self._lock:
            future = self._store[key]
            self._store[key] = _CacheEntry(result=result, run_timelim=run_timelim)
        assert isinstance(future, Future)
        future.set_result(result)  # outside lock — callbacks may acquire other locks

    def get(self, testcase: TestCase, timelim: float) -> SubmissionResult | Future[SubmissionResult] | None:
        """Look up a result for testcase at timelim.

        Returns:
            SubmissionResult  — completed result, already reclassified for timelim; use directly.
            Future            — in-flight; resolves to a reclassified SubmissionResult.
            None              — not present, or was run at a lower limit than timelim and
                                cannot be reused; caller must run the testcase synchronously.
        """
        key = testcase.reuse_key
        with self._lock:
            val = self._store.get(key)
        if val is None:
            return None
        if isinstance(val, Future):
            chained: Future[SubmissionResult] = Future()
            val.add_done_callback(lambda f: chained.set_result(_with_test_node(_reclassify(f.result(), timelim), testcase)))
            return chained
        if timelim > val.run_timelim:
            # Entry was produced at a lower limit; cannot safely reclassify upward.
            return None
        return _with_test_node(_reclassify(val.result, timelim), testcase)

import hashlib
import json
import os

class ResultCache:
    """Cross-run cache for single-testcase execution results."""

    _CACHE_DIR = '/tmp/problemtools/cache'

    def __init__(self) -> None:
        os.makedirs(self._CACHE_DIR, exist_ok=True)
        self._file_hashes: dict[str, str] = {}

    def _hash_file_content(self, path: str) -> str:
        digest = self._file_hashes.get(path)
        if digest is None:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            digest = h.hexdigest()
            self._file_hashes[path] = digest
        return digest

    def _make_key(self, sub, testcase, problem) -> str:
        h = hashlib.sha256()

        h.update(b'lang:')
        h.update(sub.language.lang_id.encode())
        for src_file in sorted(sub.src):
            h.update(os.path.basename(src_file).encode())
            h.update(self._hash_file_content(src_file).encode())

        h.update(b'\x00in:')
        h.update(self._hash_file_content(testcase.infile).encode())

        h.update(b'\x00ans:')
        h.update(self._hash_file_content(testcase.ansfile).encode())

        h.update(b'\x00val:')
        if problem.output_validators.uses_default_validator():
            h.update(b'default')
        else:
            for val in problem.output_validators._validators:
                if hasattr(val, 'src'):
                    for src_file in sorted(val.src):
                        h.update(os.path.basename(src_file).encode())
                        h.update(self._hash_file_content(src_file).encode())

        h.update(b'\x00vflags:')
        flags = (
            problem.metadata.legacy_validator_flags
            + '\x00'
            + testcase.testcasegroup.config.get('output_validator_flags', '')
        )
        h.update(flags.encode())

        h.update(b'\x00memlim:')
        h.update(str(problem.metadata.limits.memory).encode())

        return h.hexdigest()

    def lookup(
        self, sub, testcase, problem, timelim_high: float,
    ) -> SubmissionResult | None:
        key = self._make_key(sub, testcase, problem)
        path = os.path.join(self._CACHE_DIR, f'{key}.json')
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        cached_verdict: str = data['verdict']
        cached_runtime: float = data['runtime']
        cached_timelim: float = data['timelim_high']

        if cached_verdict == 'TLE':
            if cached_timelim < timelim_high:
                return None
        else:
            if cached_runtime > timelim_high:
                return None

        res = SubmissionResult(
            cached_verdict,
            score=data.get('score'),
            reason=data.get('reason'),
            additional_info=data.get('additional_info'),
        )
        res.runtime = cached_runtime
        res.precision = data.get('precision')
        return res

    def store(
        self, sub, testcase, problem,
        timelim_high: float, res_high: SubmissionResult,
    ) -> None:
        key = self._make_key(sub, testcase, problem)
        data = {
            'verdict': res_high.verdict,
            'runtime': res_high.runtime,
            'score': res_high.score,
            'reason': res_high.reason,
            'additional_info': res_high.additional_info,
            'precision': res_high.precision,
            'timelim_high': timelim_high,
        }
        path = os.path.join(self._CACHE_DIR, f'{key}.json')
        tmp = f'{path}.tmp.{os.getpid()}'
        try:
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
