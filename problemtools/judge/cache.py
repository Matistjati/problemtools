from __future__ import annotations

import copy
import hashlib
import json
import os
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import Lock
from typing import Any

from ..metadata import Metadata
from ..model import TestCase
from ..run import Program
from .result import SubmissionResult


@dataclass(frozen=True)
class CacheKey:
    input_hash: bytes
    ans_hash: bytes
    validator_flags: tuple[str, ...]


@cache
def _compute_reuse_key(infile: Path, ansfile: Path, output_validator_flags: tuple[str, ...]) -> CacheKey:
    return CacheKey(
        input_hash=hashlib.sha256(infile.read_bytes()).digest(),
        ans_hash=hashlib.sha256(ansfile.read_bytes()).digest(),
        validator_flags=output_validator_flags,
    )


def compute_reuse_key(testcase: TestCase) -> CacheKey:
    """A key identifying testcase for result-reuse purposes: same input/answer file contents
    and same output validator flags means a result can be reused. Memoized, since this hashes
    file contents on first computation for a given testcase."""
    return _compute_reuse_key(testcase.infile, testcase.ansfile, tuple(testcase.output_validator_flags))


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
        key = compute_reuse_key(testcase)
        with self._lock:
            if key in self._store:
                return False
            self._store[key] = Future()
            return True

    def complete(self, testcase: TestCase, result: SubmissionResult, run_timelim: float) -> None:
        """Store the completed result and wake any consumer waiting on the future."""
        key = compute_reuse_key(testcase)
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
        key = compute_reuse_key(testcase)
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


class ResultCache:
    """Cross-run cache for single-testcase execution results, stored as JSON under a
    temp directory. Keyed on everything that can change a testcase's result: the
    submission's language and sources, the input and answer files, the output validator's
    sources and flags, and the memory limit. The time limit is deliberately *not* part of
    the key -- it's stored alongside the result and used to decide whether an entry can
    serve a given limit (see lookup())."""

    _CACHE_DIR = '/tmp/problemtools/cache'

    def __init__(self) -> None:
        os.makedirs(self._CACHE_DIR, exist_ok=True)
        self._file_hashes: dict[Path, str] = {}

    def _hash_file_content(self, path: Path) -> str:
        digest = self._file_hashes.get(path)
        if digest is None:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            digest = h.hexdigest()
            self._file_hashes[path] = digest
        return digest

    def _hash_program(self, h: hashlib._Hash, program: Program) -> None:
        """Mix a program's identity into h: its language plus the contents of its sources.

        Programs that aren't source code (e.g. the bundled default validator) have neither
        attribute; fall back to the name, which is stable for those."""
        language = getattr(program, 'language', None)
        h.update((language.lang_id if language is not None else program.name).encode())
        for src_file in sorted(getattr(program, 'src', [])):
            src_path = Path(src_file)
            h.update(src_path.name.encode())
            h.update(self._hash_file_content(src_path).encode())

    def _make_key(self, sub: Program, testcase: TestCase, output_validator: Program, metadata: Metadata) -> str:
        h = hashlib.sha256()

        h.update(b'lang:')
        self._hash_program(h, sub)

        h.update(b'\x00in:')
        h.update(self._hash_file_content(testcase.infile).encode())

        h.update(b'\x00ans:')
        h.update(self._hash_file_content(testcase.ansfile).encode())

        h.update(b'\x00val:')
        self._hash_program(h, output_validator)

        h.update(b'\x00vflags:')
        # TestCase.output_validator_flags already has the problem-level legacy flags merged in.
        h.update('\x00'.join(testcase.output_validator_flags).encode())

        h.update(b'\x00memlim:')
        h.update(str(metadata.limits.memory).encode())

        return h.hexdigest()

    @staticmethod
    def _read_entry(path: str) -> dict[str, Any] | None:
        """The JSON entry stored at path, or None if it's missing or unreadable."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def lookup(
        self,
        sub: Program,
        testcase: TestCase,
        output_validator: Program,
        metadata: Metadata,
        timelim_high: float,
    ) -> SubmissionResult | None:
        key = self._make_key(sub, testcase, output_validator, metadata)
        data = self._read_entry(os.path.join(self._CACHE_DIR, f'{key}.json'))
        if data is None:
            return None

        cached_verdict: str = data['verdict']
        cached_runtime: float = data['runtime']
        cached_timelim: float = data['timelim_high']

        if cached_verdict == 'TLE':
            # A TLE only carries over to a limit no higher than the one it was measured at.
            if cached_timelim < timelim_high:
                return None
        else:
            # Anything else carries over as long as it fits inside the limit we're asked about.
            if cached_runtime > timelim_high:
                return None

        res = SubmissionResult(
            cached_verdict,
            score=data.get('score'),
            reason=data.get('reason'),
            additional_info=data.get('additional_info'),
        )
        res.runtime = cached_runtime
        res.max_abs_err = data.get('max_abs_err')
        res.max_rel_err = data.get('max_rel_err')
        res.max_best_err = data.get('max_best_err')
        return res

    def _stored_timelim(self, path: str) -> float:
        """The time limit the entry at path was measured at, or -inf if there is no usable entry."""
        data = self._read_entry(path)
        if data is None:
            return float('-inf')
        timelim = data.get('timelim_high')
        return timelim if isinstance(timelim, (int, float)) else float('-inf')

    def store(
        self,
        sub: Program,
        testcase: TestCase,
        output_validator: Program,
        metadata: Metadata,
        timelim_high: float,
        res_high: SubmissionResult,
    ) -> None:
        key = self._make_key(sub, testcase, output_validator, metadata)
        path = os.path.join(self._CACHE_DIR, f'{key}.json')
        if self._stored_timelim(path) > timelim_high:
            # Keep the entry measured at the higher limit: it is strictly more informative.
            # A run that finished under a high limit yields an exact runtime, reusable at every
            # limit at or above it; a TLE carries over to every limit up to the one it was
            # measured at. Either way, re-measuring under a tighter limit only loses reuse.
            return
        data = {
            'verdict': res_high.verdict,
            'runtime': res_high.runtime,
            'score': res_high.score,
            'reason': res_high.reason,
            'additional_info': res_high.additional_info,
            'max_abs_err': res_high.max_abs_err,
            'max_rel_err': res_high.max_rel_err,
            'max_best_err': res_high.max_best_err,
            'timelim_high': timelim_high,
        }
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
