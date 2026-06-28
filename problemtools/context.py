from __future__ import annotations

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re
from typing import Callable, Pattern, ParamSpec, TypeVar

_T = TypeVar('_T')
_P = ParamSpec('_P')

PROBLEM_PARTS = ['config', 'data', 'graders', 'statement', 'submissions', 'validators']


class Context:
    # Default values here must be kept in sync with the defaults in argparser().
    def __init__(
        self,
        data_filter: Pattern[str] = re.compile('.*'),
        submission_filter: Pattern[str] = re.compile('.*'),
        fixed_timelim: float | None = None,
        parts: list[str] | None = None,
        threads: int = 1,
        show_subtask_scores: bool = False,
        use_cache: bool = False,
        validation_executor: ThreadPoolExecutor | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.data_filter = data_filter
        self.submission_filter = submission_filter
        self.fixed_timelim = fixed_timelim
        self.parts: list[str] = parts if parts is not None else list(PROBLEM_PARTS)
        self.show_subtask_scores = show_subtask_scores
        self.use_cache = use_cache
        self.executor: ThreadPoolExecutor | None = ThreadPoolExecutor(threads) if threads > 1 else None
        self.validation_executor = validation_executor
        # Reports transient per-testcase progress ("Running ... on ...").  When a live
        # results table is active, it routes status through the table's caption instead
        # of writing directly to stdout (which the table's stdout redirect would mangle).
        self.status_callback = status_callback
        self._background_work: list[concurrent.futures.Future[object]] = []

    def submit_background_work(self, job: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> None:
        assert self.executor
        self._background_work.append(self.executor.submit(job, *args, **kwargs))

    def cancel_background_work(self) -> None:
        for future in self._background_work:
            future.cancel()

    def wait_for_background_work(self) -> None:
        concurrent.futures.wait(self._background_work)
