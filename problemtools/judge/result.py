from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..verifyproblem import TestCase, TestCaseGroup

Verdict = Literal['AC', 'TLE', 'OLE', 'MLE', 'RTE', 'WA', 'PAC', 'JE']


class SubmissionResult:
    def __init__(
        self,
        verdict: str,
        score: float | None = None,
        reason: str | None = None,
        additional_info: str | None = None,
    ) -> None:
        self.verdict = verdict
        self.score = score
        self.reason = reason
        self.additional_info = additional_info
        self.test_node: TestCase | TestCaseGroup | None = None
        self.runtime_testcase: TestCase | None = None
        self.runtime = -1.0
        self.validator_first = False  # Needed to work around interactive giving unreliable runtime on WA
        self.first_failure: TestCase | None = None
        self.first_failure_verdict: str | None = None
        self.group_results: dict[TestCaseGroup, SubmissionResult] | None = None
        # Float-tolerance precision tracking (default validator only). All three are max-over-tokens-
        # then-max-over-testcases. None when the float-tolerance metric doesn't apply (custom
        # validator, no float flags, or structural mismatch in a WA output).
        self.max_abs_err: float | None = None  # max |judge - team|
        self.max_abs_err_tc: TestCase | None = None
        self.max_rel_err: float | None = None  # max |judge - team| / |judge|
        self.max_rel_err_tc: TestCase | None = None
        self.max_best_err: float | None = None  # max over tokens of min(abs_err, rel_err); both tols set
        self.max_best_err_tc: TestCase | None = None

    def __str__(self) -> str:
        verdict = self.verdict
        details = []
        if verdict == 'AC' and self.score is not None:
            verdict += f' ({self.score:.0f})'
        if self.reason is not None:
            details.append(self.reason)
        if self.test_node is not None and not self.test_node.is_group:
            details.append(f'testcase: {self.test_node}')
        if self.runtime != -1:
            details.append(f'CPU: {self.runtime:.2f}s @ {self.runtime_testcase}')
        return verdict if not details else f'{verdict} [{", ".join(details)}]'
