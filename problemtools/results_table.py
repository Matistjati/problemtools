"""Live-updating rich table summarizing how each submission fared on each subtask.

Replaces the plain text table printed at the end of a submissions check with a table
that fills in row by row as submissions finish, so a long verification run stays
readable while it's in progress.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from types import TracebackType
from typing import Any, ClassVar, Self

from rich import box as rich_box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .judge import SubmissionResult, parse_float_tolerances
from .metadata import Metadata
from .model import Submission, TestDataGroup


def get_table_groups(testdata: TestDataGroup) -> list[TestDataGroup]:
    """Return the groups to show as columns: expand any root child that has subgroups."""
    result = []
    for group in testdata.get_subgroups():
        subgroups = group.get_subgroups()
        if subgroups:
            result.extend(subgroups)
        else:
            result.append(group)
    return result


class SubtaskResultsTable:
    """Live-updating table: one row per submission, one column per subtask."""

    _VERDICT_STYLE: ClassVar[dict[str, str]] = {
        'AC': 'bold green',
        'TLE': 'bold yellow',
        'OLE': 'bold yellow',
        'MLE': 'bold magenta',
        'RTE': 'bold red',
        'WA': 'bold red',
        'PAC': 'bold cyan',
        'JE': 'bold white on red',
    }

    _CATEGORY_ORDER: ClassVar[list[str]] = ['AC', 'PAC', 'WA', 'RTE', 'TLE']

    # Terminal "synchronized output" (DEC private mode 2026). Bracketing a repaint in these makes
    # supporting terminals (Windows Terminal, iTerm2, kitty, …) buffer the whole erase+redraw and
    # apply it atomically, so the table swaps in one step instead of visibly blanking and refilling
    # line by line — that line-by-line redraw is what flashes. Terminals that don't understand the
    # mode silently ignore it.
    _SYNC_BEGIN: str = '\x1b[?2026h'
    _SYNC_END: str = '\x1b[?2026l'
    # Coalesce bursts of per-testcase status updates: never repaint more than this often. Row/time-
    # limit changes bypass it (force=True) since they are infrequent and worth showing immediately.
    _MIN_PAINT_INTERVAL: float = 1.0 / 12

    @staticmethod
    def _problem_float_tolerances(groups: list[TestDataGroup], metadata: Metadata) -> tuple[float | None, float | None]:
        """Return (abs_tol, rel_tol) for the problem, taking legacy validator_flags as the baseline and
        merging in any per-group output_validator_flags overrides. If different groups disagree we keep
        the first non-None value seen — column display is informational, not authoritative."""
        abs_tol, rel_tol = parse_float_tolerances(metadata.legacy_validator_flags.split())
        for group in groups:
            group_abs, group_rel = parse_float_tolerances(group.config.get('output_validator_flags', '').split())
            if abs_tol is None and group_abs is not None:
                abs_tol = group_abs
            if rel_tol is None and group_rel is not None:
                rel_tol = group_rel
        return abs_tol, rel_tol

    def __init__(
        self,
        subtask_groups: list[TestDataGroup],
        metadata: Metadata,
        uses_default_validator: bool,
        problem_name: str = '',
        show_subtask_scores: bool = False,
    ) -> None:
        self._subtask_groups = subtask_groups
        self._is_scoring = metadata.is_scoring()
        self._problem_name = problem_name
        self._timelim: float | None = None
        self._show_subtask_scores = show_subtask_scores
        self._abs_tol, self._rel_tol = self._problem_float_tolerances(subtask_groups, metadata)
        # Display mode:
        #   'abs'    — only abs_tol; one "Abs precision" column
        #   'rel'    — only rel_tol; one "Rel precision" column
        #   'both'   — both set, equal values; one "Precision" column with per-token min
        #   'split'  — both set, different values; two columns
        #   None     — no precision column
        self._precision_mode: str | None
        if not uses_default_validator:
            self._precision_mode = None
        elif self._abs_tol is not None and self._rel_tol is not None:
            self._precision_mode = 'both' if self._abs_tol == self._rel_tol else 'split'
        elif self._abs_tol is not None:
            self._precision_mode = 'abs'
        elif self._rel_tol is not None:
            self._precision_mode = 'rel'
        else:
            self._precision_mode = None
        self._rows: list[tuple[str, float, float, list[Any]]] = []
        self._status: str = ''
        self._cached_table: Table | None = None
        self._saved_log_streams: dict[logging.StreamHandler, Any] = {}
        self._paint_lock = threading.Lock()
        self._last_paint: float = 0.0
        self.console = Console()
        self._live = Live(
            self,
            console=self.console,
            auto_refresh=False,
            redirect_stdout=True,
            redirect_stderr=True,
            # 'ellipsis' (not 'visible') while live: when the table is taller than the terminal,
            # 'visible' makes position_cursor() try to move the cursor above the top of the screen
            # to erase the previous frame. The terminal clamps that move, so the scrolled-off rows
            # are never erased and every refresh stacks another full copy of the table. 'ellipsis'
            # caps the live frame to the screen height (with a '...' hint), so in-place updates stay
            # clean. Live.stop() forces 'visible' for the single final render, so the complete table
            # still lands in the scrollback once the run finishes.
            vertical_overflow='ellipsis',
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self._cached_table is None:
            self._cached_table = self._build_table()
        if self._status:
            self._cached_table.caption = Text(self._status, style='dim')
        yield self._cached_table

    def _build_table(self) -> Table:
        tl_suffix = f' @ {self._timelim:g}s time limit' if self._timelim is not None else ''
        if self._show_subtask_scores and self._problem_name:
            title = f'{self._problem_name} raw scores{tl_suffix}'
        elif self._problem_name:
            title = f'{self._problem_name}{tl_suffix}'
        else:
            title = 'Subtask Results'
        table = Table(
            title=title,
            box=rich_box.ROUNDED,
            header_style='bold bright_cyan',
            border_style='bright_black',
            title_style='bold white',
            show_lines=False,
            expand=False,
        )
        table.add_column('Submission', style='bold white', no_wrap=True)
        for group in self._subtask_groups:
            table.add_column(group.datadir.name, justify='center', no_wrap=True)
        if self._is_scoring:
            table.add_column('Score', justify='right', style='bright_white', no_wrap=True)
        for header in self._precision_headers():
            table.add_column(header, justify='right', style='bright_white', no_wrap=True)
        by_category: dict[str, list[tuple[float, float, list[Any]]]] = {cat: [] for cat in self._CATEGORY_ORDER}
        for cat, score, max_runtime, cells in self._rows:
            by_category.setdefault(cat, []).append((score, max_runtime, cells))
        groups_present = [cat for cat in self._CATEGORY_ORDER if by_category.get(cat)]
        row_index = 0
        for g_idx, cat in enumerate(groups_present):
            # Sort by score (higher first); break ties by max runtime (lower first).
            cat_rows = sorted(by_category[cat], key=lambda x: (-x[0], x[1]))
            last_in_group = len(cat_rows) - 1
            is_last_group = g_idx == len(groups_present) - 1
            for r_idx, (_, _, cells) in enumerate(cat_rows):
                end_section = (r_idx == last_in_group) and not is_last_group
                table.add_row(*cells, style='on grey7' if row_index % 2 else '', end_section=end_section)
                row_index += 1
        return table

    def _paint(self, *, force: bool) -> None:
        """Repaint the live table, bracketed in synchronized-output markers so the redraw is atomic
        (no flashing). Non-forced paints are throttled to _MIN_PAINT_INTERVAL so that a flood of
        per-testcase status updates coalesces into a calm, readable cadence instead of flickering."""
        with self._paint_lock:
            now = time.monotonic()
            if not force and now - self._last_paint < self._MIN_PAINT_INTERVAL:
                return
            self._last_paint = now
            sync = self.console.is_terminal
            file = self.console.file
            try:
                if sync:
                    file.write(self._SYNC_BEGIN)
                    file.flush()
                self._live.refresh()
            finally:
                if sync:
                    file.write(self._SYNC_END)
                    file.flush()

    def set_status(self, msg: str) -> None:
        self._status = msg
        self._paint(force=False)

    def set_timelim(self, timelim: float) -> None:
        with self._paint_lock:
            self._timelim = timelim
            self._cached_table = None
        self._paint(force=True)

    def __enter__(self) -> Self:
        self._live.__enter__()
        for handler in logging.root.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
                and sys.__stdout__ is not None
                and handler.stream in (sys.__stdout__, sys.__stderr__)
            ):
                self._saved_log_streams[handler] = handler.stream
                handler.stream = sys.stdout  # now the Live proxy
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        for handler, stream in self._saved_log_streams.items():
            handler.stream = stream
        self._saved_log_streams.clear()
        self._live.__exit__(exc_type, exc_value, exc_traceback)

    def _precision_headers(self) -> list[str]:
        if self._precision_mode == 'abs':
            return ['Abs precision']
        if self._precision_mode == 'rel':
            return ['Rel precision']
        if self._precision_mode == 'both':
            return ['Precision']
        if self._precision_mode == 'split':
            return ['Abs precision', 'Rel precision']
        return []

    @staticmethod
    def _fmt(val: float | None) -> str:
        return '—' if val is None else f'{val:.1e}'

    def _precision_cells(self, result: SubmissionResult) -> list[str]:
        if self._precision_mode == 'abs':
            return [self._fmt(result.max_abs_err)]
        if self._precision_mode == 'rel':
            return [self._fmt(result.max_rel_err)]
        if self._precision_mode == 'both':
            return [self._fmt(result.max_best_err)]
        if self._precision_mode == 'split':
            return [self._fmt(result.max_abs_err), self._fmt(result.max_rel_err)]
        return []

    def _subtask_cell(self, res: SubmissionResult) -> Text:
        verdict = res.verdict
        style = self._VERDICT_STYLE.get(verdict, 'white')
        if verdict == 'AC':
            if self._show_subtask_scores:
                score_str = f'{res.score:.2f}' if res.score is not None else ''
                return Text.assemble((score_str, 'green'))
            elif res.runtime >= 0:
                return Text.assemble((f'{res.runtime:.2f}s', 'green'))
        return Text(f'{verdict}', style=style)

    def add_row(
        self,
        sub: Submission,
        results: list[SubmissionResult],
        category: str = 'AC',
    ) -> None:
        result = results[-1]
        # TestDataGroup is unhashable (it holds a dict and a list), so key the lookup on identity.
        group_results = {id(r.test_node): r for r in results if isinstance(r.test_node, TestDataGroup)}
        cells: list[Any] = [sub.program.name]
        for group in self._subtask_groups:
            res = group_results.get(id(group))
            cells.append(self._subtask_cell(res) if res is not None else Text('·', style='bright_black'))
        if self._is_scoring:
            cells.append(f'{result.score:.0f}' if result.score is not None else '—')
        cells.extend(self._precision_cells(result))
        sort_key = float(result.score) if result.score is not None else float('-inf')
        runtimes = [res.runtime for res in group_results.values() if res.verdict == 'AC' and res.runtime >= 0]
        max_runtime = max(runtimes) if runtimes else float('inf')
        with self._paint_lock:
            self._rows.append((category, sort_key, max_runtime, cells))
            self._status = ''
            self._cached_table = None
        self._paint(force=True)
