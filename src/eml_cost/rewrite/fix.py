"""Backward-cost guarantor — automatic compliance via in-place rewrite.

``@costlimit`` (in ``eml_cost``) is a *rejecter*: when a function
returns an expression exceeding the configured cost ceiling, it
raises :class:`eml_cost.CostLimitExceeded`. That's the right
behaviour for tests and CI; in production it can be brittle.

``@costlimit_or_fix`` is the *fixer* version. Same configuration
surface (``predicted_depth=N``, ``max_path_r=N``, ``pfaffian_r=N``)
plus a ``max_search_steps=N`` budget; if the return value exceeds
the limit, the decorator walks the rewrite graph using the
``eml_rewrite`` library and silently substitutes the first equivalent
form satisfying the limit. If no such form is found within the
budget, it raises :class:`CostFixFailed` with the failed search trail
attached so the caller can see what was tried.

    >>> from eml_rewrite import costlimit_or_fix
    >>> import sympy as sp
    >>>
    >>> @costlimit_or_fix(predicted_depth=2)
    ... def gradient(x):
    ...     return sp.exp(x) / (1 + sp.exp(x))   # textbook sigmoid; cost=3
    >>>
    >>> gradient(sp.Symbol("x"))
    1/(1 + exp(-x))    # silently rewritten under the budget

The substitution is sound by construction: the rewrite library's
three-layer non-regression (structural + domain + numerical) gates
every step, so the returned expression is symbolically equivalent
to what the wrapped function originally produced.
"""
from __future__ import annotations

import functools
import heapq
from typing import Any, Callable, TypeVar

import sympy as sp

from eml_cost import CostLimitExceeded, analyze

from .core import score, suggest
from .path import Step


__all__ = ["CostFixFailed", "costlimit_or_fix"]


_F = TypeVar("_F", bound=Callable[..., Any])


class CostFixFailed(CostLimitExceeded):
    """Raised by :func:`costlimit_or_fix` when the search budget
    exhausts without finding an under-budget equivalent.

    Inherits from :class:`eml_cost.CostLimitExceeded` so callers
    that catch the parent class also catch this. Adds:

    Attributes
    ----------
    search_trail:
        List of :class:`Step` objects walked during the failed
        search, ending at the lowest-cost expression discovered.
        Useful for debugging — shows exactly what the rewriter
        tried before giving up.
    """

    def __init__(
        self,
        expression: sp.Basic,
        axis: str,
        measured: int,
        limit: int,
        search_trail: list[Step],
    ) -> None:
        super().__init__(expression, axis, measured, limit)
        self.search_trail = search_trail


def _make_predicate(
    predicted_depth: int | None,
    max_path_r: int | None,
    pfaffian_r: int | None,
) -> Callable[[sp.Basic], bool]:
    def ok(expr: sp.Basic) -> bool:
        a = analyze(expr)
        if predicted_depth is not None and a.predicted_depth > predicted_depth:
            return False
        if max_path_r is not None and a.max_path_r > max_path_r:
            return False
        if pfaffian_r is not None and a.pfaffian_r > pfaffian_r:
            return False
        return True

    return ok


def _identify_breach(
    expr: sp.Basic,
    predicted_depth: int | None,
    max_path_r: int | None,
    pfaffian_r: int | None,
) -> tuple[str, int, int]:
    """Return (axis_name, measured, limit) for the first breached axis."""
    a = analyze(expr)
    if predicted_depth is not None and a.predicted_depth > predicted_depth:
        return "predicted_depth", a.predicted_depth, predicted_depth
    if max_path_r is not None and a.max_path_r > max_path_r:
        return "max_path_r", a.max_path_r, max_path_r
    if pfaffian_r is not None and a.pfaffian_r > pfaffian_r:
        return "pfaffian_r", a.pfaffian_r, pfaffian_r
    # Should be unreachable; predicate already returned False.
    return "predicted_depth", a.predicted_depth, predicted_depth or 0


def _search_for_fix(
    start: sp.Basic,
    predicate: Callable[[sp.Basic], bool],
    max_steps: int,
    max_frontier: int,
) -> tuple[sp.Basic | None, list[Step]]:
    """Best-first search for an equivalent expression satisfying
    ``predicate``. Returns ``(expr, trail)``: ``expr`` is the
    satisfying expression or ``None`` if none was found within budget;
    ``trail`` is the path walked (always non-empty — at least the
    seed step is present)."""
    start_cost = score(start)
    seed = Step("<start>", start, start_cost)

    if predicate(start):
        return start, [seed]

    seq = 0
    frontier: list[tuple[int, int, list[Step]]] = []
    heapq.heappush(frontier, (start_cost, seq, [seed]))
    seen: set[str] = {sp.srepr(start)}

    best_history: list[Step] = [seed]
    best_cost = start_cost

    while frontier and len(seen) < max_frontier:
        _, _, history = heapq.heappop(frontier)
        current = history[-1].expression
        current_cost = history[-1].cost

        if predicate(current):
            return current, history

        if current_cost < best_cost:
            best_cost = current_cost
            best_history = history

        if len(history) > max_steps:
            continue

        try:
            sugg = suggest(current, only_improvements=False, numerical_verify=False)
        except Exception:
            continue

        for s in sugg:
            if s.score_after > current_cost:
                continue
            key = sp.srepr(s.rewritten)
            if key in seen:
                continue
            seen.add(key)
            seq += 1
            new_history = history + [Step(s.pattern_name, s.rewritten, s.score_after)]
            heapq.heappush(frontier, (s.score_after, seq, new_history))

    return None, best_history


def costlimit_or_fix(
    *,
    predicted_depth: int | None = None,
    max_path_r: int | None = None,
    pfaffian_r: int | None = None,
    max_search_steps: int = 8,
    max_frontier: int = 256,
) -> Callable[[_F], _F]:
    """Decorator that enforces a cost ceiling on the return value of
    the wrapped function, **fixing** over-budget returns when an
    equivalent under-budget form exists in the rewrite library.

    Parameters
    ----------
    predicted_depth, max_path_r, pfaffian_r:
        Cost-axis ceilings (same semantics as :func:`eml_cost.costlimit`).
        At least one must be set; multiple are AND'd.
    max_search_steps:
        Maximum length of any single search path (number of rewrites).
        Default 8 is enough for the patterns in the standard library.
    max_frontier:
        Cap on simultaneously-tracked candidate expressions during the
        search. Default 256 keeps memory bounded.

    Behaviour:

      * **Under budget at first try.** Return value is passed through
        unchanged. No search runs.
      * **Over budget, fix found.** Return value is silently replaced
        by the first equivalent expression satisfying the limit. The
        substitution is sound — the rewrite library's three-layer
        non-regression gates every step.
      * **Over budget, no fix in budget.** Raises :class:`CostFixFailed`
        (a :class:`eml_cost.CostLimitExceeded` subclass) with
        ``search_trail`` attached: the walked path ending at the
        lowest-cost expression discovered.
      * **Non-SymPy return.** Passed through untouched.

    Raises
    ------
    ValueError
        At decoration time, when no axis is configured.
    CostFixFailed
        At call time, when no under-budget equivalent is found.
    """
    if all(v is None for v in (predicted_depth, max_path_r, pfaffian_r)):
        raise ValueError(
            "costlimit_or_fix() requires at least one of predicted_depth, "
            "max_path_r, or pfaffian_r"
        )

    predicate = _make_predicate(predicted_depth, max_path_r, pfaffian_r)

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            if not isinstance(result, sp.Basic):
                return result   # non-SymPy returns pass through untouched

            if predicate(result):
                return result   # already under budget

            fixed, trail = _search_for_fix(
                result, predicate,
                max_steps=max_search_steps,
                max_frontier=max_frontier,
            )
            if fixed is not None:
                return fixed

            axis, measured, limit = _identify_breach(
                result, predicted_depth, max_path_r, pfaffian_r,
            )
            raise CostFixFailed(result, axis, measured, limit, trail)

        return wrapper  # type: ignore[return-value]

    return decorator
