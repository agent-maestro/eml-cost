"""Cost-anneal path between two equivalent expressions.

Given ``start`` and ``end`` symbolic expressions believed to be
equivalent, ``path(start, end)`` returns the rewrite *sequence*
that walks from one to the other under the existing rewrite library
with monotonically non-increasing cost — or ``None`` if no such
path exists within the configured search budget.

Algorithm: best-first search on expression-graph. Nodes are
SymPy expressions; edges are individual rewrites from
:func:`suggest` (with ``only_improvements=False`` so we can also
admit cost-neutral rewrites that bridge the two endpoints).
Priority key = predicted cost; the search prunes any branch whose
post-rewrite cost exceeds the source-side cost (monotone-decrease
constraint).

Use cases:

  * **Pedagogy.** "Why is the simplified form simpler?" — the
    rewrite sequence is the answer, step by step.
  * **Differential rewriting.** Two equivalent expressions in
    different files; the path explains the diff.
  * **Reproducible simplification.** Save + replay a path to
    canonicalize new expressions through the same lens.

The search is bounded by ``max_steps`` (default 6) and
``max_frontier`` (default 256). On budget exhaustion the function
returns ``None`` rather than partial output, so callers can rely
on a non-``None`` return being a complete path.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import sympy as sp

from .core import score, suggest


@dataclass(frozen=True)
class Step:
    """A single rewrite in a cost-anneal path.

    Attributes
    ----------
    pattern_name:
        Name of the rewrite pattern that produced ``expression``.
        For the seed step (the input ``start``), this is
        ``"<start>"``.
    expression:
        The SymPy expression at this step.
    cost:
        Predicted EML cost of ``expression`` (per
        ``eml_cost.measure``).
    """

    pattern_name: str
    expression: sp.Basic
    cost: int


def _are_equivalent(a: sp.Basic, b: sp.Basic) -> bool:
    """Symbolic equivalence test — relies on ``sp.simplify`` proving
    ``a - b == 0``. Conservative: returns False on any failure."""
    if a == b:
        return True
    try:
        return bool(sp.simplify(a - b) == 0)
    except (TypeError, ValueError, RecursionError):
        return False


def path(
    start: sp.Basic,
    end: sp.Basic,
    *,
    max_steps: int = 6,
    max_frontier: int = 256,
) -> list[Step] | None:
    """Return the rewrite sequence from ``start`` to ``end``, or
    ``None`` if no monotone-decrease path exists within the budget.

    Parameters
    ----------
    start, end:
        SymPy expressions believed to be equivalent.
    max_steps:
        Hard cap on path length (number of rewrites). Default 6 is
        enough for the patterns in the standard library.
    max_frontier:
        Hard cap on simultaneously-tracked candidate expressions.
        Default 256 keeps memory bounded on pathological searches.
    """
    if not isinstance(start, sp.Basic) or not isinstance(end, sp.Basic):
        return None

    # Trivial: start already equals end (Python equality, not just
    # symbolic equivalence — we want zero-step paths to be cheap).
    if start == end:
        return [Step("<start>", start, score(start))]

    # Pre-flight: are they even symbolically equivalent? If not,
    # there's no path to find. Cheap guard.
    if not _are_equivalent(start, end):
        return None

    start_cost = score(start)

    # Best-first frontier. Items are (priority, sequence_id, path_so_far).
    # sequence_id breaks ties without comparing SymPy expressions.
    seq = 0
    frontier: list[tuple[int, int, list[Step]]] = []
    heapq.heappush(
        frontier,
        (start_cost, seq, [Step("<start>", start, start_cost)]),
    )
    seen: set[str] = {sp.srepr(start)}

    while frontier and len(seen) < max_frontier:
        _, _, history = heapq.heappop(frontier)
        current = history[-1].expression
        current_cost = history[-1].cost

        # Target check: Python equality only. The pre-flight has already
        # verified symbolic equivalence between ``start`` and ``end``;
        # what remains is finding the rewrite *sequence* that literally
        # produces ``end`` via library patterns. Using sp.simplify here
        # would short-circuit the search whenever simplify can prove
        # the start is already equivalent to the end (which is always,
        # by the pre-flight) — defeating the purpose.
        if current == end:
            return history

        if len(history) > max_steps:
            continue

        # Expand: take every cost-non-increasing rewrite suggestion.
        # We pass numerical_verify=False here for speed; the patterns
        # have already been proven sound elsewhere (and we re-check
        # equivalence at terminus).
        try:
            sugg = suggest(
                current,
                only_improvements=False,
                numerical_verify=False,
            )
        except Exception:
            continue

        for s in sugg:
            if s.score_after > current_cost:
                continue   # monotone-decrease guard
            key = sp.srepr(s.rewritten)
            if key in seen:
                continue
            seen.add(key)
            seq += 1
            new_history = history + [Step(s.pattern_name, s.rewritten, s.score_after)]
            heapq.heappush(frontier, (s.score_after, seq, new_history))

    return None
