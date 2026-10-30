"""Walk a SymPy expression as the DAG it is, not as the tree it prints as (0.26.0).

A program that reuses a value builds a DAG: ``t = g(s, s)`` puts the one node ``s`` under ``g``
twice, and a chain of N such reuses is one node per level whose TREE has 2^N leaves. Every detector
in this package used to recurse once per PATH to a node, so its time was the tree's size. Measured
through Monogate Forge's profiler: a scalar Kalman filter over 10 measurements (each step reuses the
previous estimate twice and the predicted covariance three times) spent 9.4 s in :func:`analyze`
alone, and each further measurement multiplied the time by about 2.8.

Every detector's value at a node depends on that node alone, so each is computed once per distinct
node here and the results are identical:

``nodes(expr)``
    Every node, each object once, in preorder (a node before its arguments, arguments in ``.args``
    order).
``fold(expr, rule)``
    ``rule(node)`` returns ``(deps, combine)``; the node's value is ``combine([value of d for d in
    deps])``. Each object is evaluated once, children first, on an explicit stack -- so a deep
    expression is no longer bounded by Python's recursion limit either (the recursive detectors
    spent two frames per level, the call and its generator).

"Seen" is object identity: two equal nodes built separately are each evaluated (same value, merely
not shared), and nothing is compared structurally just to be skipped.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import sympy as sp

Rule = Callable[[Any], "tuple[Sequence[Any], Callable[[list[Any]], Any]]"]


def nodes(expr: Any) -> Iterator[sp.Basic]:
    """Every SymPy node of ``expr``, each object once, in preorder."""
    seen: set[int] = set()
    keep: list[Any] = []            # ``seen`` holds ids: keep each visited node alive while it does
    stack = [expr]
    while stack:
        e = stack.pop()
        if not isinstance(e, sp.Basic) or id(e) in seen:
            continue
        seen.add(id(e))
        keep.append(e)
        yield e
        stack.extend(reversed(e.args))


def fold(expr: Any, rule: Rule) -> Any:
    """The value ``rule`` gives ``expr``, evaluating each distinct object once, children first."""
    values: dict[int, Any] = {}
    plans: dict[int, tuple[tuple[Any, ...], Callable[[list[Any]], Any]]] = {}
    keep: list[Any] = []            # a rule may build a dependency (``-g``): keep it alive with its id
    stack = [expr]
    while stack:
        node = stack[-1]
        key = id(node)
        if key in values:
            stack.pop()
            continue
        plan = plans.get(key)
        if plan is None:
            deps, combine = rule(node)
            plan = plans[key] = (tuple(deps), combine)
            keep.append(node)
            keep.extend(plan[0])
            pending = [d for d in plan[0] if id(d) not in values]
            if pending:
                stack.extend(reversed(pending))
                continue
        # A dependency is never an ancestor (the expression is acyclic), so every one pushed above
        # this node has been evaluated by the time it is back on top.
        values[key] = plan[1]([values[id(d)] for d in plan[0]])
        stack.pop()
    return values[id(expr)]
