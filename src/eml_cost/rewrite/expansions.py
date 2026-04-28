"""Bidirectional rewrites — go *the other way* on the standard library.

Where :func:`suggest` simplifies (canonicalizes, fuses, reduces cost),
:func:`expand` does the opposite: takes a compact canonical form and
returns equivalent expanded forms, typically with *higher* predicted
cost. Useful for:

  * **Pedagogy.** Show what ``tanh(x)`` "really is" by expanding to
    ``sinh(x)/cosh(x)`` or to its raw exp form.
  * **Paper writing.** When the canonical form is too dense for the
    target audience, surface the explicit expansion.
  * **Test generation.** Synthesize equivalent-but-uglier expressions
    for the canonical-form regression suite.
  * **Symbolic regression / discovery.** Reverse-engineer how a
    simplifier got somewhere.

The four expansion patterns mirror four of the standard rewrite
patterns ``in reverse``:

  ``tanh(x)``                  ↔  ``sinh(x)/cosh(x)``
  ``cosh(x)``                  ↔  ``(exp(x) + exp(-x))/2``
  ``sinh(x)``                  ↔  ``(exp(x) - exp(-x))/2``
  ``1/(1+exp(-x))``            ↔  ``exp(x)/(1+exp(x))``

Patterns whose forward direction *destroys* information
(``sin² + cos²`` → ``1``, ``cosh² - sinh²`` → ``1``,
``exp(log(x))`` → ``x``) have no meaningful reverse and are not
included — you can't algorithmically recover the argument from
``1`` or ``x``. Those simplifications are one-way.
"""
from __future__ import annotations

from typing import Callable, Optional

import sympy as sp

from eml_cost import measure as _eml_measure

from .core import NO_REQUIREMENT, DomainRequirement, Suggestion


__all__ = ["expand", "expand_fully", "EXPANSION_PATTERNS"]


_PatternResult = Optional[tuple[sp.Basic, DomainRequirement]]
_PatternFn = Callable[[sp.Basic], _PatternResult]


def _expand_tanh(expr: sp.Basic) -> _PatternResult:
    """tanh(x) → sinh(x)/cosh(x)"""
    if isinstance(expr, sp.tanh):
        x = expr.args[0]
        return sp.sinh(x) / sp.cosh(x), NO_REQUIREMENT
    return None


def _expand_cosh(expr: sp.Basic) -> _PatternResult:
    """cosh(x) → (exp(x) + exp(-x))/2"""
    if isinstance(expr, sp.cosh):
        x = expr.args[0]
        return (sp.exp(x) + sp.exp(-x)) / 2, NO_REQUIREMENT
    return None


def _expand_sinh(expr: sp.Basic) -> _PatternResult:
    """sinh(x) → (exp(x) - exp(-x))/2"""
    if isinstance(expr, sp.sinh):
        x = expr.args[0]
        return (sp.exp(x) - sp.exp(-x)) / 2, NO_REQUIREMENT
    return None


def _expand_sigmoid_to_textbook(expr: sp.Basic) -> _PatternResult:
    """1/(1 + exp(-x)) → exp(x)/(1 + exp(x))"""
    if not isinstance(expr, sp.Pow) or expr.args[1] != -1:
        return None
    inner = expr.args[0]
    if not isinstance(inner, sp.Add) or len(inner.args) != 2:
        return None
    has_one = any(a == sp.S.One for a in inner.args)
    exp_neg = next(
        (a for a in inner.args
         if isinstance(a, sp.exp)
         and a.args[0].could_extract_minus_sign()),
        None,
    )
    if not (has_one and exp_neg is not None):
        return None
    x = -exp_neg.args[0]   # the +x argument inside exp(-x)
    return sp.exp(x) / (1 + sp.exp(x)), NO_REQUIREMENT


EXPANSION_PATTERNS: list[tuple[str, _PatternFn]] = [
    ("tanh_to_sinh_cosh", _expand_tanh),
    ("cosh_to_exp_form",  _expand_cosh),
    ("sinh_to_exp_form",  _expand_sinh),
    ("sigmoid_to_textbook", _expand_sigmoid_to_textbook),
]


def _score(expr: sp.Basic) -> int:
    return _eml_measure(expr)


def expand(expr: sp.Basic) -> list[Suggestion]:
    """Return all available expansions of ``expr``.

    Walks the expression tree top-down, then by subterm replacement.
    Each :class:`Suggestion` carries the rewritten form and the
    cost delta — typically negative (``score_after > score_before``)
    since expansions add structure.

    For multiple matches, all are returned. Caller picks (the
    deepest expansion is via :func:`expand_fully`).
    """
    if not isinstance(expr, sp.Basic):
        return []

    out: list[Suggestion] = []
    score_before = _score(expr)

    for pattern_name, fn in EXPANSION_PATTERNS:
        result = fn(expr)
        if result is not None:
            rewritten, _req = result
            score_after = _score(rewritten)
            out.append(Suggestion(
                pattern_name=pattern_name,
                rewritten=rewritten,
                score_before=score_before,
                score_after=score_after,
                reduction=score_before - score_after,
                domain_required="",
                domain_verified=True,
                numerically_verified=True,
            ))
            continue

        # Subterm pass: walk the tree looking for expandable nodes.
        for sub in sp.preorder_traversal(expr):
            if sub == expr:
                continue
            sub_result = fn(sub)
            if sub_result is not None:
                rewritten_sub, _req = sub_result
                full = expr.xreplace({sub: rewritten_sub})
                score_after = _score(full)
                out.append(Suggestion(
                    pattern_name=pattern_name,
                    rewritten=full,
                    score_before=score_before,
                    score_after=score_after,
                    reduction=score_before - score_after,
                    domain_required="",
                    domain_verified=True,
                    numerically_verified=True,
                ))
                break

    return out


def expand_fully(expr: sp.Basic, *, max_depth: int = 6) -> sp.Basic:
    """Recursively :func:`expand` until no more expansion patterns
    apply, or ``max_depth`` is reached.

    Returns the deepest expanded form. For ``tanh(x)`` this walks
    ``tanh → sinh/cosh → (exp diff)/(exp sum) → exp-only ratio`` in
    one call — the kind of "show me everything" output a
    paper-writing or tutoring surface wants.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    current = expr
    for _ in range(max_depth):
        suggestions = expand(current)
        if not suggestions:
            break
        # Pick the suggestion with the deepest expansion (highest
        # score_after — the most "expanded" form).
        best = max(suggestions, key=lambda s: s.score_after)
        if best.rewritten == current:
            break   # no progress
        current = best.rewritten
    return current
