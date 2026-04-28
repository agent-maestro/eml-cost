"""Core rewrite engine.

Pattern library (9 patterns). Three-layer non-regression:

  - structural construction (each pattern returns ``None`` when not
    matched; only matched rewrites are even considered);
  - domain gating (rewrites whose algebraic identity requires a
    positivity or reality assumption are accepted only when SymPy's
    assumption system establishes the requirement);
  - numerical verification (each accepted rewrite is sampled at random
    points drawn from positive, negative, near-zero, and large-magnitude
    regions; mismatches are rejected).

Pattern list:

  1.  exp(x)/(1+exp(x))         -> 1/(1+exp(-x))   sigmoid canonicalization
  2.  sinh(x)/cosh(x)           -> tanh(x)         F-family fusion
  3.  (exp(x)+exp(-x))/2        -> cosh(x)         hyperbolic fusion
  4.  (exp(x)-exp(-x))/2        -> sinh(x)         hyperbolic fusion
  5.  sin^2 + cos^2             -> 1               Pythagorean identity
  6.  cosh^2 - sinh^2           -> 1               hyperbolic identity
  7.  exp(log(x)), log(exp(x))  -> x               inverse pair
                                                    (domain-gated)
  8.  log(a/b)                  -> log(a) - log(b) when score improves
                                                    (domain-gated)
  9.  log(x^n)                  -> n*log(x)        when score improves
                                                    (domain-gated)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import sympy as sp

from eml_cost import measure as _eml_measure


__all__ = [
    "Suggestion",
    "DomainRequirement",
    "NO_REQUIREMENT",
    "Counterexample",
    "best",
    "suggest",
    "score",
    "verify_equivalence",
    "find_counterexample",
]


def score(expr: sp.Basic) -> int:
    """EML predicted depth (lower is better). Wraps ``eml_cost.measure``."""
    return _eml_measure(expr)


# ---------------------------------------------------------------------------
# Domain requirement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainRequirement:
    """What must hold of the input expression for a rewrite to be sound.

    A pattern that holds unconditionally on the real (or complex) line
    returns ``NO_REQUIREMENT``. A pattern that requires a positivity or
    reality assumption on a particular subexpression carries that
    subexpression here so the engine can probe ``expr.is_positive`` /
    ``expr.is_real`` against SymPy's assumption system.
    """

    description: str = ""
    positive: tuple[sp.Basic, ...] = field(default_factory=tuple)
    real: tuple[sp.Basic, ...] = field(default_factory=tuple)
    nonzero: tuple[sp.Basic, ...] = field(default_factory=tuple)

    def is_unconditional(self) -> bool:
        return not (self.positive or self.real or self.nonzero)

    def is_satisfied(self) -> bool:
        """True when SymPy can establish all required assumptions."""
        for e in self.positive:
            if e.is_positive is not True:
                return False
        for e in self.real:
            if e.is_real is not True:
                return False
        for e in self.nonzero:
            if e.is_zero is not False:
                return False
        return True


NO_REQUIREMENT = DomainRequirement(description="")


# ---------------------------------------------------------------------------
# Suggestion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Suggestion:
    """A proposed rewrite with cost-delta and domain information."""

    pattern_name: str
    rewritten: sp.Basic
    score_before: int
    score_after: int
    reduction: int
    domain_required: str = ""
    domain_verified: bool = True
    numerically_verified: bool = True


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------
# Each detector returns ``(rewritten, requirement)`` or ``None``.


_PatternResult = Optional[tuple[sp.Basic, DomainRequirement]]
_PatternFn = Callable[[sp.Basic], _PatternResult]


def _try_sigmoid(expr: sp.Basic) -> _PatternResult:
    if not isinstance(expr, sp.Mul):
        return None
    args = expr.args
    if len(args) != 2:
        return None
    exp_arg, denom_arg = None, None
    for a in args:
        if isinstance(a, sp.exp):
            exp_arg = a
        elif isinstance(a, sp.Pow) and a.args[1] == -1:
            denom_arg = a
    if exp_arg is None or denom_arg is None:
        return None
    inner = denom_arg.args[0]
    if not isinstance(inner, sp.Add) or len(inner.args) != 2:
        return None
    has_one = any(a == sp.S.One for a in inner.args)
    has_exp = any(a == exp_arg for a in inner.args)
    if has_one and has_exp:
        return 1 / (1 + sp.exp(-exp_arg.args[0])), NO_REQUIREMENT
    return None


def _try_tanh_from_sinh_cosh(expr: sp.Basic) -> _PatternResult:
    if not isinstance(expr, sp.Mul) or len(expr.args) != 2:
        return None
    sinh_arg, recip_cosh = None, None
    for a in expr.args:
        if isinstance(a, sp.sinh):
            sinh_arg = a
        elif (isinstance(a, sp.Pow) and a.args[1] == -1
              and isinstance(a.args[0], sp.cosh)):
            recip_cosh = a
    if sinh_arg is None or recip_cosh is None:
        return None
    if sinh_arg.args[0] == recip_cosh.args[0].args[0]:
        return sp.tanh(sinh_arg.args[0]), NO_REQUIREMENT
    return None


def _try_cosh_from_exps(expr: sp.Basic) -> _PatternResult:
    """Match (exp(x)+exp(-x))/2; both Mul-of-Add form and SymPy canonical Add form."""
    # Mul(1/2, Add(exp(x), exp(-x)))
    if isinstance(expr, sp.Mul) and len(expr.args) == 2:
        half_arg, sum_arg = None, None
        for a in expr.args:
            if a == sp.Rational(1, 2):
                half_arg = a
            elif isinstance(a, sp.Add) and len(a.args) == 2:
                sum_arg = a
        if half_arg is not None and sum_arg is not None:
            a1, a2 = sum_arg.args
            if isinstance(a1, sp.exp) and isinstance(a2, sp.exp):
                x1, x2 = a1.args[0], a2.args[0]
                if (x1 + x2).simplify() == 0:
                    x_pos = x1 if not x1.could_extract_minus_sign() else x2
                    return sp.cosh(x_pos), NO_REQUIREMENT

    # SymPy canonical: Add(Mul(1/2, exp(x)), Mul(1/2, exp(-x)))
    if isinstance(expr, sp.Add) and len(expr.args) == 2:
        exps: list[sp.Basic] = []
        for a in expr.args:
            if isinstance(a, sp.Mul) and len(a.args) == 2:
                rest: sp.Basic | None = None
                has_half = False
                for ai in a.args:
                    if ai == sp.Rational(1, 2):
                        has_half = True
                    elif isinstance(ai, sp.exp):
                        rest = ai
                if has_half and rest is not None:
                    exps.append(rest)
        if len(exps) == 2:
            x1, x2 = exps[0].args[0], exps[1].args[0]
            if (x1 + x2).simplify() == 0:
                x_pos = x1 if not x1.could_extract_minus_sign() else x2
                return sp.cosh(x_pos), NO_REQUIREMENT
    return None


def _try_sinh_from_exps(expr: sp.Basic) -> _PatternResult:
    """Match (exp(x)-exp(-x))/2 in SymPy canonical form."""
    if isinstance(expr, sp.Add) and len(expr.args) == 2:
        signed: list[tuple[int, sp.Basic]] = []
        for a in expr.args:
            if isinstance(a, sp.Mul) and len(a.args) == 2:
                sign = 0
                rest: sp.Basic | None = None
                for ai in a.args:
                    if ai == sp.Rational(1, 2):
                        sign = +1
                    elif ai == sp.Rational(-1, 2):
                        sign = -1
                    elif isinstance(ai, sp.exp):
                        rest = ai
                if sign != 0 and rest is not None:
                    signed.append((sign, rest))
        if len(signed) == 2 and signed[0][0] != signed[1][0]:
            (s1, e1), (s2, e2) = signed
            x1, x2 = e1.args[0], e2.args[0]
            if (x1 + x2).simplify() == 0:
                positive = e1 if s1 > 0 else e2
                return sp.sinh(positive.args[0]), NO_REQUIREMENT
    return None


def _try_pythagorean(expr: sp.Basic) -> _PatternResult:
    if not isinstance(expr, sp.Add) or len(expr.args) != 2:
        return None
    for ai, aj in [(expr.args[0], expr.args[1]), (expr.args[1], expr.args[0])]:
        if (isinstance(ai, sp.Pow) and ai.args[1] == 2 and isinstance(ai.args[0], sp.sin)
                and isinstance(aj, sp.Pow) and aj.args[1] == 2 and isinstance(aj.args[0], sp.cos)
                and ai.args[0].args[0] == aj.args[0].args[0]):
            return sp.S.One, NO_REQUIREMENT
    return None


def _try_hyperbolic_id(expr: sp.Basic) -> _PatternResult:
    if not isinstance(expr, sp.Add) or len(expr.args) != 2:
        return None
    cosh_sq, sinh_sq_neg = None, None
    for a in expr.args:
        if isinstance(a, sp.Pow) and a.args[1] == 2 and isinstance(a.args[0], sp.cosh):
            cosh_sq = a
        elif (isinstance(a, sp.Mul) and len(a.args) == 2
              and a.args[0] == -1
              and isinstance(a.args[1], sp.Pow)
              and a.args[1].args[1] == 2
              and isinstance(a.args[1].args[0], sp.sinh)):
            sinh_sq_neg = a
    if cosh_sq is not None and sinh_sq_neg is not None:
        cosh_x = cosh_sq.args[0].args[0]
        sinh_x = sinh_sq_neg.args[1].args[0].args[0]
        if cosh_x == sinh_x:
            return sp.S.One, NO_REQUIREMENT
    return None


def _try_exp_log_inverse(expr: sp.Basic) -> _PatternResult:
    """exp(log(x)) -> x  needs x > 0;  log(exp(x)) -> x  needs x real."""
    if isinstance(expr, sp.exp) and isinstance(expr.args[0], sp.log):
        inner = expr.args[0].args[0]
        return inner, DomainRequirement(
            description=f"{inner} > 0",
            positive=(inner,),
        )
    if isinstance(expr, sp.log) and isinstance(expr.args[0], sp.exp):
        inner = expr.args[0].args[0]
        return inner, DomainRequirement(
            description=f"{inner} real",
            real=(inner,),
        )
    return None


def _try_log_split(expr: sp.Basic) -> _PatternResult:
    """log(a/b) -> log(a) - log(b)  needs a > 0 and b > 0."""
    if not isinstance(expr, sp.log):
        return None
    inner = expr.args[0]
    if isinstance(inner, sp.Mul) and len(inner.args) == 2:
        a, b = inner.args
        if isinstance(b, sp.Pow) and b.args[1] == -1:
            base = b.args[0]
            return sp.log(a) - sp.log(base), DomainRequirement(
                description=f"{a} > 0 and {base} > 0",
                positive=(a, base),
            )
    return None


def _try_log_pow(expr: sp.Basic) -> _PatternResult:
    """log(x^n) -> n*log(x)  needs x > 0 (real-valued identity)."""
    if not isinstance(expr, sp.log):
        return None
    inner = expr.args[0]
    if isinstance(inner, sp.Pow) and inner.args[1].is_Integer and inner.args[1] > 0:
        n = inner.args[1]
        x = inner.args[0]
        return n * sp.log(x), DomainRequirement(
            description=f"{x} > 0",
            positive=(x,),
        )
    return None


PATTERNS: list[tuple[str, _PatternFn]] = [
    ("sigmoid",             _try_sigmoid),
    ("tanh_from_sinh_cosh", _try_tanh_from_sinh_cosh),
    ("cosh_from_exps",      _try_cosh_from_exps),
    ("sinh_from_exps",      _try_sinh_from_exps),
    ("pythagorean",         _try_pythagorean),
    ("hyperbolic_id",       _try_hyperbolic_id),
    ("exp_log_inverse",     _try_exp_log_inverse),
    ("log_split",           _try_log_split),
    ("log_pow",             _try_log_pow),
]


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------


# Sample regions probe behaviour across the real line. Each entry is a
# tuple of (low, high) ranges; we draw uniformly within each. The
# negative and near-zero regions are the ones the historical default
# (positive-only [0.1, 5.0]) missed, which is why the log-family
# rewrites previously slipped through silently for negative inputs.
#
# The "large" region is bounded at 50 to stay well inside the safe
# range for math.exp (overflow at ~700) and other transcendentals,
# while still meaningfully exceeding the default 5.0 ceiling.
_POSITIVE_REGIONS: tuple[tuple[float, float], ...] = (
    (0.1, 5.0),       # default positive
    (1e-4, 0.01),     # near-zero positive
    (5.0, 50.0),      # moderately large positive
)
_NEGATIVE_REGIONS: tuple[tuple[float, float], ...] = (
    (-5.0, -0.1),     # negative
    (-0.01, -1e-4),   # near-zero negative
    (-50.0, -5.0),    # moderately large negative
)
_SAMPLES_PER_REGION = 3
_VERIFY_RTOL = 1e-9
_VERIFY_ATOL = 1e-9
_VERIFY_PRECISION = 30  # decimal digits for sp.N evaluation


def _free_symbols(expr: sp.Basic) -> list[sp.Symbol]:
    return sorted(
        (s for s in expr.free_symbols if isinstance(s, sp.Symbol)),
        key=lambda s: s.name,
    )


def _regions_for_symbol(sym: sp.Symbol) -> tuple[tuple[float, float], ...]:
    """Sample regions consistent with the symbol's assumptions.

    A symbol declared ``positive=True`` is never sampled in the negative
    or sub-positive zero region; a symbol declared ``negative=True`` is
    never sampled positive; a symbol declared ``nonzero=True`` skips the
    near-zero regions on the relevant side. Unconstrained symbols are
    sampled across both sides.
    """
    pos = sym.is_positive is True
    neg = sym.is_negative is True
    nonneg = sym.is_nonnegative is True
    nonpos = sym.is_nonpositive is True
    if pos:
        return _POSITIVE_REGIONS
    if neg:
        return _NEGATIVE_REGIONS
    if nonneg:
        return _POSITIVE_REGIONS  # exclude negatives; near-zero positive included
    if nonpos:
        return _NEGATIVE_REGIONS
    return _POSITIVE_REGIONS + _NEGATIVE_REGIONS


def _approx_equal(a: complex, b: complex, rtol: float, atol: float) -> bool:
    if any(math.isnan(z.real) or math.isnan(z.imag) for z in (a, b)):
        return False
    if any(math.isinf(z.real) or math.isinf(z.imag) for z in (a, b)):
        return a == b
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def _eval_high_precision(
    expr: sp.Basic,
    syms: list[sp.Symbol],
    point: tuple[float, ...],
    precision: int,
) -> tuple[complex | None, type[BaseException] | None]:
    """Evaluate ``expr`` at ``point`` using SymPy's high-precision
    arithmetic, classifying failures into two buckets:

    * ``ValueError`` — true domain error (e.g., ``log(-3)`` in the real
      branch). The expression is undefined at this point.
    * ``OverflowError`` — numerical limit (e.g., ``exp(1e8)``). The
      mathematical value exists; the chosen precision can't represent
      it.

    Returns ``(value, None)`` on success, ``(None, kind)`` on failure.
    Higher precision (default 30 decimal digits) avoids catastrophic
    cancellation that float-precision evaluation would otherwise
    misclassify as a rewrite mismatch.
    """
    subs = {s: sp.Float(v, precision) for s, v in zip(syms, point)}
    try:
        substituted = expr.subs(subs)
    except (TypeError, ValueError):
        return None, ValueError
    try:
        evaluated = substituted.evalf(precision)
    except (TypeError, ValueError):
        return None, ValueError
    # Reject results SymPy could not reduce to a numeric atom.
    if not evaluated.is_number:
        return None, ValueError
    try:
        value = complex(evaluated)
    except (TypeError, ValueError):
        # Some special "infinity" or "nan" symbolic atoms reach this branch.
        if evaluated == sp.nan:
            return None, ValueError
        if evaluated in (sp.oo, -sp.oo, sp.zoo):
            return None, OverflowError
        return None, ValueError
    if math.isnan(value.real) or math.isnan(value.imag):
        return None, ValueError
    if math.isinf(value.real) or math.isinf(value.imag):
        return None, OverflowError
    return value, None


def verify_equivalence(
    before: sp.Basic,
    after: sp.Basic,
    *,
    samples_per_region: int = _SAMPLES_PER_REGION,
    rtol: float = _VERIFY_RTOL,
    atol: float = _VERIFY_ATOL,
    precision: int = _VERIFY_PRECISION,
    seed: int | None = None,
) -> bool:
    """Numerically verify that ``before`` and ``after`` agree on a panel
    of points drawn from regions consistent with each free symbol's
    SymPy assumptions.

    Sampling respects ``is_positive`` / ``is_negative`` / etc. so that
    symbols declared positive are never substituted with negative
    values. Unconstrained symbols are probed across both sides.

    Evaluation runs at ``precision`` decimal digits (default 30) via
    SymPy's ``evalf``, sidestepping the catastrophic-cancellation false
    positives that float-precision evaluation would produce on rewrites
    that *improve* numerical stability (e.g.,
    ``cosh(x)^2 - sinh(x)^2 -> 1``).

    Returns ``False`` on the first mismatch where both sides produce
    finite numeric values that differ. A true domain mismatch (one side
    raises ``ValueError`` while the other is finite) is rejected.
    Numerical-limit asymmetry (overflow on one side) is treated as the
    rewrite being *more* numerically robust and is not a rejection.
    """
    # Fast path: SymPy's simplify can prove many universal identities
    # (Pythagorean, hyperbolic, exponential cancellation). When it
    # reduces (before - after) to zero, the rewrite is symbolically
    # equivalent on the entire domain of definition and we skip the
    # numerical sweep — which would otherwise be defeated by
    # catastrophic cancellation for rewrites that *improve* numerical
    # stability (e.g., cosh(x)^2 - sinh(x)^2 -> 1 at large x).
    try:
        difference = sp.simplify(before - after)
        if difference == 0:
            return True
    except (TypeError, ValueError, RecursionError):
        pass

    rng = random.Random(seed)
    syms = _free_symbols(before) or _free_symbols(after)
    if not syms:
        try:
            return _approx_equal(complex(sp.N(before, precision)),
                                 complex(sp.N(after, precision)),
                                 rtol, atol)
        except (TypeError, ValueError):
            return False

    region_lists = [_regions_for_symbol(s) for s in syms]
    default_value = 1.5

    # Sweep one symbol at a time across its regions; hold others fixed.
    points: list[tuple[float, ...]] = []
    for i, regions in enumerate(region_lists):
        for low, high in regions:
            for _ in range(samples_per_region):
                pt = [default_value] * len(syms)
                pt[i] = rng.uniform(low, high)
                points.append(tuple(pt))

    for point in points:
        v_before, err_before = _eval_high_precision(before, syms, point, precision)
        v_after, err_after = _eval_high_precision(after, syms, point, precision)

        if v_before is None and v_after is None:
            continue
        if err_before is ValueError and v_after is not None:
            return False
        if err_after is ValueError and v_before is not None:
            return False
        if err_before is OverflowError or err_after is OverflowError:
            continue
        # Both sides numeric.
        if not _approx_equal(v_before, v_after, rtol, atol):  # type: ignore[arg-type]
            return False

    return True


# ---------------------------------------------------------------------------
# Counterexample finder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Counterexample:
    """A specific point at which two expressions disagree.

    Attributes
    ----------
    symbols:
        Tuple of free symbols in the order matching ``point``.
    point:
        Numeric values substituted into ``symbols`` at the disagreement.
    before_value:
        Result of evaluating the *before* expression at ``point``. May
        be ``None`` if the expression is not defined there.
    after_value:
        Same, for the *after* expression.
    kind:
        One of ``"value_disagreement"`` (both sides defined, values
        differ) or ``"domain_mismatch"`` (one side raises while the
        other is finite — the rewrite changes the domain of definition).
    note:
        Human-readable one-line summary suitable for editor surfacing.
    before_expr:
        The *before* expression that was checked. Optional (defaults to
        ``None`` for backwards compatibility); always populated when
        produced by :func:`find_counterexample`. Required by
        :func:`render_test` to emit a runnable regression test.
    after_expr:
        Same, for the *after* expression.
    """

    symbols: tuple[sp.Symbol, ...]
    point: tuple[float, ...]
    before_value: complex | None
    after_value: complex | None
    kind: str
    note: str
    before_expr: sp.Basic | None = None
    after_expr: sp.Basic | None = None


def find_counterexample(
    before: sp.Basic,
    after: sp.Basic,
    *,
    samples_per_region: int = _SAMPLES_PER_REGION,
    rtol: float = _VERIFY_RTOL,
    atol: float = _VERIFY_ATOL,
    precision: int = _VERIFY_PRECISION,
    seed: int | None = None,
) -> Counterexample | None:
    """Return the first sampled point at which ``before`` and ``after``
    disagree, or ``None`` if no disagreement is found in the same
    sweep used by :func:`verify_equivalence`.

    Use this when ``verify_equivalence`` returns ``False`` and you
    want to know *why*: the returned :class:`Counterexample` carries
    the specific symbol values, both side's results, and a one-line
    note explaining the divergence.

    Useful for editor tools: when a proposed rewrite is rejected, you
    can show the user the specific input where it fails, instead of
    a generic "unsound" message.
    """
    # If simplify proves equivalence, we're done — there is no
    # counterexample, by construction.
    try:
        if sp.simplify(before - after) == 0:
            return None
    except (TypeError, ValueError, RecursionError):
        pass

    rng = random.Random(seed)
    syms = _free_symbols(before) or _free_symbols(after)
    if not syms:
        # Constant case: evaluate both and compare directly.
        try:
            cb = complex(sp.N(before, precision))
            ca = complex(sp.N(after, precision))
        except (TypeError, ValueError):
            return None
        if _approx_equal(cb, ca, rtol, atol):
            return None
        return Counterexample(
            symbols=(), point=(),
            before_value=cb, after_value=ca,
            kind="value_disagreement",
            note=f"constant disagreement: {cb} vs {ca}",
            before_expr=before, after_expr=after,
        )

    region_lists = [_regions_for_symbol(s) for s in syms]
    default_value = 1.5

    points: list[tuple[float, ...]] = []
    for i, regions in enumerate(region_lists):
        for low, high in regions:
            for _ in range(samples_per_region):
                pt = [default_value] * len(syms)
                pt[i] = rng.uniform(low, high)
                points.append(tuple(pt))

    for point in points:
        v_before, err_before = _eval_high_precision(before, syms, point, precision)
        v_after,  err_after  = _eval_high_precision(after,  syms, point, precision)

        if v_before is None and v_after is None:
            continue
        if err_before is OverflowError or err_after is OverflowError:
            continue

        if err_before is ValueError and v_after is not None:
            sym_str = ", ".join(f"{s}={p!r}" for s, p in zip(syms, point))
            return Counterexample(
                symbols=tuple(syms), point=point,
                before_value=None, after_value=v_after,
                kind="domain_mismatch",
                note=f"original undefined at ({sym_str}); "
                     f"rewrite gives {v_after} — rewrite enlarges the domain",
                before_expr=before, after_expr=after,
            )
        if err_after is ValueError and v_before is not None:
            sym_str = ", ".join(f"{s}={p!r}" for s, p in zip(syms, point))
            return Counterexample(
                symbols=tuple(syms), point=point,
                before_value=v_before, after_value=None,
                kind="domain_mismatch",
                note=f"rewrite undefined at ({sym_str}); "
                     f"original gives {v_before} — rewrite shrinks the domain",
                before_expr=before, after_expr=after,
            )

        if v_before is not None and v_after is not None:
            if not _approx_equal(v_before, v_after, rtol, atol):
                sym_str = ", ".join(f"{s}={p!r}" for s, p in zip(syms, point))
                return Counterexample(
                    symbols=tuple(syms), point=point,
                    before_value=v_before, after_value=v_after,
                    kind="value_disagreement",
                    note=f"at ({sym_str}): original={v_before}, "
                         f"rewrite={v_after}",
                    before_expr=before, after_expr=after,
                )

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _emit(
    pattern_name: str,
    rewritten: sp.Basic,
    score_before: int,
    requirement: DomainRequirement,
    *,
    domain_verified: bool,
    numerically_verified: bool,
) -> Suggestion:
    score_after = score(rewritten)
    return Suggestion(
        pattern_name=pattern_name,
        rewritten=rewritten,
        score_before=score_before,
        score_after=score_after,
        reduction=score_before - score_after,
        domain_required=requirement.description,
        domain_verified=domain_verified,
        numerically_verified=numerically_verified,
    )


def suggest(
    expr: sp.Basic,
    only_improvements: bool = True,
    *,
    include_conditional: bool = False,
    numerical_verify: bool = True,
    verify_seed: int | None = 0xEAA1,
) -> list[Suggestion]:
    """Return all matching rewrites with cost-delta and domain
    information.

    The engine walks the expression tree top-down, then by subterm
    replacement.

    ``only_improvements`` (default True) filters out rewrites whose
    ``score_after`` is not strictly less than ``score_before``.

    ``include_conditional`` (default False) controls handling of
    domain-sensitive patterns whose required assumption cannot be
    established from SymPy's assumption system. When False, such
    rewrites are rejected silently. When True, they are emitted with
    ``domain_verified=False`` and a non-empty ``domain_required``
    string so a UI can prompt the user.

    ``numerical_verify`` (default True) runs ``verify_equivalence`` on
    each accepted rewrite. Mismatching rewrites are rejected. Set to
    False only for benchmarking the structural+domain layers in
    isolation.
    """
    if not isinstance(expr, sp.Basic):
        return []

    out: list[Suggestion] = []
    score_before = score(expr)

    def consider(pattern_name: str, rewritten: sp.Basic, requirement: DomainRequirement) -> None:
        domain_ok = requirement.is_unconditional() or requirement.is_satisfied()
        if not domain_ok and not include_conditional:
            return
        # Run numerical verification only when we believe the rewrite is
        # sound (unconditional or domain-satisfied). Conditional
        # suggestions are surfaced explicitly with the requirement.
        num_ok = True
        if numerical_verify and domain_ok:
            num_ok = verify_equivalence(expr, rewritten, seed=verify_seed)
            if not num_ok:
                return
        s = _emit(
            pattern_name,
            rewritten,
            score_before,
            requirement,
            domain_verified=domain_ok,
            numerically_verified=num_ok,
        )
        if not only_improvements or s.reduction > 0:
            out.append(s)

    for pattern_name, fn in PATTERNS:
        result = fn(expr)
        if result is not None:
            rewritten, requirement = result
            consider(pattern_name, rewritten, requirement)
            continue

        for sub in sp.preorder_traversal(expr):
            if sub == expr:
                continue
            sub_result = fn(sub)
            if sub_result is not None:
                rewritten_sub, requirement = sub_result
                full = expr.xreplace({sub: rewritten_sub})
                consider(pattern_name, full, requirement)
                break

    return out


def best(expr: sp.Basic) -> sp.Basic:
    """Return the lowest-score rewrite, or the original if no improvement found."""
    suggestions = suggest(expr, only_improvements=True)
    if not suggestions:
        return expr
    chosen = min(suggestions, key=lambda s: s.score_after)
    return chosen.rewritten
