"""Core witness data types + the public ``universality_witness`` constructor.

Witness shape:

    UniversalityWitness
      |- input_expr     : the SymPy expression that was queried
      |- profile        : WitnessProfile (cost axes + fingerprint)
      |- identified     : WitnessIdentified | None (registry match if any)
      |- canonical_path : list[WitnessTreeNode] (rewrite sequence to a
      |                   lower-cost equivalent, when one exists)
      |- savings        : int (predicted_depth reduction along the path)
      |- verified_in_lean : bool (False until EML_Universality.lean lands
      |                   AND the user verifies in VS Code lean4 extension)
      |- lean_url       : str | None (only set when verified_in_lean is True)

The path is empty when the input is already canonical (no
cost-decreasing rewrite available). The identification is None
when no registry formula matches the cost axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from eml_cost import analyze, fingerprint, fingerprint_axes
from eml_discover import identify
from eml_cost.rewrite import path as walk_path


__all__ = [
    "UniversalityWitness",
    "WitnessIdentified",
    "WitnessProfile",
    "WitnessTreeNode",
    "universality_witness",
    "witness_to_dict",
    "LEAN_UNIVERSALITY_URL",
]


# Stable canonical URL for the Lean theorem the `verified_in_lean`
# flag attests to. When `verified_in_lean=True`, this URL is
# attached to the witness so downstream tools (eml-jupyter pill,
# eml-vscode hover, eml-discover-server response, eml-explore CLI)
# can link to the formal source.
LEAN_UNIVERSALITY_URL = (
    "https://github.com/almaguer1986/monogate-lean/blob/master/"
    "MonogateEML/Universality.lean"
)


# The closed set of SymPy Function classes that are EML-elementary.
# Anything outside this set (erf, gamma, polylog, zeta, elliptic_k,
# besselj, airyai, LambertW, hyper, …) is NOT covered by the
# universality theorem and must NOT carry verified_in_lean=True.
#
# Membership rule: include only standard exp / log / trig /
# hyperbolic / inverse-trig / inverse-hyperbolic primitives. The
# substrate's `is_pfaffian_not_eml` flag catches the dozen
# known-special primitives in PFAFFIAN_NOT_EML_R, but it silently
# returns False for less-common non-elementary functions (erf, Γ,
# polylog, ζ, elliptic, exp-integrals, ...) — see
# `monogate-research/exploration/deep-research-2026-04-25-overnight/
# sr134_findings.md` for the coverage-gap analysis. This strict
# allow-list closes the gap on the verified_in_lean field.
_EML_ELEMENTARY_FUNCS: frozenset[type[sp.Function]] = frozenset({
    sp.exp, sp.log,
    sp.sin, sp.cos, sp.tan,
    sp.sinh, sp.cosh, sp.tanh,
    sp.asin, sp.acos, sp.atan,
    sp.asinh, sp.acosh, sp.atanh,
})


def _is_in_eml_class_strict(expr: sp.Basic) -> bool:
    """Recursively verify every Function call in ``expr`` is one of
    the EML-elementary primitives (exp / log / trig / hyperbolic /
    inverses).

    Returns False on the first non-elementary Function encountered;
    True otherwise. Atoms (Symbol, Number, NumberSymbol like pi/E/I)
    are accepted unconditionally. Add / Mul / Pow with elementary
    args propagate via recursion.
    """
    if isinstance(expr, sp.Function):
        if type(expr) not in _EML_ELEMENTARY_FUNCS:
            return False
        return all(_is_in_eml_class_strict(arg) for arg in expr.args)
    if expr.is_Atom:
        return True
    return all(
        _is_in_eml_class_strict(a)
        for a in expr.args
        if isinstance(a, sp.Basic)
    )


@dataclass(frozen=True)
class WitnessProfile:
    """Pfaffian profile slice of the witness.

    Fields mirror :class:`eml_cost.AnalyzeResult` but are flattened
    for JSON serialisation; ``corrections`` becomes three named ints.
    """

    pfaffian_r: int
    max_path_r: int
    eml_depth: int
    structural_overhead: int
    predicted_depth: int
    is_pfaffian_not_eml: bool
    c_osc: int
    c_composite: int
    delta_fused: int
    fingerprint: str
    axes: str


@dataclass(frozen=True)
class WitnessIdentified:
    """Best registry match for the input expression.

    ``confidence`` mirrors :class:`eml_discover.Match.confidence`:
    one of ``"identical"``, ``"exact"``, ``"axes"``. Only the top
    match is captured; for the full ranked list call
    :func:`eml_discover.identify` directly.
    """

    name: str
    confidence: str
    domain: str
    citation: str
    description: str


@dataclass(frozen=True)
class WitnessTreeNode:
    """A single step in the canonical-equivalent rewrite path.

    ``pattern_name`` is the rewrite rule that produced this step
    (or ``"<start>"`` for the seed). ``cost`` is the predicted
    EML depth at that step. The path's first entry is always
    the input expression; subsequent entries are post-rewrite
    expressions with non-increasing cost.
    """

    pattern_name: str
    expression_str: str
    cost: int


@dataclass(frozen=True)
class UniversalityWitness:
    """Complete witness object — see module docstring.

    Use :func:`witness_to_dict` to serialise to a JSON-safe dict.

    ``canonical_path`` is a tuple (not a list) so the dataclass is
    *deeply* immutable — `frozen=True` only protects against
    reassignment of the field itself; a list field would still be
    mutable in place via `.append()` etc.
    """

    input_expr_str: str
    profile: WitnessProfile
    identified: WitnessIdentified | None
    canonical_path: tuple[WitnessTreeNode, ...] = field(default_factory=tuple)
    savings: int = 0
    verified_in_lean: bool = False
    lean_url: str | None = None


def universality_witness(
    expr: sp.Basic | str,
    *,
    walk_canonical: bool = True,
    canonical_target: sp.Basic | None = None,
) -> UniversalityWitness:
    """Build a universality witness for ``expr``.

    Parameters
    ----------
    expr:
        SymPy expression or sympify-able string.
    walk_canonical:
        When True (default), attempt to reduce ``expr`` to a
        lower-cost equivalent via :func:`eml_rewrite.path`. When a
        ``canonical_target`` is supplied, walk specifically to that
        target; otherwise try a small set of common simplifications.
    canonical_target:
        Optional explicit target expression to walk toward. Useful
        when the caller already knows the canonical form they want.
        When omitted, the routine asks :func:`eml_rewrite.path` to
        walk to ``eml_rewrite.best(expr)`` — i.e., whatever the
        best single rewrite produces.

    Returns
    -------
    UniversalityWitness
        Frozen dataclass with profile, identification, and path.

    Notes
    -----
    ``verified_in_lean`` is hard-coded to ``False`` here. The flag
    flips to ``True`` only via post-construction patching after the
    user verifies ``EML_Universality.lean`` in the VS Code lean4
    extension. See ``feedback_lean_writing_protocol_2026_04_25.md``.
    """
    if isinstance(expr, str):
        try:
            parsed: sp.Basic = sp.sympify(expr)
        except Exception as exc:
            raise ValueError(f"Could not parse expression: {expr!r}") from exc
    elif isinstance(expr, sp.Basic):
        parsed = expr
    else:
        raise TypeError(
            f"universality_witness expects str or sp.Basic, got {type(expr).__name__}"
        )

    a = analyze(parsed)
    fp = fingerprint(parsed)
    axes = fingerprint_axes(parsed)

    profile = WitnessProfile(
        pfaffian_r=a.pfaffian_r,
        max_path_r=a.max_path_r,
        eml_depth=a.eml_depth,
        structural_overhead=a.structural_overhead,
        predicted_depth=a.predicted_depth,
        is_pfaffian_not_eml=a.is_pfaffian_not_eml,
        c_osc=a.corrections.c_osc,
        c_composite=a.corrections.c_composite,
        delta_fused=a.corrections.delta_fused,
        fingerprint=fp,
        axes=axes,
    )

    matches = identify(parsed, max_results=1)
    identified: WitnessIdentified | None = None
    if matches:
        m = matches[0]
        identified = WitnessIdentified(
            name=m.formula.name,
            confidence=m.confidence,
            domain=getattr(m.formula, "domain", "unknown"),
            citation=getattr(m.formula, "citation", ""),
            description=getattr(m.formula, "description", ""),
        )

    path_steps: list[WitnessTreeNode] = []
    savings = 0
    if walk_canonical:
        target: sp.Basic | None = canonical_target
        if target is None:
            try:
                from eml_rewrite import best as _best
                better = _best(parsed)
                # `Basic.__eq__` returns a plain bool for structural
                # equality (no `Ne(...)` symbolic object is ever
                # constructed by `!=` on two Basic instances). We
                # want a path walk whenever `best` returned a
                # structurally different expression — even if the
                # two are mathematically equivalent (which they
                # are by definition for any rewrite).
                if better != parsed:
                    target = better
            except (ImportError, AttributeError, RecursionError):
                target = None
        if target is not None and target != parsed:
            try:
                steps = walk_path(parsed, target)
            except (ValueError, RecursionError, AttributeError):
                steps = None
            if steps is not None:
                for s in steps:
                    path_steps.append(WitnessTreeNode(
                        pattern_name=s.pattern_name,
                        expression_str=str(s.expression),
                        cost=s.cost,
                    ))
                if len(steps) >= 2:
                    savings = steps[0].cost - steps[-1].cost

    # The Lean theorem `eml_universality` (in
    # monogate-lean/MonogateEML/Universality.lean, user-verified
    # 2026-04-25) attests that every EML-elementary function admits
    # an EMLTree witness. The flag flips True only when BOTH:
    #   (a) the cost detector's `is_pfaffian_not_eml` flag is False
    #       (catches the dozen known PFAFFIAN_NOT_EML_R primitives —
    #        Bessel, Airy, LambertW, hypergeometric);
    #   (b) the strict elementary-class check passes — every Function
    #       call in the expression is exp / log / trig / hyperbolic
    #       / inverse-trig / inverse-hyperbolic. This catches the
    #       silent gap (erf, gamma, polylog, zeta, elliptic, …)
    #       the cost detector currently misses. See sr134_findings.md.
    is_within_eml_class = (
        (not profile.is_pfaffian_not_eml)
        and _is_in_eml_class_strict(parsed)
    )

    return UniversalityWitness(
        input_expr_str=str(parsed),
        profile=profile,
        identified=identified,
        canonical_path=tuple(path_steps),
        savings=savings,
        verified_in_lean=is_within_eml_class,
        lean_url=LEAN_UNIVERSALITY_URL if is_within_eml_class else None,
    )


def witness_to_dict(w: UniversalityWitness) -> dict[str, Any]:
    """Serialise a :class:`UniversalityWitness` to a JSON-safe dict.

    Round-trips through :func:`json.dumps` without further coercion.
    """
    return {
        "input_expr": w.input_expr_str,
        "profile": {
            "pfaffian_r": w.profile.pfaffian_r,
            "max_path_r": w.profile.max_path_r,
            "eml_depth": w.profile.eml_depth,
            "structural_overhead": w.profile.structural_overhead,
            "predicted_depth": w.profile.predicted_depth,
            "is_pfaffian_not_eml": w.profile.is_pfaffian_not_eml,
            "corrections": {
                "c_osc": w.profile.c_osc,
                "c_composite": w.profile.c_composite,
                "delta_fused": w.profile.delta_fused,
            },
            "fingerprint": w.profile.fingerprint,
            "axes": w.profile.axes,
        },
        "identified": (
            None if w.identified is None
            else {
                "name": w.identified.name,
                "confidence": w.identified.confidence,
                "domain": w.identified.domain,
                "citation": w.identified.citation,
                "description": w.identified.description,
            }
        ),
        "canonical_path": [
            {
                "pattern_name": n.pattern_name,
                "expression": n.expression_str,
                "cost": n.cost,
            }
            for n in w.canonical_path
        ],
        "savings": w.savings,
        "verified_in_lean": w.verified_in_lean,
        "lean_url": w.lean_url,
    }
