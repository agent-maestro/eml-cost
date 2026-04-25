"""Core detector functions: Pfaffian chain order, EML depth, structural overhead.

Convention: Khovanskii r-counting throughout.

  exp(g)        contributes 1 chain element
  ln(g)         contributes 1
  sin(g)        contributes 2 (the closed pair {sin g, cos g})
  cos(g)        contributes 2 (same pair as sin)
  tan(g)        contributes 1 (Riccati closure)
  tanh(g)       contributes 1
  atan(g)       contributes 1
  sinh / cosh   contributes 2 (via {e^g, e^-g})
  sqrt(g)       contributes 1 (non-integer power)
  Pow(b, n)     contributes 0 if n is integer; 1 otherwise
  Add, Mul      contribute 0 (composition)

Pfaffian-but-not-EML primitives (Bessel, Airy, Lambert W, hypergeometric)
contribute their registered chain order; see ``PFAFFIAN_NOT_EML_R``.
"""
from __future__ import annotations

import sympy as sp


__all__ = [
    "PFAFFIAN_NOT_EML_R",
    "pfaffian_r",
    "max_path_r",
    "eml_depth",
    "structural_overhead",
    "is_pfaffian_not_eml",
]


# Pfaffian-but-not-EML primitives mapped to chain order under standard
# defining ODEs (Khovanskii convention).
PFAFFIAN_NOT_EML_R: dict[str, int] = {
    "besselj": 3,    # {1/x, J0, J1}
    "bessely": 5,    # {1/x, ln(x), J0, Y0, Y1}
    "besseli": 3,
    "besselk": 3,
    "hankel1": 3,
    "hankel2": 3,
    "airyai": 3,
    "airybi": 3,
    "airyaiprime": 3,
    "airybiprime": 3,
    "hyper": 3,
    "LambertW": 2,
}


_TANH_LIKE = (sp.tanh, sp.atan, sp.atanh, sp.asinh, sp.acosh)
_SIN_LIKE = (sp.sin, sp.cos)
_HYPER_PAIR = (sp.sinh, sp.cosh)


# ---------------------------------------------------------------------------
# Pfaffian-but-not-EML detection
# ---------------------------------------------------------------------------


def is_pfaffian_not_eml(expr: sp.Basic) -> bool:
    """Return True if ``expr`` contains any Pfaffian-but-not-EML primitive.

    These functions (Bessel, Airy, Lambert W, hypergeometric) are Pfaffian
    in chain order but lie outside the EML-elementary class — they cannot
    be represented as a finite F16-closure tree.
    """
    if not isinstance(expr, sp.Basic):
        return False
    for sub in sp.preorder_traversal(expr):
        if hasattr(sub, "func") and getattr(sub.func, "__name__", "") in PFAFFIAN_NOT_EML_R:
            return True
    return False


# ---------------------------------------------------------------------------
# Total Pfaffian chain order (sum across the tree, deduplicated)
# ---------------------------------------------------------------------------


def _collect_chain(expr: sp.Basic, chains: set[sp.Basic]) -> None:
    if not isinstance(expr, sp.Basic):
        return
    if expr.is_Atom:
        return

    for arg in expr.args:
        _collect_chain(arg, chains)

    func = expr.func

    if func is sp.exp or func is sp.log:
        chains.add(expr)
        return

    if isinstance(expr, _SIN_LIKE):
        arg = expr.args[0]
        chains.add(sp.sin(arg))
        chains.add(sp.cos(arg))
        return

    if isinstance(expr, sp.tan):
        chains.add(sp.tan(expr.args[0]))
        return

    if isinstance(expr, _TANH_LIKE):
        chains.add(expr)
        return

    if isinstance(expr, _HYPER_PAIR):
        arg = expr.args[0]
        chains.add(sp.exp(arg))
        chains.add(sp.exp(-arg))
        return

    if func is sp.Pow:
        _, exponent = expr.args
        if exponent.is_Integer:
            return
        chains.add(expr)
        return

    fname = getattr(func, "__name__", "")
    if fname in PFAFFIAN_NOT_EML_R:
        r_value = PFAFFIAN_NOT_EML_R[fname]
        for i in range(r_value):
            chains.add(sp.Symbol(f"__chain_{fname}_{i}_{hash(expr) % 10**9}"))


def pfaffian_r(expr: sp.Basic) -> int:
    """Return total Pfaffian chain order (Khovanskii convention).

    Counts the number of distinct chain generators across the whole
    expression tree. For sequential nesting (e.g. ``exp(sin(x))``) and
    parallel chains (e.g. ``exp(x) + exp(y)``) this counts every chain
    element. Use :func:`max_path_r` for path-restricted counting.
    """
    if not isinstance(expr, sp.Basic):
        raise TypeError(f"pfaffian_r expects sp.Basic, got {type(expr).__name__}")
    chains: set[sp.Basic] = set()
    _collect_chain(expr, chains)
    return len(chains)


# ---------------------------------------------------------------------------
# Path-restricted Pfaffian chain order (max over root-to-leaf paths)
# ---------------------------------------------------------------------------


def max_path_r(expr: sp.Basic) -> int:
    """Pfaffian chain order along the deepest root-to-leaf path.

    Differs from :func:`pfaffian_r` in that ``Add`` and ``Mul`` nodes
    take the *max* over children instead of summing. For independent-variable
    products like ``f(x) * g(y)`` this is dramatically smaller than total r,
    capturing the parallel-composition behavior of EML routing depth.
    """
    if not isinstance(expr, sp.Basic):
        return 0
    if expr.is_Atom:
        return 0

    func = expr.func

    if func is sp.exp or func is sp.log:
        return 1 + max_path_r(expr.args[0])

    if isinstance(expr, _SIN_LIKE):
        return 2 + max_path_r(expr.args[0])

    if isinstance(expr, sp.tan):
        return 1 + max_path_r(expr.args[0])

    if isinstance(expr, _TANH_LIKE):
        return 1 + max_path_r(expr.args[0])

    if isinstance(expr, _HYPER_PAIR):
        return 2 + max_path_r(expr.args[0])

    if func is sp.Pow:
        base, exponent = expr.args
        if exponent.is_Integer:
            return max_path_r(base)
        return 1 + max(max_path_r(base), max_path_r(exponent))

    if func is sp.Add or func is sp.Mul:
        return max((max_path_r(a) for a in expr.args), default=0)

    fname = getattr(func, "__name__", "")
    if fname in PFAFFIAN_NOT_EML_R:
        r_value = PFAFFIAN_NOT_EML_R[fname]
        if expr.args:
            return r_value + max(
                (max_path_r(a) for a in expr.args if isinstance(a, sp.Basic)),
                default=0,
            )
        return r_value

    return max(
        (max_path_r(a) for a in expr.args if isinstance(a, sp.Basic)),
        default=0,
    )


# ---------------------------------------------------------------------------
# EML routing tree depth
# ---------------------------------------------------------------------------


def _is_lead_pattern(expr: sp.Basic) -> sp.Basic | None:
    """Return inner ``g`` if ``expr`` matches LEAd: ``log(c + exp(g))``."""
    if not isinstance(expr, sp.log):
        return None
    inner = expr.args[0]
    if not isinstance(inner, sp.Add) or len(inner.args) != 2:
        return None
    a1, a2 = inner.args
    if a1.is_constant() and isinstance(a2, sp.exp):
        return a2.args[0]
    if a2.is_constant() and isinstance(a1, sp.exp):
        return a1.args[0]
    return None


def _is_sigmoid_pattern(expr: sp.Basic) -> sp.Basic | None:
    """Return inner ``g`` if ``expr`` matches sigmoid: ``1/(1 + exp(-g))``."""
    if isinstance(expr, sp.Pow) and expr.args[1] == -1:
        inner = expr.args[0]
        if isinstance(inner, sp.Add) and len(inner.args) == 2:
            for arg in inner.args:
                if isinstance(arg, sp.exp):
                    g = arg.args[0]
                    return -g if g.could_extract_minus_sign() else g
    return None


def eml_depth(expr: sp.Basic) -> int:
    """Return EML routing tree depth.

    Models the SuperBEST routing tree:
      - exp / log: 1 level
      - sin / cos: 3 levels (Euler bypass)
      - tan: 4 levels (sin/cos via Euler)
      - tanh / atan / asinh / acosh / atanh: 1 level (F-family primitive)
      - sinh / cosh: 1 level (F-family primitive)
      - Pow: 1 level
      - Add / Mul: 1 + max over children

    F-family fusion patterns (LEAd: ``log(c + exp(g))``; sigmoid:
    ``1/(1 + exp(-g))``) collapse to 1 level + child depth.
    """
    if not isinstance(expr, sp.Basic):
        raise TypeError(f"eml_depth expects sp.Basic, got {type(expr).__name__}")
    return _eml_depth_inner(expr)


def _eml_depth_inner(expr: sp.Basic) -> int:
    if not isinstance(expr, sp.Basic):
        return 0
    if expr.is_Atom:
        return 0

    inner_g = _is_lead_pattern(expr)
    if inner_g is not None:
        return 1 + _eml_depth_inner(inner_g)

    inner_g = _is_sigmoid_pattern(expr)
    if inner_g is not None:
        return 1 + _eml_depth_inner(inner_g)

    func = expr.func

    if func is sp.exp or func is sp.log:
        return 1 + _eml_depth_inner(expr.args[0])

    if isinstance(expr, _SIN_LIKE):
        return 3 + _eml_depth_inner(expr.args[0])

    if isinstance(expr, sp.tan):
        return 4 + _eml_depth_inner(expr.args[0])

    if isinstance(expr, _TANH_LIKE):
        return 1 + _eml_depth_inner(expr.args[0])

    if isinstance(expr, _HYPER_PAIR):
        return 1 + _eml_depth_inner(expr.args[0])

    if func is sp.Pow:
        base, exponent = expr.args
        if exponent.is_Integer and exponent >= 0:
            return 1 + _eml_depth_inner(base)
        return 1 + max(_eml_depth_inner(base), _eml_depth_inner(exponent))

    if func is sp.Add or func is sp.Mul:
        return 1 + max(_eml_depth_inner(a) for a in expr.args)

    return 1 + max(
        (_eml_depth_inner(a) for a in expr.args if isinstance(a, sp.Basic)),
        default=-1,
    )


# ---------------------------------------------------------------------------
# Structural tree-overhead (Add / Mul / poly-Pow)
# ---------------------------------------------------------------------------


def structural_overhead(expr: sp.Basic) -> int:
    """Count Add / Mul / positive-integer-Pow nodes along the deepest path.

    These contribute to EML tree depth but have no Pfaffian chain analog —
    they are tree-structural overhead, not transcendental cost.
    """
    if not isinstance(expr, sp.Basic) or expr.is_Atom:
        return 0
    func = expr.func
    children_max = (
        max((structural_overhead(a) for a in expr.args), default=0)
        if expr.args
        else 0
    )
    if func is sp.Add or func is sp.Mul:
        return 1 + children_max
    if func is sp.Pow:
        exponent = expr.args[1]
        if exponent.is_Integer and exponent >= 0:
            return 1 + structural_overhead(expr.args[0])
        return children_max
    return children_max
