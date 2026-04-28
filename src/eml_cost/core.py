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
    # --- Bessel family ---
    "besselj": 3,    # {1/x, J0, J1}
    "bessely": 5,    # {1/x, ln(x), J0, Y0, Y1} — log-singular at origin
    "besseli": 3,
    "besselk": 5,    # log-singular at integer order, same as bessely.
    # K_n(x) = (1/2)·(-1)^(n+1)·I_n(x)·[2 ln(x/2) + 2γ - ψ(n+1)] + finite series
    # at integer n (DLMF 10.31.2). Chain-order additivity rule gives 5 to
    # match Y_n. Corrected from 3 in 0.14.0; see exploration/yn-higher-order-2026-04-27/.
    "hankel1": 5,    # = J_n + i·Y_n — inherits Y_n's log-singularity
    "hankel2": 5,    # = J_n - i·Y_n — inherits Y_n's log-singularity
    # --- Airy family ---
    "airyai": 3,
    "airybi": 3,
    "airyaiprime": 3,
    "airybiprime": 3,
    # --- Hypergeometric / Lambert W ---
    "hyper": 3,      # 1F1 / 2F1 — 2nd-order ODE
    "LambertW": 2,   # W(x), 1/(1+W) for the derivative
    # ---------------------------------------------------------------
    # Added in eml-cost 0.3.0 (S/R-134 follow-up). The substrate
    # previously treated these as depth-0 atoms because they weren't
    # in this registry; the strict EML-class check in eml-witness
    # 0.2.1 / monogate 2.4.3 caught the resulting verified_in_lean
    # correctness bug. Adding them here so the cost detector also
    # reports meaningful chain orders.
    # ---------------------------------------------------------------
    # erf-family — derived from integrals of exp(-t**2) etc.
    "erf": 2,        # {exp(-x**2), erf(x)}
    "erfc": 2,       # erfc = 1 - erf, same chain
    "erfi": 2,       # erfi(x) = -i*erf(ix), same chain
    "fresnels": 3,   # Fresnel S — Pfaffian chain {sin(x**2), cos(x**2), S}
    "fresnelc": 3,   # Fresnel C — partner of S
    # Gamma family — derivative chain Gamma -> psi -> psi' -> ...
    "gamma": 2,      # {Gamma(x), psi(x)}; Gamma' = Gamma * psi
    "loggamma": 2,   # log Gamma has derivative psi — same chain
    "polygamma": 3,  # psi_n includes digamma (n=0) + trigamma (n=1) + ...
    "beta": 3,       # B(x,y) = Gamma(x)*Gamma(y)/Gamma(x+y) — composition
    # Exponential / cosine integrals — extend exp/log Pfaffian chain
    "Ei": 3,         # integral of exp(t)/t; chain {exp(x), 1/x, Ei}
    "li": 3,         # li(x) = Ei(ln x); same chain order
    "Si": 3,         # integral of sin(t)/t; oscillation-augmented
    "Ci": 3,         # integral of cos(t)/t; oscillation-augmented
    "Shi": 2,        # hyperbolic sine integral
    "Chi": 2,        # hyperbolic cosine integral
    # Polylog / zeta — iterated exp/log integrals
    "polylog": 3,    # Li_2 (dilog); Li_k chain order ~ k+1 for k >= 2
    "zeta": 4,       # Riemann zeta(s) — non-trivial chain
    # Elliptic integrals — non-elementary by Liouville
    "elliptic_k": 3, # K(m) — chain {K, E, 1/m}
    "elliptic_e": 3, # E(m) — partner of K
    "elliptic_f": 4, # F(phi|m) — incomplete, depends on both args
    # ---------------------------------------------------------------
    # Added in eml-cost 0.9.0 (S/R-#10 substrate-coverage audit). The
    # 0.3.0 expansion left 36 named SymPy non-elementary functions
    # treated as depth-0 atoms. Verified by direct sympy.functions.
    # special enumeration. Chain orders below derived from each
    # function's defining ODE / closure relation per Khovanskii.
    # ---------------------------------------------------------------
    # Spherical Bessel / Hankel — besselj/bessely with sqrt(pi/(2z)) prefactor
    "jn": 3,          # spherical j_n — same chain as besselj
    "yn": 5,          # spherical y_n — log-singular like bessely
    "hn1": 5,         # spherical Hankel — composition jn ± i·yn
    "hn2": 5,
    "marcumq": 3,     # Marcum Q — integral with besseli kernel
    # Erf variants
    "erf2": 2,        # erf2(x,y) = erf(y) − erf(x), linear comb
    "erfinv": 3,      # inverse erf — extends chain by one
    "erfcinv": 3,     # erfcinv(x) = erfinv(1 − x)
    "erf2inv": 3,     # generalised inverse — same chain order
    # Exponential integrals
    "expint": 3,      # E_n(x) — generalised exp integral
    "E1": 3,          # alias for expint(1, x)
    "Li": 3,          # offset log integral, Li(x) = li(x) − li(2)
    # Gamma family extension
    "digamma": 2,     # ψ(x) = polygamma(0, x); chain {Γ, ψ}
    "trigamma": 3,    # ψ'(x) = polygamma(1, x)
    "lowergamma": 3,  # γ(s, x) — incomplete lower gamma
    "uppergamma": 3,  # Γ(s, x) — incomplete upper gamma
    "multigamma": 3,  # Γ_p(x) — multivariate gamma; product of Γ
    "harmonic": 3,    # H(n, m) — chain extends digamma
    "factorial": 2,   # n! = Γ(n+1) for non-integer extension
    "factorial2": 2,  # n!! — double factorial via Γ
    "subfactorial": 3, # !n — chain {Γ, exp, !n}
    "RisingFactorial": 2,  # (x)_n = Γ(x+n)/Γ(x)
    "FallingFactorial": 2, # x(x−1)…(x−n+1) = Γ(x+1)/Γ(x−n+1)
    # Elliptic third kind
    "elliptic_pi": 4, # Π(n; φ|m) — third kind, chain {K, E, Π}
    # Zeta family extension
    "dirichlet_eta": 4,  # η(s) = (1 − 2^(1−s))·ζ(s)
    "lerchphi": 4,    # Lerch transcendent — generalises polylog/ζ
    "stieltjes": 4,   # Stieltjes constants — derivatives of ζ at s=1
    "riemann_xi": 6,  # ξ(s) = (s(s−1)/2)·π^(−s/2)·Γ(s/2)·ζ(s).
    # Chain-additivity rule: chain_order(Γ) + chain_order(ζ) = 2 + 4 = 6.
    # Corrected from 5 in 0.13.0; see exploration/chain-5-hunt-2026-04-27/
    # for the discovery + reasoning.
    # Hypergeometric extensions
    "meijerg": 4,     # Meijer G — most general hypergeometric
    "appellf1": 3,    # Appell F1 — bivariate hypergeometric
    # Mathieu functions — periodic ODE with parameter
    "mathieuc": 4,    # cos-type Mathieu C(a, q, x)
    "mathieus": 4,    # sin-type Mathieu S(a, q, x)
    "mathieucprime": 4,  # derivative of mathieuc
    "mathieusprime": 4,  # derivative of mathieus
    # Spherical harmonics — assoc_legendre × cos/sin via Euler
    "Ynm": 3,         # Y_n^m(θ, φ) — complex spherical harmonic
    "Znm": 3,         # Z_n^m(θ, φ) — real spherical harmonic
}


_TANH_LIKE = (sp.tanh, sp.atan, sp.atanh, sp.asinh, sp.acosh, sp.asin, sp.acos)
_SIN_LIKE = (sp.sin, sp.cos)
_HYPER_PAIR = (sp.sinh, sp.cosh)


def _is_registered_pne(sub: sp.Basic) -> bool:
    """Return True if ``sub`` is a PNE primitive call that does NOT
    auto-simplify to an elementary closed form.

    Special case: SymPy keeps the canonical form of ``0F0(;;x) = exp(x)``
    as ``hyper((), (), x)`` — the parameter sequences are empty tuples
    that never trigger an automatic rewrite. The resulting expression IS
    elementary, so we don't flag it as PNE.

    Surfaced by the adversarial bench (``bench/adversarial.csv`` rows
    29 + 30); see ``bench/ADVERSARIAL_FINDINGS.md``.
    """
    if not hasattr(sub, "func"):
        return False
    fname = getattr(sub.func, "__name__", "")
    if fname not in PFAFFIAN_NOT_EML_R:
        return False
    if fname == "hyper" and len(sub.args) >= 2:
        try:
            if len(sub.args[0]) == 0 and len(sub.args[1]) == 0:
                return False
        except TypeError:
            pass
    return True


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
        if _is_registered_pne(sub):
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
    if fname in PFAFFIAN_NOT_EML_R and _is_registered_pne(expr):
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
# Per-class additivity prediction (paper-section formulation)
# ---------------------------------------------------------------------------


def predict_chain_order_via_additivity(expr: sp.Basic) -> int:
    """Predict chain order via the per-AST-node additivity rule.

    Walks the SymPy canonical form of ``expr`` and sums per-primitive-
    class contributions. The rule was derived from a 50+ expression
    probe documented in
    ``monogate-research/exploration/detector-conventions-2026-04-27/``
    and verified 18/18 exact on the PNE corpus, 44/44 on the combined
    PNE corpus (post 0.14.0 registry refit).

    Per-class contributions::

        CLASS 0  (chain 0):
          - bare variables, polynomials, integer/rational constants
          - integer-exponent ``Pow`` of a CLASS-0 base
          - ``Add`` / ``Mul`` nodes themselves

        CLASS 1  (each contributes +1):
          - ``ln``, ``exp``
          - ``tan``, ``atan``, ``atanh``, ``asinh``, ``acosh``,
            ``tanh``, ``asin``  (the ``_TANH_LIKE`` group post 0.15.1)
          - ``Pow`` with non-integer or symbolic exponent

        CLASS 2  (each contributes +2):
          - ``sin``, ``cos``  (``_SIN_LIKE``)
          - ``sinh``, ``cosh`` (``_HYPER_PAIR``)

        CLASS PNE  (each contributes registry value):
          - any registered Pfaffian-not-elementary primitive with
            chain order from :data:`PFAFFIAN_NOT_EML_R`. Distinct
            sub-expression instances each count once.

    The returned integer is comparable to :func:`pfaffian_r` for most
    expressions; the two methods agree on the 44/44 PNE corpus and on
    every expression in the detector-conventions probe. For users who
    want to verify both in parallel::

        from eml_cost import analyze, predict_chain_order_via_additivity
        agree = predict_chain_order_via_additivity(expr) == analyze(expr).pfaffian_r

    Returns 0 for atoms / numbers and any non-Basic input.
    """
    if not isinstance(expr, sp.Basic):
        try:
            expr = sp.sympify(expr)
        except (sp.SympifyError, TypeError):
            return 0

    total = 0
    # Khovanskii chain-generator dedup: multiple AST occurrences of
    # the same chain-contributing sub-expression (e.g. exp(x) in both
    # numerator and denominator) count once, not per-occurrence.
    # Matches the existing pfaffian_r engine which collects chain
    # generators into a set. _SIN_LIKE / _HYPER_PAIR canonicalize to
    # their (sin, cos) / (exp(arg), exp(-arg)) generator pair so that
    # sin(x) and cos(x) sharing an argument count as ONE chain of 2.

    seen_chains: set[sp.Basic] = set()

    def add_chain(key: sp.Basic, weight: int) -> None:
        nonlocal total
        if key not in seen_chains:
            total += weight
            seen_chains.add(key)

    for node in sp.preorder_traversal(expr):
        if not isinstance(node, sp.Basic):
            continue
        if node.is_Atom:
            continue

        func = node.func

        # CLASS 2 — sin / cos: dedup on the (sin, cos) generator pair
        # so sin(x) + cos(x) counts once.
        if isinstance(node, _SIN_LIKE):
            add_chain(("sin_cos", node.args[0]), 2)
            continue

        # CLASS 2 — sinh / cosh: same dedup pattern.
        if isinstance(node, _HYPER_PAIR):
            add_chain(("sinh_cosh", node.args[0]), 2)
            continue

        # CLASS 1 — ln / exp.
        if func is sp.exp or func is sp.log:
            add_chain(node, 1)
            continue

        # CLASS 1 — tan / atan / atanh / asinh / acosh / tanh / asin / acos.
        if func is sp.tan or isinstance(node, _TANH_LIKE):
            add_chain(node, 1)
            continue

        # CLASS 1 — Pow with non-integer or symbolic exponent.
        if func is sp.Pow:
            _, exponent = node.args
            if not exponent.is_Integer:
                add_chain(node, 1)
            # integer-exponent Pow contributes 0 (CLASS 0)
            continue

        # CLASS PNE — registered Pfaffian-not-elementary primitive.
        fname = getattr(func, "__name__", "")
        if fname in PFAFFIAN_NOT_EML_R and _is_registered_pne(node):
            add_chain(node, PFAFFIAN_NOT_EML_R[fname])
            continue

        # Add / Mul / generic container — CLASS 0 (no contribution)

    return total


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
