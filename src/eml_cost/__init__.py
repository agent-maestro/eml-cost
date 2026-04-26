"""eml-cost — Pfaffian chain order and EML routing depth for symbolic expressions.

Public API:

    >>> from eml_cost import analyze
    >>> result = analyze("exp(exp(x)) + sin(x**2)")
    >>> result.pfaffian_r
    5
    >>> result.eml_depth
    7
    >>> result.predicted_depth
    7

Drop-in measure for SymPy's :func:`sympy.simplify`:

    >>> import sympy as sp
    >>> from eml_cost import measure
    >>> x = sp.Symbol("x")
    >>> sp.simplify(sp.cos(x)**2 + sp.sin(x)**2, measure=measure)
    1

For the full Pfaffian profile breakdown::

    result.pfaffian_r            # total chain order (Khovanskii)
    result.max_path_r            # chain order along deepest path
    result.eml_depth             # EML routing tree depth
    result.structural_overhead   # tree-structural depth (Add/Mul/poly-Pow)
    result.corrections           # Corrections(c_osc, c_composite, delta_fused)
    result.predicted_depth       # max_path_r + corrections + structural
    result.is_pfaffian_not_eml   # True for Bessel, Airy, Lambert W, etc.
    result.expression            # the parsed SymPy expression
"""
from __future__ import annotations

from .analyze import AnalyzeResult, Corrections, analyze, fingerprint, measure
from .caching import FingerprintCacheInfo, cache_by_fingerprint, fingerprint_axes
from .canonicalize import analyze_canonical, canonicalize
from .guards import CostLimitExceeded, costlimit
from .core import (
    PFAFFIAN_NOT_EML_R,
    eml_depth,
    is_pfaffian_not_eml,
    max_path_r,
    pfaffian_r,
    structural_overhead,
)

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "analyze",
    "analyze_canonical",
    "canonicalize",
    "fingerprint",
    "fingerprint_axes",
    "measure",
    "costlimit",
    "cache_by_fingerprint",
    "CostLimitExceeded",
    "FingerprintCacheInfo",
    "AnalyzeResult",
    "Corrections",
    "pfaffian_r",
    "max_path_r",
    "eml_depth",
    "structural_overhead",
    "is_pfaffian_not_eml",
    "PFAFFIAN_NOT_EML_R",
]
