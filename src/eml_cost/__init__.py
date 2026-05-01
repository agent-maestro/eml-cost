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
from .estimate_time import PROXIES, TimeEstimate, estimate_time, model_metadata
from .predict_precision_loss import (
    FLOAT64_EPS,
    PrecisionLossEstimate,
    precision_loss_model_metadata,
    predict_precision_loss,
)
from .recommend_form import (
    FAMILY_RHO,
    RecommendedForm,
    SUPPORTED_FAMILIES,
    recommend_form,
)
from .dynamics import DynamicsProfile, analyze_dynamics
from .siblings import Sibling, corpus_domains, corpus_size, find_siblings
from .lint import Finding, lint_file, lint_source
from .guards import CostLimitExceeded, costlimit
from .batch import analyze_batch, cache_hit_analysis
from .profile import DEFAULT_WEIGHTS, PfaffianProfile
from .live_profile import LiveProfileResult, live_profile
from .regularizer import RegularizerConfig, RegularizerResult, regularize
from .transpile import (
    TranspileResult,
    eml_tree_to_c,
    eml_tree_to_numpy,
    eml_tree_to_python,
    eml_tree_to_sympy,
)
from .core import (
    PFAFFIAN_NOT_EML_R,
    eml_depth,
    is_pfaffian_not_eml,
    max_path_r,
    pfaffian_r,
    predict_chain_order_via_additivity,
    structural_overhead,
)

__version__ = "0.20.0"

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
    "predict_chain_order_via_additivity",
    "PFAFFIAN_NOT_EML_R",
    "PfaffianProfile",
    "DEFAULT_WEIGHTS",
    "LiveProfileResult",
    "live_profile",
    "analyze_batch",
    "cache_hit_analysis",
    "estimate_time",
    "model_metadata",
    "TimeEstimate",
    "PROXIES",
    "predict_precision_loss",
    "PrecisionLossEstimate",
    "precision_loss_model_metadata",
    "FLOAT64_EPS",
    "recommend_form",
    "RecommendedForm",
    "SUPPORTED_FAMILIES",
    "FAMILY_RHO",
    "analyze_dynamics",
    "DynamicsProfile",
    "find_siblings",
    "Sibling",
    "corpus_size",
    "corpus_domains",
    "lint_file",
    "lint_source",
    "Finding",
    "regularize",
    "RegularizerConfig",
    "RegularizerResult",
    "TranspileResult",
    "eml_tree_to_python",
    "eml_tree_to_numpy",
    "eml_tree_to_sympy",
    "eml_tree_to_c",
    "estimate_dynamics",
    "DataDynamics",
]


def __getattr__(name: str):
    """Lazy-import data_analyzer so the numpy/scipy dep stays optional."""
    if name in ("estimate_dynamics", "DataDynamics"):
        from .data_analyzer import DataDynamics, estimate_dynamics
        return {"estimate_dynamics": estimate_dynamics,
                "DataDynamics": DataDynamics}[name]
    raise AttributeError(f"module 'eml_cost' has no attribute {name!r}")
