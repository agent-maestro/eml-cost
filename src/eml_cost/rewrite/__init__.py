"""eml-rewrite — F-family fusion pattern rewriter.

Detects equivalent rewrites of symbolic expressions that strictly reduce
predicted EML cost (per ``eml-cost``). Provable non-regression: every
proposed rewrite either improves the cost or is filtered out.

Public API:

    >>> from eml_rewrite import suggest, best
    >>> import sympy as sp
    >>> x = sp.Symbol("x", real=True)
    >>> best(sp.exp(x) / (1 + sp.exp(x)))
    1/(1 + exp(-x))
    >>> # sigmoid pattern recognized; cost reduced from 3 to 2.

Command-line:

    $ eml-rewrite scan mycode.py
    $ eml-rewrite fix mycode.py
"""
from __future__ import annotations

from .core import (
    Counterexample,
    DomainRequirement,
    NO_REQUIREMENT,
    Suggestion,
    best,
    find_counterexample,
    score,
    suggest,
    verify_equivalence,
)
from .expansions import EXPANSION_PATTERNS, expand, expand_fully
from .fix import CostFixFailed, costlimit_or_fix
from .path import Step, path
from .pipeline import (
    CanonicalReport,
    RewriteResult,
    rewrite,
    to_canonical,
)
from .synthesize import render_test

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "best",
    "suggest",
    "score",
    "expand",
    "expand_fully",
    "EXPANSION_PATTERNS",
    "Suggestion",
    "DomainRequirement",
    "NO_REQUIREMENT",
    "Counterexample",
    "CostFixFailed",
    "Step",
    "verify_equivalence",
    "find_counterexample",
    "costlimit_or_fix",
    "path",
    "render_test",
    "rewrite",
    "to_canonical",
    "CanonicalReport",
    "RewriteResult",
]
