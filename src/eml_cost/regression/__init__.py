"""EML-native symbolic regression — Phase 2.

Public surface:

  - :class:`EMLNode` — tree node representation.
  - :class:`GPConfig` / :class:`GPResult` — search-engine config and output.
  - :func:`search` — run genetic-programming SR.
  - :func:`random_baseline` — random-tree-sampling baseline (no evolution).
  - :mod:`eml_cost.regression.benchmark` — Feynman SR-bench harness.

The engine is intentionally compact. It is not a competitor to
PySR/DEAP for raw performance; it exists to validate that the
chain-order regularizer steers search measurably better than
unsteered evolution. See
:func:`eml_cost.regression.benchmark.run_benchmark` for the
head-to-head comparison.
"""
from __future__ import annotations

from .gp import GPConfig, GPResult, random_baseline, search
from .nodes import (
    BINARY_OPS,
    EMLNode,
    TERMINALS_DEFAULT_CONSTS,
    UNARY_OPS,
    all_subtree_indices,
    get_subtree,
    random_tree,
    replace_subtree,
)


__all__ = [
    "EMLNode",
    "GPConfig",
    "GPResult",
    "search",
    "random_baseline",
    "random_tree",
    "all_subtree_indices",
    "get_subtree",
    "replace_subtree",
    "BINARY_OPS",
    "UNARY_OPS",
    "TERMINALS_DEFAULT_CONSTS",
]
