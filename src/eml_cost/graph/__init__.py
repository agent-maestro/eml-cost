"""eml-graph — equivalence-class graph over corpora of SymPy expressions.

Software architects have call graphs; mathematicians have equivalence
classes. ``eml-graph`` gives you both at once for a real corpus:

  - **Nodes** are unique expressions (deduplicated by Python equality).
  - **Clusters** group nodes by Pfaffian fingerprint axes — every
    member of a cluster shares a cost class.
  - **Edges** (optional, slow) connect cluster members that
    :func:`eml_rewrite.path` can walk between under monotone-decreasing
    cost.

    >>> from eml_graph import build_graph, to_dot
    >>> import sympy as sp
    >>> x, y = sp.symbols("x y")
    >>> g = build_graph([sp.sin(x), sp.sin(y), sp.exp(x), sp.cos(y)])
    >>> g.num_nodes(), g.num_classes()
    (4, 2)
    >>> dot = to_dot(g)        # Graphviz DOT source as a string

Output formats:

  - :func:`to_dot` — Graphviz DOT, no binary deps. Pipe to ``dot -Tsvg``
    for an SVG, or paste into any DOT viewer.
"""
from __future__ import annotations

from .build import (
    EquivalenceClass,
    EquivalenceGraph,
    GraphNode,
    build_graph,
)
from .render import to_dot

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "build_graph",
    "to_dot",
    "EquivalenceGraph",
    "EquivalenceClass",
    "GraphNode",
]
