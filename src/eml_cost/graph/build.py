"""Build an equivalence-class graph from a corpus of SymPy expressions.

Construction is deliberately cheap: clustering is O(n) hashing on the
fingerprint axes; edges are NOT computed at build time. Use
:meth:`EquivalenceGraph.find_path` (or :func:`to_dot` with
``include_edges=True``) to lazily compute path-based edges only when
needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import sympy as sp

from eml_cost import fingerprint, fingerprint_axes


__all__ = [
    "GraphNode",
    "EquivalenceClass",
    "EquivalenceGraph",
    "build_graph",
]


@dataclass(frozen=True)
class GraphNode:
    """A single deduplicated expression in the graph.

    Attributes
    ----------
    expression:
        The SymPy expression at this node.
    fingerprint:
        Full Pfaffian fingerprint (``p…-d…-w…-c…-h<6hex>``). Two
        nodes with the same fingerprint are guaranteed to share
        every Pfaffian profile axis AND tree-shape signature.
    axes:
        Axes-only fingerprint (``p…-d…-w…-c…``). The cluster key.
    label:
        Optional human-readable name (e.g., ``"sigmoid (canonical)"``)
        attached when the corpus is built with
        ``label_with_discover=True`` and ``eml_discover.identify``
        finds an exact registry match.
    """

    expression: sp.Basic
    fingerprint: str
    axes: str
    label: str | None = None


@dataclass
class EquivalenceClass:
    """A set of nodes sharing the same Pfaffian fingerprint axes.

    Members all have identical pfaffian_r, max_path_r, eml_depth,
    and correction sums, but may differ in tree shape or symbol
    names. Two members may or may not be symbolically equivalent —
    ``find_path`` is the way to ask whether one rewrites into the
    other under the standard library.
    """

    axes: str
    members: list[int] = field(default_factory=list)


@dataclass
class EquivalenceGraph:
    """The corpus organized as nodes plus clusters.

    Edges are not stored: they're path-based and expensive to
    compute. Use :meth:`find_path` to get the rewrite sequence
    between two nodes.
    """

    nodes: list[GraphNode] = field(default_factory=list)
    classes: dict[str, EquivalenceClass] = field(default_factory=dict)

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_classes(self) -> int:
        return len(self.classes)

    def class_sizes(self) -> list[int]:
        """Member counts per class, descending. Useful for spotting
        the largest cost-equivalence clusters in a corpus."""
        return sorted((len(c.members) for c in self.classes.values()), reverse=True)

    def class_of(self, node_idx: int) -> EquivalenceClass:
        """Return the EquivalenceClass containing the node at ``node_idx``."""
        return self.classes[self.nodes[node_idx].axes]

    def members_of(self, axes: str) -> list[GraphNode]:
        """Return the GraphNodes in the class identified by ``axes``,
        or an empty list if no such class exists."""
        cls = self.classes.get(axes)
        if cls is None:
            return []
        return [self.nodes[i] for i in cls.members]

    def find_path(self, from_idx: int, to_idx: int) -> object | None:
        """Wrap :func:`eml_rewrite.path` between two nodes' expressions.

        Returns the rewrite sequence (a list of ``Step``) or ``None``
        when no monotone-decrease path exists within the path budget.
        Imported lazily so ``eml-rewrite`` is a hard dep but not
        triggered until edges are actually needed.
        """
        from eml_rewrite import path as _path

        return _path(
            self.nodes[from_idx].expression,
            self.nodes[to_idx].expression,
        )


def _coerce(item: sp.Basic | str) -> sp.Basic | None:
    """Best-effort sympify; return None on parse failure."""
    if isinstance(item, sp.Basic):
        return item
    if isinstance(item, str):
        try:
            return sp.sympify(item)
        except Exception:
            return None
    return None


def build_graph(
    corpus: Iterable[sp.Basic | str],
    *,
    label_with_discover: bool = False,
) -> EquivalenceGraph:
    """Cluster a corpus of SymPy expressions by Pfaffian fingerprint axes.

    Parameters
    ----------
    corpus:
        Iterable of SymPy ``Basic`` instances or sympify-able strings.
        Strings that fail to parse are silently skipped (corpus
        ingestion is forgiving by design — typos in a 10K-expression
        scrape shouldn't halt the build).
    label_with_discover:
        When True, attempt to import :mod:`eml_discover` and label
        each node with the best ``identify()`` match (exact-match
        only, max 1 result). When False, or when ``eml_discover``
        is not installed, all node labels are ``None``.

    Returns
    -------
    EquivalenceGraph
        Nodes deduplicated by Python equality; clusters keyed on
        axes-only fingerprint.
    """
    identify_fn = None
    if label_with_discover:
        try:
            from eml_discover import identify as _identify
            identify_fn = _identify
        except ImportError:
            identify_fn = None

    seen: dict[sp.Basic, int] = {}
    nodes: list[GraphNode] = []

    for item in corpus:
        expr = _coerce(item)
        if expr is None:
            continue
        if expr in seen:
            continue
        try:
            fp = fingerprint(expr)
            ax = fingerprint_axes(expr)
        except Exception:
            continue

        label: str | None = None
        if identify_fn is not None:
            try:
                matches = identify_fn(expr, max_results=1)
            except Exception:
                matches = []
            for m in matches:
                if getattr(m, "confidence", "") in ("identical", "exact"):
                    label = m.formula.name
                    break

        seen[expr] = len(nodes)
        nodes.append(GraphNode(
            expression=expr,
            fingerprint=fp,
            axes=ax,
            label=label,
        ))

    classes: dict[str, EquivalenceClass] = {}
    for idx, node in enumerate(nodes):
        cls = classes.get(node.axes)
        if cls is None:
            cls = EquivalenceClass(axes=node.axes)
            classes[node.axes] = cls
        cls.members.append(idx)

    return EquivalenceGraph(nodes=nodes, classes=classes)
