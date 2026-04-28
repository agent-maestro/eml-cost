"""Render an :class:`EquivalenceGraph` to Graphviz DOT format.

DOT is a plain-text format consumable by every major graph tool
(``dot``, ``neato``, ``sfdp``, ``circo``) plus web viewers like
``edotor.net`` and the ``viz.js`` family. We emit DOT directly so
``eml-graph`` itself has no binary dependency on Graphviz.

Pipe the output through ``dot -Tsvg`` (or use ``graphviz`` Python
bindings if you have them) to get an SVG.
"""
from __future__ import annotations

import re

from .build import EquivalenceGraph


__all__ = ["to_dot"]


_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _safe_id(s: str) -> str:
    """Replace anything outside ``[A-Za-z0-9_]`` with underscores so
    the result is a legal DOT identifier."""
    return _SAFE.sub("_", s)


def _escape_label(s: str) -> str:
    """Escape backslashes and double quotes for DOT label syntax."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(
    graph: EquivalenceGraph,
    *,
    include_edges: bool = False,
    cluster_label_template: str = "{axes} ({size})",
) -> str:
    """Emit a Graphviz DOT-format string for ``graph``.

    Parameters
    ----------
    graph:
        The :class:`EquivalenceGraph` to render.
    include_edges:
        When True, compute path-based edges by invoking
        :func:`eml_rewrite.path` on every pair of nodes in the
        graph (O(n²) total). Edges typically span cluster
        boundaries, since :func:`path` requires monotone-decreasing
        cost — two same-cost-axes nodes (i.e., same cluster) can't
        have a path between them by construction. Slow for large
        graphs; intended for small, focused exhibits.
    cluster_label_template:
        Format string for cluster (subgraph) labels. Available keys:
        ``{axes}`` (the fingerprint axes string), ``{size}`` (member
        count). Default ``"{axes} ({size})"``.

    Returns
    -------
    str
        DOT source. Self-contained; no extra wrapping needed.
    """
    lines = [
        "digraph EquivalenceGraph {",
        "  rankdir=LR;",
        "  compound=true;",
        '  node [shape=box, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]

    for axes, cls in graph.classes.items():
        cluster_id = _safe_id(axes)
        cluster_label = cluster_label_template.format(
            axes=axes, size=len(cls.members),
        )
        lines.append(f"  subgraph cluster_{cluster_id} {{")
        lines.append(f'    label="{_escape_label(cluster_label)}";')
        lines.append("    style=rounded;")
        lines.append("    color=gray70;")

        for idx in cls.members:
            node = graph.nodes[idx]
            label = node.label if node.label else str(node.expression)
            lines.append(
                f'    n{idx} [label="{_escape_label(label)}"];'
            )

        lines.append("  }")

    if include_edges:
        from eml_rewrite import path as _path

        n = len(graph.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    p = _path(
                        graph.nodes[i].expression,
                        graph.nodes[j].expression,
                    )
                except Exception:
                    p = None
                if p is None:
                    continue
                weight = max(len(p) - 1, 0)
                lines.append(
                    f'  n{i} -> n{j} '
                    f'[label="{weight}", dir=both, color=gray40];'
                )

    lines.append("}")
    return "\n".join(lines) + "\n"
