"""IPython integration: ``%%eml_rewrite`` cell magic.

Load with::

    %load_ext eml_rewrite.notebook

Then any cell tagged with ``%%eml_rewrite`` is parsed for SymPy
expression literals; for each expression, the rewriter is consulted
and proposed simplifications are surfaced inline before the cell
executes its normal output. The cell *also* executes — the magic is
non-destructive.

Example::

    %%eml_rewrite
    from sympy import Symbol, sin, cos, exp
    x = Symbol("x", real=True, positive=True)
    expr = sin(x)**2 + cos(x)**2

    # Rendered output:
    # eml-rewrite: 1 improving rewrite found
    #   line 3: sin(x)**2 + cos(x)**2
    #     -> 1  (pythagorean, -5 cost units)

IPython is not a hard dependency: this module imports lazily and
``load_ipython_extension`` is a no-op outside an IPython context.
"""
from __future__ import annotations

import ast
from typing import Any

import sympy as sp

from .core import suggest


def _scan_cell(source: str) -> list[tuple[int, str, sp.Basic, Any]]:
    """For each free-floating ``ast.Expr`` in ``source``, return
    ``(line, snippet, parsed_expr, top_suggestion)``. Skips
    expressions with no improving rewrite available.

    Module-private; the magic uses this for the report it prints
    above the cell's own output.
    """
    out: list[tuple[int, str, sp.Basic, Any]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        v = node.value
        # Only attempt expressions that could plausibly be SymPy.
        try:
            parsed = sp.sympify(ast.unparse(v))
        except Exception:
            continue
        if not isinstance(parsed, sp.Basic) or parsed.is_Atom:
            continue
        sugg = suggest(parsed, only_improvements=True)
        if not sugg:
            continue
        top = min(sugg, key=lambda s: s.score_after)
        snippet = ast.unparse(v)
        line_no = getattr(v, "lineno", 0)
        out.append((line_no, snippet, parsed, top))
    return out


def _format_report(found: list[tuple[int, str, sp.Basic, Any]]) -> str:
    """Build the text report the magic prints above the cell output."""
    if not found:
        return "eml-rewrite: no improving rewrites in this cell."
    n = len(found)
    lines = [f"eml-rewrite: {n} improving rewrite{'s' if n != 1 else ''} found"]
    for line_no, snippet, _expr, top in found:
        tag = top.pattern_name
        if not top.domain_verified and top.domain_required:
            tag = f"{top.pattern_name} | conditional: {top.domain_required}"
        lines.append(f"  line {line_no}: {snippet}")
        lines.append(f"    -> {top.rewritten}  ({tag}, -{top.reduction} cost units)")
    return "\n".join(lines)


def load_ipython_extension(ipython: Any) -> None:
    """IPython entry point. Registered via ``%load_ext eml_rewrite.notebook``.

    Adds a single cell magic, ``%%eml_rewrite``, that prints the
    rewrite report above the cell's normal execution.
    """
    def _magic(line: str, cell: str) -> None:   # pragma: no cover (covered via tests)
        report = _format_report(_scan_cell(cell))
        # Print the report so it appears in the cell's output area.
        print(report)
        # Then execute the cell normally so user-defined names land
        # in the surrounding namespace.
        ipython.run_cell(cell)

    ipython.register_magic_function(_magic, magic_kind="cell", magic_name="eml_rewrite")


def unload_ipython_extension(ipython: Any) -> None:
    """Counterpart to ``load_ipython_extension``; lets users
    ``%unload_ext eml_rewrite.notebook`` without a process restart."""
    try:
        del ipython.magics_manager.magics["cell"]["eml_rewrite"]
    except (AttributeError, KeyError):
        pass
