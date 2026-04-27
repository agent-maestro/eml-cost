"""Source-file linter: warn before SymPy simplify-style calls hang.

Scans a Python file for SymPy ``simplify`` / ``factor`` / ``expand``
/ ``nsimplify`` / ``trigsimp`` calls and runs :func:`estimate_time`
on the inferred expression argument. Emits ``Finding`` records with
predicted wall-time per proxy.

The estimator was fit on E-191 / E-192 corpora at 5-fold CV
R² 0.68–0.84 across the four proxies (simplify, factor, cse,
lambdify). It is a regression model, not a guarantee — use the
predicted times as an ORDERING signal, not an absolute timeout.

Usage
-----

    >>> from eml_cost import lint_file
    >>> for finding in lint_file("mymodule.py"):    # doctest: +SKIP
    ...     print(finding.message)

Or from the shell::

    $ eml-cost lint mymodule.py
    mymodule.py:42  predicted simplify  4.5s   exp(exp(x)) + sin(x**2)
    mymodule.py:42  consider canonicalize() before simplify()
    mymodule.py:89  predicted factor   12.1s   ...

The linter is intentionally conservative: it only fires on
expression arguments it can resolve to a SymPy ``Basic`` value at
parse time (literal strings via :func:`sympy.sympify`, or local
``sp.Symbol("x") + ...`` patterns). Variables defined elsewhere
in the file are NOT followed (no whole-program flow analysis).

Source: ``estimate_time`` model in ``estimate_time.py``; corpus
in ``monogate-research/exploration/E191_estimate_time``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import sympy as sp

from .estimate_time import estimate_time

__all__ = ["Finding", "lint_file", "lint_source"]


# SymPy callables we lint. The mapping value is the matching
# ``estimate_time`` proxy name; if ``None``, we emit a "consider
# canonicalize()" advisory only.
_SIMPLIFY_NAMES = {
    "simplify": "simplify",
    "factor": "factor",
    "expand": "factor",        # similar normalisation pass
    "nsimplify": "simplify",
    "trigsimp": "simplify",
    "cse": "cse",
    "lambdify": "lambdify",
}


@dataclass(frozen=True)
class Finding:
    """One linter result.

    Attributes
    ----------
    file:
        Path of the scanned file.
    line:
        1-based line number of the offending call.
    col:
        0-based column offset of the call (``ast.AST.col_offset``).
    function:
        SymPy callable name (``"simplify"``, ``"factor"``, ...).
    proxy:
        Matching ``estimate_time`` proxy.
    expression_repr:
        ``str(expr)`` of the inferred argument; truncated to 80 chars.
    predicted_seconds:
        Predicted wall-time in seconds at the matched proxy.
    severity:
        ``"info"`` (< 0.2 s), ``"warn"`` (0.2 – 5 s), or ``"error"``
        (> 5 s — likely to hang in interactive sessions).
    message:
        Human-readable one-liner suitable for shell output.
    """

    file: str
    line: int
    col: int
    function: str
    proxy: str
    expression_repr: str
    predicted_seconds: float
    severity: str
    message: str


def _severity_for(seconds: float) -> str:
    if seconds < 0.2:
        return "info"
    if seconds < 5.0:
        return "warn"
    return "error"


def _try_resolve(node: ast.AST, source: str) -> Optional[sp.Basic]:
    """Best-effort: parse the AST node back to source text and try
    :func:`sympy.sympify`. Returns ``None`` if the expression cannot
    be resolved without runtime context.

    To avoid false-positives from SymPy's auto-Symbol behaviour
    (e.g. ``sympify("e")`` returns Euler's E; ``sympify("x")`` returns
    a fresh Symbol), we ONLY resolve:
      - String / bytes constants
      - Number-literal binary expressions (``Num``, ``BinOp`` of nums)
      - Calls to ``sp.sympify(...)`` / ``sympify(...)`` whose first
        argument is a string literal
    Bare variable names and method-chains are deliberately NOT
    resolved — they need runtime context.
    """
    # Direct string literal: ``simplify("exp(x) + 1")``.
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return sp.sympify(node.value)
        except (sp.SympifyError, ValueError, SyntaxError, TypeError):
            return None

    # Wrapped sympify: ``simplify(sp.sympify("..."))``.
    if isinstance(node, ast.Call):
        f = node.func
        if (
            (isinstance(f, ast.Attribute) and f.attr == "sympify")
            or (isinstance(f, ast.Name) and f.id == "sympify")
        ):
            if (node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                try:
                    return sp.sympify(node.args[0].value)
                except (sp.SympifyError, ValueError, SyntaxError,
                        TypeError):
                    return None

    return None


def _walk_calls(tree: ast.AST):
    """Yield (call_node, function_name) for every Call whose function
    name matches a ``_SIMPLIFY_NAMES`` key. Detects both
    ``sp.simplify(expr)`` and ``expr.simplify()``.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Pattern A: bare name, sp.NAME(...), sympy.NAME(...)
        if isinstance(func, ast.Attribute) and func.attr in _SIMPLIFY_NAMES:
            yield node, func.attr
        elif isinstance(func, ast.Name) and func.id in _SIMPLIFY_NAMES:
            yield node, func.id


def lint_source(source: str, filename: str = "<string>") -> list[Finding]:
    """Lint Python source as a string. Returns a list of findings."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [Finding(
            file=filename, line=getattr(e, "lineno", 0) or 0, col=0,
            function="<parse>", proxy="<parse>",
            expression_repr="(syntax error)",
            predicted_seconds=0.0, severity="error",
            message=f"{filename}:{e.lineno}: syntax error: {e.msg}",
        )]

    findings: list[Finding] = []
    for node, name in _walk_calls(tree):
        proxy = _SIMPLIFY_NAMES[name]
        # Pattern A: simplify(expr) — first positional arg is the target.
        # Pattern B: expr.simplify() — the receiver is the target.
        target_node: Optional[ast.AST] = None
        if isinstance(node.func, ast.Name) and node.args:
            target_node = node.args[0]
        elif isinstance(node.func, ast.Attribute):
            if node.args:
                target_node = node.args[0]
            else:
                target_node = node.func.value
        if target_node is None:
            continue

        expr = _try_resolve(target_node, source)
        if expr is None:
            continue

        try:
            est = estimate_time(expr, proxy=proxy)
        except (ValueError, TypeError) as e:
            findings.append(Finding(
                file=filename, line=node.lineno, col=node.col_offset,
                function=name, proxy=proxy,
                expression_repr=str(expr)[:80],
                predicted_seconds=0.0, severity="info",
                message=(f"{filename}:{node.lineno}: estimate_time "
                         f"failed for {name}({expr}): {e}"),
            ))
            continue

        seconds = est.predicted_ms / 1000.0
        sev = _severity_for(seconds)
        msg = (f"{filename}:{node.lineno}: predicted {name:<10} "
               f"{seconds:6.2f}s   {str(expr)[:60]}")
        findings.append(Finding(
            file=filename, line=node.lineno, col=node.col_offset,
            function=name, proxy=proxy,
            expression_repr=str(expr)[:80],
            predicted_seconds=seconds, severity=sev, message=msg,
        ))
    return findings


def lint_file(path: Union[str, Path]) -> list[Finding]:
    """Lint a single Python source file."""
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [Finding(
            file=str(p), line=0, col=0, function="<read>", proxy="<read>",
            expression_repr="(read error)",
            predicted_seconds=0.0, severity="error",
            message=f"{p}: cannot read: {e}",
        )]
    return lint_source(source, filename=str(p))
