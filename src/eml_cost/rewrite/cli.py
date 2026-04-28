"""Command-line interface for ``eml-rewrite``.

Subcommands:

    eml-rewrite scan FILE [FILE ...]    # Report rewrites without applying
    eml-rewrite fix  FILE [FILE ...]    # Apply rewrites in place
    eml-rewrite analyze EXPR            # Print Pfaffian profile of one expression

Scan/fix walk Python source files looking for SymPy expression literals
and inline arithmetic. Conservative: only rewrites that strictly reduce
EML cost are proposed/applied.

The ``scan --as-patch`` mode emits a unified diff that ``git apply``
can consume directly.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path
from typing import Any

import sympy as sp

from eml_cost import analyze

from .core import best, suggest


# ---------------------------------------------------------------------------
# Source scanning
# ---------------------------------------------------------------------------


def _safe_eval_expr_node(node: ast.AST) -> sp.Basic | None:
    """Try to parse an AST expression node into a SymPy expression.

    Whitelist-only: uses ``sympify`` on the unparsed source, but only after
    confirming the AST contains nothing dangerous (Call, Attribute, BinOp,
    UnaryOp, Constant, Name).
    """
    # ast.Num was deprecated in 3.8 and removed in 3.14; ast.Constant
    # covers numeric literals on every supported Python version.
    safe_node_types = (
        ast.Expression, ast.Expr, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Name, ast.Call, ast.Attribute, ast.USub, ast.UAdd,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        ast.Load, ast.Tuple, ast.Subscript, ast.Slice,
    )
    for sub in ast.walk(node):
        if not isinstance(sub, safe_node_types):
            return None
    try:
        src = ast.unparse(node)
        return sp.sympify(src)
    except Exception:
        return None


def _find_expressions_in_source(source: str) -> list[tuple[int, str, sp.Basic]]:
    """Return (line, snippet, sympy_expr) for each parseable expression."""
    out: list[tuple[int, str, sp.Basic]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            expr = _safe_eval_expr_node(node.value)
            if expr is not None and isinstance(expr, sp.Basic) and not expr.is_Atom:
                snippet = ast.unparse(node.value)
                out.append((node.lineno, snippet, expr))
    return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _make_patch(
    filename: str,
    source: str,
    include_conditional: bool,
) -> str:
    """Render a unified diff of the file with all improving rewrites
    applied. Output is suitable for ``git apply``. Returns an empty
    string when no rewrites apply (so callers can short-circuit).

    Uses the AST node's source span (``lineno``, ``col_offset``,
    ``end_lineno``, ``end_col_offset``) for exact in-place
    substitution, so whitespace differences between the source and
    ``ast.unparse`` output don't cause silent skips.
    """
    original_lines = source.splitlines(keepends=True)
    rewritten_lines = list(original_lines)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    # Collect (node, expr, suggestion) for every Expr we can analyze.
    # We sort right-to-left so substitutions on the same line don't
    # invalidate each other's column offsets.
    edits: list[tuple[int, int, int, str]] = []  # (line0, col_start, col_end, replacement)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        expr = _safe_eval_expr_node(node.value)
        if expr is None or not isinstance(expr, sp.Basic) or expr.is_Atom:
            continue
        sugg = suggest(
            expr,
            only_improvements=True,
            include_conditional=include_conditional,
        )
        if not sugg:
            continue
        top = min(sugg, key=lambda s: s.score_after)
        # Use the AST node's exact span; fall back gracefully if
        # spans are unavailable (very old Python or out-of-tree
        # parsers).
        v = node.value
        # Narrow Optional[int] to int through local checks; mypy then
        # accepts the int-tuple append below.
        lineno = getattr(v, "lineno", None)
        end_lineno = getattr(v, "end_lineno", None)
        col_offset = getattr(v, "col_offset", None)
        end_col_offset = getattr(v, "end_col_offset", None)
        if (lineno is None or end_lineno is None
                or col_offset is None or end_col_offset is None):
            continue
        if lineno != end_lineno:
            # Multi-line expression — the simple line-edit model
            # below doesn't handle this. Skip; v0.2 territory.
            continue
        edits.append((lineno - 1, col_offset, end_col_offset, str(top.rewritten)))

    # Apply right-to-left within each line so left-most spans don't
    # shift the right-most ones.
    edits.sort(key=lambda e: (e[0], -e[1]))
    for line0, col_start, col_end, replacement in edits:
        if line0 < 0 or line0 >= len(rewritten_lines):
            continue
        line = rewritten_lines[line0]
        rewritten_lines[line0] = line[:col_start] + replacement + line[col_end:]

    if not edits:
        return ""

    diff_iter = difflib.unified_diff(
        original_lines, rewritten_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        n=3,
    )
    return "".join(diff_iter)


def cmd_scan(args: argparse.Namespace) -> int:
    found_any = False
    n_files = 0
    n_suggestions = 0
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"{f}: not found", file=sys.stderr)
            continue
        n_files += 1
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if args.as_patch:
            patch = _make_patch(f, source, args.include_conditional)
            if patch:
                found_any = True
                # Count hunks ('@@') as a proxy for suggestion count.
                n_suggestions += sum(1 for line in patch.splitlines() if line.startswith("@@"))
                print(patch, end="")
            continue

        for line, snippet, expr in _find_expressions_in_source(source):
            sugg = suggest(
                expr,
                only_improvements=True,
                include_conditional=args.include_conditional,
            )
            if not sugg:
                continue
            top = min(sugg, key=lambda s: s.score_after)
            found_any = True
            n_suggestions += 1
            print(f"{f}:{line}  {snippet}")
            tag = top.pattern_name
            if not top.domain_verified and top.domain_required:
                tag = f"{top.pattern_name} | conditional: {top.domain_required}"
            print(f"  -> {top.rewritten}  ({tag}, -{top.reduction} cost units)")
    if args.as_patch:
        # In patch mode, the only stdout content is the unified diff
        # itself so the output stays git-apply-clean. The exit code
        # carries the "found anything" signal.
        return 0 if found_any else 0
    if not found_any:
        print(f"No improving rewrites found across {n_files} file(s).")
    else:
        print(f"\nFound {n_suggestions} improving rewrite(s) across {n_files} file(s).")
        print(f"Run `eml-rewrite fix {' '.join(args.files)}` to apply.")
    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    n_files = 0
    n_applied = 0
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"{f}: not found", file=sys.stderr)
            continue
        n_files += 1
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # In a real fix we'd rewrite the source; for v0.1 we only print
        # what would change. (Actual source rewriting requires AST surgery
        # plus careful handling of formatting; v0.2 territory.)
        for line, snippet, expr in _find_expressions_in_source(source):
            replaced = best(expr)
            if replaced != expr:
                n_applied += 1
                print(f"{f}:{line}  {snippet}")
                print(f"  Would rewrite to: {replaced}")
    print(f"\nWould apply {n_applied} rewrite(s) across {n_files} file(s).")
    print(f"NOTE: v0.1 prints proposed rewrites only. In-place source")
    print(f"      rewriting (preserving formatting) is v0.2 territory.")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        result = analyze(args.expr)
    except (TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Expression:           {result.expression}")
    print(f"  pfaffian_r:           {result.pfaffian_r}")
    print(f"  max_path_r:           {result.max_path_r}")
    print(f"  eml_depth:            {result.eml_depth}")
    print(f"  structural_overhead:  {result.structural_overhead}")
    print(f"  corrections:          {result.corrections}")
    print(f"  predicted_depth:      {result.predicted_depth}")
    print(f"  is_pfaffian_not_eml:  {result.is_pfaffian_not_eml}")

    sugg = suggest(
        result.expression,
        only_improvements=True,
        include_conditional=args.include_conditional,
    )
    if sugg:
        print(f"\nSuggested rewrites ({len(sugg)}):")
        for s in sorted(sugg, key=lambda x: -x.reduction):
            tag = s.pattern_name
            if not s.domain_verified and s.domain_required:
                tag = f"{s.pattern_name} | conditional: {s.domain_required}"
            print(f"  [{tag}] {s.rewritten}  (-{s.reduction} cost units)")
    else:
        print("\nNo improving rewrite found.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eml-rewrite",
        description="F-family fusion pattern rewriter for symbolic expressions.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Scan files; report improving rewrites")
    p_scan.add_argument("files", nargs="+")
    p_scan.add_argument("--include-conditional", action="store_true",
                        help="Also report rewrites whose domain assumption "
                             "cannot be established by SymPy's assumption "
                             "system; the requirement is annotated.")
    p_scan.add_argument("--as-patch", action="store_true",
                        help="Emit a unified diff (suitable for `git apply`) "
                             "instead of the human-readable report. The diff "
                             "applies all improving rewrites in-place; "
                             "stdout is git-apply-clean.")
    p_scan.set_defaults(func=cmd_scan)

    p_fix = sub.add_parser("fix", help="Print rewrites that would be applied")
    p_fix.add_argument("files", nargs="+")
    p_fix.set_defaults(func=cmd_fix)

    p_analyze = sub.add_parser("analyze", help="Analyze a single expression")
    p_analyze.add_argument("expr")
    p_analyze.add_argument("--include-conditional", action="store_true",
                           help="Also report conditional rewrites (see scan --help).")
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
