"""Command-line interface for eml-cost.

Surfaces the prediction trilogy at the shell:

  - ``eml-cost report EXPR [EXPR ...]``
        Pretty-prints analyze + estimate_time + predict_precision_loss
        for each expression.

  - ``eml-cost report --json EXPR [EXPR ...]``
        Same, machine-readable JSON.

  - ``eml-cost check EXPR [EXPR ...] [--max-simplify-ms N]
                       [--max-digits-lost N]``
        Same as ``report`` but exits with a non-zero status if any
        expression exceeds the given budget(s). Pre-commit usable.

  - ``eml-cost report --file PATH``
        Read one expression per line from PATH (use ``-`` for stdin).

  - ``eml-cost version``
        Print package version + a one-line summary of shipped models.

Pre-commit recipe (project-local ``.pre-commit-config.yaml``)::

    - repo: local
      hooks:
        - id: eml-cost
          name: eml-cost numerical / compile-cost gate
          entry: eml-cost check --file
          language: system
          files: ^expressions\\.txt$
          args: [--max-simplify-ms, "200", --max-digits-lost, "3"]

The CLI is a thin shell around the existing public Python API; nothing
here changes the model coefficients.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Iterable, Sequence, TextIO

import sympy as sp

from . import __version__
from .analyze import analyze
from .estimate_time import estimate_time
from .predict_precision_loss import predict_precision_loss

__all__ = ["main"]

EXIT_OK = 0
EXIT_THRESHOLD_VIOLATION = 1
EXIT_PARSE_ERROR = 2
EXIT_USAGE = 64  # match BSD sysexits.h convention


def _read_expressions(file_arg: str) -> list[str]:
    """Read one expression per line from ``file_arg`` (``-`` = stdin).

    Blank lines and lines starting with ``#`` are skipped so the file
    can hold comments.
    """
    src: TextIO
    if file_arg == "-":
        src = sys.stdin
    else:
        src = open(file_arg, "r", encoding="utf-8")
    try:
        out: list[str] = []
        for line in src:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
        return out
    finally:
        if src is not sys.stdin:
            src.close()


def _analyze_one(expr_str: str) -> dict[str, object]:
    """Run analyze + estimate_time + predict_precision_loss on one expr.

    Returns a dict ready for JSON serialization or pretty-print.
    Raises sp.SympifyError on parse failure (caller decides how to handle).
    """
    expr = sp.sympify(expr_str)

    a = analyze(expr)
    t = estimate_time(expr)  # dict of TimeEstimate per proxy
    p = predict_precision_loss(expr)

    # estimate_time returns a dict[str, TimeEstimate]; collapse to JSON-able.
    assert isinstance(t, dict)  # for the type-checker; "all" path
    times = {
        proxy: {
            "predicted_ms": est.predicted_ms,
            "ci95_ms": list(est.ci95),
            "cv_r2": est.cv_r2,
        }
        for proxy, est in t.items()
    }

    return {
        "expr": expr_str,
        "pfaffian_r": int(a.pfaffian_r),
        "max_path_r": int(a.max_path_r),
        "eml_depth": int(a.eml_depth),
        "predicted_depth": int(a.predicted_depth),
        "is_pfaffian_not_eml": bool(a.is_pfaffian_not_eml),
        "estimate_time_ms": times,
        "predict_precision_loss": {
            "predicted_max_relerr": p.predicted_max_relerr,
            "predicted_digits_lost": p.predicted_digits_lost,
            "ci95": list(p.ci95),
            "cv_r2": p.cv_r2,
        },
    }


def _format_pretty(report: dict[str, object]) -> str:
    """Render one report dict as human-readable text."""
    times = report["estimate_time_ms"]
    pl = report["predict_precision_loss"]
    assert isinstance(times, dict)
    assert isinstance(pl, dict)

    lines: list[str] = []
    lines.append(f"=== {report['expr']} ===")
    lines.append(
        f"  pfaffian_r={report['pfaffian_r']}  "
        f"max_path_r={report['max_path_r']}  "
        f"eml_depth={report['eml_depth']}  "
        f"predicted_depth={report['predicted_depth']}  "
        f"PNE={'YES' if report['is_pfaffian_not_eml'] else 'no'}"
    )
    lines.append("  estimate_time (ms, with 95% CI):")
    for proxy in ("simplify", "factor", "cse", "lambdify"):
        t = times[proxy]
        lo, hi = t["ci95_ms"]
        lines.append(
            f"    {proxy:8s}  {t['predicted_ms']:>10.3f} ms"
            f"   [{lo:>9.3f},{hi:>11.3f}]   (CV R^2 {t['cv_r2']:.2f})"
        )
    lines.append("  predict_precision_loss:")
    lines.append(
        f"    relerr      {pl['predicted_max_relerr']:.2e}"
        f"   CI95 [{pl['ci95'][0]:.1e}, {pl['ci95'][1]:.1e}]"
    )
    lines.append(
        f"    digits_lost {pl['predicted_digits_lost']:.2f}"
        f"          (CV R^2 {pl['cv_r2']:.2f})"
    )
    return "\n".join(lines)


def _gather_exprs(args: argparse.Namespace) -> list[str]:
    exprs: list[str] = list(args.expr or [])
    if args.file:
        exprs.extend(_read_expressions(args.file))
    return exprs


def _check_thresholds(
    report: dict[str, object],
    *,
    max_simplify_ms: float | None,
    max_digits_lost: float | None,
) -> list[str]:
    """Return the list of threshold-violation messages for one report."""
    violations: list[str] = []
    if max_simplify_ms is not None:
        times = report["estimate_time_ms"]
        assert isinstance(times, dict)
        ms = float(times["simplify"]["predicted_ms"])
        if ms > max_simplify_ms:
            violations.append(
                f"simplify predicted {ms:.1f} ms > budget {max_simplify_ms:.1f} ms"
            )
    if max_digits_lost is not None:
        pl = report["predict_precision_loss"]
        assert isinstance(pl, dict)
        d = float(pl["predicted_digits_lost"])
        if d > max_digits_lost:
            violations.append(
                f"precision loss predicted {d:.2f} digits > budget {max_digits_lost:.2f}"
            )
    return violations


def _cmd_report(args: argparse.Namespace, *, out: TextIO) -> int:
    exprs = _gather_exprs(args)
    if not exprs:
        print("error: no expressions provided", file=sys.stderr)
        return EXIT_USAGE

    reports: list[dict[str, object]] = []
    parse_errors: list[tuple[str, str]] = []
    for s in exprs:
        try:
            reports.append(_analyze_one(s))
        except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
            parse_errors.append((s, str(exc)))

    if args.json:
        payload: dict[str, object] = {
            "version": __version__,
            "reports": reports,
            "parse_errors": [
                {"expr": pe_expr, "error": pe_msg}
                for pe_expr, pe_msg in parse_errors
            ],
        }
        json.dump(payload, out, indent=2)
        out.write("\n")
    else:
        for r in reports:
            out.write(_format_pretty(r))
            out.write("\n")
        for pe_expr, pe_msg in parse_errors:
            out.write(f"=== {pe_expr} ===\n  PARSE ERROR: {pe_msg}\n")

    if parse_errors and not args.json:
        return EXIT_PARSE_ERROR
    return EXIT_OK


def _cmd_check(args: argparse.Namespace, *, out: TextIO) -> int:
    exprs = _gather_exprs(args)
    if not exprs:
        print("error: no expressions provided", file=sys.stderr)
        return EXIT_USAGE

    any_violation = False
    any_parse_error = False
    for s in exprs:
        try:
            r = _analyze_one(s)
        except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
            any_parse_error = True
            print(f"PARSE ERROR  {s}: {exc}", file=sys.stderr)
            continue
        violations = _check_thresholds(
            r,
            max_simplify_ms=args.max_simplify_ms,
            max_digits_lost=args.max_digits_lost,
        )
        if violations:
            any_violation = True
            for v in violations:
                print(f"FAIL  {s}: {v}", file=sys.stderr)
        else:
            print(f"OK    {s}", file=out)

    if any_parse_error:
        return EXIT_PARSE_ERROR
    if any_violation:
        return EXIT_THRESHOLD_VIOLATION
    return EXIT_OK


def _cmd_version(args: argparse.Namespace, *, out: TextIO) -> int:
    out.write(f"eml-cost {__version__}\n")
    out.write(
        "  trilogy: estimate_time (compile-time, E-191) + "
        "predict_precision_loss (runtime numerics, E-193) + "
        "see eml-cost-torch.diagnose for activations\n"
    )
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eml-cost",
        description="Pfaffian/EML cost-analysis CLI (analyze + estimate_time + predict_precision_loss).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # report --------------------------------------------------------------
    pr = sub.add_parser("report", help="Print analysis + cost predictions for one or more expressions.")
    pr.add_argument("expr", nargs="*", help="SymPy expression(s); use --file - to read stdin.")
    pr.add_argument("--file", help="Read one expression per line from PATH (- for stdin).")
    pr.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text.")

    # check ---------------------------------------------------------------
    pc = sub.add_parser("check", help="Same as report but exit non-zero if budgets are exceeded.")
    pc.add_argument("expr", nargs="*", help="SymPy expression(s); use --file - to read stdin.")
    pc.add_argument("--file", help="Read one expression per line from PATH (- for stdin).")
    pc.add_argument(
        "--max-simplify-ms",
        type=float,
        default=None,
        help="Fail if predicted simplify ms > this budget.",
    )
    pc.add_argument(
        "--max-digits-lost",
        type=float,
        default=None,
        help="Fail if predicted decimal-digits lost > this budget.",
    )

    # lint ----------------------------------------------------------------
    pl = sub.add_parser(
        "lint",
        help=("Scan a Python file for sympy.simplify/factor/expand/cse/"
              "lambdify calls and predict their wall-time before they run."),
    )
    pl.add_argument("paths", nargs="+", help="Source file(s) to lint.")
    pl.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=("Exit non-zero if any predicted simplify/factor exceeds "
              "this wall-time budget. Pre-commit usable."),
    )
    pl.add_argument(
        "--severity",
        choices=("info", "warn", "error"),
        default="info",
        help=("Only report findings at or above this severity. "
              "info: < 0.2 s; warn: 0.2-5 s; error: > 5 s."),
    )

    # version -------------------------------------------------------------
    sub.add_parser("version", help="Print eml-cost version + trilogy summary.")

    return p


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    """Entry point. Returns the exit code (does not call sys.exit)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    target_out = out if out is not None else sys.stdout

    if args.cmd == "report":
        return _cmd_report(args, out=target_out)
    if args.cmd == "check":
        return _cmd_check(args, out=target_out)
    if args.cmd == "lint":
        return _cmd_lint(args, out=target_out)
    if args.cmd == "version":
        return _cmd_version(args, out=target_out)

    parser.print_help()
    return EXIT_USAGE


_SEV_ORDER = {"info": 0, "warn": 1, "error": 2}


def _cmd_lint(args: argparse.Namespace, *, out: TextIO) -> int:
    """eml-cost lint <file...> — pre-commit linter for SymPy compile cost."""
    from .lint import lint_file

    threshold = _SEV_ORDER[args.severity]
    over_budget = False
    any_findings = False

    for path in args.paths:
        for f in lint_file(path):
            if _SEV_ORDER.get(f.severity, 0) < threshold:
                continue
            any_findings = True
            print(f.message, file=out)
            if (args.max_seconds is not None
                    and f.predicted_seconds > args.max_seconds):
                over_budget = True
                print(
                    f"  -> exceeds --max-seconds={args.max_seconds:.2f}s "
                    f"(predicted {f.predicted_seconds:.2f}s)",
                    file=out,
                )
    if not any_findings:
        print("(no findings)", file=out)
    if over_budget:
        return EXIT_THRESHOLD_VIOLATION
    return EXIT_OK


def _entrypoint() -> None:
    """Console-script entry point — calls main() and propagates exit code."""
    sys.exit(main())


if __name__ == "__main__":
    _entrypoint()
