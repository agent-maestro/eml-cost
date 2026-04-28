"""eml-explore — argparse-based CLI for the EML substrate.

The bulk of the work lives in :mod:`eml_explore.commands`; this
module only wires the argparser. Every subcommand accepts
``--json`` to emit a machine-readable JSON envelope.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ._version import __version__
from . import commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eml-explore",
        description=(
            "Cross-domain explorer for the EML substrate. "
            "Compose fingerprint, identify, path, build_graph, "
            "and universality_witness through one CLI."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"eml-explore {__version__}",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON output instead of human-readable text.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_w = sub.add_parser("witness",  help="Print universality witness for an expression.")
    p_w.add_argument("expr", help="SymPy-parseable expression string.")
    p_w.add_argument("--no-walk", action="store_true",
                     help="Skip the canonical-equivalent walk (faster).")
    p_w.set_defaults(func=commands.cmd_witness)

    p_a = sub.add_parser("analyze",  help="Pfaffian profile of an expression.")
    p_a.add_argument("expr")
    p_a.set_defaults(func=commands.cmd_analyze)

    p_i = sub.add_parser("identify", help="Registry matches for an expression.")
    p_i.add_argument("expr")
    p_i.add_argument("--max", type=int, default=5,
                     help="Max matches to return (default 5).")
    p_i.set_defaults(func=commands.cmd_identify)

    p_c = sub.add_parser("class",    help="Members of a Pfaffian-axes equivalence class.")
    p_c.add_argument("axes", help='Axes string like "p1-d2-w1-c0".')
    p_c.set_defaults(func=commands.cmd_class)

    p_corpus = sub.add_parser("corpus", help="Cluster a file of one-expression-per-line.")
    p_corpus.add_argument("file", help="Path to a text file with one SymPy expression per line.")
    p_corpus.add_argument("--top", type=int, default=10,
                          help="Number of largest clusters to print (default 10).")
    p_corpus.set_defaults(func=commands.cmd_corpus)

    p_e = sub.add_parser("example", help="Run a built-in demo.")
    p_e.add_argument("name", choices=sorted(commands.EXAMPLES.keys()),
                     help="Which built-in example to run.")
    p_e.set_defaults(func=commands.cmd_example)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("aborted.", file=sys.stderr)
        return 130
    except Exception as exc:    # noqa: BLE001 — top-level CLI handler
        if args.json:
            import json
            print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
