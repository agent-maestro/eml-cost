"""eml-explore — cross-domain explorer CLI for the EML substrate.

Subcommands:

    eml-explore witness EXPR      pretty-print the universality witness
    eml-explore analyze EXPR      Pfaffian profile only
    eml-explore identify EXPR     registry matches only
    eml-explore class AXES        every member of an equivalence class
    eml-explore corpus FILE       cluster a file of one-expr-per-line
    eml-explore example NAME      run a built-in demo (cross-domain, ...)

The CLI is a thin wrapper over `eml_witness.universality_witness` +
the existing eml-* APIs. It produces both human-readable text
output and `--json` output for downstream pipelines.
"""
from __future__ import annotations

from ._version import __version__
from .cli import main

__all__ = ["__version__", "main"]
