"""eml-witness — universality witnesses for elementary functions.

A *witness* for an expression is a finite, structured certificate
that the expression admits an EML routing tree:

    >>> from eml_witness import universality_witness
    >>> w = universality_witness("1/(1+exp(-x))")
    >>> w.profile.predicted_depth
    2
    >>> w.identified.name
    'sigmoid (canonical)'
    >>> w.verified_in_lean
    False                       # flips to True once EML_Universality.lean
                                # lands AND the user verifies in VS Code

The witness composes the existing substrate:

  - :func:`eml_cost.analyze` + :func:`eml_cost.fingerprint` — the
    Pfaffian profile (chain order, depth, corrections, fingerprint).
  - :func:`eml_discover.identify` — the closest registry match.
  - (optional) :func:`eml_rewrite.path` — the rewrite sequence
    walking the input to a canonical equivalent, if one exists.

The witness object is plain data (frozen dataclass + Pydantic-
free); serialise to JSON via :func:`witness_to_dict`.

For the Lean half of the story see
``D:/monogate-lean/MonogateEML/Universality.lean`` (planned).
``verified_in_lean`` is the link between the two halves; the
default ``False`` flips to ``True`` once the Lean file is
user-verified per ``feedback_lean_writing_protocol_2026_04_25``.
"""
from __future__ import annotations

from .core import (
    UniversalityWitness,
    WitnessIdentified,
    WitnessProfile,
    WitnessTreeNode,
    universality_witness,
    witness_to_dict,
)

__version__ = "0.2.1"

__all__ = [
    "__version__",
    "universality_witness",
    "witness_to_dict",
    "UniversalityWitness",
    "WitnessProfile",
    "WitnessIdentified",
    "WitnessTreeNode",
]
