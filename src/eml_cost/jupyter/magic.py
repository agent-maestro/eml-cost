"""IPython cell magic ``%%eml_witness``.

In a notebook cell:

    %%eml_witness
    1 / (1 + sp.exp(-sp.Symbol('x')))

The cell body is evaluated as a Python expression in the user's
namespace. The result is then run through
:func:`eml_witness.universality_witness`, and the witness is
displayed below the cell as rich HTML.

For arbitrary code that produces a SymPy expression at the end,
use ``%%eml_witness_last`` (TODO: not implemented in v0.1).
"""
from __future__ import annotations

from typing import Any

import sympy as sp


__all__ = ["EmlMagics"]


def _render(obj: Any) -> Any:
    """Build an IPython display object for a witness around ``obj``."""
    from IPython.display import HTML

    from eml_witness import universality_witness

    if not isinstance(obj, sp.Basic):
        return HTML(    # type: ignore[no-untyped-call]
            f"<div style='color:#c66;font-family:sans-serif;'>"
            f"%%eml_witness: cell did not return a SymPy expression "
            f"(got <code>{type(obj).__name__}</code>)."
            f"</div>"
        )
    try:
        w = universality_witness(obj)
    except Exception as exc:
        return HTML(    # type: ignore[no-untyped-call]
            f"<div style='color:#c66;font-family:sans-serif;'>"
            f"%%eml_witness: universality_witness failed — "
            f"<code>{type(exc).__name__}: {exc}</code>"
            f"</div>"
        )
    from ._formatter import render_witness_html
    return HTML(render_witness_html(w))    # type: ignore[no-untyped-call]


def _make_magics_class() -> Any:
    """Construct the EmlMagics class lazily so we don't require
    IPython at import time of this module."""
    from IPython.core.magic import Magics, cell_magic, magics_class

    @magics_class
    class _EmlMagics(Magics):
        @cell_magic
        def eml_witness(self, _line: str, cell: str) -> Any:    # noqa: D401
            """Evaluate the cell as an expression and show its EML witness."""
            if self.shell is None:
                # Defensive — happens only when the magic class is
                # instantiated outside an IPython shell. assert would
                # be stripped under `python -O`, hence explicit raise.
                raise RuntimeError(
                    "%%eml_witness requires an active IPython shell."
                )
            obj = self.shell.ev(cell.strip())   # type: ignore[no-untyped-call]
            return _render(obj)

    return _EmlMagics


# Lazy class — instantiated only when register_magics is called.
def __getattr__(name: str) -> Any:    # pragma: no cover — IPython-only path
    if name == "EmlMagics":
        return _make_magics_class()
    raise AttributeError(name)
