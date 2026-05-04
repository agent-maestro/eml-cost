"""Tests for the eml-jupyter display formatters.

We exercise the rendering helpers directly (no IPython required)
plus the IPython integration when available.
"""
from __future__ import annotations

import re

import pytest
import sympy as sp

from eml_cost.jupyter import (
    install_display_formatter,
    render_witness_html,
    render_witness_text,
    uninstall_display_formatter,
)
from eml_cost.witness import universality_witness


x = sp.Symbol("x")


def test_render_witness_text_includes_depth():
    w = universality_witness("1/(1+exp(-x))")
    s = render_witness_text(w)
    assert s.startswith("[EML")
    assert s.endswith("]")
    assert "d=" in s
    assert "r=" in s


def test_render_witness_text_includes_identification_when_present():
    w = universality_witness("1/(1+exp(-x))")
    s = render_witness_text(w)
    assert "id:" in s


def test_render_witness_text_includes_lean_when_verified():
    """Post-Universality.lean verification (eml-witness 0.2.0+),
    sin(x) — an EML-class expression — gets the Lean-verified
    badge in its one-line summary."""
    w = universality_witness("sin(x)")
    s = render_witness_text(w)
    assert "Lean-verified" in s


def test_render_witness_text_omits_lean_for_pfaffian_not_eml():
    """Bessel is OUTSIDE the EML class — the one-line summary
    must NOT carry the Lean-verified badge."""
    import sympy as sp_local
    w = universality_witness(sp_local.besselj(0, x))
    s = render_witness_text(w)
    assert "Lean-verified" not in s


def test_render_witness_html_returns_div():
    w = universality_witness("sin(x)")
    h = render_witness_html(w)
    assert h.startswith("<div")
    assert h.rstrip().endswith("</div>")


def test_render_witness_html_includes_input_expr():
    w = universality_witness("exp(sin(x))")
    h = render_witness_html(w)
    assert "exp(sin(x))" in h


def test_render_witness_html_includes_pfaffian_axes():
    w = universality_witness("sin(x)")
    h = render_witness_html(w)
    assert "fingerprint" in h
    assert "axes" in h
    # Axes string of the form p<n>-d<n>-w<n>-c<n>
    assert re.search(r"p\d+-d\d+-w\d+-c-?\d+", h) is not None


def test_render_witness_html_shows_lean_verified_for_eml_class():
    """sin(x) is in the EML class — HTML pill should be the
    'Lean-verified · source' green variant, not 'pending'."""
    w = universality_witness("sin(x)")
    h = render_witness_html(w)
    assert "Lean-verified" in h
    assert "pending" not in h


def test_render_witness_html_shows_lean_pending_for_pfaffian_not_eml():
    """Bessel — outside the EML class — keeps the pending pill."""
    import sympy as sp_local
    w = universality_witness(sp_local.besselj(0, x))
    h = render_witness_html(w)
    assert "pending" in h


def test_render_witness_html_escapes_ampersand_in_expression():
    """Defence against an expression that somehow contains an
    HTML-significant character via str()."""
    # SymPy strs don't normally contain &, but escape correctness
    # is worth confirming via the internal helper.
    from eml_cost.jupyter._formatter import _escape
    assert _escape("a & b") == "a &amp; b"
    assert _escape("<x>") == "&lt;x&gt;"


def test_render_witness_html_high_severity_color_on_bessel():
    """Bessel triggers Pfaffian-not-EML → high severity (deep red border)."""
    w = universality_witness(sp.besselj(0, x))
    h = render_witness_html(w)
    assert "5a2222" in h    # high-severity bg color


def test_render_witness_html_low_severity_color_on_canonical_sigmoid():
    w = universality_witness("1/(1+exp(-x))")
    h = render_witness_html(w)
    assert "1f3a1f" in h    # low-severity bg color


@pytest.mark.skip(reason="requires eml_rewrite — not yet published")
def test_render_witness_html_includes_canonical_path_for_textbook_sigmoid():
    w = universality_witness(sp.exp(x) / (1 + sp.exp(x)))
    h = render_witness_html(w)
    assert "canonical path" in h
    assert "savings" in h


# ---------- IPython integration (optional) ----------


def _has_ipython() -> bool:
    try:
        import IPython   # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_ipython(), reason="IPython not installed")
def test_install_uninstall_idempotent():
    """Installing twice + uninstalling cleanly should not error."""
    from IPython.testing.globalipapp import get_ipython

    ip = get_ipython()
    install_display_formatter(ip)
    install_display_formatter(ip)   # idempotent
    uninstall_display_formatter(ip)
    uninstall_display_formatter(ip)   # idempotent


@pytest.mark.skipif(not _has_ipython(), reason="IPython not installed")
def test_text_formatter_decorates_sympy_basic():
    """Format dispatch via the high-level DisplayFormatter.format()
    interface, which is what IPython actually invokes for cell outputs."""
    from IPython.testing.globalipapp import get_ipython

    ip = get_ipython()
    install_display_formatter(ip)
    try:
        format_dict, _ = ip.display_formatter.format(sp.sin(x))
        text = format_dict.get("text/plain", "")
        assert "[EML" in text
    finally:
        uninstall_display_formatter(ip)


@pytest.mark.skipif(not _has_ipython(), reason="IPython not installed")
def test_html_formatter_decorates_sympy_basic():
    from IPython.testing.globalipapp import get_ipython

    ip = get_ipython()
    install_display_formatter(ip)
    try:
        format_dict, _ = ip.display_formatter.format(sp.sin(x))
        html = format_dict.get("text/html", "")
        assert "<div" in html
        assert "fingerprint" in html
    finally:
        uninstall_display_formatter(ip)


@pytest.mark.skipif(not _has_ipython(), reason="IPython not installed")
def test_load_ipython_extension_registers_magic():
    from IPython.testing.globalipapp import get_ipython

    from eml_cost.jupyter import load_ipython_extension, unload_ipython_extension

    ip = get_ipython()
    load_ipython_extension(ip)
    try:
        # The magic manager should know about eml_witness.
        assert "eml_witness" in ip.magics_manager.magics["cell"]
    finally:
        unload_ipython_extension(ip)
