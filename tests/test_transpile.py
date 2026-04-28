"""Tests for ``eml_cost.transpile`` — Tool 5 of the LLM-native build.

The transpiler must:
  - Produce code that executes cleanly (the auto-generated
    verification snippet self-checks against a sample point).
  - Surface chain order, node count, and cost class in the result.
  - Auto-detect free symbols and order them alphabetically.
  - Handle constant expressions (no free symbols) without crashing.
  - Emit C output that reproduces the same expression in libm form.
"""
from __future__ import annotations

import math

import pytest
import sympy as sp

from eml_cost import (
    TranspileResult,
    eml_tree_to_c,
    eml_tree_to_numpy,
    eml_tree_to_python,
    eml_tree_to_sympy,
)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_python_returns_transpile_result(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_python(sp.sin(x))
        assert isinstance(r, TranspileResult)
        assert r.target == "python"
        assert r.function_name == "f"
        assert r.var_names == ("x",)

    def test_result_is_frozen(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_python(sp.sin(x))
        with pytest.raises(Exception):
            r.target = "numpy"  # type: ignore[misc]

    def test_chain_order_and_cost_class_populated(self) -> None:
        x, t, omega = sp.symbols("x t omega")
        expr = sp.exp(x) * sp.cos(omega * t)
        r = eml_tree_to_python(expr)
        assert r.chain_order >= 1
        assert r.node_count >= 1
        # Cost class follows the canonical ``p{r}-d{deg}-w{w}-c{c}`` form.
        assert r.cost_class.startswith("p")


# ---------------------------------------------------------------------------
# Variable detection
# ---------------------------------------------------------------------------


class TestVariableDetection:
    def test_free_symbols_alphabetised(self) -> None:
        x, t, omega = sp.symbols("x t omega")
        expr = sp.exp(x) * sp.cos(omega * t)
        r = eml_tree_to_python(expr)
        assert r.var_names == ("omega", "t", "x")

    def test_user_supplied_var_names_respected(self) -> None:
        x, t = sp.symbols("x t")
        r = eml_tree_to_python(sp.sin(x) + sp.cos(t), var_names=("t", "x"))
        assert r.var_names == ("t", "x")

    def test_no_free_symbols_constant(self) -> None:
        r = eml_tree_to_python(sp.Integer(2) + sp.Integer(3))
        assert r.var_names == ()
        assert "def f():" in r.code


# ---------------------------------------------------------------------------
# Code generation — executes cleanly
# ---------------------------------------------------------------------------


class TestPythonTarget:
    def test_sine_recovers_value(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_python(sp.sin(x))
        ns: dict = {}
        exec(r.full_source(), ns)
        # Verification didn't raise → assertion passed inside the exec.
        # Spot-check a different point.
        assert math.isclose(ns["f"](0.5), math.sin(0.5), abs_tol=1e-12)

    def test_exp_decay_recovers_value(self) -> None:
        x = sp.Symbol("x")
        expr = sp.exp(-x / 2)
        r = eml_tree_to_python(expr)
        ns: dict = {}
        exec(r.full_source(), ns)
        assert math.isclose(ns["f"](2.0), math.exp(-1.0), abs_tol=1e-12)

    def test_imports_are_minimal(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_python(sp.sin(x))
        assert r.imports == "import math"


class TestNumpyTarget:
    def test_uses_np_alias(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_numpy(sp.sin(x))
        assert "import numpy as np" in r.imports
        assert "np.sin" in r.code
        # Should NOT leak the fully qualified ``numpy.X`` form.
        assert "numpy.sin" not in r.code

    def test_executes_on_array_input(self) -> None:
        np = pytest.importorskip("numpy")
        x = sp.Symbol("x")
        r = eml_tree_to_numpy(sp.sin(x))
        ns: dict = {}
        exec(r.full_source(), ns)
        arr = np.array([0.0, 0.5, 1.0])
        result = ns["f"](arr)
        assert np.allclose(result, np.sin(arr))


class TestSympyTarget:
    def test_recreates_same_expression(self) -> None:
        x, t, omega = sp.symbols("x t omega")
        expr = sp.exp(x) * sp.cos(omega * t)
        r = eml_tree_to_sympy(expr)
        ns: dict = {}
        exec(r.full_source(), ns)
        assert ns["expr"] == expr

    def test_recreated_is_differentiable(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_sympy(sp.sin(x))
        ns: dict = {}
        exec(r.full_source(), ns)
        # diff w.r.t. the recreated local x.
        diff = ns["expr"].diff(ns["x"])
        assert diff == sp.cos(ns["x"])


class TestCTarget:
    def test_emits_math_h(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_c(sp.sin(x))
        assert "#include <math.h>" in r.imports
        # Forward-bridge note for libmonogate.h
        assert "libmonogate.h" in r.imports

    def test_function_signature_uses_double(self) -> None:
        x, y = sp.symbols("x y")
        r = eml_tree_to_c(sp.exp(x) + y)
        assert "double f(double x, double y)" in r.code

    def test_void_arg_for_constant(self) -> None:
        r = eml_tree_to_c(sp.Integer(7))
        assert "double f(void)" in r.code

    def test_verification_is_a_comment(self) -> None:
        # Cannot exec C from Python, so the verification slot must
        # be a C comment (not an assertion).
        x = sp.Symbol("x")
        r = eml_tree_to_c(sp.sin(x))
        if r.verification:
            assert r.verification.lstrip().startswith("/*")


# ---------------------------------------------------------------------------
# Sample-point + verification semantics
# ---------------------------------------------------------------------------


class TestVerification:
    def test_sample_point_avoids_zero(self) -> None:
        # Zero on a multiplicative expression would be a meaningless
        # success; the sampler picks 1.0, 1.5, 2.0, ...
        x = sp.Symbol("x")
        r = eml_tree_to_python(sp.sin(x))
        assert r.sample_point["x"] == 1.0
        assert r.expected_value is not None

    def test_diverging_expression_skips_verification(self) -> None:
        # 1/(x-1) at sample point x=1.0 diverges → expected_value is None,
        # verification snippet is empty rather than asserting NaN/inf.
        x = sp.Symbol("x")
        r = eml_tree_to_python(1 / (x - 1))
        assert r.expected_value is None
        assert r.verification == ""

    def test_verification_self_executes(self) -> None:
        x, t = sp.symbols("x t")
        r = eml_tree_to_python(sp.exp(x) * sp.sin(t))
        # full_source includes the verification assertion; if it fails,
        # exec raises AssertionError. If it raises here the transpiler
        # has a bug.
        ns: dict = {}
        exec(r.full_source(), ns)
        # Reaching this line means the auto-verification passed.

    def test_full_source_concatenates(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_python(sp.sin(x))
        full = r.full_source()
        assert "import math" in full
        assert "def f(x):" in full
        assert "math.sin(x)" in full


# ---------------------------------------------------------------------------
# Function naming
# ---------------------------------------------------------------------------


class TestFunctionName:
    def test_custom_function_name(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_python(sp.sin(x), function_name="oscillator")
        assert r.function_name == "oscillator"
        assert "def oscillator(x):" in r.code

    def test_sympy_target_default_name_is_expr(self) -> None:
        x = sp.Symbol("x")
        r = eml_tree_to_sympy(sp.sin(x))
        assert r.function_name == "expr"
        assert "expr =" in r.code
