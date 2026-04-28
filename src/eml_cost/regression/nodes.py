"""Tree representation for EML-native symbolic regression.

An ``EMLNode`` is one of:

  - **variable** — leaf naming a column of the input (e.g. ``"x"``).
  - **constant** — leaf with a real-valued ``value``.
  - **unary** — one-argument operator (sin, cos, exp, log, sqrt, neg).
  - **binary** — two-argument operator (+, -, *, /).

Trees expose three core operations:

  - :meth:`evaluate` for fast numpy-backed scoring against data.
  - :meth:`to_sympy` for handing the candidate to
    :func:`eml_cost.canonicalize` / :func:`eml_cost.regularize`.
  - :meth:`copy` for safe genetic operators.

Numerical safety
----------------

``evaluate`` returns ``np.full_like(..., np.nan)`` for any branch
that overflows, divides by zero, or feeds a non-positive argument
to ``log`` / ``sqrt``. The GP engine treats a NaN-producing
candidate as having infinite MSE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sympy as sp


__all__ = [
    "EMLNode",
    "BINARY_OPS",
    "UNARY_OPS",
    "TERMINALS_DEFAULT_CONSTS",
    "random_tree",
    "all_subtree_indices",
]


# Public op vocabularies — kept narrow on purpose. Adding more
# operators expands the search space combinatorially.
BINARY_OPS: tuple[str, ...] = ("+", "-", "*", "/")
UNARY_OPS: tuple[str, ...] = ("sin", "cos", "exp", "log", "sqrt", "neg")

# Constants the random-tree generator picks from. The numeric range
# matters: a GP that has never seen a constant near pi will have to
# discover it via mutation, which is hard.
TERMINALS_DEFAULT_CONSTS: tuple[float, ...] = (
    -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, math.pi, math.e,
)


@dataclass
class EMLNode:
    """A node in an EML symbolic-regression tree.

    Attributes
    ----------
    kind:
        ``"var"`` | ``"const"`` | ``"unary"`` | ``"binary"``.
    op:
        Operator name when ``kind`` is ``"unary"`` or ``"binary"``.
        ``None`` for terminals.
    value:
        Numeric value when ``kind == "const"``. ``None`` otherwise.
    name:
        Variable name when ``kind == "var"``. ``None`` otherwise.
    children:
        Argument list. Length 0 (terminal), 1 (unary), or 2 (binary).
    """

    kind: str
    op: Optional[str] = None
    value: Optional[float] = None
    name: Optional[str] = None
    children: list["EMLNode"] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def var(cls, name: str) -> "EMLNode":
        return cls(kind="var", name=name)

    @classmethod
    def const(cls, value: float) -> "EMLNode":
        return cls(kind="const", value=float(value))

    @classmethod
    def unary(cls, op: str, child: "EMLNode") -> "EMLNode":
        if op not in UNARY_OPS:
            raise ValueError(f"unknown unary op: {op!r}")
        return cls(kind="unary", op=op, children=[child])

    @classmethod
    def binary(cls, op: str, left: "EMLNode", right: "EMLNode") -> "EMLNode":
        if op not in BINARY_OPS:
            raise ValueError(f"unknown binary op: {op!r}")
        return cls(kind="binary", op=op, children=[left, right])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, env: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the tree against an env mapping variable names
        to numpy arrays. Numerical errors → NaN-filled output."""
        with np.errstate(all="ignore"):
            return self._eval(env)

    def _eval(self, env: dict[str, np.ndarray]) -> np.ndarray:
        if self.kind == "var":
            arr = env.get(self.name)
            if arr is None:
                raise KeyError(f"unbound variable {self.name!r}")
            return np.asarray(arr, dtype=float)
        if self.kind == "const":
            sample = next(iter(env.values()))
            return np.full(np.asarray(sample).shape, float(self.value),
                           dtype=float)

        if self.kind == "unary":
            a = self.children[0]._eval(env)
            return _apply_unary(self.op, a)

        if self.kind == "binary":
            l = self.children[0]._eval(env)
            r = self.children[1]._eval(env)
            return _apply_binary(self.op, l, r)

        raise ValueError(f"unknown node kind {self.kind!r}")

    # ------------------------------------------------------------------
    # SymPy bridge
    # ------------------------------------------------------------------
    def to_sympy(self) -> sp.Expr:
        if self.kind == "var":
            return sp.Symbol(self.name, real=True)
        if self.kind == "const":
            v = float(self.value)
            if math.isclose(v, math.pi, rel_tol=1e-9):
                return sp.pi
            if math.isclose(v, math.e, rel_tol=1e-9):
                return sp.E
            return sp.Float(v)
        if self.kind == "unary":
            inner = self.children[0].to_sympy()
            if self.op == "sin":
                return sp.sin(inner)
            if self.op == "cos":
                return sp.cos(inner)
            if self.op == "exp":
                return sp.exp(inner)
            if self.op == "log":
                return sp.log(inner)
            if self.op == "sqrt":
                return sp.sqrt(inner)
            if self.op == "neg":
                return -inner
        if self.kind == "binary":
            l = self.children[0].to_sympy()
            r = self.children[1].to_sympy()
            if self.op == "+":
                return l + r
            if self.op == "-":
                return l - r
            if self.op == "*":
                return l * r
            if self.op == "/":
                return l / r
        raise ValueError(f"cannot convert {self.kind}/{self.op}")

    # ------------------------------------------------------------------
    # Tree manipulation
    # ------------------------------------------------------------------
    def copy(self) -> "EMLNode":
        return EMLNode(
            kind=self.kind,
            op=self.op,
            value=self.value,
            name=self.name,
            children=[c.copy() for c in self.children],
        )

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def __repr__(self) -> str:
        return f"<EMLNode {self._readable()}>"

    def _readable(self) -> str:
        if self.kind == "var":
            return self.name or "?"
        if self.kind == "const":
            return f"{self.value:.3g}"
        if self.kind == "unary":
            return f"{self.op}({self.children[0]._readable()})"
        if self.kind == "binary":
            return (f"({self.children[0]._readable()}"
                    f"{self.op}{self.children[1]._readable()})")
        return "?"


# ---------------------------------------------------------------------------
# Operator dispatch — kept module-private so EMLNode stays slim.
# ---------------------------------------------------------------------------


def _apply_unary(op: str, a: np.ndarray) -> np.ndarray:
    if op == "sin":
        return np.sin(a)
    if op == "cos":
        return np.cos(a)
    if op == "exp":
        # Clip to keep float overflow at bay.
        return np.exp(np.clip(a, -50, 50))
    if op == "log":
        out = np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), np.nan)
        return out
    if op == "sqrt":
        out = np.where(a >= 0, np.sqrt(np.where(a >= 0, a, 0.0)), np.nan)
        return out
    if op == "neg":
        return -a
    raise ValueError(f"unknown unary op {op!r}")


def _apply_binary(op: str, l: np.ndarray, r: np.ndarray) -> np.ndarray:
    if op == "+":
        return l + r
    if op == "-":
        return l - r
    if op == "*":
        return l * r
    if op == "/":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(np.abs(r) > 1e-12,
                           l / np.where(np.abs(r) > 1e-12, r, 1.0),
                           np.nan)
            return out
    raise ValueError(f"unknown binary op {op!r}")


# ---------------------------------------------------------------------------
# Random tree generation (used by GP init + mutation)
# ---------------------------------------------------------------------------


def random_tree(
    rng,
    var_names: list[str],
    *,
    max_depth: int = 4,
    method: str = "grow",
    p_terminal_at_leaf: float = 1.0,
    p_terminal_internal: float = 0.3,
    constants: tuple[float, ...] = TERMINALS_DEFAULT_CONSTS,
    p_unary: float = 0.4,
    allowed_unary: tuple[str, ...] = UNARY_OPS,
) -> EMLNode:
    """Generate a random EML expression tree.

    ``method="grow"`` is the standard Koza grow: at each non-leaf
    position, choose a terminal with probability
    ``p_terminal_internal``; otherwise pick an operator. ``"full"``
    forces internal nodes everywhere until ``max_depth``.
    """
    if max_depth <= 1 or (
            method == "grow"
            and rng.random() < p_terminal_internal):
        if rng.random() < 0.5:
            return EMLNode.var(rng.choice(var_names))
        return EMLNode.const(rng.choice(constants))

    if rng.random() < p_unary:
        op = rng.choice(allowed_unary)
        child = random_tree(
            rng, var_names,
            max_depth=max_depth - 1,
            method=method,
            p_terminal_at_leaf=p_terminal_at_leaf,
            p_terminal_internal=p_terminal_internal,
            constants=constants,
            p_unary=p_unary,
            allowed_unary=allowed_unary,
        )
        return EMLNode.unary(op, child)

    op = rng.choice(BINARY_OPS)
    left = random_tree(
        rng, var_names,
        max_depth=max_depth - 1,
        method=method,
        p_terminal_at_leaf=p_terminal_at_leaf,
        p_terminal_internal=p_terminal_internal,
        constants=constants,
        p_unary=p_unary,
        allowed_unary=allowed_unary,
    )
    right = random_tree(
        rng, var_names,
        max_depth=max_depth - 1,
        method=method,
        p_terminal_at_leaf=p_terminal_at_leaf,
        p_terminal_internal=p_terminal_internal,
        constants=constants,
        p_unary=p_unary,
        allowed_unary=allowed_unary,
    )
    return EMLNode.binary(op, left, right)


def all_subtree_indices(root: EMLNode) -> list[list[int]]:
    """Return every reachable subtree path from ``root`` as a list
    of child-index lists. Path ``[]`` is the root, ``[0]`` is the
    first child, ``[0, 1]`` is the right child of the first child,
    and so on.
    """
    out: list[list[int]] = [[]]
    stack: list[tuple[EMLNode, list[int]]] = [(root, [])]
    while stack:
        node, path = stack.pop()
        for i, c in enumerate(node.children):
            sub_path = path + [i]
            out.append(sub_path)
            stack.append((c, sub_path))
    return out


def get_subtree(root: EMLNode, path: list[int]) -> EMLNode:
    node = root
    for i in path:
        node = node.children[i]
    return node


def replace_subtree(
    root: EMLNode,
    path: list[int],
    replacement: EMLNode,
) -> EMLNode:
    """Return a copy of ``root`` with the subtree at ``path``
    replaced by ``replacement`` (deep-copied)."""
    if not path:
        return replacement.copy()
    new_root = root.copy()
    parent = new_root
    for i in path[:-1]:
        parent = parent.children[i]
    parent.children[path[-1]] = replacement.copy()
    return new_root
