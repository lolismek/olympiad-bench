"""Scientific-calculator-equivalent tool for IPhO-style environments.

Scope chosen to mirror what a contestant has on the desk (TI-30, Casio fx-991):

  • Single-expression evaluation with arithmetic + common math functions.
  • Named variable slots ("memory"), usable inside subsequent expressions.
  • One linear-regression primitive (slope/intercept/r², slope stderr) — the
    TI-30 `LINREG` mode. This is the least-judgement-call fit; anything more
    (curve_fit, nonlinear optimization, matrix ops) would hand the agent the
    answer and is deliberately excluded.

Security: expressions are parsed with `ast.parse(mode="eval")` and every
node type is whitelisted. No attribute access, subscripts, comprehensions,
lambdas, keyword args, imports, or assignment. Function calls are
restricted to a closed list.
"""

from __future__ import annotations

import ast
import math
import operator as _op
from typing import Any

_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}

_UNARY_OPS = {ast.UAdd: _op.pos, ast.USub: _op.neg}

# Closed whitelist of callable names. No attribute access is permitted, so
# `math.sqrt` is unreachable; the only way to call sqrt is via this map.
_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "exp": math.exp,
    "log": math.log,       # natural log
    "ln": math.log,        # alias
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "floor": math.floor,
    "degrees": math.degrees,
    "radians": math.radians,
}

_CONSTANTS = {"pi": math.pi, "e": math.e}

_RESERVED = set(_ALLOWED_FUNCS) | set(_CONSTANTS)


class Calculator:
    """Stateful scientific calculator with named memory slots."""

    def __init__(self) -> None:
        self.variables: dict[str, float] = {}

    # -- public API ---------------------------------------------------------

    def evaluate(self, expr: str, store_as: str | None = None) -> dict[str, Any]:
        """Evaluate a single mathematical expression.

        Returns a dict `{"value": float, "stored_as": str | None}` on success,
        or `{"error": str}` on failure. The calculator is persistent; bindings
        made via `store_as` are available to subsequent `evaluate` calls.
        """
        # `^` is mathematician/calculator notation for exponentiation; in Python
        # it's XOR, which the calculator doesn't expose. Normalize up front so
        # the agent doesn't burn calls on a notation mismatch.
        if "^" in expr:
            expr = expr.replace("^", "**")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            return {"error": f"syntax error: {exc.msg}"}

        try:
            value = self._eval_node(tree.body)
        except ValueError as exc:
            return {"error": str(exc)}
        except ZeroDivisionError:
            return {"error": "division by zero"}
        except OverflowError:
            return {"error": "numeric overflow"}

        result: dict[str, Any] = {"value": float(value), "stored_as": None}
        if store_as is not None:
            if not isinstance(store_as, str) or not store_as.isidentifier():
                return {"error": f"invalid variable name: {store_as!r}"}
            if store_as in _RESERVED:
                return {"error": f"{store_as!r} shadows a built-in constant or function"}
            self.variables[store_as] = float(value)
            result["stored_as"] = store_as
        return result

    def reset(self) -> None:
        self.variables.clear()

    # -- AST walker ---------------------------------------------------------

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"only numeric literals allowed (got {node.value!r})")
            return float(node.value)

        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN_OPS:
                raise ValueError(
                    f"binary operator not allowed: {type(node.op).__name__} "
                    "(supported: + - * / // % **)"
                )
            return _BIN_OPS[type(node.op)](
                self._eval_node(node.left), self._eval_node(node.right)
            )

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPS:
                raise ValueError(
                    f"unary operator not allowed: {type(node.op).__name__}"
                )
            return _UNARY_OPS[type(node.op)](self._eval_node(node.operand))

        if isinstance(node, ast.Name):
            name = node.id
            if name in self.variables:
                return self.variables[name]
            if name in _CONSTANTS:
                return _CONSTANTS[name]
            if name in _ALLOWED_FUNCS:
                raise ValueError(f"{name!r} is a function; call it with parentheses")
            raise ValueError(f"unknown name: {name!r}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only direct function calls are allowed")
            fname = node.func.id
            if fname not in _ALLOWED_FUNCS:
                raise ValueError(f"function not allowed: {fname!r}")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            args = [self._eval_node(a) for a in node.args]
            return _ALLOWED_FUNCS[fname](*args)

        raise ValueError(
            f"disallowed expression element: {type(node).__name__}"
        )


# --- Linear regression --------------------------------------------------------

def linear_regression(xs: list[float], ys: list[float]) -> dict[str, float | int]:
    """Ordinary-least-squares fit of `y = slope · x + intercept`.

    Returns slope, intercept, r², standard errors, and n. Mirrors the
    `LINREG` primitive on TI-30 / Casio fx-991 calculators.
    """
    if not isinstance(xs, (list, tuple)) or not isinstance(ys, (list, tuple)):
        raise ValueError("xs and ys must be lists")
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"length mismatch: len(xs)={n}, len(ys)={len(ys)}")
    if n < 2:
        raise ValueError("need at least 2 points for linear regression")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    syy = sum((y - mean_y) ** 2 for y in ys)

    if sxx == 0:
        raise ValueError("all xs are identical; slope undefined")

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    r_squared = (sxy * sxy) / (sxx * syy) if syy > 0 else 1.0

    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rss = sum(r * r for r in residuals)
    if n > 2 and sxx > 0:
        s2 = rss / (n - 2)
        slope_se = math.sqrt(s2 / sxx)
        intercept_se = math.sqrt(s2 * (1.0 / n + mean_x * mean_x / sxx))
    else:
        slope_se = float("nan")
        intercept_se = float("nan")

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "slope_stderr": slope_se,
        "intercept_stderr": intercept_se,
        "n": n,
    }
