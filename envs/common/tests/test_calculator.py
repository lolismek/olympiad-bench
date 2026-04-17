"""Calculator + linreg unit tests.

Coverage: basic arithmetic, math functions, variables, error handling,
security (disallowed expression nodes), and linear regression correctness.
"""

from __future__ import annotations

import math

import pytest

from envs.common.calculator import Calculator, linear_regression


# --- basic arithmetic -------------------------------------------------------

def test_arithmetic():
    c = Calculator()
    assert c.evaluate("1 + 2 * 3")["value"] == 7.0
    assert c.evaluate("(1 + 2) * 3")["value"] == 9.0
    assert c.evaluate("2 ** 10")["value"] == 1024.0
    assert c.evaluate("7 / 2")["value"] == 3.5
    assert c.evaluate("7 // 2")["value"] == 3.0
    assert c.evaluate("7 % 2")["value"] == 1.0
    assert c.evaluate("-5")["value"] == -5.0
    assert c.evaluate("+5")["value"] == 5.0


def test_caret_normalized_to_power():
    """`^` is mathematician/calculator notation for exponentiation."""
    c = Calculator()
    assert c.evaluate("3^2")["value"] == 9.0
    assert c.evaluate("2^10")["value"] == 1024.0
    # Mixed with parens.
    assert c.evaluate("(1+1)^3")["value"] == 8.0


def test_math_functions():
    c = Calculator()
    assert c.evaluate("sqrt(16)")["value"] == 4.0
    assert c.evaluate("exp(0)")["value"] == 1.0
    assert c.evaluate("log(e)")["value"] == pytest.approx(1.0)
    assert c.evaluate("log10(1000)")["value"] == pytest.approx(3.0)
    assert c.evaluate("ln(e ** 2)")["value"] == pytest.approx(2.0)
    assert c.evaluate("sin(pi / 2)")["value"] == pytest.approx(1.0)
    assert c.evaluate("cos(0)")["value"] == 1.0
    assert c.evaluate("abs(-3.5)")["value"] == 3.5
    assert c.evaluate("round(2.7)")["value"] == 3.0


# --- variables --------------------------------------------------------------

def test_variable_storage_and_reuse():
    c = Calculator()
    r = c.evaluate("1546.0", store_as="tau1")
    assert r["value"] == 1546.0 and r["stored_as"] == "tau1"
    r2 = c.evaluate("exp(-100 / tau1)")
    assert r2["value"] == pytest.approx(math.exp(-100 / 1546.0))


def test_variable_overwrites():
    c = Calculator()
    c.evaluate("3.0", store_as="x")
    c.evaluate("5.0", store_as="x")
    assert c.evaluate("x")["value"] == 5.0


def test_invalid_variable_name_rejected():
    c = Calculator()
    assert "error" in c.evaluate("1.0", store_as="123foo")
    assert "error" in c.evaluate("1.0", store_as="has space")
    # Shadowing a built-in constant or function is rejected.
    assert "error" in c.evaluate("42", store_as="pi")
    assert "error" in c.evaluate("42", store_as="sqrt")


# --- security / disallowed constructs --------------------------------------

@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",            # dunder / import
        "x.y",                         # attribute access
        "[1, 2, 3]",                   # list literal
        "{1: 2}",                      # dict literal
        "a[0]",                        # subscript
        "lambda x: x",                 # lambda
        "(x for x in [1])",            # generator
        "x if True else y",            # ternary (BoolOp)
        "sqrt(x=1)",                   # keyword arg
        "True",                        # boolean literal
        "'hello'",                     # string literal
        "None",                        # None
        "math.sqrt(4)",                # attribute-based function call
    ],
)
def test_disallowed_expression_returns_error(expr: str):
    c = Calculator()
    r = c.evaluate(expr)
    assert "error" in r, f"{expr!r} should have been rejected: {r}"


def test_syntax_error_returns_error_not_raises():
    c = Calculator()
    r = c.evaluate("1 + + + )")
    assert "error" in r


def test_division_by_zero_returns_error():
    c = Calculator()
    assert c.evaluate("1 / 0") == {"error": "division by zero"}


def test_unknown_name_returns_error():
    c = Calculator()
    assert "error" in c.evaluate("undefined_variable * 2")


def test_reset_clears_memory():
    c = Calculator()
    c.evaluate("1.0", store_as="x")
    c.reset()
    assert "error" in c.evaluate("x")


# --- linear regression ------------------------------------------------------

def test_linreg_exact_line():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0 * x + 3.0 for x in xs]
    r = linear_regression(xs, ys)
    assert r["slope"] == pytest.approx(2.0)
    assert r["intercept"] == pytest.approx(3.0)
    assert r["r_squared"] == pytest.approx(1.0)
    assert r["n"] == 5
    # With zero residual, stderrs should be 0 (or very close).
    assert r["slope_stderr"] == pytest.approx(0.0, abs=1e-9)


def test_linreg_noisy_line_recovers_slope():
    # y = 3x + 7 + N(0, 0.5). Deterministic seed for reproducibility.
    import random

    rng = random.Random(42)
    xs = [float(i) for i in range(1, 101)]
    ys = [3.0 * x + 7.0 + rng.gauss(0, 0.5) for x in xs]
    r = linear_regression(xs, ys)
    assert abs(r["slope"] - 3.0) < 0.05
    assert abs(r["intercept"] - 7.0) < 1.0
    assert r["r_squared"] > 0.99


def test_linreg_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        linear_regression([1.0, 2.0], [1.0])


def test_linreg_single_point_raises():
    with pytest.raises(ValueError, match="at least 2 points"):
        linear_regression([1.0], [2.0])


def test_linreg_constant_x_raises():
    with pytest.raises(ValueError, match="identical"):
        linear_regression([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_linreg_mimics_tau_fit_use_case():
    """Log-linearize F(t) = F1·exp(-t/τ1) → ln F = ln F1 − t/τ1.
    Slope = −1/τ1, intercept = ln F1.
    """
    F1 = 40.0
    tau1 = 1500.0
    ts = [100.0 * i for i in range(1, 20)]
    Fs = [F1 * math.exp(-t / tau1) for t in ts]
    r = linear_regression(ts, [math.log(F) for F in Fs])
    tau1_fit = -1.0 / r["slope"]
    F1_fit = math.exp(r["intercept"])
    assert tau1_fit == pytest.approx(tau1, rel=1e-6)
    assert F1_fit == pytest.approx(F1, rel=1e-6)
