"""A very simple, safe arithmetic calculator for +, -, *, / expressions.

Used by model/generate.py to compute the real result of a <CALC>...</CALC> block
instead of trusting the model to generate the digits itself. Deliberately does NOT
use eval() -- the expression comes from model output, which is untrusted text, so
this parses it into an AST and only permits numeric literals and +-*/ operators.
"""

from __future__ import annotations

import ast
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    """Raised when an expression can't be safely parsed or evaluated."""


def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression (+, -, *, /, parens, unary minus).

    Raises CalculatorError on invalid syntax, unsupported operations, or division
    by zero -- callers decide what to do (e.g. skip injecting a result).
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"invalid expression: {expression!r}") from e
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as e:
        raise CalculatorError(f"division by zero: {expression!r}") from e


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"unsupported expression: {ast.dump(node)}")


def format_result(value: float) -> str:
    """Render a number the way GSM8K-style answers are written: bare integers
    when possible ("24" not "24.0"), otherwise a compact decimal."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}"
