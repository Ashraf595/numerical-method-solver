import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# Allows users to type functions naturally:
#   x^3 - 3x^2 + 2x - 1   instead of   x**3 - 3*x**2 + 2*x - 1
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def parse_function(function_str, local_dict):
    """
    Parse a user-entered function string into a SymPy expression.

    Supports:
      - '^' as an alias for power (in addition to '**')
      - Implicit multiplication, e.g. '3x^2' -> '3*x**2'
      - Transcendental functions: sin, cos, tan, exp, log/ln, sqrt, etc.
      - 'e' as Euler's number and 'pi' as the constant, e.g. 'e^x-3'

    Parameters:
        function_str (str): Function as typed by the user
        local_dict (dict): Symbol table, e.g. {"x": x}

    Returns:
        sympy expression
    """

    # Provide common constants so they aren't treated as free variables
    full_local_dict = {
        "e": sp.E,
        "pi": sp.pi,
        "ln": sp.log,
        "log10": lambda arg: sp.log(arg, 10),
        **local_dict,
    }

    try:
        expr = parse_expr(
            function_str,
            local_dict=full_local_dict,
            transformations=_TRANSFORMATIONS,
        )
    except Exception:
        raise ValueError(
            f"🧩 Couldn't quite parse '{function_str}' — the math "
            "notation tripped me up somewhere.\n\n"
            "A few things that usually help: use ^ or ** for powers "
            "(x^3 or x**3), write function names with parentheses "
            "(sin(x), not sinx), and make sure brackets are balanced. "
            "Example that works: x^3-3x^2+2x-1"
        )

    # Catch cases like 'sinx' (no parentheses/space) which silently
    # parse as a product of stray single-letter symbols (i*n*s*x)
    allowed_symbols = {sp.Symbol(name) for name in local_dict}
    stray = expr.free_symbols - allowed_symbols

    if stray:
        stray_list = ", ".join(sorted(f"'{s}'" for s in map(str, stray)))
        raise ValueError(
            f"🕵️ Found some mystery letter(s) I don't recognize: "
            f"{stray_list} in '{function_str}'.\n\n"
            "This usually happens when a function name is missing its "
            "parentheses — e.g. 'sinx' gets read as separate letters "
            "instead of the function sin(x). Try 'sin(x)' instead, "
            "and the same goes for cos, tan, exp, log, sqrt, etc."
        )

    return expr
