import sympy as sp
import pandas as pd

from .parser import parse_function
from .root_search import find_bracket

# Define symbolic variable
x = sp.symbols("x")


def false_position(
    function,
    a=None,
    b=None,
    tolerance=0.0001,
    max_iter=100,
    error_type="absolute",
    search_range=(-50, 50),
):
    """
    False Position (Regula Falsi) Method

    Parameters
    ----------
    function : str
        Function as a string.
        Example: "x**3 - x - 2"

    a : float
        Lower bound. If None, a bracket is found automatically by
        scanning search_range ("Any Root").

    b : float
        Upper bound. If None, same as above.

    tolerance : float
        Desired accuracy

    max_iter : int
        Maximum number of iterations

    error_type : str
        "absolute" or "relative" (percent)

    search_range : tuple
        Range to scan when a/b are None

    Returns
    -------
    root : float
        Estimated root

    table : pandas.DataFrame
        Iteration table
    """

    if error_type not in ("absolute", "relative"):
        raise ValueError("error_type must be 'absolute' or 'relative'")

    # Convert string to symbolic expression
    expr = parse_function(function, {"x": x})

    # Create numerical function
    f = sp.lambdify(x, expr, "numpy")

    # "Any Root" mode: find a bracket automatically
    if a is None or b is None:
        a, b = find_bracket(f, search_range[0], search_range[1])

    # Check interval validity
    if f(a) * f(b) >= 0:
        raise ValueError(
            f"""
Invalid Interval!

f(a) = {f(a):.6f}
f(b) = {f(b):.6f}

Choose another interval where
f(a) and f(b) have opposite signs.
"""
        )

    iterations = []

    previous_c = None

    for i in range(1, max_iter + 1):

        fa = f(a)
        fb = f(b)

        # False Position Formula
        c = (a * fb - b * fa) / (fb - fa)

        fc = f(c)

        # Approximate Error
        if previous_c is None:
            error = None
        elif error_type == "absolute":
            error = abs(c - previous_c)
        else:
            # Relative percent error
            error = abs((c - previous_c) / c) * 100 if c != 0 else 0.0

        # Decide which bound will be updated (before overwriting a/b)
        if fa * fc < 0:
            update = "b"
        else:
            update = "a"

        # Store iteration
        iterations.append(
            {
                "Iteration": i,
                "a": round(a, 8),
                "f(a)": round(fa, 8),
                "b": round(b, 8),
                "f(b)": round(fb, 8),
                "c": round(c, 8),
                "f(c)": round(fc, 8),
                "Error": error if error is None else round(error, 10),
                "Update": update,
            }
        )

        # Stop if root found
        if abs(fc) < tolerance:
            break

        # Stop if approximate error is small
        if error is not None and error < tolerance:
            break

        # Update interval
        if update == "b":
            b = c
        else:
            a = c

        previous_c = c

    table = pd.DataFrame(iterations)

    return c, table
