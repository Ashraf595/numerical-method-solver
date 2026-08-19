import sympy as sp
import pandas as pd

from .parser import parse_function
from .root_search import find_bracket

# Define the symbolic variable
x = sp.symbols("x")


def bisection(
    function,
    a=None,
    b=None,
    tolerance=0.0001,
    max_iter=100,
    error_type="absolute",
    search_range=(-50, 50),
):
    """
    Bisection Method

    Parameters:
        function (str): Mathematical function as a string
                        Example: "x**3 - x - 2"
        a (float): Lower bound. If None, a bracket is found
                   automatically by scanning search_range ("Any Root").
        b (float): Upper bound. If None, same as above.
        tolerance (float): Desired accuracy
        max_iter (int): Maximum number of iterations
        error_type (str): "absolute" or "relative" (percent)
        search_range (tuple): Range to scan when a/b are None

    Returns:
        root (float): Estimated root
        table (DataFrame): Iteration table
    """

    if error_type not in ("absolute", "relative"):
        raise ValueError("error_type must be 'absolute' or 'relative'")

    # Convert string into symbolic expression
    expr = parse_function(function, {"x": x})

    # Convert symbolic expression into a Python function
    f = sp.lambdify(x, expr, "numpy")

    # "Any Root" mode: find a bracket automatically
    if a is None or b is None:
        a, b = find_bracket(f, search_range[0], search_range[1])

    # Check if interval is valid
    if f(a) * f(b) >= 0:
        raise ValueError(
            f"""
🤔 Hmm, no sign change here!

f({a}) = {f(a):.6f}
f({b}) = {f(b):.6f}

Bisection needs f(a) and f(b) to have opposite
signs (one positive, one negative) — that's how
it knows a root is trapped between them.

Try widening the interval, or flip over to
"Any Root" mode and let the app hunt for one 🔍
"""
        )

    iterations = []

    previous_c = None

    for i in range(1, max_iter + 1):

        # Midpoint
        c = (a + b) / 2

        fa = f(a)
        fb = f(b)
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

        # Stopping conditions
        if abs(fc) < tolerance:
            break

        if error is not None and error < tolerance:
            break

        # Choose new interval
        if update == "b":
            b = c
        else:
            a = c

        previous_c = c

    table = pd.DataFrame(iterations)

    return c, table
