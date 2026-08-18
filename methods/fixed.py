import sympy as sp
import pandas as pd
import numpy as np

from .parser import parse_function

# Define symbolic variable
x = sp.symbols("x")


def fixed_point(g_function, x0, tolerance, max_iter, error_type="absolute"):
    """
    Fixed Point Iteration Method

    Parameters:
        g_function (str): Iteration function g(x)
                          Example: "(x+2)**(1/3)"
        x0 (float): Initial guess
        tolerance (float): Desired accuracy
        max_iter (int): Maximum iterations
        error_type (str): "absolute" or "relative" (percent)

    Returns:
        root (float): Estimated root
        table (DataFrame): Iteration table
    """

    if error_type not in ("absolute", "relative"):
        raise ValueError("error_type must be 'absolute' or 'relative'")

    # Convert string to symbolic expression
    expr = parse_function(g_function, {"x": x})

    # Convert symbolic expression into Python function
    g = sp.lambdify(x, expr, "numpy")

    iterations = []

    current = x0

    for i in range(1, max_iter + 1):

        next_x = g(current)

        # Detect divergence (NaN, Inf, or blowing up)
        if not np.isfinite(next_x) or abs(next_x) > 1e15:
            raise ValueError(
                "Method is diverging. Try a different g(x) or initial guess."
            )

        if error_type == "absolute":
            error = abs(next_x - current)
        else:
            error = abs((next_x - current) / next_x) * 100 if next_x != 0 else 0.0

        iterations.append(
            {
                "Iteration": i,
                "Current x": round(current, 8),
                "Next x": round(next_x, 8),
                "Error": round(error, 10)
            }
        )

        if error < tolerance:
            current = next_x
            break

        current = next_x

    table = pd.DataFrame(iterations)

    return current, table