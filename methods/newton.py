import sympy as sp
import pandas as pd
import numpy as np

from .parser import parse_function

# Define the symbolic variable
x = sp.symbols("x")


def newton_raphson(function, x0, tolerance, max_iter, error_type="absolute"):
    """
    Newton-Raphson Method

    Parameters:
        function (str): Mathematical function as a string
                        Example: "x**3 - x - 2"
        x0 (float): Initial guess
        tolerance (float): Desired accuracy
        max_iter (int): Maximum number of iterations
        error_type (str): "absolute" or "relative" (percent)

    Returns:
        root (float): Estimated root
        table (DataFrame): Iteration table
    """

    if error_type not in ("absolute", "relative"):
        raise ValueError("error_type must be 'absolute' or 'relative'")

    # Convert string to symbolic expression
    expr = parse_function(function, {"x": x})

    # Differentiate automatically
    derivative = sp.diff(expr, x)

    # Convert expressions to Python functions
    f = sp.lambdify(x, expr, "numpy")
    df = sp.lambdify(x, derivative, "numpy")

    iterations = []

    current = x0

    for i in range(1, max_iter + 1):

        fx = f(current)
        dfx = df(current)

        # Prevent division by zero
        if abs(dfx) < 1e-12:
            raise ValueError(
                "Derivative is zero. Choose another initial guess."
            )

        next_x = current - (fx / dfx)

        # Detect divergence (NaN, Inf, or blowing up)
        if not np.isfinite(next_x) or abs(next_x) > 1e15:
            raise ValueError(
                "Method is diverging. Try a different initial guess."
            )

        if error_type == "absolute":
            error = abs(next_x - current)
        else:
            error = abs((next_x - current) / next_x) * 100 if next_x != 0 else 0.0

        iterations.append(
            {
                "Iteration": i,
                "x": round(current, 8),
                "f(x)": round(fx, 8),
                "f'(x)": round(dfx, 8),
                "Next x": round(next_x, 8),
                "Error": round(error, 10)
            }
        )

        # Stop if converged
        if abs(f(next_x)) < tolerance:
            current = next_x
            break

        if error < tolerance:
            current = next_x
            break

        current = next_x

    table = pd.DataFrame(iterations)

    return current, table