import numpy as np


def find_bracket(f, search_min=-50, search_max=50, steps=4000):
    """
    Scan a wide range for the first interval where f changes sign,
    for use when the user wants "Any Root" instead of specifying
    a bracket themselves.

    Parameters:
        f (callable): Numeric function, e.g. from sp.lambdify
        search_min (float): Left edge of the search range
        search_max (float): Right edge of the search range
        steps (int): Number of sample points to scan

    Returns:
        (a, b): A bracket where f(a) and f(b) have opposite signs

    Raises:
        ValueError: If no sign change is found in the search range
    """

    xs = np.linspace(search_min, search_max, steps)

    try:
        ys = f(xs)
    except Exception:
        # Fall back to point-by-point evaluation if the function
        # can't be vectorized (e.g. uses branching sympy pieces)
        ys = np.array([f(v) for v in xs])

    ys = np.asarray(ys, dtype=float)

    # Ignore points where the function is undefined (NaN/Inf),
    # e.g. near asymptotes like tan(x) or log(x) for x <= 0
    valid = np.isfinite(ys)

    for i in range(len(xs) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        if ys[i] == 0:
            return xs[i], xs[i]
        if ys[i] * ys[i + 1] < 0:
            return float(xs[i]), float(xs[i + 1])

    raise ValueError(
        f"No root found by scanning [{search_min}, {search_max}]. "
        "Try widening the search range or use 'Root Between' with "
        "a specific interval."
    )
