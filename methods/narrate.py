import re
import sympy as sp

from .parser import parse_function

x = sp.symbols("x")


def ordinal(n):
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', 11 -> '11th', ..."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fmt(value, places=4):
    """Format a number for display: round, then drop a trailing .0"""
    rounded = round(float(value), places)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"


def _sign_word(value):
    return "< 0" if value < 0 else ("> 0" if value > 0 else "= 0")


def _substituted_expr(expr, value):
    """
    Show the expression with x replaced by a literal number, unevaluated,
    e.g. 2*x**3-2*x-5 at x=0.5 -> '2*(0.5)**3-2*(0.5)-5'
    """
    expr_str = str(expr).replace("**", "^")
    substituted = re.sub(r"\bx\b", f"({_fmt(value)})", expr_str)
    return substituted.replace("^", "**")


def narrate_bracketing(function_str, table, method_label="Bisection"):
    """
    Build a list of markdown strings narrating each iteration of a
    bracketing method (Bisection or False Position), in the style of
    a worked textbook solution.

    Parameters:
        function_str (str): The f(x) string as entered by the user
        table (DataFrame): The iteration table returned by bisection()
                           or false_position() (columns: a, f(a), b,
                           f(b), c, f(c), Error, Update)
        method_label (str): "Bisection" or "False Position", used
                            for the formula line shown

    Returns:
        list[str]: One markdown block per iteration
    """

    expr = parse_function(function_str, {"x": x})

    blocks = []

    for _, row in table.iterrows():

        n = int(row["Iteration"])
        a, fa = _fmt(row["a"]), _fmt(row["f(a)"])
        b, fb = _fmt(row["b"]), _fmt(row["f(b)"])
        c, fc = _fmt(row["c"]), _fmt(row["f(c)"])
        update = row["Update"]

        lines = [f"**{ordinal(n)} iteration:**", ""]

        lines.append(
            f"Here f({a}) = {fa} {_sign_word(row['f(a)'])} and "
            f"f({b}) = {fb} {_sign_word(row['f(b)'])}"
        )
        lines.append("")
        lines.append(f"∴ Root lies between {a} and {b}")
        lines.append("")

        if method_label == "Bisection":
            lines.append(
                f"$$c = \\dfrac{{a+b}}{{2}} = "
                f"\\dfrac{{{a} + {b}}}{{2}} = {c}$$"
            )
        else:
            fb_paren = f"({fb})" if row["f(b)"] < 0 else fb
            fa_paren = f"({fa})" if row["f(a)"] < 0 else fa
            lines.append(
                f"$$c = \\dfrac{{a \\cdot f(b) - b \\cdot f(a)}}"
                f"{{f(b) - f(a)}} = "
                f"\\dfrac{{({a})({fb}) - ({b})({fa})}}"
                f"{{{fb_paren} - {fa_paren}}} = {c}$$"
            )

        lines.append("")
        lines.append(
            f"f(c) = f({c}) = {_substituted_expr(expr, row['c'])} = "
            f"{fc} {_sign_word(row['f(c)'])}"
        )
        lines.append("")

        if update == "b":
            lines.append(f"Since f(a)·f(c) < 0, new interval: [{a}, {c}]")
        else:
            lines.append(f"Since f(a)·f(c) > 0, new interval: [{c}, {b}]")

        blocks.append("\n".join(lines))

    return blocks


def narrate_newton(function_str, table):
    """
    Narrate each iteration of Newton-Raphson.

    table columns: x, f(x), f'(x), Next x, Error
    """

    expr = parse_function(function_str, {"x": x})
    derivative = sp.diff(expr, x)

    blocks = []

    for _, row in table.iterrows():

        n = int(row["Iteration"])
        xi_raw, fxi_raw, dfxi_raw, next_xi_raw = (
            row["x"], row["f(x)"], row["f'(x)"], row["Next x"]
        )
        xi, fxi, dfxi, next_xi = (
            _fmt(xi_raw), _fmt(fxi_raw), _fmt(dfxi_raw), _fmt(next_xi_raw)
        )

        lines = [f"**{ordinal(n)} iteration:**", ""]
        lines.append(
            f"f({xi}) = {_substituted_expr(expr, xi_raw)} = {fxi}"
        )
        lines.append(
            f"f'({xi}) = {_substituted_expr(derivative, xi_raw)} = {dfxi}"
        )
        lines.append("")
        lines.append(
            f"$$x_{{{n}}} = x_{{{n-1}}} - \\dfrac{{f(x_{{{n-1}}})}}"
            f"{{f'(x_{{{n-1}}})}} = {xi} - \\dfrac{{{fxi}}}{{{dfxi}}} "
            f"= {next_xi}$$"
        )

        blocks.append("\n".join(lines))

    return blocks


def narrate_fixed_point(g_function_str, table):
    """
    Narrate each iteration of Fixed Point Iteration.

    table columns: Current x, Next x, Error
    """

    g_expr = parse_function(g_function_str, {"x": x})

    blocks = []

    for _, row in table.iterrows():

        n = int(row["Iteration"])
        current_raw, next_x_raw = row["Current x"], row["Next x"]
        current, next_x = _fmt(current_raw), _fmt(next_x_raw)

        lines = [f"**{ordinal(n)} iteration:**", ""]
        lines.append(
            f"$$x_{{{n}}} = g(x_{{{n-1}}}) = "
            f"g({current}) = {_substituted_expr(g_expr, current_raw)} = {next_x}$$"
        )

        blocks.append("\n".join(lines))

    return blocks
