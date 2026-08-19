import streamlit as st

from methods.bisection import bisection
from methods.false_position import false_position
from methods.newton import newton_raphson
from methods.fixed import fixed_point
from methods.narrate import narrate_bracketing, narrate_newton, narrate_fixed_point

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Numerical Methods Solver",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Numerical Methods Solver")
st.write("Solve nonlinear equations using different numerical methods.")

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.title("⚙️ Input Parameters")

method = st.sidebar.selectbox(
    "Select Method",
    [
        "Bisection Method",
        "False Position Method",
        "Newton-Raphson Method",
        "Fixed Point Iteration"
    ]
)

bracketing_method = method in ["Bisection Method", "False Position Method"]

st.sidebar.markdown("---")

# Function Input

if method == "Fixed Point Iteration":

    function = st.sidebar.text_input(
        "Enter g(x)",
        "(x+2)**(1/3)",
        help="Use ^ or ** for powers, e.g. (x+2)^(1/3)"
    )

else:

    function = st.sidebar.text_input(
        "Enter f(x)",
        "x^3-x-2",
        help="Use ^ or ** for powers, e.g. x^3-3x^2+2x-1"
    )

# -----------------------------------------------------
# Find: Any Root vs Root Between (bracketing methods only)
# -----------------------------------------------------

a = b = None
search_min = search_max = None

if bracketing_method:

    find_mode = st.sidebar.radio(
        "Find",
        ["Root Between", "Any Root"],
        horizontal=True
    )

    if find_mode == "Root Between":

        a = st.sidebar.number_input(
            "Lower Bound (a)",
            value=1.0
        )

        b = st.sidebar.number_input(
            "Upper Bound (b)",
            value=2.0
        )

    else:

        st.sidebar.caption(
            "Searches this range for the first sign change."
        )

        search_min = st.sidebar.number_input(
            "Search from",
            value=-50.0
        )

        search_max = st.sidebar.number_input(
            "Search to",
            value=50.0
        )

# Inputs for Newton & Fixed Point

else:

    initial_guess = st.sidebar.number_input(
        "Initial Guess",
        value=1.5
    )

# -----------------------------------------------------
# Error type
# -----------------------------------------------------

error_choice = st.sidebar.radio(
    "Error",
    ["Absolute Error", "Relative Percent Error"],
    horizontal=True
)

error_type = "absolute" if error_choice == "Absolute Error" else "relative"

# Common Inputs

tolerance = st.sidebar.number_input(
    "Tolerance" + (" (%)" if error_type == "relative" else ""),
    value=0.0001,
    format="%.6f"
)

max_iter = st.sidebar.number_input(
    "Maximum Iterations",
    value=100,
    step=1
)

show_work = st.sidebar.checkbox(
    "Show step-by-step work",
    value=True
)

solve = st.sidebar.button("🚀 Solve")

# -----------------------------------------------------
# Solve
# -----------------------------------------------------

if solve:

    try:

        # ---------------- Bisection ----------------

        if method == "Bisection Method":

            root, table = bisection(
                function,
                a,
                b,
                tolerance,
                int(max_iter),
                error_type=error_type,
                search_range=(search_min, search_max)
                if search_min is not None else (-50, 50),
            )

        # -------------- False Position -------------

        elif method == "False Position Method":

            root, table = false_position(
                function,
                a,
                b,
                tolerance,
                int(max_iter),
                error_type=error_type,
                search_range=(search_min, search_max)
                if search_min is not None else (-50, 50),
            )

        # ------------ Newton Raphson ---------------

        elif method == "Newton-Raphson Method":

            root, table = newton_raphson(
                function,
                initial_guess,
                tolerance,
                int(max_iter),
                error_type=error_type,
            )

        # ------------ Fixed Point ------------------

        elif method == "Fixed Point Iteration":

            root, table = fixed_point(
                function,
                initial_guess,
                tolerance,
                int(max_iter),
                error_type=error_type,
            )

        # -------------------------------------------

        st.success("✅ Solution Found Successfully")

        st.markdown(
            f"**Approximate root of the equation is `{root:.4f}` "
            f"(after {len(table)} iterations)**"
        )

        # ---------------- Manual Step-by-Step Work ----------------

        if show_work:

            st.subheader("✍️ Solution")

            st.markdown(f"Here **f(x) = {function}**")
            st.markdown("---")

            if method in ["Bisection Method", "False Position Method"]:
                label = "Bisection" if method == "Bisection Method" else "False Position"
                blocks = narrate_bracketing(function, table, method_label=label)

            elif method == "Newton-Raphson Method":
                blocks = narrate_newton(function, table)

            else:
                blocks = narrate_fixed_point(function, table)

            for block in blocks:
                st.markdown(block)
                st.markdown("")

            st.markdown("---")

        # ---------------- Results Summary ----------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Root", f"{root:.4f}")

        with col2:
            st.metric("Iterations", len(table))

        with col3:
            last_fc = None
            for candidate_col in ["f(c)", "f(x)"]:
                if candidate_col in table.columns:
                    last_fc = table[candidate_col].iloc[-1]
                    break
            st.metric("Function Value", f"{last_fc:.4f}" if last_fc is not None else "—")

        with col4:
            last_error = table["Error"].iloc[-1]
            error_suffix = "%" if error_type == "relative" else ""
            st.metric(
                "Error",
                f"{last_error:.4f}{error_suffix}" if last_error is not None else "—"
            )

        st.markdown("---")

        st.subheader("📊 Iteration Table")

        st.dataframe(
            table,
            use_container_width=True
        )

    except ValueError as e:

        # Our own errors already carry their own friendly tone
        st.error(str(e))

    except Exception as e:

        st.error(
            f"😵 Well, that's unexpected! Something went sideways "
            f"that I wasn't quite ready for:\n\n`{e}`\n\n"
            "Double-check your inputs, or try a simpler function "
            "to narrow down what's going on."
        )

# -----------------------------------------------------
# Footer
# -----------------------------------------------------

st.markdown("---")

st.caption(
    "Developed using Python, Streamlit, SymPy and Pandas"
)
