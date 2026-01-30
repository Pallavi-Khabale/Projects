from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Lean Six Sigma - Logistic Regression (Incentive Policy)",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "df_incentive.csv"
DEFAULT_XLSX = ROOT / "df_incentive.xlsx"


# -----------------------------
# Data helpers
# -----------------------------
def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Incentive", "Target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: Incentive, Target")

    out = df.copy()[["Incentive", "Target"]]
    out["Incentive"] = pd.to_numeric(out["Incentive"], errors="coerce")
    out["Target"] = pd.to_numeric(out["Target"], errors="coerce")
    out = out.dropna()

    out["Incentive"] = out["Incentive"].astype(float)
    out["Target"] = out["Target"].astype(int)

    if not set(out["Target"].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0/1).")

    if out["Incentive"].nunique() < 2:
        raise ValueError("Incentive must have at least 2 unique values.")

    return out


def load_default_data() -> tuple[pd.DataFrame | None, str, str]:
    """
    Loads default data from the same directory as app.py.
    Prefers CSV, falls back to XLSX.
    Returns: (df or None, status_message, source_label)
    """
    try:
        if DEFAULT_CSV.exists():
            df = pd.read_csv(DEFAULT_CSV)
            return validate_df(df), f"Loaded default dataset: {DEFAULT_CSV.name}", DEFAULT_CSV.name
        if DEFAULT_XLSX.exists():
            df = pd.read_excel(DEFAULT_XLSX)
            return validate_df(df), f"Loaded default dataset: {DEFAULT_XLSX.name}", DEFAULT_XLSX.name
        return None, "No default file found next to app.py.", "None"
    except Exception as e:
        return None, f"Default file found but could not be loaded: {e}", "Error"


def success_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Incentive")["Target"]
    tbl = pd.DataFrame({"Target": g.sum(), "Total": g.count()})
    tbl["No Target"] = tbl["Total"] - tbl["Target"]
    tbl["Success Rate %"] = (tbl["Target"] / tbl["Total"] * 100).round(1)
    return tbl.reset_index().sort_values("Incentive", ascending=True)


# -----------------------------
# Model helpers
# -----------------------------
def fit_logistic(df: pd.DataFrame, test_size: float, random_state: int) -> dict:
    X = df[["Incentive"]].values
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    b0 = float(model.intercept_[0])
    b1 = float(model.coef_[0][0])

    return {
        "model": model,
        "intercept": b0,
        "coefficient": b1,
        "accuracy_train": float(model.score(X_train, y_train)),
        "accuracy_test": float(model.score(X_test, y_test)),
    }


def incentive_for_probability(b0: float, b1: float, p: float) -> float:
    if b1 == 0:
        return np.nan
    p = float(p)
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return float((logit - b0) / b1)


def probability_for_incentive(model: LogisticRegression, x: float) -> float:
    return float(model.predict_proba(np.array([[x]], dtype=float))[:, 1][0])


# -----------------------------
# Plot helpers (NO seaborn)
# -----------------------------
def plot_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    df.boxplot(by=["Target"], column=["Incentive"], ax=ax)
    ax.set_xlabel("Target Reached (1: Yes, 0: No)")
    ax.set_ylabel("Incentive (Euros/Day)")
    ax.set_title("Incentive Distribution by Target Achievement")
    fig.suptitle("")
    fig.tight_layout()
    return fig


def plot_logistic_curve(df: pd.DataFrame, model: LogisticRegression, p_line: float):
    x_min = float(df["Incentive"].min())
    x_max = float(df["Incentive"].max())

    x_grid = np.linspace(x_min, x_max, 250).reshape(-1, 1)
    y_prob = model.predict_proba(x_grid)[:, 1]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.02, size=len(df))
    ax.scatter(df["Incentive"], df["Target"] + jitter, alpha=0.35)

    ax.plot(x_grid.flatten(), y_prob)
    ax.axhline(y=float(p_line), linestyle="--", alpha=0.7)

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Daily Incentive (Euros/Day)")
    ax.set_ylabel("Probability of meeting the productivity target")
    ax.set_title("Logistic Regression: Incentive → Probability of Hitting Target")
    fig.tight_layout()
    return fig


# -----------------------------
# Header
# -----------------------------
st.title("Lean Six Sigma — Incentive Policy Optimisation (Logistic Regression)")
st.caption("Concept-first DMAIC analysis. The app is a presentation layer for the decision logic.")


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Data")
    data_mode = st.radio("Source", ["Use default", "Upload Excel"], index=0)

    uploaded = None
    if data_mode == "Upload Excel":
        uploaded = st.file_uploader("Upload .xlsx with Incentive, Target", type=["xlsx"])

    st.header("Model")
    test_size = st.slider("Test split", 0.1, 0.5, 0.30, 0.05)
    random_state = st.number_input("Random state", 0, 9999, 0, 1)

    st.header("Target policy")
    desired_prob = st.slider("Desired probability", 0.50, 0.95, 0.75, 0.01)

    st.header("Current policy (scenario)")
    current_bonus = st.number_input("Current bonus ($/day)", min_value=0.0, value=5.0, step=1.0)


# -----------------------------
# Load data
# -----------------------------
df = None
data_source_label = "Unknown"

if data_mode == "Use default":
    df, msg, data_source_label = load_default_data()
    if df is None:
        st.error(msg)
        st.info(
            f"Fix: Put df_incentive.csv (preferred) or df_incentive.xlsx next to app.py in: {ROOT}"
        )
        st.stop()
    else:
        st.sidebar.success(msg)

else:
    if uploaded is None:
        st.info("Upload your Excel file to continue.")
        st.stop()

    try:
        raw = pd.read_excel(uploaded)
        df = validate_df(raw)
        data_source_label = "Uploaded Excel"
        st.sidebar.success("Uploaded file loaded.")
    except Exception as e:
        st.error(f"Could not load uploaded file: {e}")
        st.stop()


# -----------------------------
# Fit model + compute recommendation
# -----------------------------
results = fit_logistic(df, test_size=float(test_size), random_state=int(random_state))
b0 = results["intercept"]
b1 = results["coefficient"]

x_needed = incentive_for_probability(b0, b1, float(desired_prob))
p_current = probability_for_incentive(results["model"], float(current_bonus))


# -----------------------------
# Tabs (Overview first)
# -----------------------------
tab0, tab1, tab2, tab3 = st.tabs(["Overview", "Executive Summary", "Analysis & Visuals", "Data"])


# -----------------------------
# Overview (Scenario + Business context + Default dataset note)
# -----------------------------
with tab0:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Scenario")
        st.write(
            "You are the Regional Director of a logistics company (3PL), responsible for **22 warehouses**.\n\n"
            "In each warehouse, the site manager has defined a daily **picking productivity target** for operators. "
            "**Picking productivity** is defined as the number of cartons picked per paid hour.\n\n"
            "Your objective is to design the right incentive policy so that **75% of operators meet the target**."
        )

        st.subheader("Current incentive policy (problem)")
        st.write(
            "Operators who meet their daily target receive $5/day, on top of a daily salary of $64/day (after tax). "
            "This policy has been applied in **two warehouses**, but it is ineffective.\n\n"
            "**Only ~20% of operators are currently reaching the target.**"
        )

        st.subheader("Question")
        st.write("**What minimum daily bonus is needed to reach a 75% probability of meeting the target?**")

        st.subheader("Experiment design")
        st.write(
            "- Randomly select operators across the 22 warehouses\n"
            "- Implement a daily incentive amount varying between **$1 and $20**\n"
            "- Record whether the operator reached the target (**Target = 1**) or not (**Target = 0**)"
        )

    with right:
        st.subheader("Lean Six Sigma framing (DMAIC)")
        st.write(
            "**Define**: Incentive policy exists but target achievement remains low.\n\n"
            "**Measure**: Run a structured incentive experiment and capture binary outcomes.\n\n"
            "**Analyse**: Fit logistic regression to estimate probability of success by incentive amount.\n\n"
            "**Improve**: Choose a minimum incentive threshold that achieves the target probability.\n\n"
            "**Control**: Monitor performance and refresh the model as demand, seasonality, or workforce changes."
        )

        st.subheader("Default dataset used in this demo")
        st.write(
            f"This app loads a **default dataset** when set to **Use default**.\n\n"
            f"Current source: **{data_source_label}**\n\n"
            "To replace the default, upload your own Excel file with columns: `Incentive`, `Target`."
        )


# -----------------------------
# Executive Summary
# -----------------------------
with tab1:
    top_left, top_right = st.columns([1.35, 1], gap="large")

    with top_left:
        st.subheader("Recommendation (Main Output)")

        if np.isfinite(x_needed):
            st.markdown(
                f"""
                <div style="padding: 18px; border-radius: 14px; border: 1px solid rgba(0,0,0,0.12);">
                  <div style="font-size: 14px; opacity: 0.75;">Minimum incentive required</div>
                  <div style="font-size: 46px; font-weight: 900; line-height: 1.05; margin-top: 2px;">
                    ${x_needed:.2f}
                  </div>
                  <div style="margin-top: 8px; font-size: 16px;">
                    to achieve <b>{desired_prob*100:.0f}%</b> probability of meeting the productivity target
                  </div>
                  <div style="margin-top: 10px; font-size: 14px; opacity: 0.8;">
                    Rounded up policy suggestion: <b>${int(np.ceil(x_needed))}/day</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("Unable to compute the incentive threshold (coefficient is zero/invalid).")

        st.write("")
        st.subheader("Current policy check ($/day)")
        st.write(
            f"At the current bonus of **${current_bonus:.0f}/day**, the model estimates about "
            f"**{p_current*100:.1f}%** probability of meeting the target."
        )

    with top_right:
        st.subheader("KPIs (At a glance)")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", f"{len(df):,}")
        k2.metric("Success rate", f"{df['Target'].mean()*100:.1f}%")
        k3.metric("Train accuracy", f"{results['accuracy_train']*100:.1f}%")
        k4.metric("Test accuracy", f"{results['accuracy_test']*100:.1f}%")

        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Intercept (b0)", f"{b0:.4f}")
        m2.metric("Coefficient (b1)", f"{b1:.4f}")

        with st.expander("Lean Six Sigma summary (DMAIC)", expanded=False):
            st.write(
                "- **Define**: Incentive policy not achieving targets.\n"
                "- **Measure**: Experiment with incentive levels.\n"
                "- **Analyse**: Logistic regression probability curve.\n"
                "- **Improve**: Choose threshold incentive.\n"
                "- **Control**: Re-run periodically."
            )


# -----------------------------
# Analysis & Visuals
# -----------------------------
with tab2:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("Probability Curve")
        fig_curve = plot_logistic_curve(df, results["model"], p_line=float(desired_prob))
        st.pyplot(fig_curve, clear_figure=True)

    with c2:
        st.subheader("Incentive Distribution")
        fig_box = plot_box(df)
        st.pyplot(fig_box, clear_figure=True)

    st.write("")
    st.subheader("Success-rate table (by incentive)")
    tbl = success_table(df)
    st.dataframe(tbl, use_container_width=True)

    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download summary table (CSV)",
        data=csv,
        file_name="success_rate_by_incentive.csv",
        mime="text/csv",
    )


# -----------------------------
# Data
# -----------------------------
with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Expected columns")
    st.code("Incentive (numeric), Target (0/1)", language="text")
