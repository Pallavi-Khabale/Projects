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
    page_title="LSS — Logistic Regression (Incentive Policy)",
    layout="wide",
)

# Resolve paths relative to this app.py (NOT current working directory)
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

    # Target must be binary
    if not set(out["Target"].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0/1).")

    # If Incentive is constant, model will be meaningless
    if out["Incentive"].nunique() < 2:
        raise ValueError("Incentive must have at least 2 unique values.")

    return out


def load_default_data() -> tuple[pd.DataFrame | None, str]:
    """
    Loads default data from the same directory as app.py.
    Prefers CSV, falls back to XLSX.
    Returns: (df or None, message)
    """
    try:
        if DEFAULT_CSV.exists():
            df = pd.read_csv(DEFAULT_CSV)
            return validate_df(df), f"Loaded default CSV: {DEFAULT_CSV.name}"
        if DEFAULT_XLSX.exists():
            df = pd.read_excel(DEFAULT_XLSX)
            return validate_df(df), f"Loaded default Excel: {DEFAULT_XLSX.name}"
        return None, "No default file found next to app.py."
    except Exception as e:
        return None, f"Default file found but could not be loaded: {e}"


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
    # avoid log(0)
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return float((logit - b0) / b1)


# -----------------------------
# Plot helpers (NO seaborn)
# -----------------------------
def plot_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    df.boxplot(by=["Target"], column=["Incentive"], ax=ax)
    ax.set_xlabel("Target Reached (1: Yes, 0: No)")
    ax.set_ylabel("Incentive (Dollars/Day)")
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

    # Scatter with small jitter to prevent complete overlap
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.02, size=len(df))
    ax.scatter(df["Incentive"], df["Target"] + jitter, alpha=0.35)

    # Logistic curve
    ax.plot(x_grid.flatten(), y_prob)

    # Probability threshold line
    ax.axhline(y=float(p_line), linestyle="--", alpha=0.7)

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Productivity Incentive (Dollars/Day)")
    ax.set_ylabel("Probability of meeting the target")
    ax.set_title("Logistic Regression: Incentive → Probability of Hitting Target")
    fig.tight_layout()
    return fig


# -----------------------------
# Header
# -----------------------------
st.title("Lean Six Sigma — Incentive Policy Optimisation (Logistic Regression)")
st.caption("Concept-first DMAIC analysis. The app is just a presentation layer.")

with st.expander("Debug: paths & files (remove once stable)", expanded=False):
    st.write("Current working directory (CWD):", os.getcwd())
    st.write("App directory (ROOT):", str(ROOT))
    st.write("Files next to app.py:", sorted([p.name for p in ROOT.iterdir()]))
    st.write("DEFAULT_CSV exists:", DEFAULT_CSV.exists(), "→", str(DEFAULT_CSV))
    st.write("DEFAULT_XLSX exists:", DEFAULT_XLSX.exists(), "→", str(DEFAULT_XLSX))


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

    st.header("Target")
    desired_prob = st.slider("Desired probability", 0.50, 0.95, 0.75, 0.01)


# -----------------------------
# Load data
# -----------------------------
df = None

if data_mode == "Use default":
    df, msg = load_default_data()
    if df is None:
        st.error(msg)
        st.info(
            "Fix: Put df_incentive.csv (preferred) or df_incentive.xlsx in the SAME folder as app.py "
            f"({ROOT})."
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


# -----------------------------
# Tabs (reduce scrolling)
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Executive Summary", "Analysis & Visuals", "Data"])


# -----------------------------
# Tab 1: Executive Summary
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
        st.subheader("Interpretation")
        st.write(
            "This converts experiment results into a decision threshold. "
            "Instead of guessing bonuses, you set an incentive level that achieves a target success probability."
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

        with st.expander("Lean Six Sigma framing (DMAIC)", expanded=False):
            st.write(
                "- **Define**: Incentive policy is not achieving productivity targets.\n"
                "- **Measure**: Run incentive experiments and record binary outcomes.\n"
                "- **Analyse**: Fit logistic regression to estimate probability curve.\n"
                "- **Improve**: Choose incentive level that hits target probability.\n"
                "- **Control**: Re-evaluate quarterly as workforce and seasonality change."
            )


# -----------------------------
# Tab 2: Analysis & Visuals
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
# Tab 3: Data
# -----------------------------
with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Expected columns")
    st.code("Incentive (numeric), Target (0/1)", language="text")
