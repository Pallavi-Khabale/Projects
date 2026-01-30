import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat
import seaborn as sns
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# -----------------------------
# Lean Six Sigma: Logistic Regression (Streamlit)
# -----------------------------

st.set_page_config(
    page_title="Lean Six Sigma — Logistic Regression (Incentive Policy)",
    layout="wide",
)

st.title("Lean Six Sigma with Python — Logistic Regression 👷")
st.caption(
    "Estimate the **minimum daily bonus** needed to reach **75% probability** of hitting a productivity target "
    "using a simple logistic regression (LSS Improve step)."
)


# -----------------------------
# Utilities
# -----------------------------

def create_sample_data(seed: int = 42) -> pd.DataFrame:
    """
    Create a realistic sample dataset.
    Incentive (1..20) euros/day, Target is 0/1 whether operator met daily productivity target.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for incentive in range(1, 21):
        # S-shaped probability curve: higher incentive increases probability of hitting target
        prob = 1 / (1 + np.exp(-(incentive - 15) / 3))
        n_samples = rng.integers(14, 20)
        for _ in range(n_samples):
            target = 1 if rng.random() < prob else 0
            rows.append({"Incentive": incentive, "Target": target})

    return pd.DataFrame(rows)


def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate required columns and clean types.
    """
    required = {"Incentive", "Target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected columns: Incentive, Target")

    out = df.copy()

    # Keep only required columns (clean demo)
    out = out[["Incentive", "Target"]]

    # Coerce numeric
    out["Incentive"] = pd.to_numeric(out["Incentive"], errors="coerce")
    out["Target"] = pd.to_numeric(out["Target"], errors="coerce")

    out = out.dropna()
    out["Incentive"] = out["Incentive"].astype(float)
    out["Target"] = out["Target"].astype(int)

    # Force binary target
    if not set(out["Target"].unique()).issubset({0, 1}):
        raise ValueError("Target column must be binary (0/1).")

    return out


def success_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grouped summary by incentive.
    """
    g = df.groupby("Incentive")["Target"]
    tbl = pd.DataFrame({"Target": g.sum(), "Total": g.count()})
    tbl["No Target"] = tbl["Total"] - tbl["Target"]
    tbl["Success Rate %"] = (tbl["Target"] / tbl["Total"] * 100).round(1)
    tbl = tbl.reset_index().sort_values("Incentive", ascending=True)
    return tbl


def fit_logistic_and_stats(df: pd.DataFrame, test_size: float, random_state: int):
    """
    Fit sklearn LogisticRegression and compute:
    - coefficient, intercept
    - p-value (Wald z test approximation using Fisher Information)
    - incentive amount for 75% probability
    - train/test accuracy
    """
    X = df[["Incentive"]].values
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # p-value (approx.) using Fisher information similar to your notebook script
    # Note: This is an approximation and differs from statsmodels GLM standard output.
    decision = model.decision_function(X)  # shape: (n_samples,)
    denom = 2.0 * (1.0 + np.cosh(decision))
    denom = np.tile(denom, (X.shape[1], 1)).T  # shape: (n_samples, n_features)

    F_ij = np.dot((X / denom).T, X)  # Fisher information
    Cramer_Rao = np.linalg.inv(F_ij)
    sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))

    coef = model.coef_[0]
    z_scores = coef / sigma_estimates
    p_values = [stat.norm.sf(abs(z)) * 2 for z in z_scores]  # two-tailed

    b0 = float(model.intercept_[0])
    b1 = float(model.coef_[0][0])

    # Solve for incentive x when P=0.75: x = (log(p/(1-p)) - b0) / b1
    p = 0.75
    logit = np.log(p / (1 - p))
    incentive_75 = (logit - b0) / b1 if b1 != 0 else np.nan

    return {
        "model": model,
        "intercept": b0,
        "coefficient": b1,
        "p_value": float(p_values[0]),
        "incentive_75": float(incentive_75),
        "accuracy_train": float(model.score(X_train, y_train)),
        "accuracy_test": float(model.score(X_test, y_test)),
    }


def plot_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    df.boxplot(by=["Target"], column=["Incentive"], ax=ax)
    ax.set_xlabel("Target Reached (1: Yes, 0: No)")
    ax.set_ylabel("Incentive (Euros/Day)")
    ax.set_title("Incentive Distribution by Target Achievement")
    fig.suptitle("")
    fig.tight_layout()
    return fig


def plot_logistic_curve(df: pd.DataFrame, model: LogisticRegression, show_75_line: bool = True):
    # Scatter + logistic curve with seaborn
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(x="Incentive", y="Target", data=df, logistic=True, ax=ax)
    ax.set_xlabel("Productivity Incentive (Euros/Day)")
    ax.set_ylabel("Probability of meeting the productivity target")
    ax.set_title("Logistic Regression: Incentive vs Target Achievement")

    if show_75_line:
        ax.axhline(y=0.75, linestyle="--", alpha=0.7)
        ax.text(
            x=df["Incentive"].min(),
            y=0.76,
            s="75% probability threshold",
            fontsize=9,
            verticalalignment="bottom",
        )

    fig.tight_layout()
    return fig


# -----------------------------
# Sidebar controls (keep it minimal)
# -----------------------------

with st.sidebar:
    st.header("Data")
    mode = st.radio(
        "Choose dataset",
        ["Use sample data", "Upload Excel (.xlsx)"],
        index=0,
    )

    st.header("Model settings")
    test_size = st.slider("Test size", 0.1, 0.5, 0.30, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=0, step=1)

    st.header("Target policy")
    desired_prob = st.slider("Desired probability", 0.50, 0.95, 0.75, 0.01)


# -----------------------------
# Load data
# -----------------------------

df = None

if mode == "Use sample data":
    df = create_sample_data(seed=42)
    st.info("Using sample dataset (synthetic). Upload your own Excel to match your real experiment.")
else:
    uploaded = st.file_uploader("Upload df_incentive.xlsx", type=["xlsx"])
    if uploaded:
        try:
            raw = pd.read_excel(uploaded)
            df = validate_df(raw)
        except Exception as e:
            st.error(f"Could not read/validate the file. {e}")

# If no upload yet, stop
if df is None:
    st.stop()

# Validate sample too (for consistency)
df = validate_df(df)

# -----------------------------
# Core outputs
# -----------------------------

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Scenario (Concept)")
    st.write(
        "You run multiple warehouses and want an incentive policy that increases the share of operators "
        "meeting a daily picking productivity target.\n\n"
        "You ran an experiment where operators received different daily incentives (1–20 euros), "
        "then recorded whether they met the target (1) or not (0)."
    )

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Success rate by incentive")
    tbl = success_table(df)
    st.dataframe(tbl, use_container_width=True)

    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download summary table (CSV)",
        data=csv,
        file_name="success_rate_by_incentive.csv",
        mime="text/csv",
    )

with right:
    st.subheader("Key metrics")
    success_rate = df["Target"].mean()
    st.metric("Overall success rate", f"{success_rate*100:.1f}%")
    st.metric("Records", f"{len(df):,}")
    st.metric("Target reached", f"{int(df['Target'].sum()):,}")
    st.metric("Target not reached", f"{int((df['Target'] == 0).sum()):,}")

st.divider()

# -----------------------------
# Model + recommendation
# -----------------------------

st.subheader("Model: Logistic Regression")

results = fit_logistic_and_stats(df, test_size=test_size, random_state=int(random_state))

b0 = results["intercept"]
b1 = results["coefficient"]

# Compute incentive for chosen desired probability
p = float(desired_prob)
logit = np.log(p / (1 - p))
incentive_p = (logit - b0) / b1 if b1 != 0 else np.nan

m1, m2, m3, m4 = st.columns(4)

m1.metric("Intercept (b0)", f"{b0:.4f}")
m2.metric("Coefficient (b1)", f"{b1:.4f}")
m3.metric("p-value (approx.)", f"{results['p_value']:.2e}")
m4.metric("Test accuracy", f"{results['accuracy_test']*100:.1f}%")

st.caption(
    "Note: p-value is an approximation using Fisher Information (similar to your original notebook approach). "
    "If you want textbook GLM inference output, swap to statsmodels."
)

alpha = 0.05
if results["p_value"] < alpha:
    st.success(f"Statistically significant effect detected (p < {alpha}). Incentive amount impacts target achievement.")
else:
    st.warning(f"Not statistically significant at α = {alpha}. You may need more data or a different model.")

rec_left, rec_right = st.columns([1, 1], gap="large")

with rec_left:
    st.subheader("Recommendation")
    st.write(
        f"To reach **{p*100:.0f}% probability** of meeting the target, the estimated minimum incentive is:"
    )

    if np.isfinite(incentive_p):
        st.metric("Minimum incentive (Euros/Day)", f"{incentive_p:.2f}")
        st.metric("Rounded up (Euros/Day)", f"{int(np.ceil(incentive_p))}")
    else:
        st.error("Could not compute incentive threshold (coefficient is zero or invalid).")

with rec_right:
    st.subheader("Plots")
    fig1 = plot_box(df)
    st.pyplot(fig1, clear_figure=True)

    fig2 = plot_logistic_curve(df, results["model"], show_75_line=True)
    st.pyplot(fig2, clear_figure=True)

st.divider()

st.subheader("What to say in your portfolio (keep it concept-first)")
st.write(
    "- **Define**: Incentive policy is ineffective, only a small share hits productivity target.\n"
    "- **Measure**: Experiment with incentive levels and record binary outcome (hit target vs not).\n"
    "- **Analyse**: Fit logistic regression to quantify incentive impact and compute threshold for desired probability.\n"
    "- **Improve**: Recommend minimum incentive for a 75% (or chosen) success probability.\n"
    "- **Control**: Monitor success rate over time and rerun model quarterly as workforce/seasonality shifts."
)
