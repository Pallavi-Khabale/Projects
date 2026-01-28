# app.py
# Inventory Simulation – Streamlit Web App (4 tabs)
# Tabs: Simulator | Compare | Monte Carlo | Profit Segmentation
# Engine modules remain unchanged.


import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from inventory.inventory_models import InventoryParams
from inventory.inventory_analysis import InventorySimulation

# -----------------------------
# Global reproducibility (kept)
# -----------------------------
BASE_SEED = 1991
np.random.seed(BASE_SEED)

st.set_page_config(page_title="Inventory Simulation", layout="wide")

# -----------------------------
# Session state: click once, then auto-recompute
# -----------------------------
if "has_run" not in st.session_state:
    st.session_state.has_run = False

# -----------------------------
# Scenario presets (Hooks 1–6)
# -----------------------------
def _defaults():
    return dict(
        D=2000,
        T_total=365,
        LD=0,
        T=10,
        Q=55.0,
        initial_ioh=55.0,
        sigma=0.0,
        method_label="Simple Ordering",
    )

D0, T_total0 = 2000, 365
SCENARIOS = {
    "Custom (manual)": _defaults(),
    "Hook 1 — LD=1, Simple Ordering (clean sawtooth)": dict(
        D=D0, T_total=T_total0, LD=1, T=10, Q=55.0, initial_ioh=55.0, sigma=0.0, method_label="Simple Ordering"
    ),
    "Hook 2 — LD=2, Simple Ordering (timing breaks)": dict(
        D=D0, T_total=T_total0, LD=2, T=10, Q=55.0, initial_ioh=55.0, sigma=0.0, method_label="Simple Ordering"
    ),
    "Hook 3 — LD=2, Higher Q (fix symptom, excess IOH)": dict(
        D=D0, T_total=T_total0, LD=2, T=10, Q=60.0, initial_ioh=60.0, sigma=0.0, method_label="Simple Ordering"
    ),
    "Hook 4 — LD=2, Lead-time Ordering (fix timing)": dict(
        D=D0, T_total=T_total0, LD=2, T=10, Q=55.0, initial_ioh=55.0, sigma=0.0, method_label="Lead-time Ordering"
    ),
    "Hook 5 — EOQ-style sawtooth (manual values)": dict(
        D=D0, T_total=T_total0, LD=1, T=73, Q=400.0, initial_ioh=400.0, sigma=0.0, method_label="Simple Ordering"
    ),
    "Hook 6 — Stochastic demand, LD=5 (need safety stock)": dict(
        D=D0, T_total=T_total0, LD=5, T=10, Q=55.0, initial_ioh=55.0, sigma=2.5, method_label="Simple Ordering"
    ),
}

# -----------------------------
# Helpers
# -----------------------------
def run_one_simulation(params: InventoryParams, method_key: str) -> pd.DataFrame:
    engine = InventorySimulation(params)
    if method_key == "order_leadtime":
        return engine.simulation_2(method="order_leadtime")
    if method_key == "order":
        return engine.simulation_2(method="order")
    return engine.simulation_1()

def compute_kpis(df: pd.DataFrame, T: int) -> dict:
    stockout_days = int((df["ioh"] < 0).sum())
    min_ioh = float(df["ioh"].min())
    avg_ioh = float(df["ioh"].mean())
    max_ioh = float(df["ioh"].max())

    fill_rate_proxy = float((df["ioh"] >= 0).mean())
    max_backorder = float(max(0, -min_ioh))

    cycle_idx = ((df["time"] - 1) // max(int(T), 1)).astype(int)
    cycle_stockout = df.groupby(cycle_idx)["ioh"].apply(lambda s: (s < 0).any())
    cycle_service = float((~cycle_stockout).mean()) if len(cycle_stockout) else 1.0

    return {
        "stockout_days": stockout_days,
        "min_ioh": min_ioh,
        "avg_ioh": avg_ioh,
        "max_ioh": max_ioh,
        "fill_rate_proxy": fill_rate_proxy,
        "cycle_service": cycle_service,
        "max_backorder": max_backorder,
    }

def z_from_service_level(service_level: float) -> float:
    grid = {
        0.50: 0.000, 0.60: 0.253, 0.70: 0.524, 0.80: 0.842, 0.85: 1.036,
        0.90: 1.282, 0.95: 1.645, 0.97: 1.881, 0.98: 2.054, 0.99: 2.326, 0.995: 2.576
    }
    sl = float(np.clip(service_level, 0.50, 0.995))
    xs = sorted(grid.keys())
    if sl in grid:
        return grid[sl]
    lo = max(x for x in xs if x < sl)
    hi = min(x for x in xs if x > sl)
    return float(grid[lo] + (grid[hi] - grid[lo]) * ((sl - lo) / (hi - lo)))

def safety_stock_and_rop(D_day: float, sigma: float, LD: int, service_level: float) -> tuple[float, float, float]:
    z = z_from_service_level(service_level)
    sigma_lt = float(sigma) * float(np.sqrt(max(int(LD), 0)))
    ss = float(z) * sigma_lt
    rop = float(D_day) * float(max(int(LD), 0)) + ss
    return z, ss, rop

def plot_three_panel(df: pd.DataFrame, title: str = ""):
    fig, axes = plt.subplots(3, 1, figsize=(9, 4), sharex=True)

    df.plot(x="time", y="demand", ax=axes[0], color="r", legend=False, grid=True)
    axes[0].set_ylabel("Demand", fontsize=8)
    if title:
        axes[0].set_title(title, fontsize=10)

    df.plot.scatter(x="time", y="order", ax=axes[1], color="b")
    axes[1].set_ylabel("Orders", fontsize=8)
    axes[1].grid(True)

    df.plot(x="time", y="ioh", ax=axes[2], color="g", legend=False, grid=True)
    axes[2].set_ylabel("IOH", fontsize=8)
    axes[2].set_xlabel("Time (day)", fontsize=8)

    axes[2].set_xlim(0, int(df["time"].max()))
    for ax in axes:
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

    plt.tight_layout()
    return fig

def monte_carlo_ioh_bands(base_params: InventoryParams, method_key: str, n: int, base_seed: int) -> pd.DataFrame:
    ioh_runs = []
    time = None
    for i in range(int(n)):
        np.random.seed(int(base_seed) + i)  # engine uses np.random.*
        df_i = run_one_simulation(base_params, method_key)
        if time is None:
            time = df_i["time"].to_numpy()
        ioh_runs.append(df_i["ioh"].to_numpy())

    mat = np.vstack(ioh_runs)
    out = pd.DataFrame(
        {
            "time": time,
            "mean_ioh": mat.mean(axis=0),
            "p10_ioh": np.percentile(mat, 10, axis=0),
            "p90_ioh": np.percentile(mat, 90, axis=0),
        }
    )
    return out

def download_zip(files: dict[str, bytes], zip_name: str):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in files.items():
            zf.writestr(fname, fbytes)
    buf.seek(0)
    st.download_button(
        label="Download outputs (ZIP)",
        data=buf.getvalue(),
        file_name=zip_name,
        mime="application/zip",
    )

# -----------------------------
# Sidebar (shared inputs for sim tabs)
# -----------------------------
with st.sidebar:
    st.markdown("**Inventory Parameters**")

    scenario_name = st.selectbox("Scenario preset", list(SCENARIOS.keys()), index=0)
    preset = SCENARIOS[scenario_name]
    preset_method_label = preset.get("method_label", "Simple Ordering")

    D = st.number_input("Annual demand D (units/year)", min_value=1, value=int(preset["D"]), step=50)
    T_total = st.number_input("Horizon T_total (days)", min_value=1, value=int(preset["T_total"]), step=1)
    LD = st.number_input("Lead time LD (days)", min_value=0, value=int(preset["LD"]), step=1)
    T = st.number_input("Cycle time T (days)", min_value=1, value=int(preset["T"]), step=1)
    Q = st.number_input("Order quantity Q (units)", min_value=0.0, value=float(preset["Q"]), step=10.0, format="%.2f")
    initial_ioh = st.number_input("Initial inventory on hand", min_value=0.0, value=float(preset["initial_ioh"]), step=1.0, format="%.2f")
    sigma = st.number_input("Daily demand std. dev. σ (units/day)", min_value=0.0, value=float(preset["sigma"]), step=0.5, format="%.2f")

    method = st.radio(
        "Ordering method",
        options=["Simple Ordering", "Lead-time Ordering"],
        index=0 if preset_method_label == "Simple Ordering" else 1,
    )
    method_key = "order_leadtime" if method.startswith("Lead-time") else "order"

    st.markdown("---")
    c_run, c_reset = st.columns(2)
    with c_run:
        run = st.button("Run", type="primary", disabled=st.session_state.has_run)
    with c_reset:
        reset = st.button("Reset")

    if reset:
        st.session_state.has_run = False
    if run:
        st.session_state.has_run = True

# -----------------------------
# Main header + quick context cards
# -----------------------------
st.title("Inventory Simulation Web Application")

D_day = D / T_total

st.markdown(
    """
<style>
.quick-card{padding:.9rem 1rem;border:1px solid #eaeaea;border-radius:12px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.quick-card .label{font-size:.85rem;color:#5f6b7a;margin:0}
.quick-card .value{font-size:1.25rem;font-weight:700;margin:.15rem 0 0 0;color:#111}
.quick-card .unit{font-size:.8rem;color:#8a95a3;margin:0}
</style>
""",
    unsafe_allow_html=True,
)

def quick_card(label, value, unit=""):
    unit_html = f'<p class="unit">{unit}</p>' if unit else ""
    st.markdown(
        f'<div class="quick-card"><p class="label">{label}</p><p class="value">{value}</p>{unit_html}</div>',
        unsafe_allow_html=True,
    )

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    quick_card("Average daily demand", f"{D_day:,.2f}", "units/day")
with c2:
    quick_card("Lead time", f"{LD}", "days")
with c3:
    quick_card("Cycle time", f"{T}", "days")
with c4:
    quick_card("Order quantity Q", f"{Q:,.0f}", "units")
with c5:
    quick_card("Initial IOH", f"{initial_ioh:,.0f}", "units")
with c6:
    quick_card("Demand σ", f"{sigma:.2f}", "units/day")

# Shared params object (used across tabs once has_run is True)
params = InventoryParams(
    D=float(D),
    T_total=int(T_total),
    LD=int(LD),
    T=int(T),
    Q=float(Q),
    initial_ioh=float(initial_ioh),
    sigma=float(sigma),
)

# -----------------------------
# Tabs
# -----------------------------
tab_sim, tab_compare, tab_mc, tab_seg = st.tabs(
    ["Simulator", "Compare", "Monte Carlo", "Profit Segmentation"]
)

# =============================
# TAB 1: Simulator
# =============================
with tab_sim:
    st.subheader("Simulator")

    with st.expander("Safety Stock + ROP helper (stochastic cases)", expanded=False):
        st.caption("Simple helper based on Normal demand and independent daily variation.")
        service_level = st.slider("Target cycle service level (approx.)", 0.50, 0.995, 0.95, 0.005, key="sl_sim")
        z, ss, rop = safety_stock_and_rop(D_day=D_day, sigma=sigma, LD=LD, service_level=service_level)
        m1, m2, m3 = st.columns(3)
        m1.metric("Z-value", f"{z:.3f}")
        m2.metric("Safety stock (units)", f"{ss:,.1f}")
        m3.metric("ROP estimate (units)", f"{rop:,.1f}")
        st.write(f"ROP ≈ D_day × LD + SS = {D_day:,.2f} × {LD} + {ss:,.1f}")

    if not st.session_state.has_run:
        st.info("Set parameters in the sidebar and click **Run simulation**.")
    else:
        np.random.seed(BASE_SEED)
        df = run_one_simulation(params, method_key)
        k = compute_kpis(df, T=int(T))

        st.pyplot(plot_three_panel(df, method), clear_figure=True, use_container_width=True)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Stockout days", f"{k['stockout_days']}")
        k2.metric("Min IOH", f"{k['min_ioh']:,.0f}")
        k3.metric("Avg IOH", f"{k['avg_ioh']:,.0f}")
        k4.metric("Fill-rate proxy", f"{k['fill_rate_proxy']*100:,.1f}%")
        k5.metric("Cycle service", f"{k['cycle_service']*100:,.1f}%")
        k6.metric("Max backorder", f"{k['max_backorder']:,.0f}")

        with st.expander("View results table"):
            st.dataframe(df, use_container_width=True)

        # Export
        kpi_df = pd.DataFrame(
            [{
                "policy": method,
                "D": float(D),
                "T_total": int(T_total),
                "LD": int(LD),
                "T": int(T),
                "Q": float(Q),
                "initial_ioh": float(initial_ioh),
                "sigma": float(sigma),
                **k,
            }]
        )
        files = {
            "results.csv": df.to_csv(index=False).encode("utf-8"),
            "kpis.csv": kpi_df.to_csv(index=False).encode("utf-8"),
        }
        st.markdown("---")
        download_zip(files, "simulator_outputs.zip")
        st.success("Simulation completed.")

# =============================
# TAB 2: Compare
# =============================
with tab_compare:
    st.subheader("Compare policies side-by-side (Simple vs Lead-time)")

    if not st.session_state.has_run:
        st.info("Set parameters in the sidebar and click **Run simulation**.")
    else:
        colA, colB = st.columns(2)

        np.random.seed(BASE_SEED)
        df_simple = run_one_simulation(params, "order")
        np.random.seed(BASE_SEED)
        df_lead = run_one_simulation(params, "order_leadtime")

        k_simple = compute_kpis(df_simple, T=int(T))
        k_lead = compute_kpis(df_lead, T=int(T))

        with colA:
            st.markdown("#### Simple Ordering")
            st.pyplot(plot_three_panel(df_simple, "Simple Ordering"), clear_figure=True, use_container_width=True)
        with colB:
            st.markdown("#### Lead-time Ordering")
            st.pyplot(plot_three_panel(df_lead, "Lead-time Ordering"), clear_figure=True, use_container_width=True)

        st.markdown("#### KPI + Service metrics")
        def kpi_row(label: str, k: dict):
            c = st.columns(7)
            c[0].metric(label, "")
            c[1].metric("Stockout days", f"{k['stockout_days']}")
            c[2].metric("Min IOH", f"{k['min_ioh']:,.0f}")
            c[3].metric("Avg IOH", f"{k['avg_ioh']:,.0f}")
            c[4].metric("Fill-rate", f"{k['fill_rate_proxy']*100:,.1f}%")
            c[5].metric("Cycle service", f"{k['cycle_service']*100:,.1f}%")
            c[6].metric("Max backorder", f"{k['max_backorder']:,.0f}")

        kpi_row("Simple", k_simple)
        kpi_row("Lead-time", k_lead)

        st.markdown("#### Delta (Lead-time − Simple)")
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("Stockout days", f"{k_lead['stockout_days'] - k_simple['stockout_days']}")
        d2.metric("Min IOH", f"{k_lead['min_ioh'] - k_simple['min_ioh']:+,.0f}")
        d3.metric("Avg IOH", f"{k_lead['avg_ioh'] - k_simple['avg_ioh']:+,.0f}")
        d4.metric("Fill-rate", f"{(k_lead['fill_rate_proxy'] - k_simple['fill_rate_proxy'])*100:+,.1f}%")
        d5.metric("Cycle service", f"{(k_lead['cycle_service'] - k_simple['cycle_service'])*100:+,.1f}%")
        d6.metric("Max backorder", f"{k_lead['max_backorder'] - k_simple['max_backorder']:+,.0f}")

        with st.expander("View results tables"):
            t1, t2 = st.columns(2)
            with t1:
                st.write("Simple Ordering (df)")
                st.dataframe(df_simple, use_container_width=True)
            with t2:
                st.write("Lead-time Ordering (df)")
                st.dataframe(df_lead, use_container_width=True)

        kpi_summary = pd.DataFrame(
            [
                {"policy": "Simple Ordering", **k_simple},
                {"policy": "Lead-time Ordering", **k_lead},
            ]
        )
        files = {
            "results_simple_ordering.csv": df_simple.to_csv(index=False).encode("utf-8"),
            "results_lead_time_ordering.csv": df_lead.to_csv(index=False).encode("utf-8"),
            "kpis_comparison.csv": kpi_summary.to_csv(index=False).encode("utf-8"),
        }
        st.markdown("---")
        download_zip(files, "compare_outputs.zip")
        st.success("Comparison completed.")

# =============================
# TAB 3: Monte Carlo
# =============================
with tab_mc:
    st.subheader("Monte Carlo (confidence bands)")

    mc_n = st.slider("Monte Carlo runs (N)", 20, 300, 200, 10, key="mc_n")
    mc_policy = st.radio(
        "Policy to simulate",
        options=["Simple Ordering", "Lead-time Ordering"],
        index=0 if method_key == "order" else 1,
        key="mc_policy",
    )
    mc_method_key = "order" if mc_policy.startswith("Simple") else "order_leadtime"

    if not st.session_state.has_run:
        st.info("Set parameters in the sidebar and click **Run simulation**.")
    else:
        bands = monte_carlo_ioh_bands(params, mc_method_key, mc_n, BASE_SEED)

        fig = plt.figure(figsize=(9, 3))
        ax = plt.gca()
        ax.plot(bands["time"], bands["mean_ioh"])
        ax.plot(bands["time"], bands["p10_ioh"])
        ax.plot(bands["time"], bands["p90_ioh"])
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("IOH")
        ax.grid(True)
        ax.set_title(f"IOH bands (mean / p10 / p90) — {mc_policy}")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

        st.caption("Bands show variability in IOH across repeated runs with different seeds (deterministic series).")

        files = {
            "monte_carlo_bands.csv": bands.to_csv(index=False).encode("utf-8"),
        }
        st.markdown("---")
        download_zip(files, "monte_carlo_outputs.zip")
        st.success("Monte Carlo completed.")

# =============================
# TAB 4: Profit Segmentation
# =============================
with tab_seg:
    st.subheader("Profit Segmentation (ABC + optional XYZ)")
    st.caption(
        "Upload a SKU-level dataset to segment products by profit contribution (ABC). "
        "Optionally compute XYZ based on demand variability if you provide demand history."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="seg_upload")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        sku_col = st.text_input("SKU column name", value="sku")
    with col2:
        profit_col = st.text_input("Profit column name", value="profit")
    with col3:
        demand_col = st.text_input("Demand column name (optional, for XYZ)", value="demand")

    xyz_on = st.toggle("Also compute XYZ (requires demand column)", value=False)

    if uploaded is None:
        st.info("Upload a CSV to run segmentation. Recommended columns: sku, profit, demand (optional).")
    else:
        df_seg = pd.read_csv(uploaded)

        # Basic validation
        missing = []
        if sku_col not in df_seg.columns:
            missing.append(sku_col)
        if profit_col not in df_seg.columns:
            missing.append(profit_col)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            seg = df_seg.copy()

            # Clean numeric profit
            seg[profit_col] = pd.to_numeric(seg[profit_col], errors="coerce").fillna(0.0)

            # ABC by profit contribution
            seg = seg.groupby(sku_col, as_index=False)[profit_col].sum()
            seg = seg.sort_values(profit_col, ascending=False)
            total_profit = seg[profit_col].sum()
            seg["profit_share"] = seg[profit_col] / total_profit if total_profit != 0 else 0.0
            seg["cum_profit_share"] = seg["profit_share"].cumsum()

            # Classic cutoffs: A up to 80%, B up to 95%, C rest
            def abc_class(cum):
                if cum <= 0.80:
                    return "A"
                if cum <= 0.95:
                    return "B"
                return "C"

            seg["ABC"] = seg["cum_profit_share"].apply(abc_class)

            # Optional XYZ (needs demand history at SKU-date granularity ideally)
            # Here we compute CV = std/mean per SKU using the provided demand column
            if xyz_on:
                if demand_col not in df_seg.columns:
                    st.warning("XYZ toggled on, but demand column not found. Showing ABC only.")
                else:
                    tmp = df_seg[[sku_col, demand_col]].copy()
                    tmp[demand_col] = pd.to_numeric(tmp[demand_col], errors="coerce")
                    tmp = tmp.dropna(subset=[demand_col])

                    stats = tmp.groupby(sku_col)[demand_col].agg(["mean", "std"]).reset_index()
                    stats["cv"] = stats["std"] / stats["mean"].replace({0: np.nan})
                    stats["cv"] = stats["cv"].fillna(np.inf)

                    # Common heuristic: X <= 0.5, Y <= 1.0, Z > 1.0
                    def xyz_class(cv):
                        if cv <= 0.5:
                            return "X"
                        if cv <= 1.0:
                            return "Y"
                        return "Z"

                    stats["XYZ"] = stats["cv"].apply(xyz_class)

                    seg = seg.merge(stats[[sku_col, "mean", "std", "cv", "XYZ"]], on=sku_col, how="left")

            st.markdown("#### Segmentation output")
            st.dataframe(seg, use_container_width=True)

            # Simple summary
            s1, s2, s3 = st.columns(3)
            s1.metric("SKUs", f"{seg[sku_col].nunique():,}")
            s2.metric("Total profit", f"{total_profit:,.0f}")
            s3.metric("A-class SKUs", f"{(seg['ABC']=='A').sum():,}")

            # Export
            files = {
                "profit_segmentation.csv": seg.to_csv(index=False).encode("utf-8"),
            }
            st.markdown("---")
            download_zip(files, "profit_segmentation_outputs.zip")
            st.success("Segmentation completed.")
