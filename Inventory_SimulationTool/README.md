# Inventory Simulation - Streamlit Web App
  
The app lets you test **fixed-cycle replenishment** under deterministic and stochastic demand, visualize **Demand / Orders / IOH**, and compare **simple vs lead-time-aware** ordering.

- Click **Run simulation** once; after that, any parameter change **auto-recomputes** and redraws.
- Results are **reproducible** (fixed random seed).
- Engine modules remain **as-is**, wrapped with a minimal Streamlit UI.

## Live Demo & Deploy It Yourself

This app is deployed on **Streamlit Community Cloud**:  
👉 https://inventory-simulation-tool.streamlit.app/


## Features

### Core Simulation
- Deterministic and stochastic demand simulations
- Fixed-cycle replenishment policies
- Simple Ordering vs **Lead-time Ordering** (receipt timing aligned with cycle end)
- Compact **3-panel chart**: Demand / Orders / IOH
- Inline **Quick Context** cards (key inputs at a glance)
- KPIs:
  - Stockout days
  - Min IOH
  - Avg IOH
  - Fill-rate proxy
  - Cycle service level
  - Max backorder depth
- Auto re-run after initial click

### Advanced Analysis (Tabbed Interface)
- **Simulator tab** – single-policy exploration
- **Compare tab** – side-by-side comparison of Simple vs Lead-time Ordering
- **Monte Carlo tab** – IOH confidence bands under stochastic demand
- **Profit Segmentation tab** – ABC (and optional XYZ) classification using uploaded data

---





## Project structure

```
inventory /
├─ app.py
├─ requirements.txt
└─ inventory/
├─ init.py
├─ inventory_analysis.py
└─ inventory_models.py
```

## How to Use

```
# Run the app
streamlit run app.py
```

1. In the **sidebar**, set:
   - `D` (annual demand), `T_total` (days), `LD` (lead time), `T` (cycle), `Q` (order qty),
     `initial_ioh` (initial stock), `sigma` (daily demand std. dev.), and **Method**.
2. Click **Run simulation** (first time only).
3. Adjust any parameter: the app **auto-recomputes**.
4. Use tabs to explore:
    - Single-policy behaviour
    - Policy comparison
    - Risk under uncertainty
    - Profit-based SKU segmentation
5. Interpret the **Demand / Orders / IOH** chart.
6. Export results as CSV or ZIP

> Reproducibility: the app uses a fixed NumPy seed (`1991`).

## Ready-Made Scenarios (for demos)

> Defaults often used: `D = 2000`, `T_total = 365` ⇒ `D_day ≈ 5.48`.  
> With `T = 10` → `Q ≈ 55` and `initial_ioh ≈ 55`. Keep `sigma = 0` unless noted.

### Hook 1

**“What if your inventory touched zero without ever stocking out?”**

**Scenario 1 — Lead time = 1 (receive next day)**  
**Set:** `LD=1`, `T=10`, `Q≈55`, `initial_ioh≈55`, `sigma=0`, **Method:** Simple Ordering  
**See:** IOH drains to ~0 at cycle end; receipt next day → **no negatives** (clean sawtooth).

---

### Hook 2

**“Same policy, +1 day lead time—what breaks first?”**

**Scenario 2 — Lead time = 2 (receive two days later)**  
**Set:** `LD=2`, `T=10`, `Q≈55`, `initial_ioh≈55`, `sigma=0`, **Method:** Simple Ordering  
**See:** Receipt arrives late; IOH dips **below zero** → **stockouts** (timing issue).

---

### Hook 3

**“Can we fix stockouts by just ordering more?”**

**Scenario 3 — Keep timing, increase quantity**  
**Set:** `LD=2`, `T=10`, `Q≈ D_day × (T + (LD−1)) ≈ 60`, `initial_ioh=60`, `sigma=0`, **Method:** Simple Ordering  
**See:** Negatives avoided but **excess inventory** (higher holding costs).  
**Lesson:** Treats the **symptom** (level), not the **cause** (timing).

---

### Hook 4

**“What if we keep quantity but fix the timing?”**

**Scenario 4 — Anticipate lead time (lead-time-aware trigger)**  
**Set:** `LD=2`, `T=10`, `Q≈55`, `initial_ioh≈55`, `sigma=0`, **Method:** Lead-time Ordering  
**See:** Receipt realigns with cycle boundary; stable service without excess.  
**Interpretation:** Approx. `ROP ≈ D_day × LD`.

---

### Hook 5

**“What does the EOQ sawtooth actually look like?”**

**Scenario 5 — EOQ cycle, lead time = 1 (manual)**  
**Set:** `LD=1`, `Q=400`, `T≈73`, `initial_ioh=400`, `sigma=0`, **Method:** Simple Ordering  
**See:** Classic EOQ sawtooth; **avg IOH ≈ Q\*/2**; **no negatives**.

---

### Hook 6

**“Add uncertainty—does timing still save you?”**

**Scenario 6 — Stochastic demand (Normal), lead time = 5**  
**Set:** `LD=5`, `T=10`, `Q≈55`, `initial_ioh≈55`, `sigma=2.5`, **Method:** Simple Ordering  
**See:** IOH fluctuates; **stockouts can appear** despite fixed timing.  
**Lesson:** You need **safety stock** for demand/lead-time variability.


## Monte Carlo Analysis (Risk View)

The Monte Carlo tab runs the same policy N times with different random demand paths.

It shows:
- **Mean IOH**
- **P10 / P90 IOH bands**

This answers:
>**“How risky is this policy under demand uncertainty?”**

If lower bands dip below zero, stockout risk exists even if the average looks fine.


## Safety Stock & ROP Helper (Decision Support)

For stochastic cases, the app includes a Safety Stock + ROP helper:
- Computes safety stock using:
  - Safety Stock = Z × σ × √LD
  - ROP ≈ D_day × LD + Safety Stock
- Z-value is based on the selected service level.

**Important:**
This helper is **advisory** only.
It **does not automatically modify** the simulation or demo scenarios.

Planners apply the recommendation manually (e.g. increasing initial_ioh or Q) and re-run the simulation to evaluate trade-offs.

## Profit Segmentation (ABC / XYZ)

The Profit Segmentation tab allows uploading SKU-level data to:
- Classify items using ABC analysis based on profit contribution
- Optionally compute XYZ segmentation using demand variability (CV)

This bridges inventory policy exploration with portfolio-level prioritisation.

## About me 🤓

Supply Chain Analyst and SAP consultant with international experience working on Demand, Supply Planning and Inventory Management. 
Feel free to contact me via [LinkedIn](https://linkedin.com/in/pallavi-khabale)
