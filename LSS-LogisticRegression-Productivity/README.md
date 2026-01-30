# Lean Six Sigma with Python — Incentive Policy Optimisation (Logistic Regression)

Replacing Minitab with Python to design a data-driven incentive policy for warehouse productivity.

---

## Project Overview

In warehouse and logistics operations, incentive schemes are often implemented without clear analytical validation. Bonuses are paid, yet productivity targets remain unmet, making it difficult for leaders to determine whether incentives are effective or simply increasing cost.

This project demonstrates how **Lean Six Sigma (DMAIC)** principles combined with **logistic regression in Python** can be used to quantify the relationship between incentives and productivity, and to **estimate the minimum incentive required to achieve a target success rate**.

The Streamlit app serves only as a presentation layer.  
The focus of this project is the **analytical approach and decision logic**.

---

## Business Scenario

You are a regional director overseeing multiple warehouses in a 3PL environment.

- Each warehouse has a daily picking productivity target
- Operators receive a daily bonus if they meet the target
- An existing incentive policy performs poorly, with only ~20% of operators meeting the target
- Management wants a data-backed incentive policy rather than trial-and-error adjustments

---

## Objective

Determine the **minimum daily incentive** required to achieve a **75% probability** of operators meeting the productivity target.

---

## Lean Six Sigma Framework (DMAIC)

### Define  
Productivity targets are not being met despite an active incentive policy.

### Measure  
An experiment is conducted across warehouses:
- Daily incentive amounts are varied (e.g. €1–€20)
- Operator outcomes are recorded as binary values (target met: yes or no)

### Analyse  
A **logistic regression model** is used to:
- Model the probability of meeting the target as a function of incentive amount
- Quantify the statistical significance of incentives on productivity outcomes

### Improve  
The fitted model is solved to estimate the **incentive threshold** required to reach a 75% success probability.

### Control  
The model can be reused periodically to:
- Reassess incentive effectiveness
- Adjust thresholds as demand patterns, seasonality, or workforce composition change

---

## Analytical Approach

- Logistic regression is chosen because the outcome variable is binary
- Model coefficients measure how incentive levels influence success probability
- Statistical significance is evaluated using an approximate Wald test
- The final recommendation is derived by solving the logistic equation for a target probability

This mirrors traditional Lean Six Sigma analysis performed in tools such as Minitab, implemented using an open and reproducible Python workflow.

---

## Key Outputs

- Incentive vs productivity probability curve
- Statistical significance of incentive impact
- Recommended minimum incentive to reach a 75% success rate
- Success-rate summary by incentive level for operational transparency

---

## Technology Stack

- Python  
- pandas, numpy  
- scikit-learn (logistic regression)  
- scipy (statistical testing)  
- matplotlib, seaborn (visualisation)  
- Streamlit (presentation layer)

---

