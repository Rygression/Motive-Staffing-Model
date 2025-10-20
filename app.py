import streamlit as st
import pandas as pd

st.set_page_config(page_title="Headcount Planning Model", layout="wide")

st.title("🧮 Headcount & Productivity Model")
st.write("Adjust assumptions below to see the required headcount and ARR trajectory.")

# --- Inputs ---
st.sidebar.header("Model Inputs")
target_arr = st.sidebar.number_input("Total ARR Goal ($)", value=10_000_000, step=500_000)
quota = st.sidebar.slider("Quota per Rep ($)", 50_000, 300_000, 150_000, step=10_000)
attainment = st.sidebar.slider("Average Attainment (%)", 50, 120, 90)
ramp_months = st.sidebar.slider("Ramp Duration (months)", 1, 12, 6)
attrition = st.sidebar.slider("Annual Rep Attrition (%)", 0, 50, 10)
span = st.sidebar.slider("Manager Span of Control (# reps per mgr)", 4, 12, 8)

# --- Core Calculations ---
# Productivity after ramp and attrition
effective_productivity = quota * (attainment / 100) * ((12 - ramp_months/2) / 12) * (1 - attrition/100)
reps_needed = target_arr / effective_productivity
managers_needed = reps_needed / span

# --- Results ---
st.subheader("Results Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total ARR Goal", f"${target_arr:,.0f}")
col2.metric("Effective Annual Productivity / Rep", f"${effective_productivity:,.0f}")
col3.metric("Reps Required", f"{reps_needed:,.1f}")

st.metric("Managers Required", f"{managers_needed:,.1f}")

# --- Optional Table ---
st.subheader("Quarterly Hiring Plan (illustrative)")
df = pd.DataFrame({
    "Quarter": ["Q1", "Q2", "Q3", "Q4"],
    "Active Reps": [
        reps_needed * 0.25,
        reps_needed * 0.50,
        reps_needed * 0.75,
        reps_needed
    ],
    "ARR Delivered ($)": [
        target_arr * 0.20,
        target_arr * 0.45,
        target_arr * 0.75,
        target_arr
    ]
})
st.dataframe(df.style.format({"ARR Delivered ($)": "${:,.0f}", "Active Reps": "{:,.1f}"}))

# --- Chart ---
st.line_chart(df.set_index("Quarter")["ARR Delivered ($)"])

st.caption("Adjust the sliders in the sidebar to explore different staffing scenarios.")
