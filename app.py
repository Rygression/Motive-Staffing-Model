import datetime as dt
import streamlit as st
import pandas as pd
import altair as alt
from math import floor

# ------------------------------
# CONFIG & DEFAULTS
# ------------------------------
st.set_page_config(page_title="Headcount Growth Model", layout="wide")
st.title("👥 Interactive Headcount Growth Model")
st.caption("Monthly HC simulation with hires, attrition, promotions, team & segment subtotals through Dec 2026. Includes Conservative/Aggressive presets and a Baseline (midpoint) reset.")

# Teams ordered by promotion ladder
TEAM_LADDER = [
    {"team": "1-2 AM",     "dest": "3-4 AM",    "start": 10, "hires": 1,  "goal_2026": 20,  "attr%": 2.0, "promo_qtr_range": (0, 10)},
    {"team": "3-4 AM",     "dest": "5-9 AM",    "start": 36, "hires": 20, "goal_2026": 121, "attr%": 5.0, "promo_qtr_range": (20, 40)},
    {"team": "5-9 AM",     "dest": "10-19 AM",  "start": 15, "hires": 15, "goal_2026": 29,  "attr%": 2.0, "promo_qtr_range": (15, 30)},
    {"team": "10-19 AM",   "dest": "20-29 AM",  "start": 88, "hires": 4,  "goal_2026": 224, "attr%": 3.0, "promo_qtr_range": (5, 15)},
    {"team": "20-29 AM",   "dest": "30-49 AM",  "start": 44, "hires": 1,  "goal_2026": 87,  "attr%": 2.0, "promo_qtr_range": (5, 15)},
    {"team": "30-49 AM",   "dest": "50-99 AM",  "start": 30, "hires": 1,  "goal_2026": 86,  "attr%": 2.0, "promo_qtr_range": (5, 10)},
    {"team": "50-99 AM",   "dest": "100-149 AM","start": 27, "hires": 0,  "goal_2026": 118, "attr%": 2.0, "promo_qtr_range": (5, 8)},
    {"team": "100-149 AM", "dest": None,        "start": 10, "hires": 0,  "goal_2026": 32,  "attr%": 2.0, "promo_qtr_range": (0, 0)},
]
TEAM_ORDER = [r["team"] for r in TEAM_LADDER]

# Segment groupings
SMB_TEAMS  = ["1-2 AM", "3-4 AM", "5-9 AM"]
CMRL_TEAMS = ["10-19 AM", "20-29 AM", "30-49 AM"]
MM_TEAMS   = ["50-99 AM", "100-149 AM"]

# Precompute promo bounds/midpoints (in %)
LOW_PROMOS   = {r["team"]: r["promo_qtr_range"][0] for r in TEAM_LADDER}
HIGH_PROMOS  = {r["team"]: r["promo_qtr_range"][1] for r in TEAM_LADDER}
MID_PROMOS   = {t: (LOW_PROMOS[t] + HIGH_PROMOS[t]) / 2 for t in LOW_PROMOS}

# ------------------------------
# GLOBAL CONTROLS
# ------------------------------
with st.sidebar:
    st.header("Global Settings")
    start_month = st.date_input("Model Start Month", value=dt.date(2025, 11, 1))
    end_month = dt.date(2026, 12, 1)
    if start_month > end_month:
        st.error("Start month must be on or before Dec 2026."); st.stop()

    promo_rule = st.selectbox(
        "Promotion Timing",
        ["Quarter-Ends (Mar/Jun/Sep/Dec)", "Every 3rd Month from Start"],
        help="Choose when quarterly promotions fire."
    )

    st.markdown("---")
    st.subheader("Assumption Presets (Quarterly Promotion %)")
    preset = st.radio(
        "Preset",
        ["Custom", "Conservative (low)", "Aggressive (high)"],
        help="Only Quarterly Promotion % has explicit ranges in the brief; other inputs remain as set."
    )
    reset_mid = st.button("Reset to Baseline (Midpoint)")


# ------------------------------
# SESSION STATE PRESETS
# ------------------------------
if "_last_preset" not in st.session_state:
    st.session_state["_last_preset"] = "Custom"

def _apply_preset(kind: str):
    """Apply low/high/mid presets to promo sliders (values in %)."""
    for r in TEAM_LADDER:
        key = f"{r['team']}_promo"
        if kind == "low":
            st.session_state[key] = float(LOW_PROMOS[r["team"]])
        elif kind == "high":
            st.session_state[key] = float(HIGH_PROMOS[r["team"]])
        elif kind == "mid":
            st.session_state[key] = float(MID_PROMOS[r["team"]])

# Auto-apply radio preset
if preset != st.session_state["_last_preset"]:
    if preset.startswith("Conservative"):
        _apply_preset("low")
    elif preset.startswith("Aggressive"):
        _apply_preset("high")
    st.session_state["_last_preset"] = preset

# Apply explicit reset to midpoint (doesn't change preset label)
if reset_mid:
    _apply_preset("mid")

# ------------------------------
# TEAM INPUTS (sliders)
# ------------------------------
st.sidebar.subheader("Team Assumptions")
team_inputs = {}
for row in TEAM_LADDER:
    t = row["team"]
    dest = row["dest"]

    # Keys
    start_key = f"{t}_start"
    hires_key = f"{t}_hires"
    attr_key  = f"{t}_attr"
    promo_key = f"{t}_promo"

    # Initialize once
    if start_key not in st.session_state: st.session_state[start_key] = row["start"]
    if hires_key not in st.session_state: st.session_state[hires_key] = row["hires"]
    if attr_key  not in st.session_state: st.session_state[attr_key]  = float(row["attr%"])  # %
    if promo_key not in st.session_state: st.session_state[promo_key] = float(MID_PROMOS[t]) # start midpoint

    with st.sidebar.expander(f"{t}  →  {dest or 'n/a'}", expanded=False):
        start_val = st.slider(f"[{t}] Starting HC", 0, max_starting_hc, int(st.session_state[start_key]), key=start_key)
        hires_val = st.slider(f"[{t}] Monthly Hires", 0, max_monthly_hires, int(st.session_state[hires_key]), key=hires_key)
        attr_val  = st.slider(f"[{t}] Monthly Attrition (%)", 0.0, 10.0, float(st.session_state[attr_key]), step=0.5, key=attr_key)

        low, high = row["promo_qtr_range"]
        promo_val = st.slider(
            f"[{t}] Quarterly Promotion Rate (%)",
            0.0, 50.0, float(st.session_state[promo_key]),
            step=0.5,
            key=promo_key,
            help=f"Typical range: {low}-{high}% (quarterly). Presets and Reset affect this value."
        )

        team_inputs[t] = {
            "dest": dest,
            "start": int(start_val),
            "hires": int(hires_val),
            "attr": float(attr_val) / 100.0,       # monthly decimal
            "promo_qtr": float(promo_val) / 100.0  # quarterly decimal
        }

# ------------------------------
# SIMULATION ENGINE (rounding down each step)
# ------------------------------
def month_range(start_date: dt.date, end_date: dt.date):
    months = []
    y, m = start_date.year, start_date.month
    while (y < end_date.year) or (y == end_date.year and m <= end_date.month):
        months.append(dt.date(y, m, 1))
        m += 1
        if m == 13:
            y += 1; m = 1
    return months

def is_promo_month(d: dt.date, start: dt.date, rule: str) -> bool:
    if rule.startswith("Quarter-Ends"):
        return d.month in (3, 6, 9, 12)
    idx = (d.year - start.year) * 12 + (d.month - start.month)
    return (idx + 1) % 3 == 0

months = month_range(start_month, end_month)

# start state (integers)
current = {t: int(team_inputs[t]["start"]) for t in team_inputs}
history_rows = []

for mi, month in enumerate(months, start=1):
    # 1) Hires
    for t, p in team_inputs.items():
        current[t] = int(current[t] + p["hires"])

    # 2) Attrition
    for t, p in team_inputs.items():
        current[t] = max(0, int(floor(current[t] * (1.0 - p["attr"]))))

    # 3) Promotions (quarterly)
    if is_promo_month(month, start_month, promo_rule):
        promotions = {t: 0 for t in team_inputs}
        for t, p in team_inputs.items():
            dest = p["dest"]; r = p["promo_qtr"]
            if dest and r > 0:
                move = int(floor(current[t] * r))  # whole people only
                promotions[t] -= move
                promotions[dest] += move
        for t, delta in promotions.items():
            current[t] = max(0, int(current[t] + delta))

    # snapshot
    month_label = month.strftime("%b %Y")
    snap = {"Month": month, "MonthLabel": month_label, "MonthOrder": mi}
    for t in current:
        snap[t] = int(max(0, current[t]))
    # segment subtotals
    snap["SMB"]  = int(sum(snap[t] for t in SMB_TEAMS))
    snap["CMRL"] = int(sum(snap[t] for t in CMRL_TEAMS))
    snap["MM"]   = int(sum(snap[t] for t in MM_TEAMS))
    snap["Total"] = int(sum(snap[t] for t in team_inputs))  # base teams only
    history_rows.append(snap)

df = pd.DataFrame(history_rows)

# Display copy without datetime64; preserve sort with MonthOrder
df_display = df.drop(columns=["Month"]).copy()
month_order_labels = list(dict.fromkeys(df_display.sort_values("MonthOrder")["MonthLabel"].tolist()))

# ------------------------------
# OUTPUTS
# ------------------------------
left, right = st.columns([2, 1])

# ---- Combo chart (Teams as lines + Total as bars), container width + legend at bottom ----
with left:
    st.subheader("Headcount Projection (Teams as Lines, Total as Bars)")
    st.markdown("<div style='margin-top: 7rem;'></div>", unsafe_allow_html=True)

    team_cols = list(team_inputs.keys())

    # Long form for Altair
    lines_long = df_display.melt(
        id_vars=["MonthLabel", "MonthOrder", "Total"],
        value_vars=team_cols,
        var_name="Team",
        value_name="Headcount"
    )

    # Team ordering helpers
    order_map = {team: i for i, team in enumerate(TEAM_ORDER)}
    lines_long["TeamOrder"] = lines_long["Team"].map(order_map)

    # Chronological x-axis
    x_enc = alt.X(
        "MonthLabel:N",
        sort=month_order_labels,
        axis=alt.Axis(title="Month", labelAngle=-45, labelLimit=110)
    )

    # Lines per team (legend/color order = TEAM_ORDER; draw order by TeamOrder)
    line_chart = (
        alt.Chart(lines_long)
        .mark_line()
        .encode(
            x=x_enc,
            y=alt.Y("Headcount:Q", title="Headcount"),
            color=alt.Color(
    "Team:N",
    scale=alt.Scale(domain=TEAM_ORDER),
    legend=alt.Legend(
        title="Team",
        orient="bottom",
        direction="horizontal",
        columns=8,
        labelLimit=1800,  # 👈 prevents Altair from wrapping labels
        symbolLimit=200
    )
),
            order=alt.Order("TeamOrder:Q")
        )
    )

    # Bars for aggregate total
    bars = (
        alt.Chart(df_display)
        .mark_bar(opacity=0.25)
        .encode(
            x=x_enc,
            y=alt.Y("Total:Q", title="Headcount"),
            tooltip=[
                alt.Tooltip("MonthLabel:N", title="Month"),
                alt.Tooltip("Total:Q", title="Total", format=",.0f"),
            ],
        )
    )

    combined_chart = (bars + line_chart).properties(
        width="container",
        height=360,
        padding={"top": 10, "bottom": 10, "left": 10, "right": 10}
    ).configure_view(
        strokeWidth=0
    ).configure_legend(
        orient="bottom",
        direction="horizontal",
        columns=8,           # 👈 8 columns, 1 row
        labelFontSize=11,
        titleFontSize=12,
        symbolLimit=200,
        padding=0,
        symbolSize=80,
        labelLimit=1800      # 👈 allow legend to extend fully across the container
    )

    # keep container width so it fits the column responsively
    st.altair_chart(combined_chart, use_container_width=True)


# ---- KPI panel: ladder order + SMB/CMRL/MM subtotals + Total (no double counting) ----
with right:
    st.subheader("Key KPIs (End of 2026)")
    final_row = df_display.sort_values("MonthOrder").iloc[-1]
    st.metric("Total HC at End of 2026", f"{final_row['Total']:,}")

    goals = {r["team"]: r["goal_2026"] for r in TEAM_LADDER}

    # Base team rows in ladder order
    base_rows = []
    for t in TEAM_ORDER:
        goal = goals.get(t)
        actual = int(final_row[t])
        delta = (None if goal is None else actual - goal)
        base_rows.append({
            "Team": t,
            "Goal 12/2026": goal,
            "Actual 12/2026": actual,
            "Δ vs Goal": (None if goal is None else delta)
        })

    def subtotal_row(name, teams):
        g = sum(goals[t] for t in teams if goals.get(t) is not None)
        a = int(sum(int(final_row[t]) for t in teams))
        return {"Team": f"{name} Subtotal", "Goal 12/2026": g, "Actual 12/2026": a, "Δ vs Goal": a - g}

    smb_sub  = subtotal_row("SMB", SMB_TEAMS)
    cmrl_sub = subtotal_row("CMRL", CMRL_TEAMS)
    mm_sub   = subtotal_row("MM", MM_TEAMS)

    total_goal   = sum(g for g in goals.values() if g is not None)
    total_actual = int(final_row["Total"])  # from base teams only
    total_delta  = total_actual - total_goal
    total_row = {"Team": "Total", "Goal 12/2026": total_goal, "Actual 12/2026": total_actual, "Δ vs Goal": total_delta}

    kpi_df = pd.DataFrame(base_rows + [smb_sub, cmrl_sub, mm_sub, total_row])
    st.dataframe(kpi_df, use_container_width=True)

# ---- Full projection table (Month as string, includes segment subtotals) + CSV download ----
st.subheader("Projection Table (scrollable)")
fmt_map = {**{t: "{:,.0f}" for t in team_inputs}, "SMB": "{:,.0f}", "CMRL": "{:,.0f}", "MM": "{:,.0f}", "Total": "{:,.0f}"}
proj_df = (
    df_display
    .sort_values("MonthOrder")
    .rename(columns={"MonthLabel": "Month"})
    .drop(columns=["MonthOrder"])
)
st.dataframe(proj_df.style.format(fmt_map), use_container_width=True, height=360)

csv_bytes = proj_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="headcount_projection.csv", mime="text/csv")

# ------------------------------
# PROMOTION FLOW SNAPSHOT (Next Promo)
# ------------------------------
def next_promo_month(start: dt.date, labels: list[str], rule: str):
    # Reconstruct date sequence from labels
    seq = [dt.date(int(lbl.split()[-1]), dt.datetime.strptime(lbl.split()[0], "%b").month, 1) for lbl in labels]
    for m in seq:
        if is_promo_month(m, start, rule):
            return m
    return None

nxt = next_promo_month(start_month, month_order_labels, promo_rule)
st.markdown("### Promotion Flow (Next Promo Event Snapshot)")
if nxt is None:
    st.info("No promo month in the selected range.")
else:
    snap_lbl = nxt.strftime("%b %Y")
    snap_idx = df_display.index[df_display["MonthLabel"] == snap_lbl][0]
    base = df_display.loc[snap_idx]
    flow_rows = []
    for t, p in team_inputs.items():
        dest = p["dest"]; rate = p["promo_qtr"]
        base_hc = int(base[t])
        move = int(floor(base_hc * rate)) if dest else 0
        flow_rows.append({
            "From": t, "To": dest or "n/a",
            "HC Base": base_hc,
            "Promo Rate (qtr)": f"{rate*100:.1f}%",
            "HC Moving": move
        })
    st.dataframe(pd.DataFrame(flow_rows), use_container_width=True)

# ------------------------------
# FOOTER / NOTES
# ------------------------------
with st.expander("Model Notes & Assumptions"):
    st.markdown("""
- **Presets:** Conservative/Aggressive toggle sets all **Quarterly Promotion %** to each team's low/high bound. **Reset to Baseline** sets each to its midpoint.
- **Rounding:** All headcount math rounds **down to whole people** after each step.
- **Order per month:** (1) Hires → (2) Attrition → (3) Promotions (if promo month).
- **Segments:** Monthly **SMB/CMRL/MM** subtotals are shown.
- **Chart:** Teams as **lines** and aggregate **Total** as translucent **bars**.
""")
