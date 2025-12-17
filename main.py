
# main.py
# Streamlit + Plotly Multi-Investment Tracker (annual compounding) 
# - Persisted Calculate state so Retirement & Drawdown inputs don't force a recalculation click

from __future__ import annotations
import io
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

# -------------------------
# Data model
# -------------------------

@dataclass
class InvestmentConfig:
    name: str
    start_date: date
    end_date: date                # compounding stops here
    starting_amount: float
    monthly_installment: float
    annual_rate_pct: float
    installments_end_date: Optional[date] = None  # if None -> use end_date


# -------------------------
# Helpers
# -------------------------

def _to_date(x) -> date:
    if isinstance(x, date):
        return x
    try:
        return date.fromisoformat(str(x))
    except Exception:
        return pd.to_datetime(x).date()

def export_investments(configs: List[InvestmentConfig] | List[dict]) -> list:
    payload = []
    for c in configs:
        d = c.__dict__.copy() if isinstance(c, InvestmentConfig) else dict(c)
        for k in ("start_date", "end_date", "installments_end_date"):
            v = d.get(k)
            if isinstance(v, date):
                d[k] = v.isoformat()
        payload.append(d)
    return payload

def import_investments(raw_list: list) -> List[InvestmentConfig]:
    out: List[InvestmentConfig] = []
    for c in raw_list:
        c = dict(c)
        c["start_date"] = _to_date(c["start_date"])
        c["end_date"] = _to_date(c["end_date"])
        if c.get("installments_end_date"):
            c["installments_end_date"] = _to_date(c["installments_end_date"])
        else:
            c["installments_end_date"] = c["end_date"]
        c["starting_amount"] = float(c["starting_amount"])
        c["monthly_installment"] = float(c["monthly_installment"])
        c["annual_rate_pct"] = float(c["annual_rate_pct"])
        out.append(InvestmentConfig(**c))
    return out

# --- NEW: full scenario helpers (with retirement+inflation) ---
def export_scenario() -> str:
    """Dump the whole scenario (investments + settings) to JSON."""
    investments = export_investments(st.session_state.defaults)

    settings = {
        "last_salary": float(st.session_state.get("ret_last_salary", 1500.0)),
        "social_security_rate_pct": float(st.session_state.get("ret_ss_rate", 70.0)),
        "annual_inflation_rate_pct": float(st.session_state.get("infl_rate_pct", 2.0)),
    }
    scenario = {"investments": investments, "settings": settings}
    return json.dumps(scenario, indent=2)

def import_scenario(json_str: str) -> List[InvestmentConfig]:
    """
    Load a scenario; supports:
      - OLD format: a list of investments
      - NEW format: {"investments":[...], "settings":{...}}
    """
    raw = json.loads(json_str)

    # OLD format (list): no settings; just investments
    if isinstance(raw, list):
        cfgs = import_investments(raw)
        # leave settings untouched if not provided
        return cfgs

    # NEW format (dict)
    inv_raw = raw.get("investments", [])
    cfgs = import_investments(inv_raw)

    # pull settings if present; set sensible defaults otherwise
    settings = raw.get("settings", {})
    if "last_salary" in settings:
        st.session_state["ret_last_salary"] = float(settings["last_salary"])
    if "social_security_rate_pct" in settings:
        st.session_state["ret_ss_rate"] = float(settings["social_security_rate_pct"])
    if "annual_inflation_rate_pct" in settings:
        st.session_state["infl_rate_pct"] = float(settings["annual_inflation_rate_pct"])

    return cfgs

def month_range_inclusive(start: date, end: date) -> pd.DatetimeIndex:
    start_month = pd.Timestamp(start).normalize().replace(day=1)
    end_month = pd.Timestamp(end).normalize().replace(day=1)
    if end_month < start_month:
        return pd.DatetimeIndex([])
    return pd.date_range(start_month, end_month, freq="MS")

def build_schedule(cfg: InvestmentConfig) -> pd.DataFrame:
    """
    Annual compounding with monthly contributions (lot-by-lot aging).
    - Month 1: only the starting amount exists (no monthly installment).
    - Each month:
        1) Apply annual interest ONLY to lots whose age is a multiple of 12 months.
        2) Add this month's installment as a NEW lot (age 0) while current month <= installments_end_date.
    """
    months = month_range_inclusive(cfg.start_date, cfg.end_date)
    if len(months) == 0:
        return pd.DataFrame(columns=[
            "date","investment","month_index","opening_balance",
            "interest","installment","closing_balance","annual_rate_pct"
        ])

    r_annual = cfg.annual_rate_pct / 100.0
    inst_end = (cfg.installments_end_date or cfg.end_date)

    def ym(ts: pd.Timestamp) -> int:
        return ts.year * 12 + ts.month

    lots = [{"balance": float(cfg.starting_amount), "ym0": ym(months[0])}]
    rows = []

    for idx, dt in enumerate(months, start=1):
        current_ym = ym(dt)
        opening = sum(l["balance"] for l in lots)

        interest_this_month = 0.0
        for l in lots:
            age_months = current_ym - l["ym0"]
            if age_months >= 12 and (age_months % 12 == 0):
                interest_amt = l["balance"] * r_annual
                l["balance"] += interest_amt
                interest_this_month += interest_amt

        installment = 0.0
        if idx > 1 and dt.date() <= inst_end:
            installment = float(cfg.monthly_installment)
            lots.append({"balance": installment, "ym0": current_ym})

        closing = opening + interest_this_month + installment

        rows.append({
            "date": dt.date(),
            "investment": cfg.name,
            "month_index": idx,
            "opening_balance": opening,
            "interest": interest_this_month,
            "installment": installment,
            "closing_balance": closing,
            "annual_rate_pct": cfg.annual_rate_pct,
        })

    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "investment","start","end","months","rate",
            "initial","total_contributions","total_interest","net_gain","final_value"
        ])
    g = df.groupby("investment", as_index=False).agg(
        start=("date", "min"),
        end=("date", "max"),
        months=("month_index", "max"),
        final_value=("closing_balance", "last"),
        total_contributions=("installment", "sum"),
        total_interest=("interest", "sum"),
        initial=("opening_balance", "first"),
        rate=("annual_rate_pct", "first"),
    )
    g["net_gain"] = g["final_value"] - (g["initial"] + g["total_contributions"])
    return g[["investment","start","end","months","rate","initial",
              "total_contributions","total_interest","net_gain","final_value"]]

def to_csv_download(df: pd.DataFrame, filename: str = "investment_schedule.csv") -> Tuple[bytes, str]:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode(), filename



# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="📈 Investment Tracker", layout="wide")
st.title("📈 Investment Tracker")
st.caption("Multi investment tracker through compounding and monthly installments.")

# ---- SIDEBAR (Scenario first, then Controls) ----
with st.sidebar:
    st.header("Scenario")

    uploaded = st.file_uploader("Load scenario (.json)", type="json", key="uploader")
    if uploaded and not st.session_state.get("scenario_loaded", False):
        text = uploaded.read().decode("utf-8")
        configs_loaded = import_scenario(text)

        # seed state BEFORE any widget instantiation
        st.session_state.defaults = [c.__dict__ for c in configs_loaded]
        st.session_state["n_investments"] = len(configs_loaded)
        st.session_state["scenario_loaded"] = True

        st.success(f"Loaded {len(configs_loaded)} investments from scenario")
        st.rerun()

    # allow another upload once cleared
    if not uploaded and st.session_state.get("scenario_loaded", False):
        st.session_state.pop("scenario_loaded", None)

    # JSON download
    if st.session_state.get("defaults"):
        scenario_str = export_scenario()
        fname = st.text_input("Download scenario (.json)", value="scenario.json")
        st.download_button(
            "💾 Save current scenario",
            data=scenario_str,
            file_name=fname,
            mime="application/json",
        )

    st.header("Controls")
    # Ensure seed BEFORE rendering number_input
    if "n_investments" not in st.session_state:
        st.session_state["n_investments"] = max(1, len(st.session_state.get("defaults", [])) or 1)

    n = st.number_input(
        "Number of investments",
        min_value=1, max_value=50, step=1,
        value=st.session_state["n_investments"],
        key="n_investments",
    )

# Date bounds
today = date.today()
MIN_DATE = date(1970, 1, 1)
MAX_DATE = date(today.year + 50, 12, 31)
default_end = date(min(today.year + 1, MAX_DATE.year), today.month, 1)

# Prepare defaults
if "defaults" not in st.session_state:
    st.session_state.defaults = []

# Grow defaults to N
while len(st.session_state.defaults) < n:
    i = len(st.session_state.defaults)
    st.session_state.defaults.append({
        "name": f"Investment {i+1}",
        "start_date": date(today.year, today.month, 1),
        "end_date": default_end,
        "starting_amount": 1000.0,
        "monthly_installment": 200.0,
        "annual_rate_pct": 6.0,
        "installments_end_date": default_end,
    })

# Back-compat & type safety: ensure field exists + dates are date objects
for d in st.session_state.defaults:
    d["start_date"] = _to_date(d["start_date"])
    d["end_date"] = _to_date(d["end_date"])
    if not d.get("installments_end_date"):
        d["installments_end_date"] = d["end_date"]
    else:
        d["installments_end_date"] = _to_date(d["installments_end_date"])

# Build dynamic sections
tabs = st.tabs([f"Investment {i+1}" for i in range(n)])
configs: List[InvestmentConfig] = []

for i, tab in enumerate(tabs):
    with tab:
        d = st.session_state.defaults[i]
        name = st.text_input("Name", value=d["name"], key=f"name_{i}")

        c1, c2, c3 = st.columns(3)
        with c1:
            start_date = st.date_input(
                "Start date",
                value=d["start_date"], min_value=MIN_DATE, max_value=MAX_DATE,
                key=f"start_{i}",
            )
            starting_amount = st.number_input(
                "Starting amount", min_value=0.0,
                value=float(d["starting_amount"]), step=100.0, key=f"start_amt_{i}"
            )

        with c2:
            end_date = st.date_input(
                "Investment end date",
                value=d["end_date"], min_value=MIN_DATE, max_value=MAX_DATE,
                key=f"end_{i}",
            )
            monthly_installment = st.number_input(
                "Monthly installment", min_value=0.0,
                value=float(d["monthly_installment"]), step=50.0, key=f"inst_{i}"
            )

            inst_end_default = d.get("installments_end_date", end_date)
            early_stop_default = inst_end_default < end_date
            early_stop = st.checkbox(
                "Installments end before investment ends?",
                value=early_stop_default, key=f"earlystop_{i}"
            )
            if early_stop:
                installments_end_date = st.date_input(
                    "Installments end date",
                    value=inst_end_default,
                    min_value=start_date, max_value=end_date,
                    key=f"inst_end_{i}",
                )
            else:
                installments_end_date = end_date

        with c3:
            annual_rate_pct = st.number_input(
                "Expected annual return (%)", min_value=-100.0, max_value=100.0,
                value=float(d["annual_rate_pct"]), step=0.1, key=f"rate_{i}"
            )

        # validations
        if end_date < start_date:
            st.error("End date must be on/after start date.")
        if (end_date.year - start_date.year) > 100:
            st.error("Please pick a period of 100 years or less.")

        # persist back so changes stick when N changes
        st.session_state.defaults[i] = {
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "starting_amount": starting_amount,
            "monthly_installment": monthly_installment,
            "annual_rate_pct": annual_rate_pct,
            "installments_end_date": installments_end_date,
        }

        configs.append(InvestmentConfig(
            name=(name.strip() or f"Investment {i+1}"),
            start_date=start_date,
            end_date=end_date,
            starting_amount=starting_amount,
            monthly_installment=monthly_installment,
            annual_rate_pct=annual_rate_pct,
            installments_end_date=installments_end_date,
        ))

st.markdown("---")

# Persisted Calculate state
if "calculated" not in st.session_state:
    st.session_state.calculated = False

col_calc1, col_calc2 = st.columns([1,1])
with col_calc1:
    if st.button("🚀 Calculate / Recalculate"):
        st.session_state.calculated = True
with col_calc2:
    if st.button("🧹 Reset results"):
        st.session_state.calculated = False

if not st.session_state.calculated:
    st.info("Set inputs and click **Calculate / Recalculate**. Retirement & Drawdown inputs can be changed freely after calculation.")
    st.stop()

# Build schedules
schedules = [build_schedule(cfg) for cfg in configs]
if len(schedules) == 0 or all(s.empty for s in schedules):
    st.warning("No schedule to display. Check your dates.")
    st.stop()

df_all = pd.concat(schedules, ignore_index=True).sort_values(["date","investment"])

# KPIs
summary = summarize(df_all)
total_row = pd.DataFrame({
    "investment": ["TOTAL"],
    "start": [summary["start"].min() if not summary.empty else None],
    "end": [summary["end"].max() if not summary.empty else None],
    "months": [int(summary["months"].sum()) if not summary.empty else 0],
    "rate": [np.nan],
    "initial": [summary["initial"].sum() if not summary.empty else 0.0],
    "total_contributions": [summary["total_contributions"].sum() if not summary.empty else 0.0],
    "total_interest": [summary["total_interest"].sum() if not summary.empty else 0.0],
    "net_gain": [summary["net_gain"].sum() if not summary.empty else 0.0],
    "final_value": [summary["final_value"].sum() if not summary.empty else 0.0],
})
summary_plus = pd.concat([summary, total_row], ignore_index=True)

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Initial (sum)", f"{summary['initial'].sum():,.2f}")
with k2: st.metric("Contributions (sum)", f"{summary['total_contributions'].sum():,.2f}")
with k3: st.metric("Interest (sum)", f"{summary['total_interest'].sum():,.2f}")
with k4: st.metric("Final Value (sum)", f"{summary['final_value'].sum():,.2f}")

st.subheader("Summary by Investment")
st.dataframe(summary_plus, width="stretch",hide_index=True)

# Detailed table
st.subheader("Detailed Monthly Schedule")
gb = GridOptionsBuilder.from_dataframe(df_all)
gb.configure_default_column(filter=True, sortable=True, resizable=True)
grid_options = gb.build()
AgGrid(
    df_all,
    gridOptions=grid_options,
    height=420,
    fit_columns_on_grid_load=True,
    theme="streamlit",
)

# Download table
csv_bytes, fname = to_csv_download(df_all)
st.download_button("Download CSV", data=csv_bytes, file_name=fname, mime="text/csv")

# Chart: per investment + TOTAL as line chart (TOTAL carries forward)
st.subheader("Balances Over Time")

df_all["date"] = pd.to_datetime(df_all["date"])
cal_start = df_all["date"].min()
cal_end   = df_all["date"].max()
calendar = pd.date_range(cal_start, cal_end, freq="MS")

# TOTAL with carry-forward after investments end
totals = []
for inv, g in df_all.groupby("investment"):
    # collapse to one row per date to avoid duplicate index during reindex
    series = g.groupby("date")["closing_balance"].last()
    s = (
        series.reindex(calendar)
              .ffill()
              .fillna(0.0)
    )
    totals.append(s)

total_series = pd.concat(totals, axis=1).sum(axis=1)
total_df = total_series.reset_index()
total_df.columns = ["date", "closing_balance"]
total_df["investment"] = "TOTAL"

plot_df = pd.concat(
    [
        df_all[["date", "investment", "closing_balance"]],
        total_df[["date", "investment", "closing_balance"]],
    ],
    ignore_index=True,
)

fig = px.line(
    plot_df,
    x="date",
    y="closing_balance",
    color="investment",
    labels={"date": "Date", "closing_balance": "Closing Balance", "investment": "Investment"},
    title="Monthly Closing Balance per Investment (TOTAL carries finished balances forward)",
)
# make TOTAL more visible
fig.for_each_trace(lambda t: t.update(line=dict(width=4, dash="solid")) if t.name == "TOTAL" else ())
fig.update_traces(mode="lines")
fig.update_layout(legend_title=None)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Retirement & Drawdown
# -------------------------
st.markdown("---")
st.subheader("🏁 Retirement & Drawdown")

# Inputs (monthly salary assumed)
col_rs1, col_rs2 = st.columns(2)
with col_rs1:
    last_salary = st.number_input(
        "Last salary (monthly)",
        min_value=0.0, value=st.session_state.get("ret_last_salary", 1500.0),
        step=50.0, key="ret_last_salary"
    )
with col_rs2:
    ss_rate_pct = st.number_input(
        "Social Security rate (%)",
        min_value=0.0, max_value=100.0,
        value=st.session_state.get("ret_ss_rate", 70.0),
        step=1.0, key="ret_ss_rate"
    )
    
# Retirement monthly benefit
retirement_benefit = last_salary * (ss_rate_pct / 100.0)

# ---- Draw starts when the LAST investment ends ----
# Find the last investment end month (across all investments)
last_end_date = max(cfg.end_date for cfg in configs)
last_end_ts = pd.Timestamp(last_end_date).normalize().replace(day=1)

# Build a month calendar up to that last end
df_all["date"] = pd.to_datetime(df_all["date"])
calendar_to_last = pd.date_range(df_all["date"].min(), last_end_ts, freq="MS")

# For each investment, carry its closing balance forward (no compounding after its end)
# so we can read balances exactly at last_end_ts
balances_at_last = []
for inv, g in df_all.groupby("investment"):
    series = g.groupby("date")["closing_balance"].last()
    s = (
        series.reindex(calendar_to_last)
              .ffill()        # keep constant after investment ends
              .fillna(0.0)    # before it starts, zero
    )
    balances_at_last.append(s.rename(inv))

balances_at_last_df = pd.concat(balances_at_last, axis=1)
balances_on_last_end = balances_at_last_df.loc[last_end_ts]

# Compute portfolio total at last_end and weighted annual rate using balances at that date
rate_map = {cfg.name: float(cfg.annual_rate_pct) / 100.0 for cfg in configs}
rates_on_last_end = pd.Series(rate_map).reindex(balances_on_last_end.index).fillna(0.0)

# Remove any synthetic TOTAL if present (shouldn’t be, but safe)
if "TOTAL" in balances_on_last_end.index:
    balances_on_last_end = balances_on_last_end.drop(index="TOTAL")
    rates_on_last_end = rates_on_last_end.drop(index="TOTAL")

total_saved = float(balances_on_last_end.sum())
if total_saved > 0:
    weighted_annual_rate = float((balances_on_last_end * rates_on_last_end).sum() / total_saved)
else:
    weighted_annual_rate = 0.0

burn_rate_annual_pct = weighted_annual_rate * 100.0

# Convert to effective monthly draw rate and € draw (still nominal)
monthly_draw_rate = (1.0 + weighted_annual_rate) ** (1.0 / 12.0) - 1.0
monthly_draw_pct = monthly_draw_rate * 100.0
monthly_draw_eur = total_saved * monthly_draw_rate

# Your totals then become:
retirement_installment = retirement_benefit + monthly_draw_eur
PensionRate = retirement_installment / last_salary * 100.0


# Display
kA, kB, kD = st.columns(3)
with kA:
    st.metric("Retirement monthly benefit", f"{retirement_benefit:,.2f}")
with kB:
    st.metric("Portfolio burn rate (annual, weighted)", f"{burn_rate_annual_pct:,.2f}%")
with kD:
    st.metric("€ you could draw monthly", f"{monthly_draw_eur:,.2f}")

st.caption("Total Retirement Monthly Income")
kC,kE=st.columns(2)
with kC:
    st.metric("Total Retirement Monthly Income", f"{retirement_installment:,.2f}")
with kE:
    st.metric("Percentage of Last Salary", f"{PensionRate:,.2f}%")


# Optional helper text
st.caption(
    "Burn rate = weighted annual return across investments (weights = latest balances). "
)

# -------------------------
# Inflation Impact
# -------------------------
st.markdown("---")
st.subheader("📉 Inflation Impact")

# Inputs
c1, c2 = st.columns(2)
with c1:
    infl_rate_pct = st.number_input(
        "Annual inflation rate (%)",
        min_value=0.0, max_value=100.0,
        value=st.session_state.get("infl_rate_pct", 2.0),
        step=0.1, key="infl_rate_pct"
    )
with c2:
    st.write("Draw starts at:", last_end_date.isoformat())
    # Horizon is auto: from today to last_end_date; never negative
    horizon_years = max(0.0, (pd.Timestamp(last_end_date) - pd.Timestamp.today().normalize()).days / 365.25)

infl = infl_rate_pct / 100.0
pv_factor = (1.0 + infl) ** horizon_years if horizon_years > 0 else 1.0

# Nominal final value (sum) you already computed
final_value_sum = summary["final_value"].sum() if not summary.empty else 0.0

# PVs (today's euros), evaluated at the draw start
pv_final_value_today = final_value_sum / pv_factor if pv_factor > 0 else 0.0
real_monthly_draw_today = monthly_draw_eur / pv_factor if pv_factor > 0 else 0.0

# (Optional) PV today of a perpetuity starting at the draw start (discounted by inflation only)
monthly_infl = (1.0 + infl) ** (1.0 / 12.0) - 1.0
if infl > 0:
    pv_perpetuity_today = (monthly_draw_eur / monthly_infl) / pv_factor
else:
    pv_perpetuity_today = float("inf") if monthly_draw_eur > 0 else 0.0

d1, d2 = st.columns(2)
with d1:
    st.metric("PV of Final Value (sum) — today's €", f"{pv_final_value_today:,.2f}")
with d2:
    st.metric("Monthly draw — today's € (at draw start)", f"{real_monthly_draw_today:,.2f}")

