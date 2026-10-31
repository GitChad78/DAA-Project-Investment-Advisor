import os
import sys
from dataclasses import replace
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import Stock
from src.knapsack_bb import solve_fifo_knapsack, solve_lc_knapsack, solve_with_constraints, solve_greedy
from src.data_loader import ensure_preset_datasets_exist, fetch_live_nse_data, ASSUMED_BASIS, HISTORICAL_BASIS
from src.visualizations import render_portfolio_charts, render_scaling_benchmark_chart, render_state_space_tree

st.set_page_config(page_title="Budget-Constrained Investment Advisor", page_icon="📈", layout="wide")
st.title("📈 Budget-Constrained Investment Advisor")
st.caption("0/1 Knapsack Portfolio Optimization using Live NSE Market Data & Branch & Bound")

ensure_preset_datasets_exist()

# Sidebar Controls
st.sidebar.header("🕹️ Controls & Parameters")
dataset_option = st.sidebar.selectbox("Select Asset Dataset Source", [
    "Real-Time NSE Live Market Data (yfinance)", "Preset Benchmark (n=10)", "Preset Benchmark (n=15)", 
    "Preset Benchmark (n=20)", "Preset Benchmark (n=25)", "Preset Benchmark (n=30)", "Custom Asset Input"
])
algorithm_choice = st.sidebar.radio("Select Branch & Bound Strategy", ["LC (Least Cost / Best-First) B&B", "FIFO B&B"])

return_basis = ASSUMED_BASIS
if dataset_option == "Real-Time NSE Live Market Data (yfinance)":
    return_basis = st.sidebar.radio("Projected return basis (live data)", [ASSUMED_BASIS, HISTORICAL_BASIS])

st.sidebar.subheader("⚖️ Risk & Constraints")
risk_aversion = st.sidebar.slider("Risk aversion (λ)", 0.0, 1.0, 0.5, 0.1)
max_asset_pct = st.sidebar.slider("Max % of budget in one asset", 10, 100, 40, 5)
sector_cap_pct = st.sidebar.slider("Max % of budget in one sector (ETFs exempt)", 10, 100, 50, 5)
require_etf = st.sidebar.checkbox("Require at least one ETF", value=False)

# Dataset Resolution
stocks = []
if dataset_option == "Real-Time NSE Live Market Data (yfinance)":
    with st.spinner("Fetching live stock prices from NSE (Batch Master List)..."):
        stocks, fetch_notes = fetch_live_nse_data(return_basis)
    for note in fetch_notes: st.warning(note)
elif "Preset Benchmark" in dataset_option:
    n_val = int(dataset_option.split("=")[1].replace(")", ""))
    df = pd.read_csv(f"data/stocks_n{n_val}.csv")
    stocks = [Stock(id=int(r["id"]), name=str(r["name"]), cost=float(r["cost"]), return_val=float(r["return_val"])) for _, r in df.iterrows()]

if stocks:
    total_market_cap = sum(s.cost for s in stocks)
    budget = st.number_input("Enter Total Capital Budget Constraint W (in ₹)", min_value=10.0, max_value=total_market_cap * 2.0, value=float(round(total_market_cap * 0.45, -2)), step=500.0)

    asset_cap_amt = budget * max_asset_pct / 100
    sector_cap_amt = budget * sector_cap_pct / 100
    candidates = [replace(s, return_val=s.risk_adjusted_profit(risk_aversion)) for s in stocks if s.cost <= asset_cap_amt]

    solver = solve_lc_knapsack if algorithm_choice == "LC (Least Cost / Best-First) B&B" else solve_fifo_knapsack
    result = solve_with_constraints(candidates, budget, solver, sector_cap_amt, require_etf)
    greedy = solve_greedy(candidates, budget, sector_cap_amt, require_etf)

    feasible = result.get("feasible", True)
    by_id = {s.id: s for s in stocks}
    selected_stocks = [by_id[s.id] for s in result["selected_stocks"]]

    st.subheader("📋 Available NSE Investment Options")
    st.dataframe(pd.DataFrame([{
        "ID": s.id, "Asset Name": s.name, "Sector": s.sector, "Price (₹)": f"₹{s.cost:,.2f}", 
        "Return (%)": round(s.profit / s.cost * 100, 2), "Volatility (%)": round(s.volatility * 100, 1)
    } for s in stocks]), width="stretch")

    st.markdown("---")
    st.subheader("🎯 Recommended Portfolio Allocation")
    m1, m2, m3 = st.columns(3)
    m1.metric("Projected Value", f"₹{sum(s.return_val for s in selected_stocks):,.2f}")
    m2.metric("Net Profit", f"₹{(sum(s.return_val for s in selected_stocks) - sum(s.cost for s in selected_stocks)):,.2f}")
    m3.metric("Search Nodes Generated", result["nodes_generated"])

    if selected_stocks:
        rec_df = pd.DataFrame([{
            "ID": s.id, "Selected Asset": s.name, "Sector": s.sector, "Capital Required (₹)": f"₹{s.cost:,.2f}", "Volatility (%)": round(s.volatility * 100, 1)
        } for s in selected_stocks])
        st.dataframe(rec_df, width="stretch")
        st.download_button("📥 Download Portfolio Allocation as CSV", rec_df.to_csv(index=False).encode('utf-8'), "recommended_portfolio.csv", "text/csv")

        # Render Imported Visualizations
        render_portfolio_charts(selected_stocks)
        render_scaling_benchmark_chart()
        render_state_space_tree()
