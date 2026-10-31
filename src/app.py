import os
import sys
from dataclasses import replace
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import Stock
from src.knapsack_bb import (
    solve_fifo_knapsack,
    solve_lc_knapsack,
    solve_with_constraints,
    solve_greedy,
)

st.set_page_config(
    page_title="Budget-Constrained Investment Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Budget-Constrained Investment Advisor")
st.caption("0/1 Knapsack Portfolio Optimization using Live NSE Market Data & Branch & Bound")

ASSUMED_BASIS = "Assumed target (simulated)"
HISTORICAL_BASIS = "Trailing 1Y return (historical)"
SECTORS = ["Banking", "IT", "Energy", "FMCG", "Auto", "Infrastructure", "Telecom", "ETF", "Other"]
DEFAULT_VOL = 0.25  # used when a live ticker has too little history

# Sidebar Configuration
st.sidebar.header("🕹️ Controls & Parameters")

dataset_option = st.sidebar.selectbox(
    "Select Asset Dataset Source",
    [
        "Real-Time NSE Live Market Data (yfinance)",
        "Preset Benchmark (n=10)", 
        "Preset Benchmark (n=15)", 
        "Preset Benchmark (n=20)", 
        "Preset Benchmark (n=25)", 
        "Preset Benchmark (n=30)", 
        "Custom Asset Input"
    ]
)

algorithm_choice = st.sidebar.radio(
    "Select Branch & Bound Strategy",
    ["LC (Least Cost / Best-First) B&B", "FIFO B&B"]
)

return_basis = ASSUMED_BASIS
if dataset_option == "Real-Time NSE Live Market Data (yfinance)":
    return_basis = st.sidebar.radio(
        "Projected return basis (live data)",
        [ASSUMED_BASIS, HISTORICAL_BASIS],
        help="Assumed = fixed growth target per stock (a simulation). "
             "Historical = each stock's own trailing 1-year price change, extrapolated forward."
    )

st.sidebar.subheader("⚖️ Risk & Constraints")
risk_aversion = st.sidebar.slider(
    "Risk aversion (λ)", 0.0, 1.0, 0.5, 0.1,
    help="Optimizer score = net profit − λ × volatility × capital. 0 ignores risk."
)
max_asset_pct = st.sidebar.slider("Max % of budget in one asset", 10, 100, 40, 5)
sector_cap_pct = st.sidebar.slider("Max % of budget in one sector (ETFs exempt)", 10, 100, 50, 5)
require_etf = st.sidebar.checkbox("Require at least one ETF", value=False)

@st.cache_data(ttl=300)  # Cache live prices for 5 minutes
def fetch_live_nse_data(return_basis: str) -> tuple[list[Stock], list[str]]:
    """Fetches live prices + 1Y history from NSE via yfinance. Returns (stocks, notes)."""
    tickers_map = {
        # symbol: (name, assumed growth multiplier, sector)
        "RELIANCE.NS": ("Reliance Industries", 1.18, "Energy"),
        "TCS.NS": ("Tata Consultancy Services", 1.16, "IT"),
        "HDFCBANK.NS": ("HDFC Bank", 1.18, "Banking"),
        "INFY.NS": ("Infosys", 1.19, "IT"),
        "ICICIBANK.NS": ("ICICI Bank", 1.17, "Banking"),
        "NIFTYBEES.NS": ("Nifty 50 ETF", 1.15, "ETF"),
        "GOLDBEES.NS": ("Gold BeES ETF", 1.12, "ETF"),
        "LT.NS": ("Larsen & Toubro", 1.18, "Infrastructure"),
        "ITC.NS": ("ITC Ltd", 1.14, "FMCG"),
        "TATAMOTORS.NS": ("Tata Motors", 1.20, "Auto"),
        "BHARTIARTL.NS": ("Bharti Airtel", 1.18, "Telecom"),
        "SBIN.NS": ("State Bank of India", 1.17, "Banking")
    }

    stocks_list: list[Stock] = []
    notes: list[str] = []
    missing: list[str] = []

    try:
        data = yf.download(list(tickers_map.keys()), period="1y", progress=False)['Close']
        idx = 1
        for symbol, (name, assumed_mult, sector) in tickers_map.items():
            series = data[symbol].dropna() if symbol in data else None
            if series is None or series.empty:
                missing.append(symbol)
                continue

            price = float(series.iloc[-1])
            daily = series.pct_change().dropna()
            if len(daily) >= 30:
                volatility = float(daily.std() * np.sqrt(252))
                trailing = price / float(series.iloc[0]) - 1.0
            else:
                volatility, trailing = DEFAULT_VOL, None
                notes.append(f"{symbol}: under 30 days of history - assumed {DEFAULT_VOL:.0%} volatility and growth target.")

            if return_basis == HISTORICAL_BASIS and trailing is not None:
                multiplier = 1.0 + trailing
            else:
                multiplier = assumed_mult

            stocks_list.append(
                Stock(
                    id=idx,
                    name=f"{name} ({symbol.replace('.NS', '')})",
                    cost=round(price, 2),
                    return_val=round(price * multiplier, 2),
                    volatility=round(volatility, 4),
                    sector=sector,
                )
            )
            idx += 1

        if not stocks_list:
            raise ValueError("no price data returned")
        if missing:
            notes.append(f"No price data for {', '.join(missing)} - skipped (no placeholder price used).")
    except Exception as e:
        notes.append(f"Unable to reach live market feeds: {e}. Showing offline sample quotes (NOT live prices; volatilities are illustrative).")
        # Fallback data if network/API fails: (name, price, projected value, volatility, sector)
        fallback_data = [
            ("Reliance Industries", 2950.0, 3480.0, 0.22, "Energy"),
            ("TCS", 4120.0, 4780.0, 0.22, "IT"),
            ("HDFC Bank", 1680.0, 1980.0, 0.20, "Banking"),
            ("Infosys", 1850.0, 2200.0, 0.25, "IT"),
            ("Nifty 50 ETF", 270.0, 310.0, 0.15, "ETF"),
            ("Gold BeES ETF", 68.0, 78.0, 0.12, "ETF")
        ]
        stocks_list = [
            Stock(id=i+1, name=item[0], cost=item[1], return_val=item[2], volatility=item[3], sector=item[4])
            for i, item in enumerate(fallback_data)
        ]

    return stocks_list, notes

stocks = []
basis_note = ""

if dataset_option == "Real-Time NSE Live Market Data (yfinance)":
    with st.spinner("Fetching live stock prices from NSE via yfinance..."):
        stocks, fetch_notes = fetch_live_nse_data(return_basis)
    for note in fetch_notes:
        st.warning(note)
    if return_basis == HISTORICAL_BASIS:
        basis_note = "Projected values extrapolate each stock's trailing 1-year price change; past performance does not predict future returns."
    else:
        basis_note = "Projected values use assumed growth targets per stock (a simulation), not forecasts."

elif "Preset Benchmark" in dataset_option:
    n_val = int(dataset_option.split("=")[1].replace(")", ""))
    csv_path = f"data/stocks_n{n_val}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        stocks = [
            Stock(id=int(row["id"]), name=str(row["name"]), cost=float(row["cost"]), return_val=float(row["return_val"]))
            for _, row in df.iterrows()
        ]
        st.caption("Preset datasets are synthetic and have no volatility or sector data: a uniform 20% volatility is assumed and the sector cap is not applied.")
        basis_note = "Preset benchmark values are randomly generated for algorithm testing."
    else:
        st.error(f"Dataset file `{csv_path}` not found. Please run the benchmark suite first!")

else:
    st.sidebar.subheader("Add Custom Indian Asset")
    custom_name = st.sidebar.text_input("Asset Name", value="Custom Asset")
    custom_cost = st.sidebar.number_input("Asset Price / Min Lot (₹)", min_value=1.0, value=1000.0, step=100.0)
    custom_return = st.sidebar.number_input("Projected Value after 1 year (₹)", min_value=1.0, value=1250.0, step=100.0)
    custom_sector = st.sidebar.selectbox("Sector", SECTORS, index=SECTORS.index("Other"))
    custom_vol = st.sidebar.number_input("Volatility (% p.a.)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    
    if "custom_stocks" not in st.session_state:
        st.session_state.custom_stocks = [
            Stock(id=1, name="Nifty 50 ETF", cost=270.0, return_val=310.0, volatility=0.15, sector="ETF"),
            Stock(id=2, name="TCS", cost=4120.0, return_val=4780.0, volatility=0.22, sector="IT"),
            Stock(id=3, name="HDFC Bank", cost=1680.0, return_val=1980.0, volatility=0.20, sector="Banking")
        ]
        
    if st.sidebar.button("➕ Add Asset"):
        new_id = len(st.session_state.custom_stocks) + 1
        st.session_state.custom_stocks.append(
            Stock(id=new_id, name=custom_name, cost=custom_cost, return_val=custom_return,
                  volatility=custom_vol / 100.0, sector=custom_sector)
        )
    
    stocks = st.session_state.custom_stocks
    basis_note = "Projected values and volatilities are user-entered."

if stocks:
    # Capital Budget Input in INR
    total_market_cap = sum(s.cost for s in stocks)
    default_budget = round(total_market_cap * 0.45, -2)
    
    budget = st.number_input(
        "Enter Total Capital Budget Constraint W (in ₹)", 
        min_value=10.0, 
        max_value=total_market_cap * 2.0, 
        value=float(default_budget), 
        step=500.0
    )

    # --- Apply risk & constraints -------------------------------------------------
    has_sector_data = any(s.sector != "Other" for s in stocks)
    asset_cap_amt = budget * max_asset_pct / 100
    sector_cap_amt = budget * sector_cap_pct / 100 if has_sector_data else None

    use_etf_rule = require_etf
    if require_etf and not any(s.is_etf for s in stocks):
        st.sidebar.caption("No ETF in this dataset - ETF requirement ignored.")
        use_etf_rule = False

    # The solver maximizes Stock.return_val, so feed it the risk-adjusted net profit
    # (profit - λ·volatility·cost) instead of the gross projected value.
    candidates = [
        replace(s, return_val=s.risk_adjusted_profit(risk_aversion))
        for s in stocks if s.cost <= asset_cap_amt
    ]

    # Execute Optimization Engine
    solver = solve_lc_knapsack if algorithm_choice == "LC (Least Cost / Best-First) B&B" else solve_fifo_knapsack
    result = solve_with_constraints(candidates, budget, solver, sector_cap_amt, use_etf_rule)
    greedy = solve_greedy(candidates, budget, sector_cap_amt, use_etf_rule)

    feasible = result.get("feasible", True)
    by_id = {s.id: s for s in stocks}
    selected_stocks = [by_id[s.id] for s in result["selected_stocks"]]
    selected_ids = {s.id for s in selected_stocks}

    deployed = sum(s.cost for s in selected_stocks)
    projected_value = sum(s.return_val for s in selected_stocks)
    net_profit = projected_value - deployed
    roi = (net_profit / deployed * 100) if deployed > 0 else 0.0
    unspent = budget - deployed
    wavg_vol = (sum(s.volatility * s.cost for s in selected_stocks) / deployed * 100) if deployed > 0 else 0.0

    def asset_status(s: Stock) -> str:
        if s.id in selected_ids:
            return "✅ Selected"
        if s.cost > asset_cap_amt:
            return f"⛔ Over {max_asset_pct}% cap"
        if s.risk_adjusted_profit(risk_aversion) <= 0:
            return "⛔ Risk-adj. profit ≤ 0"
        return "—"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Available NSE Investment Options")
        inv_df = pd.DataFrame([{
            "ID": s.id, 
            "Asset Name": s.name, 
            "Sector": s.sector,
            "Price / Share (₹)": f"₹{s.cost:,.2f}", 
            "Projected Value (₹)": f"₹{s.return_val:,.2f}", 
            "Exp. Return (%)": round(s.profit / s.cost * 100, 2),
            "Volatility (%)": round(s.volatility * 100, 1),
            "Risk-adj. Profit (₹)": round(s.risk_adjusted_profit(risk_aversion), 2),
            "Status": asset_status(s)
        } for s in stocks])
        st.dataframe(inv_df, use_container_width=True)

    with col2:
        st.subheader("🎯 Recommended Portfolio Allocation")

        if not feasible:
            st.warning("An ETF is required, but none fits within the budget and per-asset cap.")
        
        # Display Metrics in INR
        m1, m2, m3 = st.columns(3)
        m1.metric("Projected Value", f"₹{projected_value:,.2f}")
        m2.metric("Net Profit", f"₹{net_profit:,.2f}")
        m3.metric("Projected ROI", f"{roi:.2f}%")

        m4, m5, m6 = st.columns(3)
        m4.metric("Capital Deployed", f"₹{deployed:,.2f}")
        m5.metric("Unspent Cash", f"₹{unspent:,.2f}")
        m6.metric("Wtd. Avg. Volatility", f"{wavg_vol:.1f}%")
        st.progress(
            min(deployed / budget, 1.0) if budget > 0 else 0.0,
            text=f"{deployed / budget * 100:.1f}% of ₹{budget:,.2f} budget deployed"
        )

        # Tree Pruning Metrics
        st.markdown("---")
        p1, p2 = st.columns(2)
        p1.metric("Search Tree Nodes Generated", result["nodes_generated"])
        p2.metric("Subtrees Pruned", result["nodes_pruned"])

        if selected_stocks:
            rec_df = pd.DataFrame([{
                "ID": s.id, 
                "Selected Asset": s.name, 
                "Sector": s.sector,
                "Capital Required (₹)": f"₹{s.cost:,.2f}", 
                "Projected Value (₹)": f"₹{s.return_val:,.2f}",
                "Net Profit (₹)": f"₹{s.profit:,.2f}",
                "Volatility (%)": round(s.volatility * 100, 1)
            } for s in selected_stocks])
            st.dataframe(rec_df, use_container_width=True)
        elif feasible:
            st.warning(
                "No asset passes the budget, per-asset cap and risk filters. "
                "Try a larger budget, a lower risk aversion, or higher caps."
            )

    # --- Greedy baseline comparison -----------------------------------------------
    if feasible:
        st.markdown("---")
        st.subheader("⚖️ Branch & Bound vs Greedy Baseline")
        bb_score = result["max_profit"]
        greedy_score = greedy["max_profit"]
        gap = bb_score - greedy_score

        g1, g2, g3 = st.columns(3)
        g1.metric("B&B Risk-adj. Profit", f"₹{bb_score:,.2f}")
        g2.metric("Greedy Risk-adj. Profit", f"₹{greedy_score:,.2f}")
        if gap > 1e-9:
            g3.metric(
                "B&B Advantage", f"₹{gap:,.2f}",
                delta=f"{gap / greedy_score * 100:.1f}% vs greedy" if greedy_score > 0 else None
            )
        else:
            g3.metric("B&B Advantage", "₹0.00")
            st.caption("Greedy matched the optimal portfolio on this input.")

        greedy_names = ", ".join(s.name for s in greedy["selected_stocks"]) or "nothing"
        st.caption(f"Greedy (picks by value/cost ratio while it fits) selected: {greedy_names}")

    st.caption(
        "⚠️ Educational project, not financial advice. " + basis_note +
        " Volatility is annualised from daily prices, and the weighted average ignores correlation between assets."
    )
