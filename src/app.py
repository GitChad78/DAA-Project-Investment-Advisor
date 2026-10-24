import os
import sys
import pandas as pd
import streamlit as st
import yfinance as yf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import Stock
from src.knapsack_bb import solve_fifo_knapsack, solve_lc_knapsack

st.set_page_config(
    page_title="Budget-Constrained Investment Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Budget-Constrained Investment Advisor")
st.caption("0/1 Knapsack Portfolio Optimization using Live NSE Market Data & Branch & Bound")

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

@st.cache_data(ttl=300)  # Cache live prices for 5 minutes
def fetch_live_nse_data() -> list[Stock]:
    """Fetches real-time live stock prices from NSE via yfinance."""
    tickers_map = {
        "RELIANCE.NS": ("Reliance Industries", 1.18),
        "TCS.NS": ("Tata Consultancy Services", 1.16),
        "HDFCBANK.NS": ("HDFC Bank", 1.18),
        "INFY.NS": ("Infosys", 1.19),
        "ICICIBANK.NS": ("ICICI Bank", 1.17),
        "NIFTYBEES.NS": ("Nifty 50 ETF", 1.15),
        "GOLDBEES.NS": ("Gold BeES ETF", 1.12),
        "LT.NS": ("Larsen & Toubro", 1.18),
        "ITC.NS": ("ITC Ltd", 1.14),
        "TATAMOTORS.NS": ("Tata Motors", 1.20),
        "BHARTIARTL.NS": ("Bharti Airtel", 1.18),
        "SBIN.NS": ("State Bank of India", 1.17)
    }

    stocks_list = []
    
    try:
        data = yf.download(list(tickers_map.keys()), period="1d", progress=False)['Close']
        idx = 1
        for symbol, (name, target_multiplier) in tickers_map.items():
            if symbol in data and not data[symbol].empty:
                price = float(data[symbol].iloc[-1])
            else:
                price = 1000.0  # Fallback price
            
            projected_return = price * target_multiplier
            stocks_list.append(
                Stock(id=idx, name=f"{name} ({symbol.replace('.NS', '')})", cost=round(price, 2), return_val=round(projected_return, 2))
            )
            idx += 1
    except Exception as e:
        st.warning(f"Unable to reach live market feeds: {e}. Falling back to default market quotes.")
        # Fallback dictionary if network/API fails
        fallback_data = [
            ("Reliance Industries", 2950.0, 3480.0),
            ("TCS", 4120.0, 4780.0),
            ("HDFC Bank", 1680.0, 1980.0),
            ("Infosys", 1850.0, 2200.0),
            ("Nifty 50 ETF", 270.0, 310.0),
            ("Gold BeES ETF", 68.0, 78.0)
        ]
        stocks_list = [
            Stock(id=i+1, name=item[0], cost=item[1], return_val=item[2])
            for i, item in enumerate(fallback_data)
        ]
        
    return stocks_list

stocks = []

if dataset_option == "Real-Time NSE Live Market Data (yfinance)":
    with st.spinner("Fetching live stock prices from NSE via yfinance..."):
        stocks = fetch_live_nse_data()

elif "Preset Benchmark" in dataset_option:
    n_val = int(dataset_option.split("=")[1].replace(")", ""))
    csv_path = f"data/stocks_n{n_val}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        stocks = [
            Stock(id=int(row["id"]), name=str(row["name"]), cost=float(row["cost"]), return_val=float(row["return_val"]))
            for _, row in df.iterrows()
        ]
    else:
        st.error(f"Dataset file `{csv_path}` not found. Please run the benchmark suite first!")

else:
    st.sidebar.subheader("Add Custom Indian Asset")
    custom_name = st.sidebar.text_input("Asset Name", value="Custom Asset")
    custom_cost = st.sidebar.number_input("Asset Price / Min Lot (₹)", min_value=1.0, value=1000.0, step=100.0)
    custom_return = st.sidebar.number_input("Target Valuation Return (₹)", min_value=1.0, value=1250.0, step=100.0)
    
    if "custom_stocks" not in st.session_state:
        st.session_state.custom_stocks = [
            Stock(id=1, name="Nifty 50 ETF", cost=270.0, return_val=310.0),
            Stock(id=2, name="TCS", cost=4120.0, return_val=4780.0),
            Stock(id=3, name="HDFC Bank", cost=1680.0, return_val=1980.0)
        ]
        
    if st.sidebar.button("➕ Add Asset"):
        new_id = len(st.session_state.custom_stocks) + 1
        st.session_state.custom_stocks.append(
            Stock(id=new_id, name=custom_name, cost=custom_cost, return_val=custom_return)
        )
    
    stocks = st.session_state.custom_stocks

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

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Available NSE Investment Options")
        inv_df = pd.DataFrame([{
            "ID": s.id, 
            "Asset Name": s.name, 
            "Price / Share (₹)": f"₹{s.cost:,.2f}", 
            "Projected Return (₹)": f"₹{s.return_val:,.2f}", 
            "Return/Cost Ratio": round(s.ratio, 3)
        } for s in stocks])
        st.dataframe(inv_df, use_container_width=True)

    # Execute Optimization Engine
    if algorithm_choice == "LC (Least Cost / Best-First) B&B":
        result = solve_lc_knapsack(stocks, budget)
    else:
        result = solve_fifo_knapsack(stocks, budget)

    with col2:
        st.subheader("🎯 Recommended Portfolio Allocation")
        
        # Display Metrics in INR
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Total Return", f"₹{result['max_profit']:,.2f}")
        m2.metric("Capital Deployed", f"₹{result['total_cost']:,.2f}", delta=f"Out of ₹{budget:,.2f}")
        
        roi = ((result['max_profit'] - result['total_cost']) / result['total_cost'] * 100) if result['total_cost'] > 0 else 0.0
        m3.metric("Projected ROI", f"{roi:.2f}%")

        # Tree Pruning Metrics
        st.markdown("---")
        p1, p2 = st.columns(2)
        p1.metric("Search Tree Nodes Generated", result["nodes_generated"])
        p2.metric("Subtrees Pruned", result["nodes_pruned"])

        selected_stocks = result["selected_stocks"]
        if selected_stocks:
            rec_df = pd.DataFrame([{
                "ID": s.id, 
                "Selected Asset": s.name, 
                "Capital Required (₹)": f"₹{s.cost:,.2f}", 
                "Projected Yield (₹)": f"₹{s.return_val:,.2f}"
            } for s in selected_stocks])
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.warning("Capital budget is too low to deploy into any available investment positions.")
