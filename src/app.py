import os
import sys
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import Stock
from src.knapsack_bb import solve_fifo_knapsack, solve_lc_knapsack

st.set_page_config(
    page_title="Budget-Constrained Investment Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("Budget-Constrained Investment Advisor")
st.caption("0/1 Knapsack Optimization using FIFO & Least-Cost (LC) Branch and Bound")

# Sidebar Configuration
st.sidebar.header("Controls & Parameters")

dataset_option = st.sidebar.selectbox(
    "Select Dataset Source",
    ["Preset Sample (n=10)", "Preset Sample (n=15)", "Preset Sample (n=20)", "Preset Sample (n=25)", "Preset Sample (n=30)", "Custom Input"]
)

algorithm_choice = st.sidebar.radio(
    "Select Branch & Bound Strategy",
    ["LC (Least Cost / Best-First) B&B", "FIFO B&B"]
)

# Load / Parse Stocks Data
stocks = []

if "Preset Sample" in dataset_option:
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
    st.sidebar.subheader("Add Custom Stock")
    custom_cost = st.sidebar.number_input("Stock Price / Cost", min_value=1.0, value=100.0, step=10.0)
    custom_return = st.sidebar.number_input("Expected Return", min_value=1.0, value=130.0, step=10.0)
    
    if "custom_stocks" not in st.session_state:
        st.session_state.custom_stocks = [
            Stock(id=1, name="AAPL", cost=150.0, return_val=195.0),
            Stock(id=2, name="GOOGL", cost=200.0, return_val=270.0),
            Stock(id=3, name="MSFT", cost=120.0, return_val=150.0)
        ]
        
    if st.sidebar.button("➕ Add Stock"):
        new_id = len(st.session_state.custom_stocks) + 1
        st.session_state.custom_stocks.append(
            Stock(id=new_id, name=f"STK_{new_id:02d}", cost=custom_cost, return_val=custom_return)
        )
    
    stocks = st.session_state.custom_stocks

if stocks:
    # Capital Budget Input
    total_market_cap = sum(s.cost for s in stocks)
    default_budget = round(total_market_cap * 0.5, 2)
    
    budget = st.number_input(
        "Enter Capital Budget Constraint (W)", 
        min_value=1.0, 
        max_value=total_market_cap * 2.0, 
        value=default_budget, 
        step=50.0
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Available Stock Inventory")
        inv_df = pd.DataFrame([{"ID": s.id, "Name": s.name, "Cost ($)": s.cost, "Expected Return ($)": s.return_val, "Return/Cost Ratio": round(s.ratio, 3)} for s in stocks])
        st.dataframe(inv_df, use_container_width=True)

    # Execute Optimization Engine
    if algorithm_choice == "LC (Least Cost / Best-First) B&B":
        result = solve_lc_knapsack(stocks, budget)
    else:
        result = solve_fifo_knapsack(stocks, budget)

    with col2:
        st.subheader("Optimal Portfolio Recommendation")
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Return", f"${result['max_profit']:,.2f}")
        m2.metric("Capital Deployed", f"${result['total_cost']:,.2f} / ${budget:,.2f}")
        m3.metric("ROI", f"{((result['max_profit'] - result['total_cost'])/result['total_cost']*100 if result['total_cost'] > 0 else 0):.2f}%")

        # Tree Pruning Metrics
        p1, p2 = st.columns(2)
        p1.metric("Nodes Generated", result["nodes_generated"])
        p2.metric("Nodes Pruned", result["nodes_pruned"])

        selected_stocks = result["selected_stocks"]
        if selected_stocks:
            rec_df = pd.DataFrame([{"ID": s.id, "Name": s.name, "Cost ($)": s.cost, "Expected Return ($)": s.return_val} for s in selected_stocks])
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.warning("Budget is too low to include any stock positions.")
