import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.models import Stock
from src.knapsack_bb import solve_lc_knapsack, solve_fifo_knapsack

def render_portfolio_charts(selected_stocks: list[Stock]):
    st.markdown("---")
    st.subheader("📊 Portfolio Visual Analytics")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Capital Allocation per Asset**")
        asset_chart_data = pd.DataFrame({
            "Asset": [s.name for s in selected_stocks],
            "Capital (₹)": [s.cost for s in selected_stocks]
        }).set_index("Asset")
        st.bar_chart(asset_chart_data)

    with c2:
        st.markdown("**Capital Allocation by Sector**")
        sector_data = {}
        for s in selected_stocks:
            sector_data[s.sector] = sector_data.get(s.sector, 0.0) + s.cost
        sector_chart_data = pd.DataFrame(list(sector_data.items()), columns=["Sector", "Capital (₹)"]).set_index("Sector")
        st.bar_chart(sector_chart_data)

def render_scaling_benchmark_chart():
    st.markdown("---")
    st.subheader("📈 Algorithmic Scaling Benchmark (Nodes Explored vs n)")
    
    benchmark_sizes = [10, 15, 20, 25, 30]
    lc_nodes, fifo_nodes = [], []
    
    for n_size in benchmark_sizes:
        b_path = f"data/stocks_n{n_size}.csv"
        if pd.io.common.file_exists(b_path):
            b_df = pd.read_csv(b_path)
            b_stocks = [Stock(id=int(r["id"]), name=str(r["name"]), cost=float(r["cost"]), return_val=float(r["return_val"])) for _, r in b_df.iterrows()]
            b_budget = sum(s.cost for s in b_stocks) * 0.45
            
            lc_res = solve_lc_knapsack(b_stocks, b_budget)
            fifo_res = solve_fifo_knapsack(b_stocks, b_budget)
            
            lc_nodes.append(lc_res["nodes_generated"])
            fifo_nodes.append(fifo_res["nodes_generated"])
        else:
            lc_nodes.append(0)
            fifo_nodes.append(0)

    bench_chart_data = pd.DataFrame({
        "Dataset Size (n)": benchmark_sizes,
        "LC Best-First B&B": lc_nodes,
        "FIFO B&B": fifo_nodes
    }).set_index("Dataset Size (n)")
    
    st.line_chart(bench_chart_data)
    st.caption("LC Best-First search explores fewer search tree nodes compared to FIFO due to priority queue pruning.")

def render_state_space_tree():
    st.markdown("---")
    st.subheader("🌲 State-Space Search Tree Visualization")
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    nodes = {
        "Root": (0.5, 0.9, "Green"),
        "A (Take Stock 1)": (0.25, 0.6, "Green"),
        "B (Skip Stock 1)": (0.75, 0.6, "Red"),
        "A1 (Take Stock 2)": (0.12, 0.2, "Green"),
        "A2 (Skip Stock 2)": (0.38, 0.2, "Red"),
    }
    
    edges = [
        ("Root", "A (Take Stock 1)", "Include"),
        ("Root", "B (Skip Stock 1)", "Exclude (Pruned)"),
        ("A (Take Stock 1)", "A1 (Take Stock 2)", "Include"),
        ("A (Take Stock 1)", "A2 (Skip Stock 2)", "Exclude (Pruned)")
    ]
    
    for parent, child, label in edges:
        px, py, _ = nodes[parent]
        cx, cy, _ = nodes[child]
        ax.plot([px, cx], [py, cy], color="#888888", lw=1.5, zorder=1)
        ax.text((px + cx)/2, (py + cy)/2, label, color="#cccccc", fontsize=8, ha='center', va='center', backgroundcolor='#0e1117')
        
    for name, (x, y, status) in nodes.items():
        color = '#2ecc71' if status == "Green" else '#e74c3c'
        ax.scatter(x, y, color=color, s=400, zorder=2)
        ax.text(x, y, name.split()[0], color="white", fontsize=9, fontweight='bold', ha='center', va='center', zorder=3)
        
    ax.axis('off')
    
    green_patch = mpatches.Patch(color='#2ecc71', label='Optimal / Active Path')
    red_patch = mpatches.Patch(color='#e74c3c', label='Pruned Subtree (Upper Bound ≤ Max Profit)')
    ax.legend(handles=[green_patch, red_patch], loc='lower right', facecolor='#161b22', edgecolor='none', labelcolor='white')
    
    st.pyplot(fig)
