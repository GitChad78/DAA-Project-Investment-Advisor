import time
import random
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import Stock
from src.knapsack_bb import solve_fifo_knapsack, solve_lc_knapsack


def generate_synthetic_stocks(n: int, seed: int = 42) -> list[Stock]:
    """Generates synthetic stock datasets of size n."""
    random.seed(seed + n)
    stocks = []
    for i in range(1, n + 1):
        cost = round(random.uniform(50.0, 500.0), 2)
        # Expected return calculated with variation relative to cost
        return_val = round(cost * random.uniform(1.05, 1.45), 2)
        stocks.append(Stock(id=i, name=f"STK_{i:02d}", cost=cost, return_val=return_val))
    return stocks


def run_benchmarks():
    sample_sizes = [10, 15, 20, 25, 30]
    results = []

    os.makedirs("data", exist_ok=True)
    os.makedirs("benchmarks/plots", exist_ok=True)

    print("Running Benchmarks across dataset sizes n = [10, 15, 20, 25, 30]...\n")

    for n in sample_sizes:
        stocks = generate_synthetic_stocks(n)
        
        # Save dataset to CSV
        df_stocks = pd.DataFrame([{"id": s.id, "name": s.name, "cost": s.cost, "return_val": s.return_val} for s in stocks])
        df_stocks.to_csv(f"data/stocks_n{n}.csv", index=False)

        # Capacity set to ~50% of total stock valuation sum
        capacity = round(sum(s.cost for s in stocks) * 0.5, 2)

        # Benchmark FIFO
        start = time.perf_counter()
        fifo_res = solve_fifo_knapsack(stocks, capacity)
        fifo_time = (time.perf_counter() - start) * 1000  # ms

        # Benchmark LC
        start = time.perf_counter()
        lc_res = solve_lc_knapsack(stocks, capacity)
        lc_time = (time.perf_counter() - start) * 1000  # ms

        results.append({
            "n": n,
            "Algorithm": "FIFO B&B",
            "Execution Time (ms)": round(fifo_time, 3),
            "Nodes Generated": fifo_res["nodes_generated"],
            "Nodes Pruned": fifo_res["nodes_pruned"]
        })

        results.append({
            "n": n,
            "Algorithm": "LC B&B",
            "Execution Time (ms)": round(lc_time, 3),
            "Nodes Generated": lc_res["nodes_generated"],
            "Nodes Pruned": lc_res["nodes_pruned"]
        })

        print(f"n={n:2d} | FIFO: {fifo_time:6.2f} ms | Nodes Gen: {fifo_res['nodes_generated']:5d} | Pruned: {fifo_res['nodes_pruned']:5d}")
        print(f"     | LC  : {lc_time:6.2f} ms | Nodes Gen: {lc_res['nodes_generated']:5d} | Pruned: {lc_res['nodes_pruned']:5d}")
        print("-" * 65)

    df_results = pd.DataFrame(results)
    df_results.to_csv("benchmarks/benchmark_results.csv", index=False)

    # Plot Visualizations
    sns.set_theme(style="darkgrid")
    
    # 1. Execution Time Plot
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_results, x="n", y="Execution Time (ms)", hue="Algorithm", marker="o", linewidth=2.5)
    plt.title("Execution Time Comparison: FIFO vs LC Branch & Bound")
    plt.xlabel("Number of Stocks (n)")
    plt.ylabel("Execution Time (ms)")
    plt.tight_layout()
    plt.savefig("benchmarks/plots/runtime_comparison.png")
    plt.close()

    # 2. State Space Nodes Generated Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_results, x="n", y="Nodes Generated", hue="Algorithm")
    plt.title("State Space Search Tree Size: Nodes Generated")
    plt.xlabel("Number of Stocks (n)")
    plt.ylabel("Generated Nodes Count")
    plt.tight_layout()
    plt.savefig("benchmarks/plots/nodes_generated_comparison.png")
    plt.close()

    print("\nBenchmark completed successfully! Datasets saved in data/ and charts in benchmarks/plots/")


if __name__ == "__main__":
    run_benchmarks()
