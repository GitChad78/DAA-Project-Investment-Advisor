# 📈 Budget-Constrained Investment Advisor

A professional **0/1 Knapsack Portfolio Optimization** tool built for Design and Analysis of Algorithms (DAA), leveraging live NSE market data via `yfinance` and high-performance Branch & Bound search strategies.

---

## 🚀 Key Features
* **Dual Branch & Bound Strategies:** Implements both **LC Best-First Search (Max-Heap)** and **FIFO Search (Queue)** with greedy upper-bound pruning.
* **Live NSE Market Data:** Dynamically queries an expanded hybrid master list of ~35 liquid Nifty stocks and ETFs across multiple sectors.
* **Advanced Portfolio Constraints:** Handles capital budget bounds ($W$), per-asset percentage caps, sector limits, and mandatory ETF allocation.
* **Visual Analytics & Benchmarking:** Features interactive Streamlit charts, capital allocation breakdowns, State-Space search tree diagrams, and automated scaling performance plots.

---

## 🛠️ Project Structure
```text
DAA-Project-Investment-Advisor/
├── data/               # Preset benchmark CSV datasets (n=10 to 30)
├── benchmarks/         # Automated benchmarking suite and evaluation plots
├── docs/               # Theoretical analysis and viva documentation
├── src/                # Modular Python application (app, models, solvers, data loader, visuals)
├── requirements.txt    # Project dependencies
└── README.md           # Project guide
