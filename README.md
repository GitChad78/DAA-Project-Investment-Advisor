# Budget-Constrained Investment Advisor (0/1 Knapsack via Branch & Bound)

**Group:** G12 | **Focus:** Design & Analysis of Algorithms (DAA) & Optimization Theory

## Project Overview
This project solves the Portfolio Optimization problem by modeling stock selection under a fixed capital budget as a **0/1 Knapsack Problem**. It evaluates and contrasts two Branch and Bound strategies:
1. **FIFO Branch & Bound** (Breadth-First Search using a Standard Queue)
2. **LC (Least Cost / Best-First) Branch & Bound** (Best-First Search using a Max-Heap Priority Queue with Fractional Knapsack Upper Bounding)

## Repository Structure
DAA-Project-Investment-Advisor/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/            # Core algorithm code & Streamlit UI
├── data/           # Synthetic datasets for benchmarking
├── benchmarks/     # Test runners and Matplotlib chart generator scripts
├── docs/           # Theoretical report & presentation slides
└── tests/          # Unit tests for FIFO and LC algorithms
