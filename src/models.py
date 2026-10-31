from dataclasses import dataclass
from typing import List

@dataclass
class Stock:
    id: int
    name: str
    cost: float            # price per share (₹)
    return_val: float      # projected value of one share after holding period (₹)
    volatility: float = 0.20   # annualised volatility as a decimal (0.20 = 20%)
    sector: str = "Other"      # sector category (e.g., Banking, IT, ETF)

    @property
    def profit(self) -> float:
        """Net projected gain per share (projected value minus price paid)."""
        return self.return_val - self.cost

    @property
    def is_etf(self) -> bool:
        return self.sector == "ETF"

    def risk_adjusted_profit(self, risk_aversion: float) -> float:
        """Net profit minus a volatility penalty: profit - lambda * volatility * cost."""
        return self.profit - risk_aversion * self.volatility * self.cost

    @property
    def ratio(self) -> float:
        """Value to cost ratio, for greedy upper bounding."""
        return self.return_val / self.cost if self.cost > 0 else 0.0

class Node:
    def __init__(self, level: int, profit: float, weight: float, bound: float, taken: List[bool]):
        self.level = level
        self.profit = profit
        self.weight = weight
        self.bound = bound
        self.taken = taken

    def __lt__(self, other: 'Node') -> bool:
        return self.bound > other.bound
