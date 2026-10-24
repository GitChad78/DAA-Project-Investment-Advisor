from dataclasses import dataclass
from typing import List

@dataclass
class Stock:
    id: int
    name: str
    cost: float
    return_val: float

    @property
    def ratio(self) -> float:
        """Value to cost ratio, for the greedy upper bounding"""
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
  
