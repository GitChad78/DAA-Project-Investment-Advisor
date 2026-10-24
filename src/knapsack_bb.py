import heapq
from collections import deque
from typing import List, Tuple, Dict, Any
from src.models import Stock, Node

def calculate_upper_bound(node: Node, capacity: float, stocks: List[Stock]) -> float:
    """Calculates Fractional Knapsack upper bound for pruning subtrees."""
    if node.weight >= capacity:
        return 0.0

    profit_bound = node.profit
    total_weight = node.weight
    n = len(stocks)
    idx = node.level + 1

    # Greedily take full stocks by best return-to-cost ratio
    while idx < n and total_weight + stocks[idx].cost <= capacity:
        total_weight += stocks[idx].cost
        profit_bound += stocks[idx].return_val
        idx += 1

    # Take fractional portion of the next stock if capacity remains
    if idx < n:
        profit_bound += (capacity - total_weight) * stocks[idx].ratio

    return profit_bound


def solve_fifo_knapsack(stocks: List[Stock], capacity: float) -> Dict[str, Any]:
    """FIFO Branch and Bound 0/1 Knapsack Solver (Queue BFS)."""
    sorted_stocks = sorted(stocks, key=lambda s: s.ratio, reverse=True)
    n = len(sorted_stocks)

    queue = deque()
    
    root_bound = calculate_upper_bound(Node(-1, 0, 0, 0, []), capacity, sorted_stocks)
    root = Node(-1, 0, 0, root_bound, [])
    queue.append(root)

    max_profit = 0.0
    best_taken = []
    
    nodes_generated = 1
    nodes_pruned = 0

    while queue:
        curr = queue.popleft()

        if curr.bound <= max_profit:
            nodes_pruned += 1
            continue

        next_idx = curr.level + 1
        if next_idx >= n:
            continue

        item = sorted_stocks[next_idx]

        # Branch 1: Include stock
        if curr.weight + item.cost <= capacity:
            incl_profit = curr.profit + item.return_val
            incl_weight = curr.weight + item.cost
            incl_taken = curr.taken + [True]
            incl_bound = calculate_upper_bound(
                Node(next_idx, incl_profit, incl_weight, 0, incl_taken), capacity, sorted_stocks
            )

            if incl_profit > max_profit:
                max_profit = incl_profit
                best_taken = incl_taken + [False] * (n - len(incl_taken))

            if incl_bound > max_profit:
                queue.append(Node(next_idx, incl_profit, incl_weight, incl_bound, incl_taken))
                nodes_generated += 1
            else:
                nodes_pruned += 1

        # Branch 2: Exclude stock
        excl_taken = curr.taken + [False]
        excl_bound = calculate_upper_bound(
            Node(next_idx, curr.profit, curr.weight, 0, excl_taken), capacity, sorted_stocks
        )

        if excl_bound > max_profit:
            queue.append(Node(next_idx, curr.profit, curr.weight, excl_bound, excl_taken))
            nodes_generated += 1
        else:
            nodes_pruned += 1

    selected_dict = {sorted_stocks[i].id: best_taken[i] for i in range(n)}
    selected_stocks = [s for s in stocks if selected_dict.get(s.id, False)]
    total_cost = sum(s.cost for s in selected_stocks)

    return {
        "max_profit": max_profit,
        "total_cost": total_cost,
        "selected_stocks": selected_stocks,
        "nodes_generated": nodes_generated,
        "nodes_pruned": nodes_pruned,
        "algorithm": "FIFO Branch and Bound"
    }


def solve_lc_knapsack(stocks: List[Stock], capacity: float) -> Dict[str, Any]:
    """LC (Least Cost / Best-First) Branch and Bound 0/1 Knapsack Solver (Max-Heap)."""
    sorted_stocks = sorted(stocks, key=lambda s: s.ratio, reverse=True)
    n = len(sorted_stocks)

    heap = []
    
    root_bound = calculate_upper_bound(Node(-1, 0, 0, 0, []), capacity, sorted_stocks)
    root = Node(-1, 0, 0, root_bound, [])
    heapq.heappush(heap, root)

    max_profit = 0.0
    best_taken = []
    
    nodes_generated = 1
    nodes_pruned = 0

    while heap:
        curr = heapq.heappop(heap)

        if curr.bound <= max_profit:
            nodes_pruned += 1
            continue

        next_idx = curr.level + 1
        if next_idx >= n:
            continue

        item = sorted_stocks[next_idx]

        # Branch 1: Include stock
        if curr.weight + item.cost <= capacity:
            incl_profit = curr.profit + item.return_val
            incl_weight = curr.weight + item.cost
            incl_taken = curr.taken + [True]
            incl_bound = calculate_upper_bound(
                Node(next_idx, incl_profit, incl_weight, 0, incl_taken), capacity, sorted_stocks
            )

            if incl_profit > max_profit:
                max_profit = incl_profit
                best_taken = incl_taken + [False] * (n - len(incl_taken))

            if incl_bound > max_profit:
                heapq.heappush(heap, Node(next_idx, incl_profit, incl_weight, incl_bound, incl_taken))
                nodes_generated += 1
            else:
                nodes_pruned += 1

        # Branch 2: Exclude stock
        excl_taken = curr.taken + [False]
        excl_bound = calculate_upper_bound(
            Node(next_idx, curr.profit, curr.weight, 0, excl_taken), capacity, sorted_stocks
        )

        if excl_bound > max_profit:
            heapq.heappush(heap, Node(next_idx, curr.profit, curr.weight, excl_bound, excl_taken))
            nodes_generated += 1
        else:
            nodes_pruned += 1

    selected_dict = {sorted_stocks[i].id: best_taken[i] for i in range(n)}
    selected_stocks = [s for s in stocks if selected_dict.get(s.id, False)]
    total_cost = sum(s.cost for s in selected_stocks)

    return {
        "max_profit": max_profit,
        "total_cost": total_cost,
        "selected_stocks": selected_stocks,
        "nodes_generated": nodes_generated,
        "nodes_pruned": nodes_pruned,
        "algorithm": "LC Best-First Branch and Bound"
    }
