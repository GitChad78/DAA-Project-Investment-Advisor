import heapq
from collections import deque
from typing import List, Dict, Any, Optional
from src.models import Stock, Node

def _sector_ok(chosen: List[Stock], item: Stock, sector_cap: Optional[float]) -> bool:
    """True if adding `item` keeps its sector's spend within `sector_cap` (₹). ETFs are exempt."""
    if sector_cap is None or item.is_etf:
        return True
    spent = sum(s.cost for s in chosen if s.sector == item.sector)
    return spent + item.cost <= sector_cap

def calculate_upper_bound(node: Node, capacity: float, stocks: List[Stock]) -> float:
    """Calculates Fractional Knapsack upper bound for pruning subtrees."""
    if node.weight >= capacity:
        return 0.0

    profit_bound = node.profit
    total_weight = node.weight
    n = len(stocks)
    idx = node.level + 1

    while idx < n and total_weight + stocks[idx].cost <= capacity:
        total_weight += stocks[idx].cost
        profit_bound += stocks[idx].return_val
        idx += 1

    if idx < n:
        profit_bound += (capacity - total_weight) * stocks[idx].ratio

    return profit_bound

def solve_fifo_knapsack(stocks: List[Stock], capacity: float, sector_cap: Optional[float] = None) -> Dict[str, Any]:
    """FIFO Branch and Bound 0/1 Knapsack Solver (Queue BFS)."""
    sorted_stocks = sorted(stocks, key=lambda s: s.ratio, reverse=True)
    n = len(sorted_stocks)

    queue = deque()
    root_bound = calculate_upper_bound(Node(-1, 0, 0, 0, []), capacity, sorted_stocks)
    queue.append(Node(-1, 0, 0, root_bound, []))

    max_profit = 0.0
    best_taken = [False] * n
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
        chosen = [sorted_stocks[i] for i, t in enumerate(curr.taken) if t] if sector_cap is not None else []
        if curr.weight + item.cost <= capacity and _sector_ok(chosen, item, sector_cap):
            incl_profit = curr.profit + item.return_val
            incl_weight = curr.weight + item.cost
            incl_taken = curr.taken + [True]
            incl_bound = calculate_upper_bound(Node(next_idx, incl_profit, incl_weight, 0, incl_taken), capacity, sorted_stocks)

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
        excl_bound = calculate_upper_bound(Node(next_idx, curr.profit, curr.weight, 0, excl_taken), capacity, sorted_stocks)

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

def solve_lc_knapsack(stocks: List[Stock], capacity: float, sector_cap: Optional[float] = None) -> Dict[str, Any]:
    """LC (Least Cost / Best-First) Branch and Bound 0/1 Knapsack Solver (Max-Heap)."""
    sorted_stocks = sorted(stocks, key=lambda s: s.ratio, reverse=True)
    n = len(sorted_stocks)

    heap = []
    root_bound = calculate_upper_bound(Node(-1, 0, 0, 0, []), capacity, sorted_stocks)
    heapq.heappush(heap, Node(-1, 0, 0, root_bound, []))

    max_profit = 0.0
    best_taken = [False] * n
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
        chosen = [sorted_stocks[i] for i, t in enumerate(curr.taken) if t] if sector_cap is not None else []
        if curr.weight + item.cost <= capacity and _sector_ok(chosen, item, sector_cap):
            incl_profit = curr.profit + item.return_val
            incl_weight = curr.weight + item.cost
            incl_taken = curr.taken + [True]
            incl_bound = calculate_upper_bound(Node(next_idx, incl_profit, incl_weight, 0, incl_taken), capacity, sorted_stocks)

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
        excl_bound = calculate_upper_bound(Node(next_idx, curr.profit, curr.weight, 0, excl_taken), capacity, sorted_stocks)

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

def solve_with_constraints(
    stocks: List[Stock],
    capacity: float,
    solver=solve_lc_knapsack,
    sector_cap: Optional[float] = None,
    require_etf: bool = False,
) -> Dict[str, Any]:
    pool = [s for s in stocks if s.return_val > 0]

    if not require_etf:
        return solver(pool, capacity, sector_cap)

    best = None
    nodes_generated = nodes_pruned = 0
    for etf in (s for s in stocks if s.is_etf and s.cost <= capacity):
        rest = [s for s in pool if s.id != etf.id]
        res = solver(rest, capacity - etf.cost, sector_cap)
        nodes_generated += res["nodes_generated"]
        nodes_pruned += res["nodes_pruned"]
        value = res["max_profit"] + etf.return_val
        if best is None or value > best["max_profit"]:
            best = {
                "max_profit": value,
                "total_cost": res["total_cost"] + etf.cost,
                "selected_stocks": res["selected_stocks"] + [etf],
                "algorithm": res["algorithm"],
            }

    if best is None:
        return {"max_profit": 0.0, "total_cost": 0.0, "selected_stocks": [],
                "nodes_generated": 0, "nodes_pruned": 0, "algorithm": "n/a", "feasible": False}

    best.update(nodes_generated=nodes_generated, nodes_pruned=nodes_pruned, feasible=True)
    return best

def solve_greedy(
    stocks: List[Stock],
    capacity: float,
    sector_cap: Optional[float] = None,
    require_etf: bool = False,
) -> Dict[str, Any]:
    ordered = sorted(stocks, key=lambda s: s.ratio, reverse=True)
    chosen: List[Stock] = []
    spent = 0.0

    if require_etf:
        first_etf = next((s for s in ordered if s.is_etf and s.cost <= capacity), None)
        if first_etf:
            chosen.append(first_etf)
            spent += first_etf.cost

    for s in ordered:
        if s.return_val <= 0 or any(c.id == s.id for c in chosen):
            continue
        if spent + s.cost <= capacity and _sector_ok(chosen, s, sector_cap):
            chosen.append(s)
            spent += s.cost

    return {
        "max_profit": sum(s.return_val for s in chosen),
        "total_cost": spent,
        "selected_stocks": chosen,
        "algorithm": "Greedy (by value/cost ratio)",
    }
