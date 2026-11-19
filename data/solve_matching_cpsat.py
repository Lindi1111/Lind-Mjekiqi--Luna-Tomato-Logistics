"""
ESA SpOC 4 - Luna Tomato Logistics
Exact(-ish) solver for the max-weight 3-dimensional matching (Matching I / II)
using Google OR-Tools CP-SAT.

Usage:
    python solve_matching_cpsat.py matching-i.txt out_matching_i.json 600
    python solve_matching_cpsat.py matching-ii.txt out_matching_ii.json 1800
"""
import sys
import json
import time
from collections import defaultdict

from ortools.sat.python import cp_model


def load_matching(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b, c, w = line.split()
            rows.append((int(a), int(b), int(c), float(w)))
    return rows


def validate_solution(rows, selected):
    used_e, used_l, used_d = set(), set(), set()
    total = 0.0
    for i in selected:
        e, l, d, w = rows[i]
        if e in used_e or l in used_l or d in used_d:
            return False, 0.0
        used_e.add(e)
        used_l.add(l)
        used_d.add(d)
        total += w
    return True, total


def solve(rows, time_limit, num_workers=8, hint=None, log=True):
    n = len(rows)
    model = cp_model.CpModel()

    x = [model.new_bool_var(f"x{i}") for i in range(n)]

    by_e, by_l, by_d = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, (e, l, d, w) in enumerate(rows):
        by_e[e].append(i)
        by_l[l].append(i)
        by_d[d].append(i)

    for group in (by_e, by_l, by_d):
        for idxs in group.values():
            if len(idxs) > 1:
                model.add_at_most_one(x[i] for i in idxs)

    weights_int = [round(rows[i][3] * 1000) for i in range(n)]
    model.maximize(sum(weights_int[i] * x[i] for i in range(n)))

    if hint:
        hint_set = set(hint)
        for i in range(n):
            model.add_hint(x[i], 1 if i in hint_set else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = log

    status = solver.solve(model)
    selected = [i for i in range(n) if solver.value(x[i]) == 1]
    return selected, solver, status


def greedy(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i][3], reverse=True)
    used_e, used_l, used_d = set(), set(), set()
    selected = []
    for i in order:
        e, l, d, w = rows[i]
        if e in used_e or l in used_l or d in used_d:
            continue
        selected.append(i)
        used_e.add(e)
        used_l.add(l)
        used_d.add(d)
    return selected


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    time_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    warm_json = sys.argv[5] if len(sys.argv) > 5 else None

    rows = load_matching(in_path)
    n = len(rows)
    print(f"Loaded {n} candidate assignments from {in_path}")

    g_sel = greedy(rows)
    ok, g_total = validate_solution(rows, g_sel)
    print(f"Greedy baseline: selected={len(g_sel)} total={g_total:.3f} valid={ok}")

    hint_sel = g_sel
    if warm_json:
        with open(warm_json) as f:
            prev = json.load(f)
        prev_sel = [i for i, v in enumerate(prev["decisionVector"]) if v == 1]
        ok2, prev_total = validate_solution(rows, prev_sel)
        print(f"Warm start from {warm_json}: selected={len(prev_sel)} total={prev_total:.3f} valid={ok2}")
        if ok2 and prev_total > g_total:
            hint_sel = prev_sel

    t0 = time.time()
    selected, solver, status = solve(rows, time_limit=time_limit, num_workers=num_workers, hint=hint_sel, log=False)
    print(f"CP-SAT status: {solver.status_name(status)} in {time.time()-t0:.1f}s")

    ok, total = validate_solution(rows, selected)
    print(f"CP-SAT solution: selected={len(selected)} total={total:.3f} valid={ok}")
    print(f"Best objective bound: {solver.best_objective_bound/1000.0:.3f}")

    _, hint_total = validate_solution(rows, hint_sel)
    if total < hint_total:
        print("CP-SAT solution worse than warm-start hint, keeping hint.")
        selected = hint_sel
        total = hint_total

    decision_vector = [0] * n
    for i in selected:
        decision_vector[i] = 1

    with open(out_path, "w") as f:
        json.dump({
            "decisionVector": decision_vector,
            "total_weight": total,
            "selected_count": len(selected),
        }, f)

    print(f"Wrote solution to {out_path}")


if __name__ == "__main__":
    main()
