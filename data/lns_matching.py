"""
ESA SpOC 4 - Luna Tomato Logistics
External Large Neighborhood Search (LNS) for the max-weight 3D matching.

Idea: CP-SAT over the FULL problem plateaus because a 25k/92k-variable
search has to spend most of its time on parts of the solution that are
already good. Instead, repeatedly:
  1. Take the current best solution.
  2. Randomly "destroy" a small fraction of the selected transfers.
  3. Collect every candidate transfer (selected or not) whose e/l/d ids
     are NOT blocked by the remaining (kept) selection.
  4. Re-solve this much smaller subproblem to full optimality with CP-SAT.
  5. Recombine with the kept transfers -> a new global solution that is
     provably >= the previous one (the destroyed set is always a feasible
     candidate for the subproblem).
Repeat for many rounds; each round only takes a second or two since the
subproblem is tiny compared to the full instance.
"""
import sys
import json
import random
import time
from collections import defaultdict

from ortools.sat.python import cp_model

from solve_matching_cpsat import load_matching, validate_solution, greedy


def build_index(rows):
    by_e, by_l, by_d = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, (e, l, d, w) in enumerate(rows):
        by_e[e].append(i)
        by_l[l].append(i)
        by_d[d].append(i)
    return by_e, by_l, by_d


def solve_subproblem(rows, candidate_idx, time_limit=5.0, num_workers=4):
    model = cp_model.CpModel()
    x = {i: model.new_bool_var(f"x{i}") for i in candidate_idx}

    by_e, by_l, by_d = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in candidate_idx:
        e, l, d, w = rows[i]
        by_e[e].append(i)
        by_l[l].append(i)
        by_d[d].append(i)

    for group in (by_e, by_l, by_d):
        for idxs in group.values():
            if len(idxs) > 1:
                model.add_at_most_one(x[i] for i in idxs)

    weights_int = {i: round(rows[i][3] * 1000) for i in candidate_idx}
    model.maximize(sum(weights_int[i] * x[i] for i in candidate_idx))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = False
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return {i for i in candidate_idx if solver.value(x[i]) == 1}


def lns_round(rows, selected_set, rng, destroy_frac, time_limit, num_workers, targeted=False):
    selected = list(selected_set)
    n_destroy = max(1, int(len(selected) * destroy_frac))
    if targeted:
        # Bias destruction towards low-weight selected transfers: these are
        # the most likely to be "wrong" choices blocking a better swap.
        weighted = sorted(selected, key=lambda i: rows[i][3])
        pool = weighted[: max(n_destroy * 4, n_destroy)]
        n_pick = min(n_destroy, len(pool))
        removed = set(rng.sample(pool, n_pick))
        remaining_needed = n_destroy - len(removed)
        if remaining_needed > 0:
            rest = [i for i in selected if i not in removed]
            removed |= set(rng.sample(rest, min(remaining_needed, len(rest))))
    else:
        removed = set(rng.sample(selected, n_destroy))
    kept = selected_set - removed

    kept_e = {rows[i][0] for i in kept}
    kept_l = {rows[i][1] for i in kept}
    kept_d = {rows[i][2] for i in kept}

    candidates = set(removed)
    for i, (e, l, d, w) in enumerate(rows):
        if i in selected_set:
            continue
        if e in kept_e or l in kept_l or d in kept_d:
            continue
        candidates.add(i)

    sub_result = solve_subproblem(rows, candidates, time_limit=time_limit, num_workers=num_workers)
    if sub_result is None:
        return selected_set, 0, len(candidates)

    new_selected = kept | sub_result
    return new_selected, len(candidates), len(candidates)


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    total_time = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
    warm_json = sys.argv[4] if len(sys.argv) > 4 else None
    destroy_frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
    sub_time_limit = float(sys.argv[6]) if len(sys.argv) > 6 else 4.0
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    targeted = (sys.argv[8] == "1") if len(sys.argv) > 8 else False

    rows = load_matching(in_path)
    n = len(rows)
    print(f"Loaded {n} candidate assignments from {in_path}", flush=True)

    if warm_json:
        with open(warm_json) as f:
            prev = json.load(f)
        selected = {i for i, v in enumerate(prev["decisionVector"]) if v == 1}
    else:
        selected = set(greedy(rows))

    ok, total = validate_solution(rows, list(selected))
    print(f"Starting solution: selected={len(selected)} total={total:.3f} valid={ok}", flush=True)

    rng = random.Random(seed)
    t0 = time.time()
    best_total = total
    rounds = 0
    improved_rounds = 0

    while time.time() - t0 < total_time:
        try:
            new_selected, n_free, n_cand = lns_round(
                rows, selected, rng, destroy_frac, sub_time_limit, num_workers=4, targeted=targeted
            )
            ok, new_total = validate_solution(rows, list(new_selected))
        except Exception as ex:
            print(f"  [round {rounds}] exception, skipping round: {ex!r}", flush=True)
            rounds += 1
            continue
        rounds += 1
        if ok and new_total > best_total + 1e-9:
            improved_rounds += 1
            selected = new_selected
            best_total = new_total
            decision_vector = [0] * n
            for i in selected:
                decision_vector[i] = 1
            tmp = out_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "decisionVector": decision_vector,
                    "total_weight": best_total,
                    "selected_count": len(selected),
                }, f)
            import os
            os.replace(tmp, out_path)
            print(f"  [round {rounds}] t={time.time()-t0:.1f}s cand={n_cand} "
                  f"-> IMPROVED total={best_total:.3f} selected={len(selected)}", flush=True)
        elif rounds % 20 == 0:
            print(f"  [round {rounds}] t={time.time()-t0:.1f}s cand={n_cand} "
                  f"no improvement (best={best_total:.3f})", flush=True)

    print(f"Done: {rounds} rounds, {improved_rounds} improved, "
          f"final total={best_total:.3f}", flush=True)

    ok, final_total = validate_solution(rows, list(selected))
    decision_vector = [0] * n
    for i in selected:
        decision_vector[i] = 1
    with open(out_path, "w") as f:
        json.dump({
            "decisionVector": decision_vector,
            "total_weight": final_total,
            "selected_count": len(selected),
        }, f)
    print(f"Wrote final solution to {out_path}: total={final_total:.3f} valid={ok}", flush=True)


if __name__ == "__main__":
    main()
