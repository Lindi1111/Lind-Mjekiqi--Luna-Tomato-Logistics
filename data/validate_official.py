"""
Validates a solved_matching_*.json decision vector using the EXACT fitness
function from the official ESA evaluator scripts (matching-i-eval.py /
matching-ii-eval.py), adapted to read from our local instance files
instead of the ESA server's internal data path.
"""
import sys
import json


class matching:
    def __init__(self, edges):
        self.edges = edges

    def fitness(self, x):
        used_a = set()
        used_b = set()
        used_c = set()
        total = 0.0
        for bit, (a, b, c, w) in zip(x, self.edges):
            if bit == 0:
                continue
            if a in used_a or b in used_b or c in used_c:
                return [0]
            used_a.add(a)
            used_b.add(b)
            used_c.add(c)
            total += w
        return [-total]

    def get_bounds(self):
        m = len(self.edges)
        return ([0] * m, [1] * m)


def parse_input(filepath):
    edges = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            a, b, c, w = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            edges.append((a, b, c, w))
    return edges


def main():
    instance_path = sys.argv[1]
    solution_json = sys.argv[2]

    edges = parse_input(instance_path)
    udp = matching(edges)

    with open(solution_json) as f:
        sol = json.load(f)
    x = sol["decisionVector"]

    assert len(x) == len(edges), f"length mismatch: {len(x)} vs {len(edges)}"

    fit = udp.fitness(x)
    print(f"Instance: {instance_path} ({len(edges)} transfers)")
    print(f"Decision vector length: {len(x)}, selected: {sum(x)}")
    print(f"OFFICIAL fitness() output (negated, PyGMO minimization convention): {fit}")
    print(f"=> ESA leaderboard score would show: {fit[0]}")
    if fit[0] == 0:
        print("INVALID SOLUTION (conflict detected or zero score)")
    else:
        print(f"Objective (mass delivered, to maximize): {-fit[0]:.3f}")


if __name__ == "__main__":
    main()
