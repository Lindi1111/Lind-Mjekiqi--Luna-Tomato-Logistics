"""
ESA SpOC 4 - Luna Tomato Logistics
Author: Lind Mjekiqi

Greedy baseline for maximum-weight
three-dimensional matching.
"""

from pathlib import Path


def load_matching(file_path):
    rows = []

    with open(file_path, "r") as file:
        for line in file:
            e, l, d, weight = line.split()
            rows.append((int(e), int(l), int(d), float(weight)))

    return rows


def greedy_matching(rows):
    selected = [0] * len(rows)

    used_e = set()
    used_l = set()
    used_d = set()

    order = sorted(
        range(len(rows)),
        key=lambda i: rows[i][3],
        reverse=True
    )

    for i in order:
        e, l, d, weight = rows[i]

        if e in used_e or l in used_l or d in used_d:
            continue

        selected[i] = 1
        used_e.add(e)
        used_l.add(l)
        used_d.add(d)

    return selected


def validate_solution(rows, solution):
    if len(rows) != len(solution):
        return False

    if any(value not in (0, 1) for value in solution):
        return False

    used_e = set()
    used_l = set()
    used_d = set()

    for i, selected in enumerate(solution):
        if selected == 0:
            continue

        e, l, d, _ = rows[i]

        if e in used_e or l in used_l or d in used_d:
            return False

        used_e.add(e)
        used_l.add(l)
        used_d.add(d)

    return True


def main():
    data_path = Path("data")

    for filename in ["matching-i.txt", "matching-ii.txt"]:
        file_path = data_path / filename

        if not file_path.exists():
            print(f"Dataset not found: {file_path}")
            continue

        rows = load_matching(file_path)
        solution = greedy_matching(rows)

        total_weight = sum(
            row[3]
            for row, selected in zip(rows, solution)
            if selected
        )

        print(f"\nDataset: {filename}")
        print(f"Selected transfers: {sum(solution)}")
        print(f"Total weight: {total_weight:.3f}")
        print(f"Decision vector length: {len(solution)}")
        print(f"Valid solution: {validate_solution(rows, solution)}")


if __name__ == "__main__":
    main()
