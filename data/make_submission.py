"""
Builds ESA-submission-ready JSON files from our internal solved_*.json
decision vectors, in the exact format required by the challenge:

[
  {
    "decisionVector": [...],
    "problem": "<problem-id>",
    "challenge": "spoc-4-luna-tomato-logistics"
  }
]
"""
import json
import sys


def build(in_path, problem_id, out_path):
    with open(in_path) as f:
        sol = json.load(f)
    payload = [{
        "decisionVector": sol["decisionVector"],
        "problem": problem_id,
        "challenge": "spoc-4-luna-tomato-logistics",
    }]
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {out_path}: problem={problem_id} length={len(sol['decisionVector'])} "
          f"selected={sum(sol['decisionVector'])} total_weight={sol.get('total_weight')}")


if __name__ == "__main__":
    build("solved_matching_i.json", "matching-i", "submission_matching-i.json")
    build("solved_matching_ii.json", "matching-ii", "submission_matching-ii.json")
