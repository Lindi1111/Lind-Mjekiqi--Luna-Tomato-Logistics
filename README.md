# ESA SpOC 4 – Luna Tomato Logistics
**Author:** Lind Mjekiqi

## Project Overview

This repository documents our approach to the **ESA SpOC 4 – Luna Tomato Logistics** optimization challenge.

The project focuses on developing, testing, and improving optimization algorithms for maximum-weight three-dimensional matching problems.

Our goal is to produce high-quality, feasible solutions and systematically improve their objective values.

## Problem Description

Each candidate assignment is represented by four values:

`e l d weight`

Where:
- `e` represents the first matching dimension.
- `l` represents the second matching dimension.
- `d` represents the third matching dimension.
- `weight` represents the assignment's objective value.

The objective is to maximize the total weight of selected assignments.

### Constraints

Each selected assignment must satisfy the following conditions:

1. Every `e` value can be selected at most once.
2. Every `l` value can be selected at most once.
3. Every `d` value can be selected at most once.
4. The decision vector must contain only binary values (0 or 1).

## Datasets

| Dataset | Candidate Assignments |
|---|---:|
| Matching I | 25,000 |
| Matching II | 92,103 |

Both datasets passed initial structural validation with zero invalid rows.

## Baseline Algorithm

Our initial implementation uses a **Greedy Matching Algorithm**.

The algorithm:

1. Sorts candidate assignments by descending weight.
2. Iterates through the sorted assignments.
3. Selects an assignment only if none of its three identifiers has been used.
4. Produces a binary decision vector.

## Baseline Results

| Metric | Matching I | Matching II |
|---|---:|---:|
| Selected assignments | 3,870 | 7,759 |
| Total objective weight | 29,791.692 | 63,313.022 |
| Decision vector length | 25,000 | 92,103 |
| Constraint validation | PASS | PASS |

**Note:** These are locally computed baseline objective values, not official ESA leaderboard scores.

## Development Environment

- Python
- Kaggle Notebooks
- GitHub

## Next Steps

- Save and document the baseline implementation.
- Develop improved optimization heuristics.
- Compare improved solutions against the baseline.
- Validate every generated decision vector.
- Prepare submission files according to the official challenge requirements.

## Project Status

**Current phase:** Baseline implementation and validation completed.

Further optimization and official submission testing are pending.
