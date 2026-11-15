ESA SpOC 4 – Luna Tomato Logistics

Advanced: Trajectory Matching

Current Status

The Advanced problem is under development. No complete trajectory solution has been validated yet.

1. Data Preparation

The following datasets have been loaded and checked:

Earth orbits: 400

Moon orbits: 400

LTL transport capacities: 160,000

Basic numerical and consistency checks passed.

2. Mathematical Model

We are implementing the Bi-Circular Problem (BCP) model.

Components implemented in the Kaggle notebook:

Earth and Moon positions in the rotating reference frame

Gravitational acceleration from Earth and Moon

Rotating-frame acceleration terms

Time-dependent Sun position

Solar gravitational contribution with an indirect term

3. Initial Tests

Individual component tests have passed.

These tests confirm basic numerical consistency but do not yet verify the complete model against ESA's official equations.

4. Remaining Tasks

Verify the complete BCP equations.

Implement numerical trajectory propagation.

Search for feasible Earth-to-Moon transfers.

Optimize transfer parameters and fuel consumption.

Validate orbital constraints.

Calculate delivered mass.

Prepare and validate the Advanced submission.

5. Submission Status

The Advanced problem has not yet been submitted to ESA.
