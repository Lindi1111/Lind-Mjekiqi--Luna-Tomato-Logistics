"""
ESA SpOC 4 - Luna Tomato Logistics - Advanced (Trajectory Matching)

Self-contained re-implementation of the official evaluator's physics
(BCP dynamics + Earth/Moon orbital-element extraction), using only
numpy/scipy, since the official evaluator's dependencies (pykep, heyoka)
cannot be built on this machine (no C++ toolchain, no wheels for the
installed Python version).

The equations and unit conventions below are copied verbatim from the
official evaluation script downloaded from:
https://api.optimize.esa.int/media/problems/spoc-4-luna-tomato-logistics-trajectory-matching-*.py

Physical reasoning (verified by hand, see PROGRESS notes):
- Position in the barycentric rotating frame equals position in any
  "instantaneous inertial" frame sharing the same origin & z-axis
  (rotation only changes orientation, not the vector's own components'
  magnitude/shape-defining properties such as (a, e, i), since those are
  invariant to a rotation about z).
- Velocity must be de-rotated: v_inertial = v_rotating + omega x r,
  with omega = +z (nondimensional angular rate of the rotating frame),
  evaluated at the BARYCENTRIC position (not the body-relative one).
- The body itself (Earth or Moon) moves on a circular orbit around the
  barycenter at the same rate, with velocity omega x r_body; subtracting
  that gives the velocity relative to the body, consistent with ic2par.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Constants (verbatim from the official evaluator)
# ---------------------------------------------------------------------------
L = 3.84405000e8          # m
T = 3.7567696752e5        # s
V = L / T                 # m/s

MU_SUN = 1.32712440041279419e20
MU_EARTH = 398600435507000.0
MU_MOON = 4902800118000.0

MU = MU_MOON / (MU_MOON + MU_EARTH)           # CR3BP_MU_EARTH_MOON
MU_S = MU_SUN / (MU_MOON + MU_EARTH)          # BCP_MU_S
RHO_S = 3.88811143e2                           # BCP_RHO_S
OMEGA_S = -9.25195985e-1                       # BCP_OMEGA_S

EARTH_RADIUS = 6378137.0      # m (standard WGS84 eq. radius, used by pykep)
MOON_RADIUS = 1737400.0       # m
EARTH_IMPACT_R = (EARTH_RADIUS + 99000.0) / L       # nondim
MOON_IMPACT_R = (MOON_RADIUS + 30000.0) / L         # nondim

EARTH_CENTER = np.array([-MU, 0.0, 0.0])
MOON_CENTER = np.array([1.0 - MU, 0.0, 0.0])

G0 = 9.80665
ISP = 311.0
M_WET = 5000.0
M_DRY = 500.0

SEC2DAY = 1.0 / 86400.0


# ---------------------------------------------------------------------------
# BCP dynamics (exact copy of the official equations of motion)
# ---------------------------------------------------------------------------
def bcp_rhs(t, state):
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - (1.0 - MU)) ** 2 + y ** 2 + z ** 2)

    x_sun = RHO_S * np.cos(OMEGA_S * t)
    y_sun = RHO_S * np.sin(OMEGA_S * t)
    r_sun = np.sqrt((x - x_sun) ** 2 + (y - y_sun) ** 2 + z ** 2)

    vxdot = (
        2 * vy + x
        - (1 - MU) * (x + MU) / r1 ** 3
        - MU * (x + MU - 1) / r2 ** 3
        - MU_S / r_sun ** 3 * (x - x_sun)
        - MU_S / RHO_S ** 2 * np.cos(OMEGA_S * t)
    )
    vydot = (
        -2 * vx + y
        - (1 - MU) * y / r1 ** 3
        - MU * y / r2 ** 3
        - MU_S / r_sun ** 3 * (y - y_sun)
        - MU_S / RHO_S ** 2 * np.sin(OMEGA_S * t)
    )
    vzdot = (
        -(1 - MU) * z / r1 ** 3
        - MU * z / r2 ** 3
        - MU_S * z / r_sun ** 3
    )
    return [vx, vy, vz, vxdot, vydot, vzdot]


def _earth_impact_event(t, state):
    x, y, z = state[:3]
    return (x + MU) ** 2 + y ** 2 + z ** 2 - EARTH_IMPACT_R ** 2


_earth_impact_event.terminal = True
_earth_impact_event.direction = -1


def _moon_impact_event(t, state):
    x, y, z = state[:3]
    return (x - 1 + MU) ** 2 + y ** 2 + z ** 2 - MOON_IMPACT_R ** 2


_moon_impact_event.terminal = True
_moon_impact_event.direction = -1


def propagate(state0, t0, duration, rtol=1e-12, atol=1e-13):
    """Propagate a BCP state for `duration` (nondim time) starting at t0.

    Returns (state_final, impacted: bool). If an Earth/Moon impact is
    detected, returns (None, True).
    """
    if duration <= 0:
        return np.array(state0, dtype=float), False

    sol = solve_ivp(
        bcp_rhs, (t0, t0 + duration), state0,
        method="DOP853", rtol=rtol, atol=atol,
        events=[_earth_impact_event, _moon_impact_event],
        dense_output=False,
    )
    if sol.status != 0 or len(sol.t_events[0]) > 0 or len(sol.t_events[1]) > 0:
        return None, True
    return sol.y[:, -1], False


# ---------------------------------------------------------------------------
# Orbital elements <-> Cartesian state (standard 2-body formulas)
# ---------------------------------------------------------------------------
def rv_to_elements(r, v, mu):
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    energy = 0.5 * np.dot(v, v) - mu / r_norm
    a = -mu / (2.0 * energy)
    e_vec = np.cross(v, h) / mu - r / r_norm
    e = np.linalg.norm(e_vec)
    i = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))
    return a, e, i


def elements_to_rv(a, e, inc, raan, argp, nu, mu):
    p = a * (1.0 - e ** 2)
    r_mag = p / (1.0 + e * np.cos(nu))
    r_pf = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    R = np.array([
        [cO * cw - sO * ci * sw, -cO * sw - sO * ci * cw, sO * si],
        [sO * cw + cO * ci * sw, -sO * sw + cO * ci * cw, -cO * si],
        [si * sw, si * cw, ci],
    ])
    r_body = R @ r_pf
    v_body = R @ v_pf
    return r_body, v_body


# ---------------------------------------------------------------------------
# Barycentric rotating frame  <->  body-centered "instantaneous inertial" SI
# ---------------------------------------------------------------------------
def barycentric_to_body_elements(state, body_center, mu_body_si):
    """state: 6-vector (nondim, barycentric rotating frame).
    body_center: nondim barycentric position of Earth or Moon, e.g. EARTH_CENTER.
    Returns (a, e, i) in SI units (a in meters)."""
    x, y, z, vx, vy, vz = state
    body_vy = body_center[0]  # see module docstring: omega x r_body = (0, x_body, 0)

    pos_si = (np.array([x, y, z]) - body_center) * L
    v_inertial_derot = np.array([vx - y, vy + x, vz])
    v_rel_si = (v_inertial_derot - np.array([0.0, body_vy, 0.0])) * V

    return rv_to_elements(pos_si, v_rel_si, mu_body_si)


def body_elements_to_barycentric_state(a, e, inc, raan, argp, nu, body_center, mu_body_si):
    """Inverse of barycentric_to_body_elements: build a full barycentric
    rotating-frame state (nondim) whose (a, e, i) relative to the given
    body match the requested elements. raan/argp/nu are free parameters
    (they do not affect a, e, i and may be chosen arbitrarily)."""
    r_si, v_si = elements_to_rv(a, e, inc, raan, argp, nu, mu_body_si)

    pos_bary = r_si / L + body_center
    body_vy = body_center[0]
    v_inertial_derot = v_si / V + np.array([0.0, body_vy, 0.0])
    x_b, y_b, _ = pos_bary
    vx = v_inertial_derot[0] + y_b
    vy = v_inertial_derot[1] - x_b
    vz = v_inertial_derot[2]
    return np.concatenate([pos_bary, [vx, vy, vz]])


def earth_elements_to_state(aE, eE, iE, raan, argp, nu):
    return body_elements_to_barycentric_state(aE, eE, iE, raan, argp, nu, EARTH_CENTER, MU_EARTH)


def state_to_earth_elements(state):
    return barycentric_to_body_elements(state, EARTH_CENTER, MU_EARTH)


def moon_elements_to_state(aL, eL, iL, raan, argp, nu):
    return body_elements_to_barycentric_state(aL, eL, iL, raan, argp, nu, MOON_CENTER, MU_MOON)


def state_to_moon_elements(state):
    return barycentric_to_body_elements(state, MOON_CENTER, MU_MOON)


# ---------------------------------------------------------------------------
# Delivered mass model
# ---------------------------------------------------------------------------
def mass_from_dv(dv_si):
    return M_WET * np.exp(-dv_si / (ISP * G0)) - M_DRY


def solve_insertion_dv(position_bary_nondim, v_before_bary_nondim, target_a, target_e, target_i,
                        mu_body_si, body_center, v_guess=None, max_tries=6):
    """Find the minimal-change velocity v_after (barycentric, nondim) at the
    given (fixed) position such that the resulting Moon-relative elements
    match (target_a, target_e, target_i). Returns (dv_vector_nondim, v_after)
    or (None, None) if no solution is found."""
    pos = np.asarray(position_bary_nondim, dtype=float)

    def residual(v_after):
        full_state = np.concatenate([pos, v_after])
        a, e, i = barycentric_to_body_elements(full_state, body_center, mu_body_si)
        if not np.isfinite(a) or a <= 0:
            return np.array([1e6, 1e6, 1e6])
        return np.array([
            (a - target_a) / max(target_a, 1.0),
            (e - target_e),
            (i - target_i),
        ])

    guesses = []
    if v_guess is not None:
        guesses.append(np.asarray(v_guess, dtype=float))
    # fall back guesses: try a circular-velocity-like vector in a few directions
    r_rel = (pos - body_center)
    r_rel_si = r_rel * L
    r_norm_si = np.linalg.norm(r_rel_si)
    if r_norm_si > 0 and target_a > 0:
        energy = -mu_body_si / (2 * target_a)
        v2 = 2 * (energy + mu_body_si / r_norm_si)
        if v2 > 0:
            vmag_si = np.sqrt(v2)
            vmag_nd = vmag_si / V
            for axis in (np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])):
                tang = np.cross(axis, r_rel)
                n = np.linalg.norm(tang)
                if n > 1e-9:
                    guesses.append(tang / n * vmag_nd)

    best = None
    for g in guesses[:max_tries]:
        sol = least_squares(residual, g, method="lm", max_nfev=2000)
        if sol.cost < 1e-14:
            v_after = sol.x
            if best is None or np.linalg.norm(v_after) < np.linalg.norm(best):
                best = v_after
    if best is None:
        return None, None
    dv = best - np.asarray(v_before_bary_nondim, dtype=float)
    return dv, best
