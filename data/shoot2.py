"""Improved shooting method using Levenberg-Marquardt least-squares on the
miss VECTOR (3 components), which converges far better than Nelder-Mead on
a scalar distance for this kind of 2-point boundary value targeting problem.
"""
import numpy as np
from scipy.optimize import least_squares
import bcp_lib as bl
from transfer_design import build_departure_state, hohmann_dv0_mag


def miss_vector(params, pos0, vel0, dv0_mag_nd_nom, t1_nom, t0=0.0):
    dvx, dvy, dvz, t1_scale = params
    dv0 = dv0_mag_nd_nom * np.array([dvx, dvy, dvz])
    vel_after = vel0 + dv0
    state_dep = np.concatenate([pos0, vel_after])
    t1 = t1_nom * max(t1_scale, 0.05)
    final_state, impacted = bl.propagate(state_dep, t0, t1, rtol=1e-11, atol=1e-12)
    if impacted or final_state is None:
        return np.array([1e3, 1e3, 1e3])
    pos1 = final_state[:3]
    return (pos1 - bl.MOON_CENTER)  # nondim; ~1e-3 nondim = ~380 km


def shoot_lm(aE, eE, iE, raan_e=0.0, argp_e=np.pi, nu_e=0.0, t0=0.0, verbose=False,
             n_restarts=12, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    state0 = build_departure_state(aE, eE, iE, raan_e, argp_e, nu_e)
    pos0, vel0 = state0[:3], state0[3:]

    dv0_mag_si, half_period_si, r_p = hohmann_dv0_mag(aE, eE)
    dv0_mag_nd = dv0_mag_si / bl.V
    t1_nom = half_period_si / bl.T

    vel0_dir = vel0 / np.linalg.norm(vel0)

    best = None
    for trial in range(n_restarts):
        if trial == 0:
            direction = vel0_dir
            tscale0 = 0.5
            mscale0 = 1.0
        else:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            tscale0 = rng.uniform(0.15, 1.2)
            mscale0 = rng.uniform(0.7, 1.6)

        x0 = np.array([direction[0] * mscale0, direction[1] * mscale0, direction[2] * mscale0, tscale0])
        try:
            sol = least_squares(
                miss_vector, x0, args=(pos0, vel0, dv0_mag_nd, t1_nom, t0),
                method="trf", max_nfev=400, xtol=1e-13, ftol=1e-13,
            )
        except Exception as ex:
            if verbose:
                print(f"  trial {trial}: exception {ex}")
            continue
        miss_km = np.linalg.norm(sol.fun) * bl.L / 1000.0
        if verbose:
            print(f"  trial {trial}: miss={miss_km:.1f} km")
        if best is None or miss_km < best[0]:
            best = (miss_km, sol.x)
        if miss_km < 1.0:
            break

    miss_km, x = best
    dvx, dvy, dvz, t1_scale = x
    dv0_nd = dv0_mag_nd * np.array([dvx, dvy, dvz])
    vel_after = vel0 + dv0_nd
    state_dep = np.concatenate([pos0, vel_after])
    t1 = t1_nom * max(t1_scale, 0.05)
    final_state, impacted = bl.propagate(state_dep, t0, t1, rtol=1e-12, atol=1e-13)

    return {
        "miss_km": miss_km,
        "dv0_nd": dv0_nd,
        "dv0_si": np.linalg.norm(dv0_nd) * bl.V,
        "pos0": pos0, "vel_after0": vel_after,
        "final_state": final_state, "impacted": impacted,
        "t1_nd": t1, "t0": t0,
        "raan_e": raan_e, "argp_e": argp_e, "nu_e": nu_e,
    }


if __name__ == "__main__":
    import sys
    E = np.loadtxt("Earth_orbits.txt", skiprows=1)
    idxE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    aE, eE, iE = E[idxE, 1:]
    print(f"Earth orbit {idxE}: a={aE:.0f} e={eE:.4f} i={iE:.4f}")
    r = shoot_lm(aE, eE, iE, verbose=True, n_restarts=15)
    print(f"BEST miss={r['miss_km']:.3f} km, dv0={r['dv0_si']:.1f} m/s, t1_days={r['t1_nd']*bl.T/86400:.2f}, impacted={r['impacted']}")
