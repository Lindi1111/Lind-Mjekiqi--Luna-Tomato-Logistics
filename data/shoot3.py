"""Shooting method v3: free departure true-anomaly + full Monte-Carlo warm
start before Levenberg-Marquardt / trust-region refinement. Also supports
a non-zero mid-course impulse (3-impulse) for extra flexibility.
"""
import numpy as np
from scipy.optimize import least_squares
import bcp_lib as bl
from transfer_design import hohmann_dv0_mag


def state_from_params(aE, eE, iE, raan_e, argp_e, nu_e):
    return bl.earth_elements_to_state(aE, eE, iE, raan_e, argp_e, nu_e)


def miss_vector_full(params, aE, eE, iE, dv0_mag_nd_nom, t1_nom, t0=0.0, rtol=1e-9, atol=1e-10,
                      target_dist_nd=0.0):
    nu_e, dvx, dvy, dvz, t1_scale = params
    state0 = state_from_params(aE, eE, iE, 0.0, np.pi, nu_e)
    pos0, vel0 = state0[:3], state0[3:]
    dv0 = dv0_mag_nd_nom * np.array([dvx, dvy, dvz])
    vel_after = vel0 + dv0
    state_dep = np.concatenate([pos0, vel_after])
    t1 = t1_nom * max(abs(t1_scale), 0.02)
    final_state, impacted = bl.propagate(state_dep, t0, t1, rtol=rtol, atol=atol)
    if impacted or final_state is None:
        return np.array([1e2, 1e2, 1e2])
    pos1 = final_state[:3]
    rel = pos1 - bl.MOON_CENTER
    if target_dist_nd <= 0.0:
        return rel
    dist = np.linalg.norm(rel)
    direction = rel / dist if dist > 1e-12 else np.array([1.0, 0.0, 0.0])
    target_point = bl.MOON_CENTER + direction * target_dist_nd
    return pos1 - target_point


def monte_carlo_search(aE, eE, iE, dv0_mag_nd, t1_nom, n_samples=3000, rng=None, t0=0.0,
                        target_dist_nd=0.0):
    if rng is None:
        rng = np.random.default_rng(1)
    best = None
    for _ in range(n_samples):
        nu_e = rng.uniform(0, 2 * np.pi)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        mscale = rng.uniform(0.6, 1.8)
        tscale = rng.uniform(0.1, 1.5)
        params = np.array([nu_e, *(direction * mscale), tscale])
        resid = miss_vector_full(params, aE, eE, iE, dv0_mag_nd, t1_nom, t0, rtol=1e-7, atol=1e-8,
                                  target_dist_nd=target_dist_nd)
        miss = np.linalg.norm(resid)
        if best is None or miss < best[0]:
            best = (miss, params.copy())
    return best


def design(aE, eE, iE, aL=None, t0=0.0, n_mc=3000, verbose=False, rng=None):
    dv0_mag_si, half_period_si, r_p = hohmann_dv0_mag(aE, eE)
    dv0_mag_nd = dv0_mag_si / bl.V
    t1_nom = half_period_si / bl.T
    target_dist_nd = (aL / bl.L) if aL else 0.0

    miss0, x0 = monte_carlo_search(aE, eE, iE, dv0_mag_nd, t1_nom, n_samples=n_mc, rng=rng, t0=t0,
                                    target_dist_nd=target_dist_nd)
    if verbose:
        print(f"  MC best: miss={miss0 * bl.L / 1000:.0f} km  x0={x0}")

    sol = least_squares(
        miss_vector_full, x0, args=(aE, eE, iE, dv0_mag_nd, t1_nom, t0, 1e-11, 1e-12, target_dist_nd),
        method="trf", max_nfev=300, xtol=1e-14, ftol=1e-14,
    )
    miss_km = np.linalg.norm(sol.fun) * bl.L / 1000.0
    if verbose:
        print(f"  refined: miss={miss_km:.3f} km x={sol.x}")

    nu_e, dvx, dvy, dvz, t1_scale = sol.x
    state0 = state_from_params(aE, eE, iE, 0.0, np.pi, nu_e)
    pos0, vel0 = state0[:3], state0[3:]
    dv0 = dv0_mag_nd * np.array([dvx, dvy, dvz])
    vel_after = vel0 + dv0
    state_dep = np.concatenate([pos0, vel_after])
    t1 = t1_nom * max(abs(t1_scale), 0.02)
    final_state, impacted = bl.propagate(state_dep, t0, t1, rtol=1e-12, atol=1e-13)

    return {
        "miss_km": miss_km, "nu_e": nu_e, "t0": t0,
        "dv0_nd": dv0, "dv0_si": np.linalg.norm(dv0) * bl.V,
        "pos0": pos0, "vel_after0": vel_after,
        "final_state": final_state, "impacted": impacted,
        "t1_nd": t1,
    }


if __name__ == "__main__":
    import sys
    E = np.loadtxt("Earth_orbits.txt", skiprows=1)
    idxE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_mc = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    aE, eE, iE = E[idxE, 1:]
    print(f"Earth orbit {idxE}: a={aE:.0f} e={eE:.4f} i={iE:.4f}")
    r = design(aE, eE, iE, verbose=True, n_mc=n_mc)
    print(f"FINAL miss={r['miss_km']:.3f} km dv0={r['dv0_si']:.1f} m/s t1_days={r['t1_nd']*bl.T/86400:.2f} impacted={r['impacted']}")
