"""Shooting method v4: joint targeting of the FULL arrival position against
a free point on the target Moon orbit (parameterized by raan_l, argp_l,
nu_l). Once position matches, the required arrival velocity is read off
directly from the target orbit's own velocity at that point (no separate
nonlinear insertion solve needed) -- DV2 = v_target - v_trajectory.
"""
import numpy as np
from scipy.optimize import least_squares
import bcp_lib as bl
from transfer_design import hohmann_dv0_mag


def departure_state(aE, eE, iE, nu_e, raan_e=0.0, argp_e=np.pi):
    return bl.earth_elements_to_state(aE, eE, iE, raan_e, argp_e, nu_e)


def residual(params, aE, eE, iE, aL, eL, iL, dv0_mag_nd_nom, t1_nom, t0=0.0,
             rtol=1e-9, atol=1e-10):
    nu_e, dvx, dvy, dvz, t1_scale, raan_l, argp_l, nu_l = params
    state0 = departure_state(aE, eE, iE, nu_e)
    pos0, vel0 = state0[:3], state0[3:]
    dv0 = dv0_mag_nd_nom * np.array([dvx, dvy, dvz])
    vel_after = vel0 + dv0
    state_dep = np.concatenate([pos0, vel_after])
    t1 = t1_nom * max(abs(t1_scale), 0.02)
    final_state, impacted = bl.propagate(state_dep, t0, t1, rtol=rtol, atol=atol)
    if impacted or final_state is None:
        return np.array([1e2, 1e2, 1e2])
    pos1 = final_state[:3]
    target_state = bl.moon_elements_to_state(aL, eL, iL, raan_l, argp_l, nu_l)
    target_pos = target_state[:3]
    return pos1 - target_pos


def monte_carlo(aE, eE, iE, aL, eL, iL, dv0_mag_nd, t1_nom, n_samples, rng, t0=0.0,
                 tscale_max=1.5):
    best = None
    for _ in range(n_samples):
        params = np.array([
            rng.uniform(0, 2 * np.pi),
            *(rng.normal(size=3) * rng.uniform(0.6, 1.8) / np.sqrt(3)),
            rng.uniform(0.05, tscale_max),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0, 2 * np.pi),
        ])
        res = residual(params, aE, eE, iE, aL, eL, iL, dv0_mag_nd, t1_nom, t0, rtol=1e-7, atol=1e-8)
        miss = np.linalg.norm(res)
        if best is None or miss < best[0]:
            best = (miss, params.copy())
    return best


def design(aE, eE, iE, aL, eL, iL, t0=0.0, n_mc=4000, n_restarts=3, verbose=False, rng=None,
           mc_tscale_max=1.5):
    if rng is None:
        rng = np.random.default_rng(3)
    dv0_mag_si, half_period_si, r_p = hohmann_dv0_mag(aE, eE)
    dv0_mag_nd = dv0_mag_si / bl.V
    t1_nom = half_period_si / bl.T

    def dv_cost(x):
        nu_e, dvx, dvy, dvz, t1_scale, raan_l, argp_l, nu_l = x
        state0 = departure_state(aE, eE, iE, nu_e)
        vel0 = state0[3:]
        dv0 = dv0_mag_nd * np.array([dvx, dvy, dvz])
        vel_after0 = vel0 + dv0
        t1 = t1_nom * max(abs(t1_scale), 0.02)
        final_state, impacted = bl.propagate(
            np.concatenate([state0[:3], vel_after0]), t0, t1, rtol=1e-12, atol=1e-13)
        if impacted or final_state is None:
            return None
        target_state = bl.moon_elements_to_state(aL, eL, iL, raan_l, argp_l, nu_l)
        dv2 = target_state[3:] - final_state[3:]
        total_si = (np.linalg.norm(dv0) + np.linalg.norm(dv2)) * bl.V
        return total_si

    best_overall = None       # (miss_km, x) - fallback if nothing fully converges
    best_feasible = None      # (total_dv_si, miss_km, x) among converged (miss<1 km) solutions
    for attempt in range(n_restarts):
        miss0, x0 = monte_carlo(aE, eE, iE, aL, eL, iL, dv0_mag_nd, t1_nom, n_mc, rng, t0,
                                 tscale_max=mc_tscale_max)
        if verbose:
            print(f"  attempt {attempt}: MC best miss={miss0 * bl.L / 1000:.0f} km")
        sol = least_squares(
            residual, x0, args=(aE, eE, iE, aL, eL, iL, dv0_mag_nd, t1_nom, t0, 1e-11, 1e-12),
            method="trf", max_nfev=400, xtol=1e-14, ftol=1e-14,
        )
        miss_km = np.linalg.norm(sol.fun) * bl.L / 1000.0
        if verbose:
            print(f"    refined miss={miss_km:.4f} km")
        if best_overall is None or miss_km < best_overall[0]:
            best_overall = (miss_km, sol.x.copy())
        if miss_km < 1.0:
            cost = dv_cost(sol.x)
            if cost is not None and (best_feasible is None or cost < best_feasible[0]):
                best_feasible = (cost, miss_km, sol.x.copy())
                if verbose:
                    print(f"    -> feasible, total_dv={cost:.1f} m/s")

    if best_feasible is not None:
        _, miss_km, x = best_feasible
    else:
        miss_km, x = best_overall
    nu_e, dvx, dvy, dvz, t1_scale, raan_l, argp_l, nu_l = x

    state0 = departure_state(aE, eE, iE, nu_e)
    pos0, vel0 = state0[:3], state0[3:]
    dv0 = dv0_mag_nd * np.array([dvx, dvy, dvz])
    vel_after0 = vel0 + dv0
    state_dep = np.concatenate([pos0, vel_after0])
    t1 = t1_nom * max(abs(t1_scale), 0.02)
    final_state, impacted = bl.propagate(state_dep, t0, t1, rtol=1e-12, atol=1e-13)

    target_state = bl.moon_elements_to_state(aL, eL, iL, raan_l, argp_l, nu_l)
    v_target = target_state[3:]
    vel1 = final_state[3:] if final_state is not None else None
    dv2_nd = (v_target - vel1) if vel1 is not None else None

    return {
        "miss_km": miss_km, "impacted": impacted,
        "nu_e": nu_e, "t0": t0, "t1_nd": t1,
        "dv0_nd": dv0, "dv0_si": np.linalg.norm(dv0) * bl.V,
        "dv2_nd": dv2_nd, "dv2_si": (np.linalg.norm(dv2_nd) * bl.V) if dv2_nd is not None else None,
        "pos0": pos0, "vel_after0": vel_after0,
        "final_state": final_state, "target_state": target_state,
        "raan_l": raan_l, "argp_l": argp_l, "nu_l": nu_l,
    }


if __name__ == "__main__":
    import sys
    E = np.loadtxt("Earth_orbits.txt", skiprows=1)
    M = np.loadtxt("Moon_orbits.txt", skiprows=1)
    idxE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    idxL = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_mc = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    aE, eE, iE = E[idxE, 1:]
    aL, eL, iL = M[idxL, 1:]
    print(f"Earth {idxE}: a={aE:.0f} e={eE:.4f} i={iE:.4f}")
    print(f"Moon {idxL}: a={aL:.0f} e={eL:.4f} i={iL:.4f}")
    r = design(aE, eE, iE, aL, eL, iL, verbose=True, n_mc=n_mc)
    if r["dv2_si"] is None:
        print("FAILED")
    else:
        total = r["dv0_si"] + r["dv2_si"]
        mass = bl.mass_from_dv(total)
        print(f"miss={r['miss_km']:.4f} km dv0={r['dv0_si']:.1f} dv2={r['dv2_si']:.1f} total={total:.1f} m/s mass={mass:.1f} kg t1_days={r['t1_nd']*bl.T/86400:.2f} impacted={r['impacted']}")
