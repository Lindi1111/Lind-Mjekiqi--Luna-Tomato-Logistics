"""Shooting method v5: wide coast-time search (hours to ~150 days) to find
low-energy transfers, since direct short Hohmann-like transfers need DV
well beyond what the rocket equation can afford (5000 kg wet mass gives
~3000-4000 m/s of usable characteristic velocity before delivered mass
goes negative).
"""
import numpy as np
from scipy.optimize import least_squares
import bcp_lib as bl
from transfer_design import hohmann_dv0_mag


def departure_state(aE, eE, iE, nu_e, raan_e=0.0, argp_e=np.pi):
    return bl.earth_elements_to_state(aE, eE, iE, raan_e, argp_e, nu_e)


def residual(params, aE, eE, iE, aL, eL, iL, dv0_mag_nd_nom, t0=0.0, rtol=1e-9, atol=1e-10):
    nu_e, dvx, dvy, dvz, t1_nd, raan_l, argp_l, nu_l = params
    state0 = departure_state(aE, eE, iE, nu_e)
    pos0, vel0 = state0[:3], state0[3:]
    dv0 = dv0_mag_nd_nom * np.array([dvx, dvy, dvz])
    vel_after = vel0 + dv0
    state_dep = np.concatenate([pos0, vel_after])
    final_state, impacted = bl.propagate(state_dep, t0, max(abs(t1_nd), 0.01), rtol=rtol, atol=atol)
    if impacted or final_state is None:
        return np.array([1e2, 1e2, 1e2])
    pos1 = final_state[:3]
    target_state = bl.moon_elements_to_state(aL, eL, iL, raan_l, argp_l, nu_l)
    target_pos = target_state[:3]
    return pos1 - target_pos


DAY_ND = 86400.0 / bl.T  # one day in nondim time units


def monte_carlo(aE, eE, iE, aL, eL, iL, dv0_mag_nd, n_samples, rng, t0=0.0,
                 t1_days_range=(0.3, 150.0), mscale_range=(0.3, 2.5)):
    best = None
    for _ in range(n_samples):
        t1_days = np.exp(rng.uniform(np.log(t1_days_range[0]), np.log(t1_days_range[1])))
        params = np.array([
            rng.uniform(0, 2 * np.pi),
            *(rng.normal(size=3) * rng.uniform(*mscale_range) / np.sqrt(3)),
            t1_days * DAY_ND,
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0, 2 * np.pi),
        ])
        res = residual(params, aE, eE, iE, aL, eL, iL, dv0_mag_nd, t0, rtol=1e-7, atol=1e-8)
        miss = np.linalg.norm(res)
        if best is None or miss < best[0]:
            best = (miss, params.copy())
    return best


def dv_cost(x, aE, eE, iE, aL, eL, iL, dv0_mag_nd, t0=0.0):
    nu_e, dvx, dvy, dvz, t1_nd, raan_l, argp_l, nu_l = x
    state0 = departure_state(aE, eE, iE, nu_e)
    vel0 = state0[3:]
    dv0 = dv0_mag_nd * np.array([dvx, dvy, dvz])
    vel_after0 = vel0 + dv0
    final_state, impacted = bl.propagate(
        np.concatenate([state0[:3], vel_after0]), t0, max(abs(t1_nd), 0.01), rtol=1e-12, atol=1e-13)
    if impacted or final_state is None:
        return None, None
    target_state = bl.moon_elements_to_state(aL, eL, iL, raan_l, argp_l, nu_l)
    dv2 = target_state[3:] - final_state[3:]
    total_si = (np.linalg.norm(dv0) + np.linalg.norm(dv2)) * bl.V
    return total_si, final_state


def design(aE, eE, iE, aL, eL, iL, t0=0.0, n_mc=1500, n_restarts=4, verbose=False, rng=None,
           t1_days_range=(0.3, 150.0), mscale_range=(0.3, 2.5)):
    if rng is None:
        rng = np.random.default_rng(5)
    dv0_mag_si, _, _ = hohmann_dv0_mag(aE, eE)
    dv0_mag_nd = dv0_mag_si / bl.V

    best_overall = None
    best_feasible = None
    for attempt in range(n_restarts):
        miss0, x0 = monte_carlo(aE, eE, iE, aL, eL, iL, dv0_mag_nd, n_mc, rng, t0,
                                 t1_days_range, mscale_range)
        if verbose:
            print(f"  attempt {attempt}: MC best miss={miss0 * bl.L / 1000:.0f} km, t1={x0[4]*bl.T/86400:.1f} days")
        sol = least_squares(
            residual, x0, args=(aE, eE, iE, aL, eL, iL, dv0_mag_nd, t0, 1e-11, 1e-12),
            method="trf", max_nfev=400, xtol=1e-14, ftol=1e-14,
        )
        miss_km = np.linalg.norm(sol.fun) * bl.L / 1000.0
        if verbose:
            print(f"    refined miss={miss_km:.4f} km, t1={sol.x[4]*bl.T/86400:.2f} days")
        if best_overall is None or miss_km < best_overall[0]:
            best_overall = (miss_km, sol.x.copy())
        if miss_km < 1.0:
            cost, _ = dv_cost(sol.x, aE, eE, iE, aL, eL, iL, dv0_mag_nd, t0)
            if cost is not None and (best_feasible is None or cost < best_feasible[0]):
                best_feasible = (cost, miss_km, sol.x.copy())
                if verbose:
                    print(f"    -> feasible, total_dv={cost:.1f} m/s")

    if best_feasible is not None:
        total_dv, miss_km, x = best_feasible
    else:
        miss_km, x = best_overall
        total_dv = None

    nu_e, dvx, dvy, dvz, t1_nd, raan_l, argp_l, nu_l = x
    state0 = departure_state(aE, eE, iE, nu_e)
    pos0, vel0 = state0[:3], state0[3:]
    dv0 = dv0_mag_nd * np.array([dvx, dvy, dvz])
    vel_after0 = vel0 + dv0
    final_state, impacted = bl.propagate(
        np.concatenate([pos0, vel_after0]), t0, max(abs(t1_nd), 0.01), rtol=1e-12, atol=1e-13)
    target_state = bl.moon_elements_to_state(aL, eL, iL, raan_l, argp_l, nu_l)
    dv2 = (target_state[3:] - final_state[3:]) if final_state is not None else None

    return {
        "miss_km": miss_km, "impacted": impacted, "total_dv_si": total_dv,
        "nu_e": nu_e, "t0": t0, "t1_nd": t1_nd,
        "dv0_nd": dv0, "dv0_si": np.linalg.norm(dv0) * bl.V,
        "dv2_nd": dv2, "dv2_si": (np.linalg.norm(dv2) * bl.V) if dv2 is not None else None,
        "pos0": pos0, "vel_after0": vel_after0,
        "final_state": final_state, "target_state": target_state,
        "raan_l": raan_l, "argp_l": argp_l, "nu_l": nu_l,
    }


if __name__ == "__main__":
    import sys
    E = np.loadtxt("Earth_orbits.txt", skiprows=1)
    M = np.loadtxt("Moon_orbits.txt", skiprows=1)
    idxE = int(sys.argv[1]) if len(sys.argv) > 1 else 304
    idxL = int(sys.argv[2]) if len(sys.argv) > 2 else 234
    n_mc = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
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
