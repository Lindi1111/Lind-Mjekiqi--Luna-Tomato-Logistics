"""Shooting method: find a departure impulse (direction+magnitude) and coast
time that brings the spacecraft close to the Moon's center, starting from a
given Earth orbit. Then refine with a local optimizer minimizing miss distance.
"""
import numpy as np
from scipy.optimize import minimize
import bcp_lib as bl
from transfer_design import build_departure_state, hohmann_dv0_mag


def miss_distance(params, pos0, vel0, dv0_mag_nd_nom, t1_nom, t0=0.0):
    mag_scale, theta, phi, t1_scale = params
    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    dv0 = dv0_mag_nd_nom * mag_scale * direction
    vel_after = vel0 + dv0
    state_dep = np.concatenate([pos0, vel_after])
    t1 = t1_nom * t1_scale
    final_state, impacted = bl.propagate(state_dep, t0, t1, rtol=1e-10, atol=1e-11)
    if impacted or final_state is None:
        return 1e9
    pos1 = final_state[:3]
    return np.linalg.norm(pos1 - bl.MOON_CENTER) * bl.L / 1000.0  # km


def shoot(aE, eE, iE, raan_e=0.0, argp_e=np.pi, nu_e=0.0, t0=0.0, verbose=False):
    state0 = build_departure_state(aE, eE, iE, raan_e, argp_e, nu_e)
    pos0, vel0 = state0[:3], state0[3:]

    dv0_mag_si, half_period_si, r_p = hohmann_dv0_mag(aE, eE)
    dv0_mag_nd = dv0_mag_si / bl.V
    t1_nom = half_period_si / bl.T

    vel0_dir = vel0 / np.linalg.norm(vel0)
    theta0 = np.arccos(np.clip(vel0_dir[2], -1, 1))
    phi0 = np.arctan2(vel0_dir[1], vel0_dir[0])

    best = None
    # coarse grid warm-up over magnitude & time scale around the nominal guess
    for mscale in (0.85, 1.0, 1.15, 1.3):
        for tscale in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
            d = miss_distance([mscale, theta0, phi0, tscale], pos0, vel0, dv0_mag_nd, t1_nom, t0)
            if best is None or d < best[0]:
                best = (d, [mscale, theta0, phi0, tscale])
    if verbose:
        print(f"  coarse best: dist={best[0]:.0f} km params={best[1]}")

    x0 = np.array(best[1])
    res = minimize(
        miss_distance, x0, args=(pos0, vel0, dv0_mag_nd, t1_nom, t0),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-3, "maxiter": 2000, "maxfev": 2000},
    )
    if verbose:
        print(f"  refined: dist={res.fun:.3f} km params={res.x}")

    mag_scale, theta, phi, t1_scale = res.x
    direction = np.array([
        np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta),
    ])
    dv0_nd = dv0_mag_nd * mag_scale * direction
    vel_after = vel0 + dv0_nd
    state_dep = np.concatenate([pos0, vel_after])
    t1 = t1_nom * t1_scale
    final_state, impacted = bl.propagate(state_dep, t0, t1, rtol=1e-12, atol=1e-13)

    return {
        "miss_km": res.fun,
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
    r = shoot(aE, eE, iE, verbose=True)
    print(f"miss={r['miss_km']:.1f} km, dv0={r['dv0_si']:.1f} m/s, t1_days={r['t1_nd']*bl.T/86400:.2f}")
