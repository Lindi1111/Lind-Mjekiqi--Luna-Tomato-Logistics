"""
Experimental single-transfer designer for the Advanced problem.

Strategy: patched-2-body (Earth-centered) initial guess for a Hohmann-like
departure burn (tangential at periapsis, raising apogee towards the Moon's
distance), then full BCP propagation + a coarse search over coast time to
find a close lunar approach, followed by a least-squares insertion burn to
hit the exact target Moon orbital elements.
"""
import numpy as np
import bcp_lib as bl


def hohmann_dv0_mag(aE, eE, mu_e=bl.MU_EARTH, r_target=bl.L):
    r_p = aE * (1.0 - eE)
    v1 = np.sqrt(mu_e * (2.0 / r_p - 1.0 / aE))
    a_t = (r_p + r_target) / 2.0
    v2 = np.sqrt(mu_e * (2.0 / r_p - 1.0 / a_t))
    period_t = 2 * np.pi * np.sqrt(a_t ** 3 / mu_e)
    return v2 - v1, period_t / 2.0, r_p


def build_departure_state(aE, eE, iE, raan=0.0, argp=0.0, nu=0.0):
    state0 = bl.earth_elements_to_state(aE, eE, iE, raan, argp, nu)
    return state0


def try_transfer(aE, eE, iE, aL, eL, iL, t0=0.0, raan_e=0.0, argp_e=0.0, nu_e=0.0,
                  dv0_scale=1.0, dt_scale_list=None, verbose=False):
    state0 = build_departure_state(aE, eE, iE, raan_e, argp_e, nu_e)
    pos0, vel0 = state0[:3], state0[3:]

    dv0_mag_si, half_period_si, r_p = hohmann_dv0_mag(aE, eE)
    dv0_mag_si *= dv0_scale

    # tangential direction = normalized velocity (in the SI Earth-centered
    # equivalent frame, i.e. the derotated v used inside earlier construction);
    # easiest: perturb in the direction of vel0 itself (already matches orbit
    # shape at this true anomaly).
    vel0_si_dir = vel0.copy()
    vnorm = np.linalg.norm(vel0_si_dir)
    dv0_nondim = (dv0_mag_si / bl.V) * (vel0_si_dir / vnorm)

    vel_after0 = vel0 + dv0_nondim
    state_departure = np.concatenate([pos0, vel_after0])

    if dt_scale_list is None:
        dt_scale_list = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0]

    best = None
    for s in dt_scale_list:
        dt_si = half_period_si * s
        dt_nd = dt_si / bl.T
        final_state, impacted = bl.propagate(state_departure, t0, dt_nd)
        if impacted or final_state is None:
            if verbose:
                print(f"  scale={s:.2f}: impact")
            continue
        pos1 = final_state[:3]
        dist_to_moon_km = np.linalg.norm(pos1 - bl.MOON_CENTER) * bl.L / 1000.0
        if verbose:
            print(f"  scale={s:.2f}: dist_to_moon={dist_to_moon_km:.0f} km")
        if best is None or dist_to_moon_km < best[0]:
            best = (dist_to_moon_km, s, final_state, dt_nd)

    if best is None:
        return None

    dist_km, s, final_state, dt_nd = best
    pos1, vel1 = final_state[:3], final_state[3:]

    dv2_nd, v_after = bl.solve_insertion_dv(
        pos1, vel1, aL, eL, iL, bl.MU_MOON, bl.MOON_CENTER, v_guess=vel1,
    )
    if dv2_nd is None:
        if verbose:
            print(f"  best dist={dist_km:.0f} km, no insertion solution found")
        return None

    dv0_si = dv0_mag_si
    dv2_si = np.linalg.norm(dv2_nd) * bl.V
    total_dv = dv0_si + dv2_si
    mass = bl.mass_from_dv(total_dv)

    return {
        "dv0_si": dv0_si,
        "dv2_si": dv2_si,
        "total_dv": total_dv,
        "mass": mass,
        "dt_nd": dt_nd,
        "dt_days": dt_nd * bl.T / 86400.0,
        "closest_km": dist_km,
        "pos0": pos0, "vel_after0": vel_after0,
        "pos1": pos1, "vel1": vel1, "dv2_nd": dv2_nd,
        "raan_e": raan_e, "argp_e": argp_e, "nu_e": nu_e, "t0": t0,
    }


if __name__ == "__main__":
    import sys
    E = np.loadtxt("Earth_orbits.txt", skiprows=1)
    M = np.loadtxt("Moon_orbits.txt", skiprows=1)

    idxE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    idxL = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    aE, eE, iE = E[idxE, 1:]
    aL, eL, iL = M[idxL, 1:]
    print(f"Earth orbit {idxE}: a={aE:.0f} m e={eE:.4f} i={iE:.4f} rad")
    print(f"Moon orbit {idxL}: a={aL:.0f} m e={eL:.4f} i={iL:.4f} rad")

    res = try_transfer(aE, eE, iE, aL, eL, iL, verbose=True)
    if res is None:
        print("NO FEASIBLE TRANSFER FOUND")
    else:
        print(f"dv0={res['dv0_si']:.1f} m/s dv2={res['dv2_si']:.1f} m/s total={res['total_dv']:.1f} m/s")
        print(f"mass delivered (uncapped) = {res['mass']:.2f} kg, transfer time = {res['dt_days']:.2f} days")
        print(f"closest approach used = {res['closest_km']:.0f} km")
