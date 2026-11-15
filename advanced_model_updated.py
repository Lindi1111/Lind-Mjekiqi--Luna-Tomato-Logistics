"""
ESA SpOC 4 - Luna Tomato Logistics
Advanced: Trajectory Matching

Preliminary BCP dynamics implementation.

IMPORTANT:
The equations have not yet been verified against
the complete official ESA specification.
No feasible transfer has been validated.
"""

import numpy as np

MU = 0.01215058439470971
MU_S = 3.3294604877306713e5
RHO_S = 3.88811143e2
OMEGA_S = -9.25195985e-1

EARTH_POS = np.array([-MU, 0.0, 0.0])
MOON_POS = np.array([1.0 - MU, 0.0, 0.0])

def sun_position(t):
    """
    Pozicioni i Diellit në sistemin rrotullues BCP,
    në njësi jodimensionale.
    """
    theta = OMEGA_S * t

    return np.array([
        RHO_S * np.cos(theta),
        RHO_S * np.sin(theta),
        0.0
    ])


def earth_moon_gravity(position):
    """
    Përshpejtimi gravitacional nga Toka dhe Hëna,
    në njësi jodimensionale.
    """
    r_earth = position - EARTH_POS
    r_moon = position - MOON_POS

    d_earth = np.linalg.norm(r_earth)
    d_moon = np.linalg.norm(r_moon)

    if d_earth == 0 or d_moon == 0:
        raise ValueError("Pozicioni përkon me qendrën e Tokës ose Hënës.")

    a_earth = -(1.0 - MU) * r_earth / d_earth**3
    a_moon = -MU * r_moon / d_moon**3

    return a_earth + a_moon


def rotating_frame_acceleration(position, velocity):
    """
    Termat e përshpejtimit në sistemin rrotullues.
    """
    x, y, z = position
    vx, vy, vz = velocity

    return np.array([
        x + 2.0 * vy,
        y - 2.0 * vx,
        0.0
    ])


def sun_gravity(position, t):
    """
    Kontributi gravitacional i Diellit në modelin BCP.
    Njësitë janë jodimensionale.
    """
    r_sun = sun_position(t)
    relative = position - r_sun

    distance = np.linalg.norm(relative)

    if distance == 0:
        raise ValueError("Pozicioni përkon me qendrën e Diellit.")

    # Tërheqja direkte e Diellit
    direct = -MU_S * relative / distance**3

    # Termi indirekt në sistemin Tokë–Hënë
    indirect = -MU_S * r_sun / RHO_S**3

    return direct + indirect


def bcp_dynamics(t, state):
    """
    Modeli dinamik BCP në koordinata rrotulluese.

    state = [x, y, z, vx, vy, vz]
    """
    position = state[:3]
    velocity = state[3:]

    a_gravity = earth_moon_gravity(position)
    a_rotating = rotating_frame_acceleration(position, velocity)
    a_sun = sun_gravity(position, t)

    acceleration = a_gravity + a_rotating + a_sun

    return np.concatenate([velocity, acceleration])




def orbital_elements_from_state(state, body_center, gm):
    """
    Llogarit a, e, i nga një gjendje në sistemin rrotullues.
    Supozon bosht rrotullimi z dhe shpejtësi këndore 1.
    """
    r = np.asarray(state[:3]) - np.asarray(body_center)
    v_rot = np.asarray(state[3:])

    omega_cross_r = np.array([-r[1], r[0], 0.0])
    v = v_rot + omega_cross_r

    r_norm = np.linalg.norm(r)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    e_vector = np.cross(v, h) / gm - r / r_norm
    e = np.linalg.norm(e_vector)

    energy = np.dot(v, v) / 2 - gm / r_norm
    a = -gm / (2 * energy)

    i = np.arctan2(np.linalg.norm(h[:2]), h[2])

    return np.array([a, e, i])
