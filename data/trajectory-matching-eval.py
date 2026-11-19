import sys
import numpy  as np
import pykep  as pk
import heyoka as hy
from copy import deepcopy

L = 3.84405000e8
T = 3.7567696752e5
V = L / T

MU_SUN   = 1.32712440041279419e20
MU_EARTH = 398600435507000.0
MU_MOON  = 4902800118000.0
CR3BP_MU_EARTH_MOON = MU_MOON / (MU_MOON + MU_EARTH)
BCP_MU_S    = MU_SUN / (MU_MOON + MU_EARTH)
BCP_RHO_S   = 3.88811143E2
BCP_OMEGA_S = -9.25195985E-01

class ltl:
    def __init__(self, path):
        self.path = path
        self.dim  = 8400

        self._load_data()

    def _load_data(self):
        self.earth_data = np.loadtxt(self.path + "Earth_orbits.txt", skiprows=1)[:, 1:]
        self.moon_data  = np.loadtxt(self.path + "Moon_orbits.txt", skiprows=1)[:, 1:]
        ltl_data = np.loadtxt(self.path + "LTL.txt", skiprows=1)[:, :]
        self.ltl_dict = {(int(a), int(b)): weight for a, b, weight in ltl_data}

    def _real_tomatoe_mass(self, idl, idd, mass, dt):    
        idl = int(idl)
        idd = int(idd)
        cld = self.ltl_dict[(idl,idd)]
        if (200 - dt) * cld < mass:
            return (200-dt) * cld
        else:
            return mass

    def _validate_matching(self, matching):
        earth = set()
        moon  = set()
        dest  = set()
        for (e,m,d) in matching:
            if e in earth or m in moon or d in dest:
                return False
            earth.add(e)
            moon.add(m)
            dest.add(d)
        return True

    def _match_orbit(self, elements, a, e, i):
        return (np.abs(elements[0]  - a) / L < 1e-6) \
            and (np.abs(elements[1] - e)     < 1e-6) \
            and (np.abs(elements[2] - i)     < 1e-6)
            
    def _validate_transfer(self, idE, idL, t0, posvel, DVs, Ts):
        posvel0  = deepcopy(posvel) 
        aE,eE,iE = self.earth_data[idE]
        aL,eL,iL = self.moon_data[idL]

        # Does the initial state match the Earth orbit?
        if not self._match_orbit(state2earth(posvel0), aE, eE, iE):
            return False 
    
        # Propagate the trajectory and see if it impacts Earth or Moon
        posvel1 = propagate(posvel0, t0, DVs, Ts)
        if len(posvel1) == 0: # trajectory impacted Earth or Moon
            return False
    
        # Does the final state match the Moon orbit?
        if not self._match_orbit(state2moon(posvel1), aL, eL, iL):
            return False 

        return sum([np.sqrt(DV[0]**2 + DV[1]**2 + DV[2]**2) for DV in DVs]) * V, sum(Ts) * T * pk.SEC2DAY

    
    def fitness(self, x):
        sol = np.array(x)
        if len(sol) % 21 != 0:
            return [0]   
        sol = sol.reshape(-1, 21)
    
        # We collected the edges of the matching and their accumulated mass.
        matching     = []
        tomatoe_mass = 0
        for line in sol:
            ide, idl, idd, t0, x0, y0, z0, vx0, vy0, vz0, DVx0, DVy0, DVz0, DVx1, DVy1, DVz1, DVx2, DVy2, DVz2, T1, T2 = line
            if ide < 0:
                continue
            
            matching.append((int(ide), int(idl), int(idd)))        
            DVs = [[DVx0, DVy0, DVz0], 
                   [DVx1, DVy1, DVz1], 
                   [DVx2, DVy2, DVz2]]
        
            Ts = [T1, T2]
            posvel = [[x0, y0, z0], [vx0, vy0, vz0]]

            # Check if the transfer is correct
            res = self._validate_transfer(int(ide), int(idl), t0, posvel, DVs, Ts)            
            if res is False:
                return [0]
        
            DV, DT = res  # DT is in days        
            mass = np.exp(-DV/311./pk.G0) * 5000 - 500.        
            tomatoe_mass += self._real_tomatoe_mass(idl, idd, mass, DT)

        # Check if the matching is correct.
        if not self._validate_matching(matching):
            return [0]
        return [-float(tomatoe_mass)]

    def get_bounds(self):
        return ([0,0,0] + [-sys.maxsize/2] * (self.dim-3), [400,400,400] + [sys.maxsize/2] * (self.dim-3) )

def bcp_dyn():
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

    mu        = hy.par[0]
    mu_sun    = hy.par[1]
    rho_sun   = hy.par[2]
    omega_sun = hy.par[3]

    # Distances to primaries
    r1 = hy.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = hy.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    # Sun position
    x_sun = rho_sun * hy.cos(omega_sun * hy.time)
    y_sun = rho_sun * hy.sin(omega_sun * hy.time)

    # Distance to the Sun
    r_sun = hy.sqrt((x - x_sun)**2 + (y - y_sun)**2 + z**2)

    # Equations of motion
    xdot = vx
    ydot = vy
    zdot = vz

    vxdot = (
        2*vy
        + x
        - (1 - mu)*(x + mu)/r1**3
        - mu*(x + mu - 1)/r2**3
        - mu_sun/r_sun**3 * (x - x_sun)
        - mu_sun/rho_sun**2 * hy.cos(omega_sun * hy.time)
    )

    vydot = (
        -2*vx
        + y
        - (1 - mu)*y/r1**3
        - mu*y/r2**3
        - mu_sun/r_sun**3 * (y - y_sun)
        - mu_sun/rho_sun**2 * hy.sin(omega_sun * hy.time)
    )

    vzdot = (
        -(1 - mu)*z/r1**3
        - mu*z/r2**3
        - mu_sun/r_sun**3 * z
    )

    # Return the dynamical system
    return [
        (x, xdot),
        (y, ydot),
        (z, zdot),
        (vx, vxdot),
        (vy, vydot),
        (vz, vzdot),
    ]

def state2earth(posvel):
    """
    Convert a BCP state vector (non dimensional) to orbital elements in the Earth system (SI).
    """
    [x, y, z], [vx, vy, vz] = posvel
    # Derotate the velocities (becoming absolute velocity in the inertial non-rotating system)
    vx_EF = (vx - y) * V
    vy_EF = (vy + x) * V
    vz_EF = vz * V
    # Subtract the Earth velocity (becoming relative velocity in the inertial non-rotating system)
    vy_EF = vy_EF - (-CR3BP_MU_EARTH_MOON) * V
    # Translate the position
    x_EF = (x + CR3BP_MU_EARTH_MOON) * L
    y_EF = y * L
    z_EF = z * L
    return pk.ic2par([x_EF, y_EF, z_EF], [vx_EF, vy_EF, vz_EF], MU_EARTH)[:3]

def state2moon(posvel):
    """
    Convert a BCP state vector (non dimensional) to orbital elements in the Moon system (SI).
    """
    [x, y, z], [vx, vy, vz] = posvel
    # Derotate the velocities (becoming absolute velocity in the inertial non-rotating system)
    vx_MF = (vx - y) * V
    vy_MF = (vy + x) * V
    vz_MF = vz * V
    # Subtract the Moon velocity (becoming relative velocity in the inertial non-rotating system)
    vy_MF = vy_MF - (1.0 - CR3BP_MU_EARTH_MOON) * V
    # Translate the position
    x_MF = (x - 1 + CR3BP_MU_EARTH_MOON) * L
    y_MF = y * L
    z_MF = z * L
    el = pk.ic2par([x_MF, y_MF, z_MF], [vx_MF, vy_MF, vz_MF], MU_MOON)
    return el
    
def propagate(posvel, t0, DVs, Ts):
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    dyn           = bcp_dyn()
    earth_impacts = []
    moon_impacts  = []

    # Define a callback for the Earth impact event.
    def cb_E(ta, time, d_sgn):
        earth_impacts.append(time)
    ev_E = hy.nt_event(
        (x + hy.par[0]) ** 2 + y**2 + z**2 - (pk.EARTH_RADIUS + 99000) ** 2 / L**2,
        callback = cb_E,
    )

    # Define a callback for the Moon impact event.
    def cb_M(ta, time, d_sgn):
        moon_impacts.append(time)
    ev_M = hy.nt_event(
        (x - 1 + hy.par[0]) ** 2 + y**2 + z**2 - (1737400.0 + 30000) ** 2 / L**2,
        callback = cb_M,
    )

    ta = hy.taylor_adaptive(dyn, tol=1e-16, nt_events=[ev_E, ev_M])
    ta.pars[:] = [
        CR3BP_MU_EARTH_MOON,
        BCP_MU_S,
        BCP_RHO_S,
        BCP_OMEGA_S,
    ]    
    ta.time = t0

    # build initial state
    posvel0 = deepcopy(posvel)
    posvel0[1][0] += DVs[0][0]
    posvel0[1][1] += DVs[0][1]
    posvel0[1][2] += DVs[0][2]    
    ta.state[:6] = posvel0[0] + posvel0[1]

    # propagate for time and add further DVs
    for i, time in enumerate(Ts):
        ta.propagate_for(time)
        ta.state[3] = ta.state[3] + DVs[i+1][0]
        ta.state[4] = ta.state[4] + DVs[i+1][1]
        ta.state[5] = ta.state[5] + DVs[i+1][2]

    # we return an empty state if there was an impact with Earth or Moon, otherwise we return the new state 
    if len(earth_impacts) > 0 or len(moon_impacts) > 0:
        return []
    return [list(ta.state[:3]), list(ta.state[3:6])]
    
#
# Build to UDP.
#
udp = ltl("data/spoc4/ltl/")
