"""Two-body Keplerian propagation.

Given an inertial (ECI) state at an epoch, propagate under a point-mass
Earth with the WGS84 gravitational parameter. Only bound (elliptic) orbits
are supported: parabolic / hyperbolic trajectories and degenerate states
raise ``OrbitError``, which the HTTP layer maps to 422.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# WGS84 constants
MU_EARTH = 3.986004418e14  # m^3 s^-2
MU_EARTH_KM = 398600.4418  # km^3 s^-2
EARTH_OMEGA = 7.2921151467e-5  # rad/s, WGS84
EARTH_RADIUS_KM = 6378.137  # WGS84 semi-major axis (km)
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_E2 = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)  # eccentricity squared

# Solver tolerances.
_NEWTON_TOL = 1.0e-10
_NEWTON_MAX_ITER = 80

# Orbits with eccentricity closer to 1 than this are treated as parabolic.
_ECC_PARABOLIC = 1.0 - 1.0e-9


class OrbitError(ValueError):
    """Raised for degenerate or unsupported orbital states."""


@dataclass
class TwoBodyOrbit:
    r0: np.ndarray  # km, ECI position at epoch
    v0: np.ndarray  # km/s, ECI velocity at epoch

    def __post_init__(self) -> None:
        r = np.asarray(self.r0, dtype=float)
        v = np.asarray(self.v0, dtype=float)
        if r.shape != (3,) or v.shape != (3,):
            raise OrbitError("position and velocity must be 3-vectors")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise OrbitError("position/velocity contain non-finite values")

        rmag = float(np.linalg.norm(r))
        vmag = float(np.linalg.norm(v))
        if rmag < 1.0e-9:
            raise OrbitError("position vector is zero")
        if vmag < 1.0e-12:
            raise OrbitError("velocity vector is zero: orbit is degenerate")

        h = np.cross(r, v)
        hmag = float(np.linalg.norm(h))
        if hmag < 1.0e-10:
            raise OrbitError(
                "orbit is degenerate: position and velocity are collinear"
            )

        energy = 0.5 * vmag * vmag - MU_EARTH_KM / rmag
        if energy >= -1.0e-9:
            raise OrbitError(
                "orbit is not bound (parabolic/hyperbolic): only elliptic "
                "orbits are supported"
            )

        a = -MU_EARTH_KM / (2.0 * energy)
        if not (a > 0.0) or not math.isfinite(a):
            raise OrbitError("could not determine a positive semi-major axis")

        n = math.sqrt(MU_EARTH_KM / a ** 3)  # rad/s mean motion
        period = 2.0 * math.pi / n

        # Eccentricity vector.
        evec = ((vmag * vmag - MU_EARTH_KM / rmag) * r - np.dot(r, v) * v) / MU_EARTH_KM
        ecc = float(np.linalg.norm(evec))
        if ecc >= _ECC_PARABOLIC:
            raise OrbitError(
                f"eccentricity {ecc:.6g} too close to 1: orbit not safely elliptic"
            )
        if ecc >= 1.0:
            raise OrbitError("orbit is hyperbolic: only elliptic orbits are supported")

        # Orbital plane basis: node/perigee-independent perifocal-like frame
        # derived directly from h and e vectors (works also at e == 0).
        p_hat = evec / ecc if ecc > 1.0e-14 else self._circular_inplane(r, h)
        q_hat = h / hmag
        # q_hat = h_hat cross p_hat completes the perifocal frame
        q_hat = np.cross(h / hmag, p_hat)

        # Initial eccentric anomaly from E = atan2(e_dot, r_dot) where
        # r = a (cos E - e) p + b sin E q.
        r_p = float(np.dot(r, p_hat))
        r_q = float(np.dot(r, q_hat))
        b = a * math.sqrt(max(0.0, 1.0 - ecc * ecc))
        cos_e = ecc + r_p / a
        sin_e = r_q / b if b > 0.0 else 0.0
        e0 = math.atan2(sin_e, cos_e)
        m0 = e0 - ecc * math.sin(e0)

        self.r = r
        self.v = v
        self.rmag = rmag
        self.a = a
        self.e = ecc
        self.n = n
        self.period = period
        self.periapsis = a * (1.0 - ecc)
        self.apoapsis = a * (1.0 + ecc)
        self.p = p_hat
        self.q = q_hat
        self.b = b
        self.m0 = m0
        self._e_prev = e0

    @staticmethod
    def _circular_inplane(r: np.ndarray, h: np.ndarray) -> np.ndarray:
        """In-plane reference direction for circular orbits."""
        p = r / np.linalg.norm(r)
        return p

    @staticmethod
    def _solve_kepler(mean_anom: float, ecc: float, guess: float) -> tuple[float, bool]:
        """Solve M = E - e sin E by Newton-Raphson. Returns (E, ok)."""
        # Wrap M into [-pi, pi] and seed E accordingly.
        m = (mean_anom + math.pi) % (2.0 * math.pi) - math.pi
        e = guess
        # Keep the warm start near the wrapped solution to avoid divergence.
        if abs(e - m) > math.pi:
            e = m + ecc * math.sin(m)
        for _ in range(_NEWTON_MAX_ITER):
            f = e - ecc * math.sin(e) - m
            fp = 1.0 - ecc * math.cos(e)
            de = f / fp
            e -= de
            if abs(de) < _NEWTON_TOL:
                return e, True
        # Fallback: classic bisection on a wide interval.
        lo, hi = m - ecc - 0.1, m + ecc + 0.1
        flo = lo - ecc * math.sin(lo) - m
        fhi = hi - ecc * math.sin(hi) - m
        if flo * fhi > 0.0:
            return e, False
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fm = mid - ecc * math.sin(mid) - m
            if abs(fm) < _NEWTON_TOL:
                return mid, True
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi), True

    def state_at(self, dt: float, use_warm_start: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return (position km, velocity km/s) in the same inertial frame,
        dt seconds after epoch. Raises OrbitError on numerical failure."""
        m = self.m0 + self.n * dt
        guess = self._e_prev if use_warm_start else m
        ea, ok = self._solve_kepler(m, self.e, guess)
        if not ok:
            raise OrbitError(f"Kepler solver failed at t = t0 + {dt:.3f} s")
        self._e_prev = ea

        cos_e, sin_e = math.cos(ea), math.sin(ea)
        one_minus_e_cos = 1.0 - self.e * cos_e
        pos = self.a * (cos_e - self.e) * self.p + self.b * sin_e * self.q
        vel = (
            self.a
            * self.n
            / one_minus_e_cos
            * (-sin_e * self.p + math.sqrt(max(0.0, 1.0 - self.e * self.e)) * cos_e * self.q)
        )
        return pos, vel
