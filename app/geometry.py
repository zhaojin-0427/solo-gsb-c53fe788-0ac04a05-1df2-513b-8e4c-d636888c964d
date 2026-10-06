"""Coordinate frames and topocentric geometry.

- ECI (Earth-Centered Inertial, TEME-like) satellite states are rotated to
  ECEF with a fixed WGS84 Earth rotation rate (GMST angle).
- The station geodetic position is converted to ECEF once with the WGS84
  ellipsoid (station fixed to Earth).
- Topocentric east/north/up basis gives elevation, azimuth and their rates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .mechanics import EARTH_E2, EARTH_OMEGA, EARTH_RADIUS_KM, TwoBodyOrbit
from .time_utils import gmst_rad


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_km: float) -> np.ndarray:
    """WGS84 geodetic latitude/longitude/height(km) -> ECEF (km)."""
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    n = EARTH_RADIUS_KM / math.sqrt(1.0 - EARTH_E2 * sin_lat * sin_lat)
    x = (n + alt_km) * cos_lat * math.cos(lon_rad)
    y = (n + alt_km) * cos_lat * math.sin(lon_rad)
    z = (n * (1.0 - EARTH_E2) + alt_km) * sin_lat
    return np.array([x, y, z])


def enu_basis(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Rows are the local east, north, up unit vectors (ECEF components)."""
    slat, clat = math.sin(lat_rad), math.cos(lat_rad)
    slon, clon = math.sin(lon_rad), math.cos(lon_rad)
    east = np.array([-slon, clon, 0.0])
    north = np.array([-slat * clon, -slat * slon, clat])
    up = np.array([clat * clon, clat * slon, slat])
    return np.vstack([east, north, up])


@dataclass
class Station:
    lat_deg: float
    lon_deg: float
    alt_km: float

    def __post_init__(self) -> None:
        self.lat = math.radians(self.lat_deg)
        self.lon = math.radians(self.lon_deg)
        self.ecef = geodetic_to_ecef(self.lat, self.lon, self.alt_km)
        self.enu = enu_basis(self.lat, self.lon)


def eci_to_ecef(pos_eci: np.ndarray, when: datetime) -> np.ndarray:
    # ECEF x-axis lags the inertial x-axis by GMST: rotate by -theta.
    theta = gmst_rad(when)
    return _rz(-theta) @ pos_eci


def ecef_velocity_from_eci(pos_eci: np.ndarray, vel_eci: np.ndarray, when: datetime) -> np.ndarray:
    """Rotate inertial velocity to the rotating ECEF frame, subtracting the
    Earth rotation term w x r (fixed WGS84 rotation rate)."""
    theta = gmst_rad(when)
    r_ecef = _rz(-theta) @ pos_eci
    omega = np.array([0.0, 0.0, EARTH_OMEGA])
    return _rz(-theta) @ vel_eci - np.cross(omega, r_ecef)


@dataclass
class Observation:
    elevation_deg: float
    azimuth_deg: float  # clockwise from true north, [0, 360)
    range_km: float
    elevation_rate_deg_s: float
    azimuth_rate_deg_s: float


def observe(
    orbit: TwoBodyOrbit,
    dt: float,
    when: datetime,
    station: Station,
) -> Observation:
    """Full topocentric observation of the satellite at time ``when``
    (``dt`` seconds after the orbit epoch)."""
    pos_i, vel_i = orbit.state_at(dt)
    sat_ecef = eci_to_ecef(pos_i, when)
    sat_vel_ecef = ecef_velocity_from_eci(pos_i, vel_i, when)

    rho = sat_ecef - station.ecef
    rng = float(np.linalg.norm(rho))
    if rng < 1.0e-9:
        raise ValueError("satellite coincides with the station")

    enu = station.enu @ rho
    east, north, up = enu
    horiz = math.hypot(east, north)
    elev = math.asin(min(1.0, max(-1.0, up / rng)))
    az = math.atan2(east, north) % (2.0 * math.pi)

    # Rates: differentiate rho_enu. The station is fixed in ECEF.
    rho_dot_ecef = sat_vel_ecef
    enu_dot = station.enu @ rho_dot_ecef
    de, dn, du = enu_dot

    # d(el)/dt = (rho_hat . u_dot - sin(el) * (rho_hat . rho_dot_hat))/rho
    # Simplified via rho^2 * d(el)/dt:
    cos_el = horiz / rng
    if horiz < 1.0e-9:
        el_rate = 0.0
        az_rate = 0.0
    else:
        el_rate = (du * rng - up * float(np.dot(rho, rho_dot_ecef)) / rng) / (rng * rng)
        # azimuth from atan2(e, n): az_dot = (n*de - e*dn)/(e^2+n^2)
        az_rate = (north * de - east * dn) / (horiz * horiz)

    return Observation(
        elevation_deg=math.degrees(elev),
        azimuth_deg=math.degrees(az),
        range_km=rng,
        elevation_rate_deg_s=math.degrees(el_rate),
        azimuth_rate_deg_s=math.degrees(az_rate),
    )
