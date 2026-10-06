"""Shared helpers for the test-suite."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from app.geometry import Station, _rz
from app.mechanics import MU_EARTH_KM
from app.time_utils import gmst_rad

EPOCH = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
ALT_KM = 420.0
STATION = dict(latitude_deg=39.9, longitude_deg=116.4, altitude_km=0.05)


def circular_leo_state(
    alt_km: float = ALT_KM,
    inc_deg: float = 51.6,
    station_lat: float = 39.9,
    station_lon: float = 116.4,
    target_offset_s: float = 300.0,
):
    """Return (epoch, position_km, velocity_km_s) for a circular orbit whose
    ground track passes near the given station ~target_offset_s after epoch."""
    a = 6378.137 + alt_km
    inc = math.radians(inc_deg)
    n = math.sqrt(MU_EARTH_KM / a ** 3)
    sta = Station(station_lat, station_lon, 0.05)

    def state(raan, u):
        cr, sr = math.cos(raan), math.sin(raan)
        ci, si = math.cos(inc), math.sin(inc)
        cu, su = math.cos(u), math.sin(u)
        r = a * np.array([cr * cu - sr * su * ci,
                          sr * cu + cr * su * ci, su * si])
        v = a * n * np.array([-cr * su - sr * cu * ci,
                              -sr * su + cr * cu * ci, cu * si])
        return r, v

    when = EPOCH + timedelta(seconds=target_offset_s)
    target_dir_eci = _rz(gmst_rad(when)) @ (sta.ecef / np.linalg.norm(sta.ecef))

    best = (-2.0, 0.0, 0.0)
    for raan_deg in range(0, 360, 2):
        for u_deg in range(0, 360, 2):
            r, _ = state(math.radians(raan_deg),
                         math.radians(u_deg) + n * target_offset_s)
            dot = float(np.dot(r / np.linalg.norm(r), target_dir_eci))
            if dot > best[0]:
                best = (dot, math.radians(raan_deg), math.radians(u_deg))
    _, raan, u = best
    r, v = state(raan, u)
    return EPOCH, r.tolist(), v.tolist()


def make_body(**overrides):
    epoch, pos, vel = circular_leo_state()
    body = {
        "epoch": epoch.isoformat().replace("+00:00", "Z"),
        "start": epoch.isoformat().replace("+00:00", "Z"),
        "end": (epoch + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "min_elevation_deg": 10.0,
        "satellite": {"position_km": pos, "velocity_km_s": vel},
        "station": dict(STATION),
    }
    body.update(overrides)
    return body
