"""FastAPI application for satellite pass prediction."""
from __future__ import annotations

import math

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .geometry import Station
from .mechanics import EARTH_RADIUS_KM, OrbitError, TwoBodyOrbit
from .passes import find_passes
from .relay import StationInput, plan_relay
from .schemas import (
    PassRequest,
    PassResponse,
    RelayRequest,
    RelayResponse,
)

app = FastAPI(
    title="Satellite Pass Observation API",
    version="1.0.0",
    description=(
        "Predict visibility windows of an Earth satellite from a ground "
        "station. Two-body Kepler propagation, fixed WGS84 Earth rotation, "
        "adaptive stepping with bisection-level crossing accuracy."
    ),
)


@app.exception_handler(OrbitError)
async def orbit_error_handler(request: Request, exc: OrbitError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "type": "orbit_degenerate_or_invalid",
                    "loc": ["body", "satellite"],
                    "msg": str(exc),
                }
            ]
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/passes", response_model=PassResponse)
def passes(req: PassRequest):
    orbit = TwoBodyOrbit(
        np.array(req.satellite.position_km, dtype=float),
        np.array(req.satellite.velocity_km_s, dtype=float),
    )

    # Physical reality check: an ellipse whose perigee is inside the Earth
    # cannot be propagated meaningfully for observations.
    min_surface = EARTH_RADIUS_KM - 14.0  # account for deepest ocean trench
    if orbit.periapsis < min_surface:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "type": "orbit_impact_earth",
                        "loc": ["body", "satellite"],
                        "msg": (
                            f"orbit perigee radius {orbit.periapsis:.3f} km is "
                            f"below the Earth surface (~{EARTH_RADIUS_KM:.3f} km); "
                            "two-body propagation is not physically meaningful"
                        ),
                    }
                ]
            },
        )

    station = Station(
        lat_deg=req.station.latitude_deg,
        lon_deg=req.station.longitude_deg,
        alt_km=req.station.altitude_km,
    )

    result = find_passes(
        orbit=orbit,
        station=station,
        epoch=req.epoch,
        t_start=req.start,
        t_end=req.end,
        min_elevation_deg=req.min_elevation_deg,
    )
    span_s = (req.end - req.start).total_seconds()
    uncovered_s = sum(
        (u.end - u.start).total_seconds() for u in result.uncovered
    )
    diagnostics = {
        "orbit": {
            "semi_major_axis_km": orbit.a,
            "eccentricity": orbit.e,
            "period_s": orbit.period,
            "perigee_radius_km": orbit.periapsis,
            "apogee_radius_km": orbit.apoapsis,
            "perigee_altitude_km": orbit.periapsis - EARTH_RADIUS_KM,
            "apogee_altitude_km": orbit.apoapsis - EARTH_RADIUS_KM,
        },
        "coarse_samples": result.coarse_samples,
        "dense_samples": result.dense_samples,
        "search_span_s": span_s,
        "uncovered_seconds": uncovered_s,
        "coverage_fraction": max(0.0, 1.0 - uncovered_s / span_s)
        if span_s > 0 else 1.0,
        "notes": (
            "Elevation crossings refined by bisection; peaks by golden-section "
            "search. Half-open search interval [start, end). Below-threshold "
            "intervals count as covered only when a Lipschitz rate bound "
            "proves them invisible; unprovable brackets are bisected down to "
            "the 1 us reporting resolution to expose hidden crossings, and "
            "whatever stays unresolvable (grazing contacts) is reported in "
            "uncovered_intervals instead of being silently skipped."
        ),
    }

    return PassResponse(
        status=result.status,
        count=len(result.windows),
        windows=[
            {
                "start": w.start,
                "end": w.end,
                "peak_time": w.peak_time,
                "peak_elevation_deg": w.peak_elevation_deg,
                "peak_azimuth_deg": _nan_none(w.peak_azimuth_deg),
                "rise_azimuth_deg": _nan_none(w.rise_azimuth_deg),
                "set_azimuth_deg": _nan_none(w.set_azimuth_deg),
                "max_range_km": w.max_range_km,
                "duration_s": w.duration_s,
                "start_open": w.start_open,
                "end_open": w.end_open,
                "truncated": w.truncated,
            }
            for w in result.windows
        ],
        uncovered_intervals=[
            {"start": u.start, "end": u.end, "reason": u.reason}
            for u in result.uncovered
        ],
        search_start=req.start,
        search_end=req.end,
        min_elevation_deg=req.min_elevation_deg,
        diagnostics=diagnostics,
    )


def _nan_none(x: float):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


@app.post("/relay-plan", response_model=RelayResponse)
def relay_plan(req: RelayRequest):
    orbit = TwoBodyOrbit(
        np.array(req.satellite.position_km, dtype=float),
        np.array(req.satellite.velocity_km_s, dtype=float),
    )

    # Same physical reality check as /passes: a perigee inside the Earth
    # makes the two-body propagation meaningless for every station.
    min_surface = EARTH_RADIUS_KM - 14.0  # account for deepest ocean trench
    if orbit.periapsis < min_surface:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "type": "orbit_impact_earth",
                        "loc": ["body", "satellite"],
                        "msg": (
                            f"orbit perigee radius {orbit.periapsis:.3f} km is "
                            f"below the Earth surface (~{EARTH_RADIUS_KM:.3f} km); "
                            "two-body propagation is not physically meaningful"
                        ),
                    }
                ]
            },
        )

    inputs = [
        StationInput(
            id=s.id,
            station=Station(
                lat_deg=s.latitude_deg,
                lon_deg=s.longitude_deg,
                alt_km=s.altitude_km,
            ),
            min_elevation_deg=s.min_elevation_deg,
        )
        for s in req.stations
    ]

    result = plan_relay(
        orbit=orbit,
        station_inputs=inputs,
        epoch=req.epoch,
        t_start=req.start,
        t_end=req.end,
    )

    span_s = (req.end - req.start).total_seconds()
    unknown_s = sum((u.end - u.start).total_seconds() for u in result.unknown)
    relay_s = result.plan.duration_s if result.plan else 0.0

    return RelayResponse(
        status=result.status,
        plan=(
            None
            if result.plan is None
            else {
                "start": result.plan.start,
                "end": result.plan.end,
                "duration_s": result.plan.duration_s,
                "switch_count": len(result.plan.switches),
                "segments": [
                    {
                        "station_id": seg.station_id,
                        "start": seg.start,
                        "end": seg.end,
                        "duration_s": (seg.end - seg.start).total_seconds(),
                    }
                    for seg in result.plan.segments
                ],
                "switches": [
                    {
                        "time": sw.time,
                        "from_station_id": sw.from_station_id,
                        "to_station_id": sw.to_station_id,
                    }
                    for sw in result.plan.switches
                ],
            }
        ),
        stations=[
            {
                "id": sr.id,
                "status": sr.status,
                "min_elevation_deg": sr.min_elevation_deg,
                "proven_windows": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "start_open": w.start_open,
                        "end_open": w.end_open,
                        "truncated": w.truncated,
                    }
                    for w in sr.windows
                ],
                "uncovered_intervals": [
                    {"start": u.start, "end": u.end, "reason": u.reason}
                    for u in sr.uncovered
                ],
            }
            for sr in result.stations
        ],
        unknown_intervals=[
            {"start": u.start, "end": u.end, "reason": u.reason}
            for u in result.unknown
        ],
        search_start=req.start,
        search_end=req.end,
        diagnostics={
            "search_span_s": span_s,
            "proven_visible_seconds": result.proven_visible_seconds,
            "proven_visible_fraction": (
                result.proven_visible_seconds / span_s if span_s > 0 else 1.0
            ),
            "unknown_seconds": unknown_s,
            "relay_covered_seconds": relay_s,
            "relay_coverage_fraction": relay_s / span_s if span_s > 0 else 1.0,
            "station_count": len(inputs),
        },
    )
