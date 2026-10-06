"""HTTP-level tests for POST /relay-plan (and /passes regression)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from conftest import EPOCH, circular_leo_state

client = TestClient(app)

ISO = lambda dt: dt.isoformat().replace("+00:00", "Z")


def relay_body(stations, **overrides):
    epoch, pos, vel = circular_leo_state(target_offset_s=300.0)
    body = {
        "epoch": ISO(epoch),
        "start": ISO(epoch),
        "end": ISO(epoch + timedelta(hours=2)),
        "satellite": {"position_km": pos, "velocity_km_s": vel},
        "stations": stations,
    }
    body.update(overrides)
    return body


def station(sid, lat, lon, alt=0.05, el=10.0):
    return {
        "id": sid,
        "latitude_deg": lat,
        "longitude_deg": lon,
        "altitude_km": alt,
        "min_elevation_deg": el,
    }


# Chain geometry around the first pass (surveyed from find_passes):
#   (40,110) visible ~38..440 s,  BJ(39.9,116.4) ~100..509 s,
#   (40,125) ~194..572 s,         (40,135) ~350..584 s,
#   (25,128) ~390..776 s.
W2 = dict(lat=40.0, lon=110.0)
BJ = dict(lat=39.9, lon=116.4)
EAST = dict(lat=40.0, lon=125.0)
EAST2 = dict(lat=40.0, lon=135.0)
SE = dict(lat=25.0, lon=128.0)


# ------------------------------------------------------------------- happy
def test_health_still_ok():
    assert client.get("/health").status_code == 200


def test_relay_single_station_matches_passes_semantics():
    body = relay_body([station("BJ", **BJ)],
                      end=ISO(EPOCH + timedelta(hours=1)))
    r = client.post("/relay-plan", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "complete"
    p = d["plan"]
    assert p["switch_count"] == 0
    assert [s["station_id"] for s in p["segments"]] == ["BJ"]
    # exactly one station block, status complete, windows sorted
    assert [s["id"] for s in d["stations"]] == ["BJ"]
    assert d["stations"][0]["status"] == "complete"
    wins = d["stations"][0]["proven_windows"]
    assert wins == sorted(wins, key=lambda w: w["start"])
    # plan span equals that station's only proven window
    assert p["start"] == wins[0]["start"]
    assert p["end"] == wins[0]["end"]
    assert d["unknown_intervals"] == []
    diag = d["diagnostics"]
    assert diag["station_count"] == 1
    assert abs(diag["relay_covered_seconds"] - p["duration_s"]) < 1e-9


def test_relay_handover_chain_switch_in_overlap():
    body = relay_body([
        station("W2", **W2),
        station("BJ", **BJ),
        station("E", **EAST),
        station("SE", **SE),
    ])
    d = client.post("/relay-plan", json=body).json()
    p = d["plan"]
    ids = [s["station_id"] for s in p["segments"]]
    assert ids == ["W2", "SE"]  # minimum-switch chain
    assert p["switch_count"] == 1
    sw = p["switches"][0]
    assert (sw["from_station_id"], sw["to_station_id"]) == ("W2", "SE")
    # every switch lies inside a strictly-positive overlap of a proven
    # window from each side
    def windows_of(sid):
        block = next(s for s in d["stations"] if s["id"] == sid)
        return block["proven_windows"]
    t = sw["time"]
    a = next(w for w in windows_of("W2") if w["start"] <= t < w["end"])
    b = next(w for w in windows_of("SE") if w["start"] <= t < w["end"])
    # switch inside/out at the overlap boundary; overlap has positive length
    assert a["start"] <= t < a["end"] and b["start"] <= t < b["end"]
    assert min(a["end"], b["end"]) > max(a["start"], b["start"])
    # segments partition the plan span and are time-contiguous
    assert p["segments"][0]["start"] == p["start"]
    assert p["segments"][-1]["end"] == p["end"]
    for x, y in zip(p["segments"], p["segments"][1:]):
        assert x["end"] == y["start"]
    # the relay lasts longer than W2 covers on its own
    single = client.post("/relay-plan", json=relay_body([station("W2", **W2)]))
    span_single = single.json()["plan"]["duration_s"]
    assert p["duration_s"] > span_single


def test_relay_picks_longest_chain_with_fewer_switches():
    """W2 alone ends ~440 s; the chain W2->...->E2 reaches ~584 s. The
    longest possible continuous chain must win even if more switches."""
    body = relay_body([
        station("W2", **W2),
        station("BJ", **BJ),
        station("E", **EAST),
        station("E2", **EAST2),
    ])
    p = client.post("/relay-plan", json=body).json()["plan"]
    assert p["end"][11:19] == "00:09:44"  # E2's set (~584 s)
    # chain reaches beyond anything a single station covers on its own
    d = client.post("/relay-plan", json=relay_body([station("BJ", **BJ)])).json()
    assert p["duration_s"] > d["plan"]["duration_s"]


def test_plan_endpoint_includes_station_level_fields():
    body = relay_body([station("BJ", **BJ)])
    d = client.post("/relay-plan", json=body).json()
    block = d["stations"][0]
    assert set(block) == {
        "id", "status", "min_elevation_deg",
        "proven_windows", "uncovered_intervals",
    }
    w = block["proven_windows"][0]
    assert set(w) == {"start", "end", "start_open", "end_open", "truncated"}
    assert block["min_elevation_deg"] == 10.0


# ----------------------------------------------------------- empty/unknown
def test_no_windows_anywhere_returns_empty_plan_complete():
    # An 89 deg minimum elevation: provably no visibility for any station.
    body = relay_body([
        station("A", 39.9, 116.4, el=89.0),
        station("B", 40.0, 125.0, el=89.0),
    ])
    r = client.post("/relay-plan", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "complete"
    assert d["plan"] is None
    assert d["unknown_intervals"] == []
    for s in d["stations"]:
        assert s["proven_windows"] == []
        assert s["status"] == "complete"
    assert d["diagnostics"]["relay_covered_seconds"] == 0.0


def test_grazing_unresolved_reported_as_unknown_not_invisible():
    """Threshold a hair above a station's peak: no proven window, but the
    grazing contact is unresolvable -> unknown interval, status failed,
    never 'definitely invisible'."""
    # Find the first BJ peak via /passes, then bracket it.
    base = relay_body([station("BJ", **BJ)])
    p0 = client.post("/passes", json={
        "epoch": base["epoch"], "start": base["start"],
        "end": base["end"], "min_elevation_deg": 10.0,
        "satellite": base["satellite"],
        "station": {"latitude_deg": 39.9, "longitude_deg": 116.4,
                    "altitude_km": 0.05},
    }).json()
    w0 = p0["windows"][0]
    peak = w0["peak_elevation_deg"]
    tp = datetime.fromisoformat(w0["peak_time"].replace("Z", "+00:00"))
    body = relay_body(
        [station("BJ", **BJ, el=peak + 1e-7)],
        start=ISO(tp - timedelta(seconds=30)),
        end=ISO(tp + timedelta(seconds=30)),
    )
    d = client.post("/relay-plan", json=body).json()
    assert d["plan"] is None
    assert d["status"] == "failed"
    assert len(d["unknown_intervals"]) >= 1
    u = d["unknown_intervals"][0]
    us, ue = (datetime.fromisoformat(u[k].replace("Z", "+00:00"))
              for k in ("start", "end"))
    assert us < ue and us <= tp <= ue
    assert "BJ" in u["reason"]


def test_one_station_uncovered_another_visible_keeps_relay():
    """Station X is given an unresolvable grazing threshold at the same
    time BJ proves visible: the relay still covers; the gap is not unknown
    because BJ proves it."""
    p0 = client.post("/passes", json={
        "epoch": ISO(EPOCH), "start": ISO(EPOCH),
        "end": ISO(EPOCH + timedelta(hours=2)),
        "min_elevation_deg": 10.0,
        "satellite": relay_body([])["satellite"],
        "station": {"latitude_deg": 39.9, "longitude_deg": 116.4,
                    "altitude_km": 0.05},
    }).json()
    w0 = p0["windows"][0]
    peak = w0["peak_elevation_deg"]
    tp = datetime.fromisoformat(w0["peak_time"].replace("Z", "+00:00"))
    body = relay_body(
        [station("BJ", **BJ),
         station("X", **BJ, el=peak + 1e-7)],
        start=ISO(tp - timedelta(seconds=30)),
        end=ISO(tp + timedelta(seconds=30)),
    )
    d = client.post("/relay-plan", json=body).json()
    assert d["status"] == "complete"
    assert d["plan"] is not None
    assert d["unknown_intervals"] == []
    assert [s["station_id"] for s in d["plan"]["segments"]] == ["BJ"]


# ------------------------------------------------------------------- 422
def test_duplicate_station_ids_422():
    body = relay_body([
        station("DUP", **BJ),
        station("DUP", **EAST),
        station("OK", **SE),
    ])
    r = client.post("/relay-plan", json=body)
    assert r.status_code == 422
    msg = str(r.json()["detail"])
    assert "unique" in msg.lower()


def test_empty_station_list_422():
    body = relay_body([])
    assert client.post("/relay-plan", json=body).status_code == 422


def test_bad_orbit_422():
    body = relay_body([station("BJ", **BJ)])
    body["satellite"]["velocity_km_s"] = [
        x * 1000.0 for x in body["satellite"]["velocity_km_s"]
    ]
    r = client.post("/relay-plan", json=body)
    assert r.status_code == 422
    assert r.json()["detail"][0]["type"] in (
        "orbit_degenerate_or_invalid",)


def test_perigee_inside_earth_422():
    body = relay_body([station("BJ", **BJ)])
    body["satellite"]["position_km"] = [
        x * 1000.0 for x in body["satellite"]["position_km"]
    ]
    assert client.post("/relay-plan", json=body).status_code == 422


def test_search_outside_epoch_range_422():
    body = relay_body(
        [station("BJ", **BJ)],
        start=ISO(EPOCH + timedelta(days=4000)),
        end=ISO(EPOCH + timedelta(days=4000, hours=1)),
    )
    assert client.post("/relay-plan", json=body).status_code == 422


def test_span_over_seven_days_422():
    body = relay_body(
        [station("BJ", **BJ)],
        end=ISO(EPOCH + timedelta(days=7, seconds=1)),
    )
    assert client.post("/relay-plan", json=body).status_code == 422


def test_time_order_422():
    body = relay_body(
        [station("BJ", **BJ)],
        start=ISO(EPOCH + timedelta(hours=1)),
        end=ISO(EPOCH),
    )
    assert client.post("/relay-plan", json=body).status_code == 422


def test_naive_datetime_422():
    body = relay_body([station("BJ", **BJ)], epoch="2025-06-01T00:00:00")
    assert client.post("/relay-plan", json=body).status_code == 422


def test_bad_station_range_422():
    body = relay_body([station("BJ", 91.0, 116.4)])
    assert client.post("/relay-plan", json=body).status_code == 422


def test_bad_per_station_elevation_422():
    body = relay_body([station("BJ", **BJ, el=90.0)])
    assert client.post("/relay-plan", json=body).status_code == 422


def test_extra_field_422():
    body = relay_body([station("BJ", **BJ)])
    body["stations"][0]["unknown"] = 1
    assert client.post("/relay-plan", json=body).status_code == 422


def test_unknown_toplevel_field_422():
    body = relay_body([station("BJ", **BJ)])
    body["nope"] = 1
    assert client.post("/relay-plan", json=body).status_code == 422


# ------------------------------------------------- /passes stays unchanged
def test_passes_response_shape_unchanged():
    body = relay_body([])
    r = client.post("/passes", json={
        "epoch": body["epoch"], "start": body["start"], "end": body["end"],
        "min_elevation_deg": 10.0,
        "satellite": body["satellite"],
        "station": {"latitude_deg": 39.9, "longitude_deg": 116.4,
                    "altitude_km": 0.05},
    })
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {
        "status", "count", "windows", "uncovered_intervals",
        "search_start", "search_end", "min_elevation_deg", "diagnostics",
    }
    assert set(d["windows"][0]) == {
        "start", "end", "peak_time", "peak_elevation_deg",
        "peak_azimuth_deg", "rise_azimuth_deg", "set_azimuth_deg",
        "max_range_km", "duration_s", "start_open", "end_open", "truncated",
    }
