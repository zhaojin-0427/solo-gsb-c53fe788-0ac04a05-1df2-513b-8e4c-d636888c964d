"""Unit tests for the relay-chain optimizer (synthetic proven windows)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.relay import (
    RelayResult,
    StationInput,
    StationRelayResult,
    StationUncovered,
    UnknownInterval,
    _Node,
    _best_chain,
    _build_components,
    _chain_to_plan,
    _cluster_station_windows,
    plan_relay,
)

T0 = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


def node(sid, s, e, start_open=False, end_open=False, truncated=False):
    return _Node(sid, float(s), float(e), start_open, end_open, truncated)


def chain_ids(chain):
    return [n.station_id for n in chain]


# ------------------------------------------------------------ components
def test_components_merge_on_positive_overlap_other_station():
    nodes = [node("A", 0, 10), node("B", 9, 20), node("C", 19, 30)]
    comps = _build_components(nodes)
    assert len(comps) == 1
    chain = _best_chain(comps[0])
    assert chain_ids(chain) == ["A", "B", "C"]


def test_components_split_on_endpoint_touch_different_stations():
    nodes = [node("A", 0, 10), node("B", 10, 20)]
    comps = _build_components(nodes)
    assert len(comps) == 2


def test_components_join_on_endpoint_touch_same_station():
    nodes = [node("A", 0, 10), node("A", 10, 20)]
    # same-station abutting windows are clustered into one proven span,
    # which of course forms a single component
    clustered = _cluster_station_windows(nodes)
    comps = _build_components(clustered)
    assert len(comps) == 1
    assert comps[0][0].s == 0.0 and comps[0][0].e == 20.0


def test_clustered_same_station_span_needs_no_switch():
    nodes = _cluster_station_windows([node("A", 0, 10), node("A", 10, 25)])
    chain = _best_chain(_build_components(nodes)[0])
    assert chain_ids(chain) == ["A"]
    assert chain[0].s == 0.0 and chain[0].e == 25.0


def test_components_bridge_merges_several_components():
    # B's window bridges A's and C's previously separate components.
    nodes = [node("A", 0, 5), node("C", 15, 20), node("B", 4, 16)]
    comps = _build_components(nodes)
    assert len(comps) == 1


# --------------------------------------------------------- min switches
def test_min_switches_prefers_long_window_over_chain():
    # A starts at the component origin and reaches the long window; the
    # minimum-switch route is A->B (1 switch), not A->C->D (2).
    nodes = [
        node("A", 0, 8),
        node("B", 1, 30),
        node("C", 7, 20),
        node("D", 19, 30),
    ]
    chain = _best_chain(_build_components(nodes)[0])
    assert chain_ids(chain) == ["A", "B"]


def test_station_spanning_component_alone_wins():
    # One station genuinely spans the whole component: zero switches beats
    # every multi-station chain.
    nodes = [
        node("B", 0, 30),
        node("A", 0, 8),
        node("C", 7, 20),
        node("D", 19, 30),
    ]
    chain = _best_chain(_build_components(nodes)[0])
    assert chain_ids(chain) == ["B"]


def test_lexicographic_tie_break_equal_switches():
    # A and Z both span the whole component: same switch count (0), the
    # lexicographically smaller station ID wins.
    nodes = [node("Z", 0, 30), node("A", 0, 30)]
    chain = _best_chain(_build_components(nodes)[0])
    assert chain_ids(chain) == ["A"]


def test_lexicographic_tie_break_two_station_chain():
    nodes = [
        node("A", 0, 15),
        node("B", 0, 15),
        node("C", 14, 30),
        node("D", 14, 30),
    ]
    chain = _best_chain(_build_components(nodes)[0])
    assert chain_ids(chain) == ["A", "C"]


def test_no_backwards_handover():
    # B starts inside A but ends before A: using it cannot extend, so the
    # covering chain stays A->C.
    nodes = [node("A", 0, 20), node("B", 5, 10), node("C", 19, 30)]
    chain = _best_chain(_build_components(nodes)[0])
    assert chain_ids(chain) == ["A", "C"]


def test_same_station_abut_needs_no_switch():
    nodes = _cluster_station_windows([
        node("A", 0, 10), node("A", 10, 25), node("B", 24, 30),
    ])
    chain = _best_chain(_build_components(nodes)[0])
    assert chain_ids(chain) == ["A", "B"]


def test_segments_partition_component_and_switches_at_rise():
    nodes = [node("A", 0, 12), node("B", 10, 25), node("C", 24, 30)]
    chain = _best_chain(_build_components(nodes)[0])
    plan = _chain_to_plan(chain, T0)
    assert len(plan.segments) == 3
    assert plan.segments[0].start == T0
    assert plan.segments[-1].end == T0 + timedelta(seconds=30)
    # segments partition without gaps/overlaps
    for a, b in zip(plan.segments, plan.segments[1:]):
        assert a.end == b.start
    # switch at the incoming window's rise
    assert plan.switches[0].time == T0 + timedelta(seconds=10)
    assert plan.switches[0].from_station_id == "A"
    assert plan.switches[0].to_station_id == "B"


def test_longest_component_wins_even_if_second():
    from app.relay import _build_components, _best_chain, _chain_to_plan

    short = [node("A", 0, 5), node("B", 4, 11)]  # length 11
    longc = [node("C", 100, 120), node("D", 119, 140)]  # length 40
    plans = [_chain_to_plan(_best_chain(c), T0) for c in (short, longc)]
    best = max(plans, key=lambda p: p.duration_s)
    assert best.start == T0 + timedelta(seconds=100)
    assert best.switches[0].to_station_id == "D"


# -------------------------------------------------- plan_relay integration
class _FakeOrbit:
    pass


class _TaggedStation:
    def __init__(self, tag):
        self.tag = tag


def _make_result(sid, windows, uncovered=(), status=None):
    from app.relay import ProvenWindow
    wins = [
        ProvenWindow(sid, T0 + timedelta(seconds=a), T0 + timedelta(seconds=b),
                     False, False, False)
        for a, b in windows
    ]
    uncov = [
        StationUncovered(T0 + timedelta(seconds=a),
                         T0 + timedelta(seconds=b), why)
        for a, b, why in uncovered
    ]
    if status is None:
        status = "complete" if not uncov else (
            "partial" if wins else "failed")
    return StationRelayResult(sid, status, 10.0, wins, uncov)


def test_plan_relay_uses_find_passes_outputs(monkeypatch):
    import app.relay as relay

    def fake_find_passes(orbit, station, epoch, t_start, t_end,
                         min_elevation_deg):
        table = {
            "A": ([(0.0, 12.0)], []),
            "B": ([(10.0, 25.0)], []),
            "C": ([(24.0, 30.0)], []),
        }
        wins, unc = table[station.tag]
        return _make_result(station.tag, wins, unc)

    monkeypatch.setattr(relay, "find_passes", fake_find_passes)

    def station(tag):
        return StationInput(tag, _TaggedStation(tag), 10.0)

    res = plan_relay(
        orbit=_FakeOrbit(),
        station_inputs=[station("A"), station("B"), station("C")],
        epoch=T0,
        t_start=T0,
        t_end=T0 + timedelta(seconds=60),
    )
    assert isinstance(res, RelayResult)
    assert [g.station_id for g in res.plan.segments] == ["A", "B", "C"]
    assert res.status == "complete"
    assert res.unknown == []
    assert res.plan.duration_s == 30.0


def test_unknown_interval_when_nobody_proven_but_some_uncovered(monkeypatch):
    import app.relay as relay

    def fake_find_passes(orbit, station, epoch, t_start, t_end,
                         min_elevation_deg):
        table = {
            "A": ([(0.0, 10.0)], []),
            # B cannot decide on [40, 50); nobody proves visibility there.
            "B": ([], [(40.0, 50.0, "grazing unresolved")]),
        }
        wins, unc = table[station.tag]
        return _make_result(station.tag, wins, unc)

    monkeypatch.setattr(relay, "find_passes", fake_find_passes)

    def station(tag):
        return StationInput(tag, _TaggedStation(tag), 10.0)

    res = plan_relay(
        orbit=_FakeOrbit(),
        station_inputs=[station("A"), station("B")],
        epoch=T0,
        t_start=T0,
        t_end=T0 + timedelta(seconds=60),
    )
    assert res.status == "partial"
    assert len(res.unknown) == 1
    u = res.unknown[0]
    assert u.start == T0 + timedelta(seconds=40)
    assert u.end == T0 + timedelta(seconds=50)
    assert "B" in u.reason


def test_uncovered_inside_anothers_window_is_not_unknown(monkeypatch):
    """One station uncovered at a time another station proves visible:
    the relay can continue, no unknown interval is emitted."""
    import app.relay as relay

    def fake_find_passes(orbit, station, epoch, t_start, t_end,
                         min_elevation_deg):
        table = {
            "A": ([(0.0, 60.0)], [(20.0, 25.0, "eval failed")]),
            "B": ([], [(20.0, 25.0, "grazing unresolved")]),
        }
        wins, unc = table[station.tag]
        return _make_result(station.tag, wins, unc)

    monkeypatch.setattr(relay, "find_passes", fake_find_passes)

    def station(tag):
        return StationInput(tag, _TaggedStation(tag), 10.0)

    res = plan_relay(
        orbit=_FakeOrbit(),
        station_inputs=[station("A"), station("B")],
        epoch=T0,
        t_start=T0,
        t_end=T0 + timedelta(seconds=60),
    )
    assert res.status == "complete"
    assert res.unknown == []
    assert [g.station_id for g in res.plan.segments] == ["A"]


def test_empty_plan_when_nobody_visible_and_all_proven(monkeypatch):
    import app.relay as relay

    def fake_find_passes(orbit, station, epoch, t_start, t_end,
                         min_elevation_deg):
        return _make_result(station.tag, [], [])

    monkeypatch.setattr(relay, "find_passes", fake_find_passes)

    def station(tag):
        return StationInput(tag, _TaggedStation(tag), 10.0)

    res = plan_relay(
        orbit=_FakeOrbit(),
        station_inputs=[station("A")],
        epoch=T0,
        t_start=T0,
        t_end=T0 + timedelta(seconds=60),
    )
    assert res.plan is None
    assert res.status == "complete"  # proven invisible, not "failed"
    assert res.unknown == []


def test_no_claim_when_nobody_visible_and_uncovered(monkeypatch):
    import app.relay as relay

    def fake_find_passes(orbit, station, epoch, t_start, t_end,
                         min_elevation_deg):
        return _make_result(
            station.tag, [], [(30.0, 35.0, "grazing unresolved")])

    monkeypatch.setattr(relay, "find_passes", fake_find_passes)

    def station(tag):
        return StationInput(tag, _TaggedStation(tag), 10.0)

    res = plan_relay(
        orbit=_FakeOrbit(),
        station_inputs=[station("A")],
        epoch=T0,
        t_start=T0,
        t_end=T0 + timedelta(seconds=60),
    )
    assert res.plan is None
    assert res.status == "failed"
    assert len(res.unknown) == 1
