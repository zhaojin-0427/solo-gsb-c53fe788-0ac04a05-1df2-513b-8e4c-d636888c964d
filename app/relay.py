"""Multi-station relay planning.

Given one satellite state and several ground stations (each with its own
minimum elevation), the same :func:`~app.passes.find_passes` machinery is
run per station. Only its *proven* outputs are used:

- every returned window is a **proven-visible** half-open interval
  ``[start, end)`` (``start_open`` / ``end_open`` mark clipping by the search
  boundaries, ``truncated`` a cut caused by a numerical failure);
- every entry in ``uncovered_intervals`` is a time span that is neither
  proven visible nor proven invisible.

A relay chain must keep at least one station provably visible at every
instant. Consecutive responsible stations differ, and a handover is only
legal when the two proven windows overlap for a **strictly positive**
duration (a shared endpoint is not an overlap). A station's own windows are
allowed to abut (their shared endpoint needs no handover).

Endpoint representation
-----------------------
UTC timestamps are reported at microsecond resolution, so all planning
arithmetic is done in **integer microseconds after the search start** and
every comparison is exact -- no epsilon tolerance anywhere:

- two endpoints are the same instant iff their microsecond counts are
  equal, so same-station windows join only when they truly abut or overlap
  (a 1 us hole is a real gap and breaks continuity);
- a cross-station overlap is positive iff it spans at least one full
  microsecond, so windows overlapping by exactly 1 us *can* hand over,
  while windows merely touching at an endpoint cannot;
- an uncovered/unknown span is kept iff it is at least 1 us wide, so a
  1 us unknown sliver is never swallowed into a ``complete`` status;
- plan durations are compared in whole microseconds: a 1 us difference
  already decides the longest plan, and only exactly equal durations fall
  through to the switch-count / start-time / station-ID tie-breaks.

The set of all proven windows is split into connected components under
those relations; each component is one continuously coverable span. Inside
a component the chain with the fewest switches is found on a small DAG
(window -> later-ending, positively-overlapping window of another station);
equal switch counts are adjudicated lexicographically by the station-ID
sequence, so the result is stable. Across components the longest span wins;
equal spans prefer fewer switches, then the earlier start, then station IDs.

Time that no station proves visible but some station reports as uncovered is
returned in ``unknown_intervals`` -- never as "definitely invisible" and
never as coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from .geometry import Station
from .mechanics import TwoBodyOrbit
from .passes import _merge, _subtract, find_passes

_US_PER_S = 1_000_000


def _to_us(dt: datetime, t_start: datetime) -> int:
    """Exact integer microseconds from ``t_start`` to ``dt``.

    ``timedelta`` stores days/seconds/microseconds as integers, so this is
    the lossless form of the microsecond-resolution UTC timestamps.
    """
    d = dt - t_start
    return (d.days * 86400 + d.seconds) * _US_PER_S + d.microseconds


def _from_us(us: int, t_start: datetime) -> datetime:
    return t_start + timedelta(microseconds=us)


# --------------------------------------------------------------- data types
@dataclass
class StationInput:
    id: str
    station: Station
    min_elevation_deg: float


@dataclass
class ProvenWindow:
    station_id: str
    start: datetime
    end: datetime
    start_open: bool
    end_open: bool
    truncated: bool


@dataclass
class StationUncovered:
    start: datetime
    end: datetime
    reason: str


@dataclass
class StationRelayResult:
    id: str
    status: str  # complete | partial | failed, straight from find_passes
    min_elevation_deg: float
    windows: list[ProvenWindow] = field(default_factory=list)
    uncovered: list[StationUncovered] = field(default_factory=list)


@dataclass
class RelaySegment:
    station_id: str
    start: datetime
    end: datetime


@dataclass
class RelaySwitch:
    time: datetime
    from_station_id: str
    to_station_id: str


@dataclass
class RelayPlan:
    start: datetime
    end: datetime
    segments: list[RelaySegment]
    switches: list[RelaySwitch]

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


@dataclass
class UnknownInterval:
    start: datetime
    end: datetime
    reason: str


@dataclass
class RelayResult:
    status: str  # complete | partial | failed
    plan: Optional[RelayPlan]
    stations: list[StationRelayResult]
    unknown: list[UnknownInterval]
    search_start: datetime
    search_end: datetime
    proven_visible_seconds: float


# ------------------------------------------------------------- window graph
@dataclass(eq=False)
class _Node:
    """A proven-visible span in integer microseconds after the search start.

    A node may cluster several abutting windows of one station (which need
    no handover between them); ``members`` are the original windows.
    """
    station_id: str
    s: int
    e: int
    start_open: bool
    end_open: bool
    truncated: bool
    members: list = field(default_factory=list)

    def __post_init__(self):
        if not self.members:
            self.members = [self]


def _cluster_station_windows(nodes: list[_Node]) -> list[_Node]:
    """Join windows of one station that overlap or abut into one continuous
    proven span: handing over to oneself at a shared endpoint is not a
    switch. The shared endpoint must be the *same* microsecond -- a 1 us
    hole is a real gap and keeps the spans separate. Windows of one station
    have disjoint interiors, so a simple sweep suffices."""
    out: list[_Node] = []
    by_station: dict[str, list[_Node]] = {}
    for n in nodes:
        by_station.setdefault(n.station_id, []).append(n)
    for sid, group in by_station.items():
        group.sort(key=lambda n: (n.s, n.e))
        cur: Optional[_Node] = None
        for n in group:
            if cur is not None and n.s <= cur.e:
                cur.e = max(cur.e, n.e)
                cur.end_open = cur.end_open or n.end_open
                cur.truncated = cur.truncated or n.truncated
                cur.members.extend(n.members)
            else:
                if cur is not None:
                    out.append(cur)
                cur = _Node(n.station_id, n.s, n.e, n.start_open,
                            n.end_open, n.truncated, list(n.members))
        if cur is not None:
            out.append(cur)
    return out


def _build_components(nodes: list[_Node]) -> list[list[_Node]]:
    """Group proven spans into continuously coverable components.

    A span joins a component when either (a) one of *another* station's
    spans in it overlaps with strictly positive duration (at least one
    microsecond), or (b) one of the *same* station's spans abuts/overlaps
    (already clustered, so this only covers identical/duplicate geometry).
    A span can bridge several existing components at once; they are then
    merged.
    """
    comps: list[dict] = []
    for idx in sorted(range(len(nodes)), key=lambda i: (nodes[i].s, nodes[i].e)):
        w = nodes[idx]
        hits = []
        for c in comps:
            same_station_end = c["stat_end"].get(w.station_id)
            if same_station_end is not None and same_station_end >= w.s:
                hits.append(c)  # abutting/overlapping span of same station
            elif c["end"] > w.s:
                # Union reaches beyond this span's start; every span so far
                # starts no later than this one (sweep order), so the
                # extending span starts before w ends and overlaps it for
                # a positive duration. It must be another station, since
                # one station's proven spans have disjoint interiors.
                hits.append(c)
        if not hits:
            comps.append({
                "end": w.e,
                "stat_end": {w.station_id: w.e},
                "nodes": [w],
            })
            continue
        head = hits[0]
        for c in hits[1:]:
            head["nodes"].extend(c["nodes"])
            head["end"] = max(head["end"], c["end"])
            for sid, end in c["stat_end"].items():
                prev = head["stat_end"].get(sid)
                head["stat_end"][sid] = end if prev is None else max(prev, end)
            comps.remove(c)
        head["nodes"].append(w)
        head["end"] = max(head["end"], w.e)
        prev = head["stat_end"].get(w.station_id)
        head["stat_end"][w.station_id] = w.e if prev is None else max(prev, w.e)
    return [c["nodes"] for c in comps]


@dataclass
class _Label:
    hops: int
    chain_ids: tuple
    prev: Optional[_Node]


def _best_chain(comp: list[_Node]) -> Optional[list[_Node]]:
    """Minimum-switch station chain covering the whole component.

    Edge u -> v exists iff the stations differ, v starts no earlier than u,
    v ends strictly later, and the windows overlap for a positive duration
    (at least one microsecond -- touching endpoints do not qualify).
    End strictly increases along an edge, so processing by end order is a
    topological sweep. Labels are (switch count, station-ID tuple); the
    tuple gives the deterministic tie-break.
    """
    cstart = min(w.s for w in comp)
    cend = max(w.e for w in comp)
    labels: dict[_Node, _Label] = {}

    def better(cand: _Label, old: Optional[_Label]) -> bool:
        return old is None or (cand.hops, cand.chain_ids) < (old.hops, old.chain_ids)

    # Every span starting at the component origin can be the first station
    # with zero switches. Seeds must exist before edge relaxation, otherwise
    # an incoming edge could pre-label an origin span with hops > 0.
    for w in comp:
        if w.s == cstart:
            labels[w] = _Label(0, (w.station_id,), None)

    for u in sorted(comp, key=lambda w: (w.e, w.s)):
        lu = labels.get(u)
        if lu is None:
            continue
        for v in comp:
            if v.station_id == u.station_id:
                continue
            if v.s < u.s:
                continue  # would hand over backwards in time
            if v.e <= u.e:
                continue  # must extend the reach
            overlap = min(u.e, v.e) - max(u.s, v.s)
            if overlap <= 0:
                continue  # endpoints touching is not a positive overlap
            cand = _Label(lu.hops + 1, lu.chain_ids + (v.station_id,), u)
            if better(cand, labels.get(v)):
                labels[v] = cand

    terminals = [w for w in comp if w.e == cend]
    reachable = [w for w in terminals if w in labels]
    if not reachable:
        return None
    last = min(reachable, key=lambda w: (labels[w].hops, labels[w].chain_ids))

    chain: list[_Node] = []
    cur: Optional[_Node] = last
    while cur is not None:
        chain.append(cur)
        cur = labels[cur].prev
    chain.reverse()

    # Handover instants are the rise of the incoming window; they must be
    # strictly ordered (equal starts would mean a zero-length segment,
    # which the minimum-hop optimum never contains).
    for a, b in zip(chain, chain[1:]):
        if b.s <= a.s:
            return None
    return chain


def _chain_to_plan(chain: list[_Node], t_start: datetime) -> RelayPlan:
    cstart = min(w.s for w in chain)  # == chain[0].s
    cend = max(w.e for w in chain)
    boundaries = [cstart] + [w.s for w in chain[1:]] + [cend]
    segments = [
        RelaySegment(
            station_id=w.station_id,
            start=_from_us(boundaries[i], t_start),
            end=_from_us(boundaries[i + 1], t_start),
        )
        for i, w in enumerate(chain)
    ]
    switches = [
        RelaySwitch(
            time=_from_us(b.s, t_start),
            from_station_id=a.station_id,
            to_station_id=b.station_id,
        )
        for a, b in zip(chain, chain[1:])
    ]
    return RelayPlan(
        start=_from_us(cstart, t_start),
        end=_from_us(cend, t_start),
        segments=segments,
        switches=switches,
    )


# ------------------------------------------------------------- unknown time
def _merge_uncovered(raw: list[tuple[int, int, str]]):
    """Merge (start, end, 'station: reason') triples that overlap or abut
    (shared endpoint microsecond); every distinct contributor reason is
    kept on the merged interval. A 1 us hole stays a hole."""
    if not raw:
        return []
    raw = sorted(raw, key=lambda x: (x[0], x[1]))
    out: list[list] = [[raw[0][0], raw[0][1], [raw[0][2]]]]
    for a, b, why in raw[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
            if why not in out[-1][2]:
                out[-1][2].append(why)
        else:
            out.append([a, b, [why]])
    return [(a, b, "; ".join(reasons)) for a, b, reasons in out]


# ------------------------------------------------------------------- public
def plan_relay(
    orbit: TwoBodyOrbit,
    station_inputs: list[StationInput],
    epoch: datetime,
    t_start: datetime,
    t_end: datetime,
) -> RelayResult:
    epoch = epoch.astimezone(timezone.utc)
    t_start = t_start.astimezone(timezone.utc)
    t_end = t_end.astimezone(timezone.utc)
    span_us = _to_us(t_end, t_start)

    station_results: list[StationRelayResult] = []
    nodes: list[_Node] = []
    raw_uncovered: list[tuple[int, int, str]] = []

    for sin in station_inputs:
        res = find_passes(
            orbit=orbit,
            station=sin.station,
            epoch=epoch,
            t_start=t_start,
            t_end=t_end,
            min_elevation_deg=sin.min_elevation_deg,
        )
        wins: list[ProvenWindow] = []
        for w in res.windows:
            wins.append(ProvenWindow(
                station_id=sin.id,
                start=w.start,
                end=w.end,
                start_open=w.start_open,
                end_open=w.end_open,
                truncated=w.truncated,
            ))
            s = max(0, _to_us(w.start, t_start))
            e = min(span_us, _to_us(w.end, t_start))
            if e <= s:
                continue  # empty half-open span: proves nothing
            nodes.append(_Node(
                station_id=sin.id,
                s=s,
                e=e,
                start_open=w.start_open,
                end_open=w.end_open,
                truncated=w.truncated,
            ))
        uncovered: list[StationUncovered] = []
        for u in res.uncovered:
            a = max(0, _to_us(u.start, t_start))
            b = min(span_us, _to_us(u.end, t_start))
            if b <= a:
                continue  # nothing representable at microsecond resolution
            uncovered.append(StationUncovered(
                start=_from_us(a, t_start),
                end=_from_us(b, t_start),
                reason=u.reason,
            ))
            raw_uncovered.append((a, b, f"station '{sin.id}': {u.reason}"))
        station_results.append(StationRelayResult(
            id=sin.id, status=res.status,
            min_elevation_deg=sin.min_elevation_deg,
            windows=wins, uncovered=uncovered))

    # ---- longest continuously coverable span, with stable tie-breaks ----
    clustered = _cluster_station_windows(nodes)
    best_chain: Optional[list[_Node]] = None
    best_key = None
    for comp in _build_components(clustered):
        chain = _best_chain(comp)
        if chain is None:
            continue
        cstart = min(w.s for w in chain)
        cend = max(w.e for w in chain)
        duration = cend - cstart
        ids = tuple(w.station_id for w in chain)
        # Longest duration (exact microseconds); then fewest switches,
        # earliest start, lexicographically smallest ID sequence.
        if best_chain is None or _plan_better(
                duration, len(chain) - 1, cstart, ids,
                best_key[0], best_key[1], best_key[2], best_key[3]):
            best_chain = chain
            best_key = (duration, len(chain) - 1, cstart, ids)
    plan = _chain_to_plan(best_chain, t_start) if best_chain else None

    # ---- unknown intervals: uncovered by everyone, proven by no one ------
    visible = _merge([(n.s, n.e) for n in nodes])
    proven_visible_s = sum(b - a for a, b in visible) / _US_PER_S
    merged_unknown = _merge_uncovered(raw_uncovered)
    unknown_pieces = _subtract(
        [(a, b) for a, b, _ in merged_unknown], visible)
    reason_at = {}
    for a, b, why in merged_unknown:
        reason_at[(a, b)] = why
    unknown = []
    for a, b in unknown_pieces:
        if b <= a:
            continue  # a 1 us sliver is kept; only empty pieces are dropped
        why = "; ".join(
            w for (x, y), w in sorted(reason_at.items())
            if y > a and x < b
        )
        unknown.append(UnknownInterval(
            start=_from_us(a, t_start),
            end=_from_us(b, t_start),
            reason=why or "unresolved by all stations",
        ))

    # ---- honest aggregate status ----------------------------------------
    if not nodes:
        # Nothing proven visible anywhere: "complete" only when the whole
        # span is proven invisible; any unprovable gap makes it "failed".
        status = "failed" if unknown else "complete"
    else:
        status = "partial" if unknown else "complete"

    return RelayResult(
        status=status,
        plan=plan,
        stations=station_results,
        unknown=unknown,
        search_start=t_start,
        search_end=t_end,
        proven_visible_seconds=proven_visible_s,
    )


def _plan_better(dur, switches, start, ids,
                 b_dur, b_switches, b_start, b_ids) -> bool:
    # All quantities are exact (integer microseconds / counts), so a 1 us
    # duration difference already decides; only true ties fall through.
    if dur != b_dur:
        return dur > b_dur
    if switches != b_switches:
        return switches < b_switches
    if start != b_start:
        return start < b_start
    return ids < b_ids
