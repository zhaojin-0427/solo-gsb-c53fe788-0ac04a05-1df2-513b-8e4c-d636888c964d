"""UTC time helpers and sidereal time.

All API times are UTC. Internally we work in TT-like "seconds since epoch":
the propagated state is only used relative to the supplied epoch, so no
leap-second distinction is needed. GMST is computed from the Julian Date of
UT1 (we treat UTC as UT1, the standard practical approximation).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Julian Date of the Unix epoch (1970-01-01 00:00 UTC).
JD_UNIX_EPOCH = 2440587.5
SECONDS_PER_DAY = 86400.0

# J2000.0 epoch (2000-01-01 12:00 UTC), Julian Date and Unix seconds.
JD_J2000 = 2451545.0
UNIX_J2000 = (JD_J2000 - JD_UNIX_EPOCH) * SECONDS_PER_DAY
DAYS_PER_JULIAN_CENTURY = 36525.0


def to_utc(dt: datetime) -> datetime:
    """Normalize an aware datetime to UTC. Naive datetimes are rejected
    beforehand by the request model."""
    return dt.astimezone(timezone.utc)


def unix_seconds(dt: datetime) -> float:
    return dt.astimezone(timezone.utc).timestamp()


def julian_date(dt: datetime) -> float:
    return JD_UNIX_EPOCH + unix_seconds(dt) / SECONDS_PER_DAY


def iso_utc(dt: datetime) -> str:
    """Format a UTC datetime as ...Z with milliseconds."""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def gmst_rad(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time (radians, [0, 2pi)) at the given UTC time.

    IAU 1982 model expressed for an arbitrary UT1 instant. With ``d`` the
    integer(ish) UT1 day number from J2000 and ``f`` the fraction of day
    (Astronomical Almanac, Vallado):

      GMST = 67310.54841 + (876600h + 8640184.812866) T_u
             + 0.093104 T_u^2 - 6.2e-6 T_u^3   (seconds of sidereal time)

    where T_u is Julian centuries from J2000.0 measured at the *current*
    instant. This gives 18h41m50.543s at J2000 (2000-01-01T12:00Z).

    The day count from J2000 is formed directly in seconds. Forming the
    full Julian date first (``JD_UNIX_EPOCH + sec/86400`` ~ 2.46e6) would
    quantize the instant to ~40 us (double precision), and the station
    lever arm (r_station / range) amplifies each such GMST step to
    ~1e-6 deg of elevation noise on close passes -- enough to hide or
    invent grazing crossings.
    """
    d = (unix_seconds(dt) - UNIX_J2000) / SECONDS_PER_DAY
    tu = d / DAYS_PER_JULIAN_CENTURY
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * tu
        + 0.093104 * tu * tu
        - 6.2e-6 * tu * tu * tu
    )
    # Seconds of time -> radians (24 sidereal hours == 2*pi).
    rad = (gmst_sec / SECONDS_PER_DAY) * 2.0 * 3.141592653589793
    return rad % (2.0 * 3.141592653589793)
