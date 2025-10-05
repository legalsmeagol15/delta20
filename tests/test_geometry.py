# tests/test_geometry.py

import math
import pytest

from delta20.geometry import (
    get_dot_product,
    get_cross_product,
    get_normalized,
    get_lat_long,
    get_vector_length,
    get_vector,
    get_shortest_arc,
)

EPS = 1e-12
ANGLE_EPS = 1e-10  # radians


# --- Helpers (test-local) ---


def almost(a, b, eps=EPS):
    return abs(a - b) <= eps


def angdiff(a, b):
    """Smallest signed difference a-b wrapped to (-pi, pi]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


# --- get_dot_product ---

def test_dot_product_basic_identities():
    x = (1.0, 0.0, 0.0)
    y = (0.0, 1.0, 0.0)
    z = (0.0, 0.0, 1.0)
    assert get_dot_product(x, x) == 1.0
    assert get_dot_product(x, y) == 0.0
    assert get_dot_product(y, z) == 0.0
    assert get_dot_product(x, z) == 0.0
    assert get_dot_product(x, y) == get_dot_product(y, x)


# --- get_cross_product ---

def test_cross_product_right_handed_and_normalized():
    x = (1.0, 0.0, 0.0)
    y = (0.0, 1.0, 0.0)
    z = (0.0, 0.0, 1.0)
    c_norm = get_cross_product(x, y, normalize=True)

    # The c vector should be normal to the x and y
    assert almost(get_dot_product(x, c_norm), 0.0)
    assert almost(get_dot_product(y, c_norm), 0.0)

    # The c vector should essentially be z
    assert almost(get_dot_product(z, c_norm), 1.0)


# --- get_normalized ---
def test_get_normalized_returns_unit_vector():
    v = (2.0, -3.0, 6.0)
    n = get_normalized(*v)
    length = math.sqrt(get_dot_product(n, n))
    assert almost(length, 1.0)


def test_get_normalized_raises_on_zero_vector():
    with pytest.raises(ZeroDivisionError):
        get_normalized(0.0, 0.0, 0.0)


# --- get_lat_long ---

def test_cardinal_directions_equator_and_poles():
    # +x (equator, lon=0)
    lat, lon = get_lat_long(1.0, 0.0, 0.0)
    assert almost(lat, 0.0)
    assert almost(angdiff(lon, 0.0), 0.0)

    # +z (equator, lon=+pi/2 = 90E)
    lat, lon = get_lat_long(0.0, 0.0, 1.0)
    assert almost(lat, 0.0)
    assert almost(angdiff(lon, math.pi / 2), 0.0)

    # -x (equator, lon=pi or -pi → both acceptable)
    lat, lon = get_lat_long(-1.0, 0.0, 0.0)
    assert almost(lat, 0.0)
    assert min(abs(angdiff(lon, math.pi)), abs(
        angdiff(lon, -math.pi))) <= ANGLE_EPS

    # -z (equator, lon=-pi/2 = 90W)
    lat, lon = get_lat_long(0.0, 0.0, -1.0)
    assert almost(lat, 0.0)
    assert almost(angdiff(lon, -math.pi / 2), 0.0)

    # North pole (+y): lon arbitrary but finite
    lat, lon = get_lat_long(0.0, 1.0, 0.0)
    assert almost(lat, math.pi / 2)
    assert math.isfinite(lon)

    # South pole (-y): lon arbitrary but finite
    lat, lon = get_lat_long(0.0, -1.0, 0.0)
    assert almost(lat, -math.pi / 2)
    assert math.isfinite(lon)


def test_longitude_wrap_equivalence_at_idl():
    # Vector at -x should yield lon near ±pi; both wraps are equivalent
    lat1, lon1 = get_lat_long(-1.0, 0.0, 0.0)
    # lon near 0; used only for diff function
    lat2, lon2 = get_lat_long(1.0, 0.0, 0.0)
    # Just ensure lon1 is effectively ±pi
    assert min(abs(angdiff(lon1, math.pi)), abs(
        angdiff(lon1, -math.pi))) <= ANGLE_EPS
    assert almost(lat1, 0.0)
    assert almost(lat2, 0.0)  # sanity


def test_normalizes_non_unit_inputs():
    # 10x a unit direction should produce identical (lat, lon)
    vx, vy, vz = get_vector(math.radians(20.0), math.radians(30.0))
    lat1, lon1 = get_lat_long(vx, vy, vz)
    lat2, lon2 = get_lat_long(10 * vx, 10 * vy, 10 * vz)
    assert almost(lat1, lat2)
    assert almost(angdiff(lon1, lon2), 0.0)


def test_round_trip_with_get_vec_generic_angles():
    cases = [
        (math.radians(37.5), math.radians(-122.25)),
        (math.radians(-10.0), math.radians(179.9)),
        (math.radians(45.0), math.radians(89.999)),
        (math.radians(-60.0), math.radians(-179.5)),
    ]
    for lat, lon in cases:
        x, y, z = get_vector(lat, lon)
        lat2, lon2 = get_lat_long(x, y, z)
        assert abs(lat2 - lat) <= ANGLE_EPS
        assert abs(angdiff(lon2, lon)) <= ANGLE_EPS


def test_near_poles_stability_and_bounds():
    # Very close to North pole
    lat = math.radians(89.999999)
    lon = math.radians(23.0)
    x, y, z = get_vector(lat, lon)
    lat2, lon2 = get_lat_long(x, y, z)
    assert abs(lat2 - lat) < 1e-8
    assert math.isfinite(lon2)
    assert -math.pi <= lon2 <= math.pi
    assert y > 0.999999999 - EPS  # strong north component

    # Very close to South pole
    lat = math.radians(-89.999999)
    lon = math.radians(-130.0)
    x, y, z = get_vector(lat, lon)
    lat2, lon2 = get_lat_long(x, y, z)
    assert abs(lat2 - lat) < 1e-8
    assert math.isfinite(lon2)
    assert -math.pi <= lon2 <= math.pi
    assert y < -0.999999999 + EPS  # strong south component


def test_zero_vector_raises():
    with pytest.raises(ZeroDivisionError):
        get_lat_long(0.0, 0.0, 0.0)


# --- get_vec ---

@pytest.mark.parametrize(
    "lat_deg, lon_deg, expected",
    [
        # Equator & prime meridian → +x
        (0.0, 0.0, (1.0, 0.0, 0.0)),
        # 90°E → +z
        (0.0, 90.0, (0.0, 0.0, 1.0)),
        # 180° / −180° → −x
        (0.0, 180.0, (-1.0, 0.0, 0.0)),
        (0.0, -180.0, (-1.0, 0.0, 0.0)),
        # 90°W → −z
        (0.0, -90.0, (0.0, 0.0, -1.0)),
        # North pole (lon irrelevant) → +y
        (90.0, 0.0, (0.0, 1.0, 0.0)),
        (90.0, 123.4, (0.0, 1.0, 0.0)),
        # South pole → -y
        (-90.0, 0.0, (0.0, -1.0, 0.0)),
        (-90.0, -77.7, (0.0, -1.0, 0.0)),
    ],
)
def test_get_vec_cardinals(lat_deg, lon_deg, expected):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x, y, z = get_vector(lat, lon)

    # Unit length
    assert almost(get_vector_length(x, y, z), 1.0)

    # Expected axes (within tolerance)
    assert almost(x, expected[0])
    assert almost(y, expected[1])
    assert almost(z, expected[2])


def test_longitude_wrap_equivalence():
    # lon = π and lon = -π should give the same vector (−x, 0, 0) at lat=0
    v_pos = get_vector(0.0, math.pi)
    v_neg = get_vector(0.0, -math.pi)
    assert almost(v_pos[0], v_neg[0]) and almost(
        v_pos[1], v_neg[1]) and almost(v_pos[2], v_neg[2])


def test_equator_quadrature_orthogonality():
    # At the equator, lon 0 and lon 90° should be orthogonal
    v0 = get_vector(0.0, 0.0)                 # +x
    v90 = get_vector(0.0, math.pi / 2.0)       # +z
    dot = v0[0] * v90[0] + v0[1] * v90[1] + v0[2] * v90[2]
    assert almost(dot, 0.0)


def test_near_poles_stability():
    # Near North pole: y ~ 1, x/z small; ensure unit length and finite components
    lat = math.radians(89.999999)
    lon = math.radians(37.0)
    v = get_vector(lat, lon)
    assert almost(get_vector_length(*v), 1.0)
    assert all(math.isfinite(c) for c in v)
    assert v[1] > 0.999999999  # strongly “north”


def test_random_unit_length_and_bounds():
    # A few random angles stay unit length and within [-1,1] per component
    rng = [(13.0, 27.0), (-45.0, 123.0), (22.5, -179.9), (-70.0, 89.0)]
    for lat_deg, lon_deg in rng:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        x, y, z = get_vector(lat, lon)
        x, y, z = get_normalized(x, y, z)
        assert get_vector_length(x, y, z) - 1.0 <= EPS
        assert -1.0 - EPS <= x <= 1.0 + EPS
        assert -1.0 - EPS <= y <= 1.0 + EPS
        assert -1.0 - EPS <= z <= 1.0 + EPS

# --- get_shortest_arc ---


def test_equator_basic_bearings():
    start = get_vector(0.0, 0.0)

    az = get_shortest_arc((0.0, 1.0, 0.0), start)  # to North pole
    assert almost(az, 0.0)

    az = get_shortest_arc(get_vector(0.0, math.pi / 2), start)  # to +z
    assert almost(az, math.pi / 2)

    az = get_shortest_arc(get_vector(0.0, -math.pi / 2), start)  # to -z
    assert almost(az, 3 * math.pi / 2)

    az = get_shortest_arc((0.0, -1.0, 0.0), start)  # to South pole
    assert almost(az, math.pi)


def local_axes_at(p):
    """y-axis is North. Return (north_hat, east_hat) at point p on the unit sphere."""
    yhat = (0.0, 1.0, 0.0)
    k = get_dot_product(yhat, p)
    north = (yhat[0] - k * p[0], yhat[1] - k * p[1], yhat[2] - k * p[2])
    north = get_normalized(*north)
    east = get_cross_product(p, north, normalize=True)
    return north, east


@pytest.mark.parametrize(
    "lat_a, lon_a, lat_b, lon_b",
    [
        (math.radians(10.0), math.radians(0.0),
         math.radians(0.0), math.radians(90.0)),
        (math.radians(-20.0), math.radians(45.0),
         math.radians(35.0), math.radians(-120.0)),
        (math.radians(45.0), math.radians(179.0),
         math.radians(-30.0), math.radians(-179.5)),
    ],
)
def test_reverse_bearings_match_great_circle_tangents(lat_a, lon_a, lat_b, lon_b):
    '''
    We can't just ask whether a bearing is equal to the reverse bearing less pi, because except 
    for bearings along meridians and lines of latitude, that pattern doesn't hold. This is 
    because directions n/s aren't actually parallel except at the equator.
    '''
    a = get_vector(lat_a, lon_a)
    b = get_vector(lat_b, lon_b)

    # Great-circle normal (orientation matters)
    n = get_cross_product(a, b, normalize=True)

    # Bearings
    az_ab = get_shortest_arc(b, a)  # A -> B
    az_ba = get_shortest_arc(a, b)  # B -> A

    # Local frames
    nA, eA = local_axes_at(a)
    nB, eB = local_axes_at(b)

    # Tangent vectors implied by the azimuths (North=0, clockwise; so components are cos/sin)
    u_ab = (
        math.cos(az_ab) * nA[0] + math.sin(az_ab) * eA[0],
        math.cos(az_ab) * nA[1] + math.sin(az_ab) * eA[1],
        math.cos(az_ab) * nA[2] + math.sin(az_ab) * eA[2],
    )
    u_ba = (
        math.cos(az_ba) * nB[0] + math.sin(az_ba) * eB[0],
        math.cos(az_ba) * nB[1] + math.sin(az_ba) * eB[1],
        math.cos(az_ba) * nB[2] + math.sin(az_ba) * eB[2],
    )
    u_ab = get_normalized(*u_ab)
    u_ba = get_normalized(*u_ba)

    # Expected great-circle tangents
    t_ab = get_cross_product(n, a, normalize=True)      # along A->B
    t_ba = get_cross_product(
        (-n[0], -n[1], -n[2]), b, normalize=True)  # along B->A

    # Each azimuth-implied tangent must align to its expected GC tangent
    dot1 = get_dot_product(u_ab, t_ab)
    dot2 = get_dot_product(u_ba, t_ba)
    assert abs(1.0 - dot1) <= EPS
    assert abs(1.0 - dot2) <= EPS


@pytest.mark.parametrize(
    "lon_a_deg, lon_b_deg",
    [
        (0.0, 45.0),
        (10.0, 170.0),
        (-90.0, 60.0),
        (179.9, -179.9),
    ]
)
def test_reverse_bearing_on_equator(lon_a_deg, lon_b_deg):
    """Along the equator (lat=0), initial bearings are π apart."""
    lat_a = lat_b = 0.0
    lon_a = math.radians(lon_a_deg)
    lon_b = math.radians(lon_b_deg)
    A = get_vector(lat_a, lon_a)
    B = get_vector(lat_b, lon_b)

    az_ab = get_shortest_arc(B, A)
    az_ba = get_shortest_arc(A, B)

    assert abs(angdiff(az_ab, az_ba + math.pi)) <= EPS


@pytest.mark.parametrize(
    "lat_a_deg, lat_b_deg, lon_deg",
    [
        (30.0, -20.0, 0.0),
        (-45.0, 10.0, 90.0),
        (60.0, -30.0, -135.0),
        (10.0, 80.0, 179.0),  # avoid exact pole/antipodal
    ]
)
def test_reverse_bearing_on_meridian(lat_a_deg, lat_b_deg, lon_deg):
    """Along a meridian (same longitude), initial bearings are π apart."""
    lat_a = math.radians(lat_a_deg)
    lat_b = math.radians(lat_b_deg)
    lon = math.radians(lon_deg)

    # Ensure not identical or antipodal
    assert abs(lat_a - lat_b) > 1e-9
    assert abs(abs(lat_a - lat_b) - math.pi) > 1e-6

    A = get_vector(lat_a, lon)
    B = get_vector(lat_b, lon)

    az_ab = get_shortest_arc(B, A)
    az_ba = get_shortest_arc(A, B)

    assert abs(angdiff(az_ab, az_ba + math.pi)) <= EPS


'''


def test_get_shortest_arc_basic_bearings_from_equator():
    a = vec(0.0, 0.0)  # equator, lon 0
    # to North pole -> azimuth 0
    bN = (0.0, 0.0, 1.0)
    az = get_shortest_arc(bN, a)
    assert abs(az - 0.0) <= ANGLE_EPS

    # to East along equator -> azimuth 90° (π/2)
    bE = vec(0.0, math.pi / 2)
    az = get_shortest_arc(bE, a)
    assert abs(az - math.pi / 2) <= ANGLE_EPS

    # to West along equator -> azimuth 270° (3π/2)
    bW = vec(0.0, -math.pi / 2)
    az = get_shortest_arc(bW, a)
    assert abs(((az - 3 * math.pi / 2 + 2 * math.pi) %
               (2 * math.pi))) <= ANGLE_EPS

    # to South pole -> azimuth 180° (π)
    bS = (0.0, 0.0, -1.0)
    az = get_shortest_arc(bS, a)
    assert abs(az - math.pi) <= ANGLE_EPS






def test_get_shortest_arc_near_pole_uses_fallback_axis():
    # Start very close to North pole, head toward lon=90 equator; azimuth should be ~East (π/2)
    start = vec(math.radians(89.9999), 0.0)
    goal = vec(0.0, math.pi / 2)
    az = get_shortest_arc(goal, start)
    assert abs(az - math.pi / 2) < 1e-4  # looser near-pole tolerance


def test_get_shortest_arc_raises_on_identical_or_antipodal():
    a = vec(0.3, -1.2)
    # identical
    with pytest.raises(ValueError):
        get_shortest_arc(a, a)
    # antipodal (negate)
    b = (-a[0], -a[1], -a[2])
    with pytest.raises(ValueError):
        get_shortest_arc(b, a)
'''
