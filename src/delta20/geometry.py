from delta20.defs import VertexIdx
from typing import Tuple
from math import atan2, asin, cos, sin, sqrt, pi
from delta20.defs import VertexIdx
from delta20.precomputed.canonical_d20 import CANONICAL_VERTS

N = 0
NW = 1
W = 2
SW = 3
S = 4
SE = 5
E = 6
NE = 7


def get_dot_product(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def get_cross_product(a: Tuple[float, float, float], b: Tuple[float, float, float], normalize=True) -> Tuple[float, float, float]:
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    if normalize:
        return get_normalized(x, y, z)
    else:
        return (x, y, z)


def get_shortest_arc(goal: Tuple[float, float, float],
                     start: Tuple[float, float, float]) -> float:
    """
    Initial great-circle azimuth from 'start' to 'goal' (North=0, East=π/2, clockwise), [0, 2π).
    Uses spherical forward-azimuth formula; y-axis is North/South.
    Raises ValueError if identical or antipodal.
    """
    lat1, lon1 = get_lat_long(*start)
    lat2, lon2 = get_lat_long(*goal)

    # Antipodal check (robust-ish). If dot is about -1, we are looking at the antipode. Any arc
    # will be a valid arc.
    dot = get_dot_product(start, goal)
    if dot <= -1.0 + 1e-15:
        raise ValueError(
            "Great-circle direction undefined for antipodal points.")

    # Also undefined if start and goal are the same
    elif dot > 1.0 - 1e-15:
        raise ValueError(
            "Great-circle direction undefined for identical points.")

    dlon = (lon2 - lon1 + pi) % (2 * pi) - pi  # wrap to (-π, π]
    # Spherical forward azimuth (North=0, clockwise)
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * \
        cos(lat2) * cos(dlon)
    az = atan2(x, y)  # (-π, π]
    if az < 0.0:
        az += 2.0 * pi
    return az


def get_face_center(face: Tuple[VertexIdx, VertexIdx, VertexIdx], normalize: bool = True) -> Tuple[float, float, float]:
    '''
    Returns a directional center for a face. Specifically, computes the mean direction of the 
    three unit vertex vectors. Note: this is the chordal/Euclidean center, normalized to project 
    back onto the unit sphere. It is not the spherical (surface-area) centroid of the geodesic 
    triangle, but it’s stable, fast, and close for our faces.
    '''
    # fmt: off
    v0, v1, v2 = CANONICAL_VERTS[face[0]], CANONICAL_VERTS[face[1]], CANONICAL_VERTS[face[2]]
    # fmt: on

    x = (v0[0] + v1[0] + v2[0]) / 3.0
    y = (v0[1] + v1[1] + v2[1]) / 3.0
    z = (v0[2] + v1[2] + v2[2]) / 3.0
    if normalize:
        return get_normalized(x, y, z)
    return (x, y, z)


def get_vector_length(x, y, z) -> float:
    return sqrt(x * x + y * y + z * z)


def get_normalized(x: float, y: float, z: float) -> Tuple[float, float, float]:
    '''
    Returns the normalized vector. Does no 0-checking - beware of trying to normalize a (0,0,0) 
    vector.
    '''
    mag = get_vector_length(x, y, z)
    return x / mag, y / mag, z / mag


def get_lat_long(vx: float, vy: float, vz: float) -> Tuple[float, float]:
    """
    Convert a 3D vector (x, y, z) to (lat, lon) in radians for a sphere whose
    North–South axis is aligned to the +y / -y axis.


    Conventions
    - Latitude lat ∈ [-pi/2, pi/2]; +lat toward +y (North), -lat toward -y (South).
    - Longitude lon ∈ (-pi, pi]; rotation around the y-axis with positive eastward:
    * lon = 0 -> +x (prime meridian)
    * lon = +pi/2 -> +z (90°E)
    * lon = pi/-pi -> -x (IDL)
    * lon = -pi/2 -> -z (90°W)


    The input does not need to be unit length; it is normalized internally.
    """
    r = get_vector_length(vx, vy, vz)
    x, y, z = vx / r, vy / r, vz / r

    lat = asin(y)
    lon = atan2(z, x)  # angle in the x–z plane, positive toward +z (east)
    return lat, lon


def get_vector(lat: float, lon: float) -> Tuple[float, float, float]:
    """
    Unit vector for (lat, lon) with the N/S axis aligned to +y (North) / -y (South).
    - lat ∈ [-π/2, π/2], lon ∈ (-π, π]
    - lon=0 (the Prime Meridian) falls on +x, lon=π/2 points to +z (Eastern Hemisphere). lat=π/2 points +y (North Pole)
    """
    cl = cos(lat)
    x = cl * cos(lon)
    y = sin(lat)       # North–South axis
    z = cl * sin(lon)

    # tiny renormalization for numerical safety
    r = get_vector_length(x, y, z)
    return (x / r, y / r, z / r)
