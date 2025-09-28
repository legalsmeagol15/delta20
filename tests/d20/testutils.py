from math import sqrt
from typing import Dict, Tuple
from delta20.precomputed.canonical_d20 import CANONICAL_VERTS, CANONICAL_FACES, CANONICAL_NEIGHBORS

# Tunable tolerances
EPSILON = 1e-12


def get_canonicals():
    # Return the three read-only proxies
    return CANONICAL_VERTS, CANONICAL_FACES, CANONICAL_NEIGHBORS


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def is_ccw_on_sphere(p0, p1, p2, eps=EPSILON):
    r = (p0[0] + p1[0] + p2[0], p0[1] + p1[1] + p2[1], p0[2] + p1[2] + p2[2])
    n = cross(vec_sub(p1, p0), vec_sub(p2, p0))
    s = dot(n, r)
    # scale-aware threshold
    thresh = eps * norm(n) * norm(r)
    return s > thresh


def undirected_edge_key(a, b):
    # return a sorted pair for undirected edge sets
    return (a, b) if a < b else (b, a)
