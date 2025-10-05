import pytest
from testutils import get_canonicals, undirected_edge_key, EPSILON
import math
from delta20.packing import unpack_face_idx
from delta20.geometry import get_lat_long
from delta20.precomputed.raw_d20 import raw_faces, raw_vertices
from typing import Tuple


def test_counts_and_euler():
    verts, faces, neighbors = get_canonicals()
    V = len(verts)
    F = len(faces)

    # Unique undirected edges from faces
    edges = set()
    for (v0, v1, v2) in faces.values():
        edges.add(undirected_edge_key(v0, v1))
        edges.add(undirected_edge_key(v1, v2))
        edges.add(undirected_edge_key(v2, v0))
    E = len(edges)

    assert V == 12
    assert F == 20
    assert E == 30
    assert V - E + F == 2


def test_every_edge_has_two_incident_faces():
    _, faces, _ = get_canonicals()
    edge_to_faces = {}
    for fid, (a, b, c) in faces.items():
        for u, v in ((a, b), (b, c), (c, a)):
            k = undirected_edge_key(u, v)
            edge_to_faces.setdefault(k, set()).add(fid)
    assert all(len(inc) == 2 for inc in edge_to_faces.values())


def test_orientation_ordered_neighbors():
    verts, faces, neighbors = get_canonicals()

    for fid, (v0, v1, v2) in faces.items():
        n0, n1, n2 = neighbors[fid]
        _, _, _, south = unpack_face_idx(fid)
        _, _, _, south_n0 = unpack_face_idx(n0)

        # n0 must be opposite polarity (across e0)
        assert bool(south_n0) is not bool(south)

        # Build this face's undirected edges by numbering convention:
        # e0 opposite v0 -> (v1, v2)
        # e1 opposite v1 -> (v2, v0)
        # e2 opposite v2 -> (v0, v1)
        e0 = undirected_edge_key(v1, v2)
        e1 = undirected_edge_key(v2, v0)
        e2 = undirected_edge_key(v0, v1)

        # Each neighbor must share the corresponding edge
        def edge_set(fid_):
            a, b, c = faces[fid_]
            return {
                undirected_edge_key(a, b),
                undirected_edge_key(b, c),
                undirected_edge_key(c, a),
            }

        assert e0 in edge_set(n0)
        assert e1 in edge_set(n1)
        assert e2 in edge_set(n2)

        # Symmetry: each neighbor must list this face somewhere too
        assert fid in neighbors[n0]
        assert fid in neighbors[n1]
        assert fid in neighbors[n2]


def test_apex_matches_orientation_bit():

    verts, faces, _ = get_canonicals()

    for fid, (v0, v1, v2) in faces.items():
        lod, d20, path, south = unpack_face_idx(fid)
        south = bool(south)
        apex_y = verts[v0][1]

        if south:
            assert apex_y < 0.0
        else:
            assert apex_y > 0.0


def test_raw_faces_caps_are_polar_aligned():
    EPS = 1e-12

    def _y(v_idx):
        return raw_vertices[v_idx][1]  # y is north/south axis

    def _is_north_cap_face(face):
        v0, v1, v2 = face
        y0, y1, y2 = _y(v0), _y(v1), _y(v2)
        # Apex must be the northernmost; flanks share latitude (within EPS)
        return (y0 > y1 and y0 > y2) and (abs(y1 - y2) <= EPS)

    def _is_south_cap_face(face):
        v0, v1, v2 = face
        y0, y1, y2 = _y(v0), _y(v1), _y(v2)
        # Apex must be the southernmost; flanks share latitude (within EPS)
        return (y0 < y1 and y0 < y2) and (abs(y1 - y2) <= EPS)

    # First 5: north cap
    for i in range(5):
        assert _is_north_cap_face(
            raw_faces[i]), f"raw_faces[{i}] is not north-polar aligned"

    # Last 5: south cap
    for i in range(len(raw_faces) - 5, len(raw_faces)):
        assert _is_south_cap_face(
            raw_faces[i]), f"raw_faces[{i}] is not south-polar aligned"
