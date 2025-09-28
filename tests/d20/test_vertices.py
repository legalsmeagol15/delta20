import math
import pytest
from testutils import get_canonicals, norm, undirected_edge_key, EPSILON


def test_vertices_are_normalized():
    verts, _, _ = get_canonicals()
    for v in verts.values():
        assert abs(norm(v) - 1.0) <= EPSILON


def test_edges_are_uniform():
    verts, faces, _ = get_canonicals()
    EDGE_EPS = 1e-9

    def chord_len(u, v):
        ux, uy, uz = verts[u]
        vx, vy, vz = verts[v]
        dx, dy, dz = ux - vx, uy - vy, uz - vz
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    edges = set()
    for (a, b, c) in faces.values():
        edges.add(undirected_edge_key(a, b))
        edges.add(undirected_edge_key(b, c))
        edges.add(undirected_edge_key(c, a))

    base_edge = next(iter(edges))
    base_len = chord_len(*base_edge)

    for e in edges:
        assert abs(chord_len(*e) - base_len) <= EDGE_EPS
