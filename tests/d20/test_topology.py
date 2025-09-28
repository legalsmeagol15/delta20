import pytest
from testutils import get_canonicals, undirected_edge_key


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
