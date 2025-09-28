import pytest
from testutils import get_canonicals, undirected_edge_key


def test_referential_integrity():
    verts, faces, neighbors = get_canonicals()

    # every neighbor face id exists
    for fid, nbrs in neighbors.items():
        assert fid in faces
        for n in nbrs:
            assert n in faces

    # every vertex referenced by faces exists
    for (v0, v1, v2) in faces.values():
        assert v0 in verts and v1 in verts and v2 in verts


def test_face_vertex_uniqueness():
    _, faces, _ = get_canonicals()
    for (v0, v1, v2) in faces.values():
        assert len({v0, v1, v2}) == 3


def test_neighbor_shares_exact_opposite_edge():
    _, faces, neighbors = get_canonicals()
    for fid, (a, b, c) in faces.items():
        n0, n1, n2 = neighbors[fid]

        # edge opposite v0 is (v1, v2)
        assert undirected_edge_key(b, c) in {
            undirected_edge_key(*pair)
            for pair in [(faces[n0][0], faces[n0][1]),
                         (faces[n0][1], faces[n0][2]),
                         (faces[n0][2], faces[n0][0])]
        }
        # edge opposite v1 is (v2, v0)
        assert undirected_edge_key(c, a) in {
            undirected_edge_key(*pair)
            for pair in [(faces[n1][0], faces[n1][1]),
                         (faces[n1][1], faces[n1][2]),
                         (faces[n1][2], faces[n1][0])]
        }
        # edge opposite v2 is (v0, v1)
        assert undirected_edge_key(a, b) in {
            undirected_edge_key(*pair)
            for pair in [(faces[n2][0], faces[n2][1]),
                         (faces[n2][1], faces[n2][2]),
                         (faces[n2][2], faces[n2][0])]
        }
