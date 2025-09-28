import pytest
from testutils import get_canonicals, undirected_edge_key


def test_neighbor_symmetry_and_edge_sharing():
    verts, faces, neighbors = get_canonicals()

    for fid, (v0, v1, v2) in faces.items():
        n0, n1, n2 = neighbors[fid]

        # neighbors must exist and not be self
        assert len({n0, n1, n2}) == 3
        assert fid not in (n0, n1, n2)

        # Edge opposite v0 is (v1,v2), and neighbor across e0 shares that edge reversed
        for edge_index, nbr in enumerate((n0, n1, n2)):
            assert nbr in faces, f"Neighbor {nbr} missing from faces"
            a, b, c = faces[fid]
            if edge_index == 0:
                shared = undirected_edge_key(b, c)
            elif edge_index == 1:
                shared = undirected_edge_key(c, a)
            else:
                shared = undirected_edge_key(a, b)

            na, nb, nc = faces[nbr]
            nbr_edges = {
                undirected_edge_key(na, nb),
                undirected_edge_key(nb, nc),
                undirected_edge_key(nc, na),
            }
            assert shared in nbr_edges

        # symmetry: if A lists B as a neighbor, B lists A somewhere too
        for nbr in (n0, n1, n2):
            assert fid in neighbors[nbr]
