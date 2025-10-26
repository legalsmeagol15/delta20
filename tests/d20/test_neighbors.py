import pytest
from testutils import get_canonicals, undirected_edge_key
from delta20.packing import pack_face_idx, unpack_face_idx
from delta20.indexing import find_neighbor, build_path, face_idx_to_str, FaceIdx
from delta20.precomputed.canonical_d20 import CANONICAL_FACES_INDEXED


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


MAX_LEVELS = 23


def test_find_neighbors_calculated_by_hand():

    # Test the neighbor there and back again.
    def test_get_neighbor_and_back(start: FaceIdx, edge: int, expected: FaceIdx):
        n, ret_edge = find_neighbor(start, edge)

        assert n == expected, f"first hop error:\nfound:\t\t{face_idx_to_str(n)}\nexpected:\t{face_idx_to_str(expected)}\n"
        n_rev, ret_edge_rev = find_neighbor(n, ret_edge)
        assert n_rev == start, f"first reverse error:\nfound:\t\t{face_idx_to_str(n_rev)}\nstart:\t{face_idx_to_str(start)}\n"
        assert ret_edge_rev == edge, f"reverse edge error:\nfound:{ret_edge_rev}\nexpected(original):{edge}"

    # Check correct results at the d20 level
    test_get_neighbor_and_back(
        pack_face_idx(0, 0, build_path(0), False),
        1,
        pack_face_idx(0, 1, build_path(0), False))
    test_get_neighbor_and_back(
        pack_face_idx(0, 0, build_path(0), False),
        0,
        pack_face_idx(0, 6, build_path(0), True))
    test_get_neighbor_and_back(
        pack_face_idx(0, 0, build_path(0), False),
        2,
        pack_face_idx(0, 4, build_path(0), False))

    # And in reverse
    n, ret_edge = find_neighbor(CANONICAL_FACES_INDEXED[1], 2)
    assert n == CANONICAL_FACES_INDEXED[0]
    assert ret_edge == 1
    n, ret_edge = find_neighbor(CANONICAL_FACES_INDEXED[6], 0)
    assert n == CANONICAL_FACES_INDEXED[0]
    assert ret_edge == 0
    n, ret_edge = find_neighbor(CANONICAL_FACES_INDEXED[4], 1)
    assert n == CANONICAL_FACES_INDEXED[0]
    assert ret_edge == 2

    # Try LOD = 1 neighbor finding

    # Into a central triangle
    a = pack_face_idx(1, 0, build_path(0), False)
    n, ret_edge = find_neighbor(a, 0)
    assert n == pack_face_idx(1, 0, build_path(3), True)
    assert ret_edge == 0

    # Across a parent triangle's boundary
    n, ret_edge = find_neighbor(a, 1)
    assert n == pack_face_idx(1, 1, build_path(0), False)
    assert ret_edge == 2
    n, ret_edge = find_neighbor(a, 2)
    assert n == pack_face_idx(1, 4, build_path(0), False)
    assert ret_edge == 1

    # Out of a central triangle
    a = pack_face_idx(1, 0, build_path(3), True)
    n, ret_edge = find_neighbor(a, 0)

    assert n == pack_face_idx(1, 0, build_path(0), False)
    assert ret_edge == 0
    n, ret_edge = find_neighbor(a, 2)
    assert n == pack_face_idx(1, 0, build_path(2), False)
    assert ret_edge == 2
    n, ret_edge = find_neighbor(a, 1)
    assert n == pack_face_idx(1, 0, build_path(1), False)
    assert ret_edge == 1

    # A few deeper checks, with lod=4. This is all done comparing against some indices set mapped
    # out by hand.
    tri = {}
    for t in (3022, 3021, 3012, 3011, 3020, 3033, 3010, 3002, 3001, 3000):
        path = build_path(t)
        tri[t] = pack_face_idx(4, 7, path, is_south=True)
    for t in (3023, 3030, 3013, 3031, 3032, 3003):
        path = build_path(t)
        tri[t] = pack_face_idx(4, 7, path, is_south=False)

    def test(a: int, b: int, edge):
        test_get_neighbor_and_back(tri[a], edge, tri[b])

    test(3022, 3023, 2)
    test(3021, 3030, 2)
    test(3032, 3033, 2)
    test(3001, 3032, 0)
    test(3000, 3003, 0)
    test(3030, 3033, 0)
    test(3010, 3032, 1)
    test(3021, 3023, 1)
    test(3031, 3033, 1)

    # Check the hops across a triangle edge
    for t in (2100, 2101, 2102, 2110, 2133, 2120, 2111, 2112, 2121, 2122):
        path = build_path(t)
        tri[t] = pack_face_idx(4, 7, path, is_south=False)
    for t in (2103, 2132, 2131, 2113, 2130, 2123):
        path = build_path(t)
        tri[t] = pack_face_idx(4, 7, path, is_south=True)

    test(3011, 2100, 2)
    test(3010, 2101, 2)
    test(3001, 2110, 2)
    test(3000, 2111, 2)

    # Check the hops into another d20 face(face 0 -> face1, over d20 edge 1)
    # (0023, 0030, 0013, 0031, 0032, 0003)
    for t in ("0023", "0030", "0013", "0031", "0032", "0003"):
        path = build_path(t)
        tri[t] = pack_face_idx(lod=4, d20=8, path=path, is_south=False)
    # (0022, 0021, 0012, 0011, 0020, 0033, 0010, 0002, 0001, 0000)
    for t in ("0022", "0021", "0012", "0011", "0020", "0033", "0010", "0002", "0001", "0000"):
        path = build_path(t)
        tri[t] = pack_face_idx(lod=4, d20=8, path=path, is_south=True)
    for t in (2200, 2201, 2202, 2210, 2233, 2220, 2211, 2212, 2221, 2222):
        path = build_path(t)
        tri[t] = pack_face_idx(lod=4, d20=7, path=path, is_south=False)
    for t in (2203, 2232, 2231, 2213, 2230, 2223):
        path = build_path(t)
        tri[t] = pack_face_idx(lod=4, d20=7, path=path, is_south=True)

    # TODO: look at the interesting pattern formed with a hop over edge 1 for counter-polars
    test(2200, "0022", 1)
    test(2202, "0020", 1)
    test(2220, "0002", 1)
    test(2222, "0000", 1)
