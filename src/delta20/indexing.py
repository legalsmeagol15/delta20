from __future__ import annotations
from typing import Dict, Tuple, List, Mapping, Sequence
from math import sqrt, hypot, atan2
from delta20.packing import build_path, get_pos, pack_face_idx, unpack_face_idx, face_idx_to_str
from delta20.precomputed.canonical_d20 import CANONICAL_FACES_INDEXED
from delta20.precomputed.raw_d20 import raw_neighbors

NORTH = 0b0
SOUTH = 0b1

FaceIdx = int
VertexIdx = int


def _get_nbr_chars(is_south: bool, pos: int, edge: int, copolar: bool) -> Tuple[bool, int, int]:
    '''
    Returns the is_south, pos, and return edge of the neighbor. Central triangle hops are ruled 
    out already.
    '''
    assert edge >= 0 and edge <= 2
    assert pos >= 0 and pos <= 3
    if edge == 0:
        return not is_south, 2 if pos == 1 else 1, 0
    if edge == 1:
        if copolar:
            return is_south, 0 if pos == 0 else 1, 2
        else:
            return not is_south, 2 if pos == 0 else 0, 1
    if edge == 2:
        if copolar:
            return is_south, 0 if pos == 0 else 2, 1
        else:
            return not is_south, 1 if pos == 0 else 0, 2


def find_neighbor(face_idx: FaceIdx, edge: int) -> Tuple[FaceIdx, int]:
    '''
    Returns the neighbor's face_idx across the given edge, plus the edge that the neigbor would 
    use to return.    
    '''
    # TODO: There is an even more efficient solution here. Once we have established the neighbor-
    # ancestor of a given triangle, we know that the ascent path will comprise solely 0s and a
    # single number in a pattern that is inverse of the descent path's 0s and its number. Ie:
    # 2200 crosses to a new d20 and ascends as 0022
    # 2222 becomes 0000
    # 2022 becomes 0200
    # This is true at least for contra-polar triangles. For co-polar triangles, I suspect that
    # the inversion pattern holds true but the neighbor number is opposite. For example:
    # 2200 becomes 0011
    # 2222 becomes 0000 (still)
    # 2022 becomes 0100
    # But I haven't proven this for co-polar triangles yet.

    # TODO: this code is hot and could benefit from Cython or mypyc or some other means of optimization.
    assert edge >= 0 and edge <= 2
    orig_lod, orig_d20, orig_path, is_south = unpack_face_idx(face_idx)
    assert orig_lod >= 0 and orig_lod < 23

    # Step #1 - the neighbors of LOD=0 d20 faces are precomputed.
    if orig_lod == 0:
        nbr_d20 = raw_neighbors[orig_d20][edge]
        nbr_is_south = (CANONICAL_FACES_INDEXED[nbr_d20] & 1) != 0
        _, _, nbr_edge = \
            _get_nbr_chars(is_south, 0, edge, is_south == nbr_is_south)
        return pack_face_idx(0, nbr_d20, 0, nbr_is_south), nbr_edge
    lod, d20, path = orig_lod, orig_d20, orig_path
    # Step #2 - we don't really care about LODs higher than the given face_idx's LOD.
    path >>= (23 - orig_lod) * 2

    # Step #3 - descend until we find the ancestor of the neighbor.
    nbr_is_south = nbr_edge = None
    nbr_d20 = d20
    path_rev = 0
    pos = None
    while lod > 0:
        pos = path & 3
        path >>= 2
        path_rev = (path_rev << 2) | pos

        if pos == 3:
            nbr_is_south, nbr_pos, nbr_edge = not is_south, edge, edge
            break
        elif pos == edge:
            nbr_is_south, nbr_pos, nbr_edge = not is_south, 3, edge
            break
        # elif edge == 0:
        #     nbr_is_south, nbr_pos, nbr_edge = not is_south, 2 if pos == 1 else 1, 0
        #     break
        lod -= 1

    # Step #4 - did we drop all the way to the d20 level because we never crossed over a corner/
    # central edge? Then we find the neighbor in the precomputed tables.
    if lod == 0:
        nbr_d20 = raw_neighbors[d20][edge]
        nbr_is_south = (CANONICAL_FACES_INDEXED[nbr_d20] & 1) != 0
        _, nbr_pos, nbr_edge = \
            _get_nbr_chars(is_south, pos, edge, nbr_is_south == is_south)
        lod = 1

    # Step #5 - ascend, figuring out the neighbor's path by reflecting across the indicated edge.
    nbr_path = path
    while lod <= orig_lod:
        lod += 1
        nbr_path = (nbr_path << 2) | nbr_pos
        path_rev >>= 2
        pos = path_rev & 3
        _, nbr_pos, _ = \
            _get_nbr_chars(is_south, pos, edge, nbr_is_south == is_south)

    # Step #6 - shift the neighbor's path to its proper position regarding unused LOD levels.
    nbr_path <<= (23 - orig_lod) * 2

    # Done.
    return pack_face_idx(orig_lod, nbr_d20, nbr_path, nbr_is_south), nbr_edge


if __name__ == '__main__':

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
