import pytest
from testutils import get_canonicals, undirected_edge_key
from delta20.packing import pack_face_idx, unpack_face_idx
from delta20.indexing import find_neighbor


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


def build_path(digits_msb_first):
    """Compose a 46-bit path from MSB→LSB base-4 digits."""
    p = 0
    for d in digits_msb_first:
        p = (p << 2) | (d & 3)
    # Fill remaining digits with zeros to 23 if caller provided fewer
    for _ in range(len(digits_msb_first), MAX_LEVELS):
        p <<= 2
    return p


CASES = [
    # Diverge immediately via center (pos==3 at top)
    ("center-hit_immediate", 4, 0, [3, 0, 1, 2], False, 1, True, False),
    # Diverge immediately via edge-match (pos==edge at top)
    ("edge-match_immediate", 4, 0, [1, 1, 1, 1], False, 1, True, False),

    # Forced D20 on equator (no digit==3 or ==edge anywhere)
    ("equator_forced_D20_e1", 4, 6, [0, 2, 0, 2], False, 1, True, True),
    ("equator_forced_D20_e2", 4, 11, [2, 0, 2, 0], True, 2, True, True),

    # Late divergence (deep level), south cap
    ("south_cap_late_edge-match", 4, 19, [0, 1, 2, 0], True, 2, True, False),

    # Caps → D20 hop (no digit==3 or ==edge), no flip on caps
    ("north_cap_forced_D20_e1", 4, 4, [0, 0, 0, 0], False, 1, False, True),
    ("south_cap_forced_D20_e1", 4, 16, [2, 2, 2, 2], True, 1, False, True),

    # Edge 0 hop inside face (branch should trigger; flip expected)
    ("edge0_inside_face", 4, 0, [0, 2, 0, 2], False, 0, True, False),

    # Mixed: north cap, divergence not at top, edge=2
    ("north_cap_late_div_e2", 4, 2, [1, 2, 1, 2], False, 2, True, False),

    # Equator, edge 0 hop
    ("equator_edge0", 4, 9, [0, 2, 0, 2], True, 0, True, False),
]


@pytest.mark.parametrize("desc,lod,d20,digs,south,edge,expect_flip,expect_lod0", CASES, ids=[c[0] for c in CASES])
def test_black_box_neighbors(desc, lod, d20, digs, south, edge, expect_flip, expect_lod0):
    # Build full path (MSB→LSB), pad remaining digits with zeros
    path = build_path(digs)

    fid = pack_face_idx(lod=lod, d20=d20, path=path, south=south)

    nbr, ret_edge = find_neighbor(fid, edge)
    # 1) Return edge must be identical in this indexing scheme
    assert ret_edge == edge, f"{desc}: return edge mismatch"

    # 2) Invertibility across the same edge
    fid2, ret2 = find_neighbor(nbr, ret_edge)
    assert fid2 == fid, f"{desc}: not invertible across edge {edge}"
    assert ret2 == edge, f"{desc}: return edge not preserved on inverse hop"

    # 3) D20 hop detection: internal divergence → same d20; forced D20 → different d20
    _, d20_nbr, _, south_nbr = unpack_face_idx(nbr)
    internal_divergence = (d20_nbr == d20)
    assert internal_divergence != expect_lod0, f"{desc}: expected LOD-0 hop={expect_lod0}, got internal={internal_divergence}"

    # 4) Polarity rules: flip or not per case
    flipped = (south_nbr != south)
    assert flipped == expect_flip, f"{desc}: expected flip={expect_flip}, got {flipped}"

    # 5) Extra sanity for equator/caps at D20 hops
    if expect_lod0:
        # is it an equator face?
        if 5 <= d20 <= 14:
            # equator band: flip on any edge
            assert flipped, f"{desc}: equator D20 hop should flip"
        else:
            # caps: no flip at base hop
            assert not flipped, f"{desc}: cap D20 hop should not flip"
