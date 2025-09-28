import importlib
import random
import sys
import pytest
from delta20.packing import pack_face_idx, pack_vertex_idx, unpack_face_idx, unpack_vertex_idx
from delta20.precomputed.canonical_d20 import CANONICAL_FACES
from testutils import get_canonicals

# Assumptions (conservative and easy to tweak):
# - LOD uses 5 bits: 0..31
# - Face d20 is a base-face index: 0..19
# - Vertex d20 may be the "unowned" sentinel (e.g., 0b11111 = 31) for unit D20; we still fuzz 0..31
# - Face path on the unit D20 is 0 (no subdivisions); we sample only 0 here to stay compatible
MAX_LOD = 22
MAX_D20 = 19
PATH_SAMPLES = [0]  # expand later if your packer allows arbitrary paths here
VERTEX_INDEX_SAMPLES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # unit D20 seeds

# -----------------------------------------------------------------------


def test_face_pack_unpack_edge_cases():
    cases = [
        (0, 0, 0, False),
        (0, 0, 0, True),
        (MAX_LOD, 0, 0, False),
        (MAX_LOD, MAX_D20, 0, True),
    ]
    for lod, d20, path, south in cases:
        fid = pack_face_idx(lod=lod, d20=d20, path=path, south=south)
        lod2, d202, path2, south2 = unpack_face_idx(fid)
        assert (lod, d20, path, bool(south)) == (
            lod2, d202, path2, bool(south2))


def test_face_pack_unpack_random_roundtrip():
    rng = random.Random(0xD20D20)
    for _ in range(200):
        lod = rng.randint(0, MAX_LOD)
        d20 = rng.randint(0, MAX_D20)
        path = rng.choice(PATH_SAMPLES)
        south = rng.choice([False, True])
        fid = pack_face_idx(lod=lod, d20=d20, path=path, south=south)
        assert isinstance(fid, int)
        fields = unpack_face_idx(fid)
        assert (lod, d20, path, south) == (
            fields[0], fields[1], fields[2], bool(fields[3]))


def test_face_flip_south_preserves_other_fields():
    # Use canonical faces to get “known good” IDs, then flip their south flag
    for fid in CANONICAL_FACES.keys():
        lod, d20, path, south = unpack_face_idx(fid)
        fid_flip = pack_face_idx(
            lod=lod, d20=d20, path=path, south=not bool(south))
        lod2, d202, path2, south2 = unpack_face_idx(fid_flip)
        assert (lod2, d202, path2) == (lod, d20, path)
        assert bool(south2) is (not bool(south))


def test_vertex_pack_unpack_edge_cases():
    cases = [
        (0, 0, 0),
        (0, MAX_D20, 0),
        (MAX_LOD, 0, 11),
        (MAX_LOD, MAX_D20, 5),
    ]
    for lod, d20, index in cases:
        vid = pack_vertex_idx(lod=lod, d20=d20, index=index)
        lod2, d202, index2 = unpack_vertex_idx(vid)
        assert (lod, d20, index) == (lod2, d202, index2)


def test_vertex_pack_unpack_random_roundtrip():
    rng = random.Random(0xFACEFACE)
    for _ in range(200):
        lod = rng.randint(0, MAX_LOD)
        d20 = rng.randint(0, MAX_D20)
        index = rng.choice(VERTEX_INDEX_SAMPLES)
        vid = pack_vertex_idx(lod=lod, d20=d20, index=index)
        assert isinstance(vid, int)
        lod2, d202, index2 = unpack_vertex_idx(vid)
        assert (lod, d20, index) == (lod2, d202, index2)


def test_pack_rejects_out_of_range_values():
    # Test for face_idx
    with pytest.raises(Exception):
        pack_face_idx(lod=-1, d20=0, path=0, south=False)
    with pytest.raises(Exception):
        pack_face_idx(lod=MAX_LOD + 1, d20=0, path=0, south=False)
    with pytest.raises(Exception):
        pack_face_idx(lod=0, d20=MAX_D20 + 1, path=0, south=False)

    # Test for vertex_idx
    with pytest.raises(Exception):
        pack_vertex_idx(lod=-1, d20=0, index=0)
    with pytest.raises(Exception):
        pack_vertex_idx(lod=MAX_LOD + 1, d20=0, index=0)
    with pytest.raises(Exception):
        pack_vertex_idx(lod=0, d20=MAX_D20 + 1, index=0)
    with pytest.raises(Exception):
        pack_vertex_idx(lod=0, d20=0, index=-1)

# -----------------------------------------------------------------------
# Test fields as they exist to ensure correct packing.


UINT64_MAX = (1 << 64) - 1


def test_face_idx_round_trip_on_canonicals():
    _, faces, _ = get_canonicals()
    for fid in faces.keys():
        # adjust to your actual return type
        lod, d20, path, south = unpack_face_idx(fid)

        rid = pack_face_idx(lod=lod, d20=d20, path=path, south=south)
        assert isinstance(rid, int) and 0 <= rid <= UINT64_MAX
        assert rid == fid


def test_vertex_idx_round_trip_on_canonicals():
    verts, _, _ = get_canonicals()
    for vid in verts.keys():
        fields = unpack_vertex_idx(vid)
        assert isinstance(fields, tuple)
        rid = pack_vertex_idx(*fields)
        assert isinstance(rid, int) and 0 <= rid <= UINT64_MAX
        assert rid == vid


def test_public_proxies_are_same_across_imports():
    # Importing the package twice (without reload) should yield identical proxy objects
    import delta20.precomputed.canonical_d20 as m1
    m2 = importlib.import_module("delta20.precomputed.canonical_d20")
    assert m1.CANONICAL_VERTS is m2.CANONICAL_VERTS
    assert m1.CANONICAL_FACES is m2.CANONICAL_FACES
    assert m1.CANONICAL_NEIGHBORS is m2.CANONICAL_NEIGHBORS
