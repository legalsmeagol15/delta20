import importlib
import random
import re
import sys
import pytest
from delta20.packing import pack_face_idx, pack_vertex_idx, unpack_face_idx, unpack_vertex_idx, face_idx_to_str
from delta20.precomputed.canonical_d20 import CANONICAL_FACES
from testutils import get_canonicals

# Assumptions (conservative and easy to tweak):
# - LOD uses 5 bits: 0..31
# - Face d20 is a base-face index: 0..19
# - Vertex d20 may be the "unowned" sentinel (e.g., 0b11111 = 31) for unit D20; we still fuzz 0..31
# - Face path on the unit D20 is 0 (no subdivisions); we sample only 0 here to stay compatible
MAX_LOD = 22
MAX_D20 = 19
MAX_PATH = (1 << 46) - 1
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


def test_face_bitfield_boundaries():
    # Low edges
    fid = pack_face_idx(lod=0, d20=0, path=0, south=False)
    assert 0 <= fid <= UINT64_MAX
    assert unpack_face_idx(fid) == (0, 0, 0, False)

    # High edges
    fid = pack_face_idx(lod=MAX_LOD, d20=MAX_D20, path=0, south=True)
    assert unpack_face_idx(fid) == (MAX_LOD, MAX_D20, 0, True)

    # Path high bits (max 46-bit value)
    fid = pack_face_idx(lod=MAX_LOD, d20=MAX_D20, path=MAX_PATH, south=False)
    lod2, d202, path2, south2 = unpack_face_idx(fid)
    assert (lod2, d202, path2, south2) == (MAX_LOD, MAX_D20, MAX_PATH, False)


def test_vertex_bitfield_boundaries():
    # Low edges
    vid = pack_vertex_idx(lod=0, d20=0, index=0)
    assert unpack_vertex_idx(vid) == (0, 0, 0)

    # High edges (vertex d20 may allow 31 sentinel; keep if you support it)
    vid = pack_vertex_idx(lod=MAX_LOD, d20=MAX_D20, index=(1 << 51) - 1)
    lod2, d202, idx2 = unpack_vertex_idx(vid)
    assert lod2 == MAX_LOD and d202 == MAX_D20 and idx2 == (1 << 51) - 1

# --- Tests for to_str function ---


MAX_LEVELS = 23  # 46 bits → 23 base-4 digits


def extract_field(s: str, name: str) -> str:
    m = re.search(rf"{name}=([^,]+)", s)
    assert m, f"expected field {name} in: {s}"
    return m.group(1).strip()


def extract_path_str(s: str) -> str:
    m = re.search(r"path=([0-3]*)", s)
    assert m is not None, f"expected path=... in: {s}"
    return m.group(1)


def decode_flags(s: str) -> int:
    f = extract_field(s, "flags")
    # Accept '0b0/0b1', '0/1', or 'False/True'
    if f.startswith("0b"):
        return int(f, 2)
    if f in ("0", "1"):
        return int(f)
    if f in ("False", "True"):
        return 1 if f == "True" else 0
    pytest.fail(f"Unrecognized flags format: {f}")


def base4_digits_from_path(path: int, lod: int) -> str:
    # MSB-first digits: level 0..lod-1 absolute index (0 = topmost)
    # shift per digit = 2 * (22 - L_abs)
    out = []
    for L_abs in range(lod):
        d = (path >> (2 * (22 - L_abs))) & 0b11
        out.append(str(d))
    return "".join(out)


@pytest.mark.parametrize("lod,d20,south", [
    (0, 0, False),
    (1, 4, True),
    (7, 12, False),
    (12, 19, True),
    (22, 3, False),
])
def test_format_and_fields(lod, d20, south):
    # Construct a path that exercises digits 0..3 in a simple cycle
    path = 0
    for L_abs in range(MAX_LEVELS):
        digit = (L_abs % 4)
        path = (path << 2) | digit
    # Pack and stringify
    fid = pack_face_idx(lod=lod, d20=d20, path=path, south=south)
    s = face_idx_to_str(fid)

    # lod/d20 fields
    assert extract_field(s, "lod") == str(lod)
    assert extract_field(s, "d20") == str(d20)

    # path length must equal lod
    p = extract_path_str(s)
    assert len(p) == lod

    # flags render to either 0b0/0b1 or False/True/0/1; decode to {0,1}
    f = decode_flags(s)
    assert f in (0, 1)
    assert f == (1 if south else 0)


@pytest.mark.parametrize("lod", [0, 1, 5, 11, 22])
def test_digits_match_bit_extraction(lod):
    # Build a path with a known MSB-first pattern: (3,2,1,0,3,2,1,0,...)
    path = 0
    for L_abs in range(MAX_LEVELS):
        digit = (3 - (L_abs % 4))
        path = (path << 2) | digit

    fid = pack_face_idx(lod=lod, d20=7, path=path, south=False)
    s = face_idx_to_str(fid)

    rendered = extract_path_str(s)
    expected = base4_digits_from_path(path, lod)
    assert rendered == expected


def test_lod_zero_has_empty_path_and_valid_flags():
    fid0 = pack_face_idx(lod=0, d20=0, path=0, south=False)
    s0 = face_idx_to_str(fid0)
    assert extract_path_str(s0) == ""
    assert decode_flags(s0) == 0

    fid1 = pack_face_idx(lod=0, d20=19, path=0, south=True)
    s1 = face_idx_to_str(fid1)
    assert extract_path_str(s1) == ""
    assert decode_flags(s1) == 1


@pytest.mark.parametrize("lod", [3, 9, 17, 22])
def test_all_digit_values_appear_in_rendered_path(lod):
    # Make a path whose first `lod` digits (MSB→LSB) are 0,1,2,3,0,1,...
    path = 0
    pattern = [0, 1, 2, 3]
    digits = []
    for L_abs in range(MAX_LEVELS):
        d = pattern[L_abs % 4]
        digits.append(d)
        path = (path << 2) | d

    fid = pack_face_idx(lod=lod, d20=5, path=path, south=False)
    s = face_idx_to_str(fid)
    p = extract_path_str(s)

    # Compare against the intended prefix
    expected = "".join(str(d) for d in digits[:lod])
    assert p == expected


def test_extreme_lod_max():
    lod = 22
    # Construct a path with alternating 3 and 0 in MSB-first order
    path = 0
    for L_abs in range(MAX_LEVELS):
        d = 3 if (L_abs % 2 == 0) else 0
        path = (path << 2) | d

    fid = pack_face_idx(lod=lod, d20=9, path=path, south=True)
    s = face_idx_to_str(fid)

    p = extract_path_str(s)
    assert len(p) == lod
    assert set(p).issubset(set("03"))
    assert decode_flags(s) == 1
