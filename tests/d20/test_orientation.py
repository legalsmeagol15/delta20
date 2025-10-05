import pytest
from delta20.packing import pack_vertex_idx, unpack_face_idx
from testutils import get_canonicals, is_ccw_on_sphere


def test_faces_are_ccw_and_apex_polarity_matches_orientation_y_up():
    verts, faces, _ = get_canonicals()
    north_count = 0
    south_count = 0

    for fid, (v0, v1, v2) in faces.items():
        p0, p1, p2 = verts[v0], verts[v1], verts[v2]
        assert is_ccw_on_sphere(p0, p1, p2)

        # polarity bit from packed face id
        lod, d20, path, south = unpack_face_idx(
            fid)  # adjust to your signature

        # with y-up, the apex is the pole-most vertex by y
        apex_y = p0[1]
        if south:
            south_count += 1
            assert apex_y < 0.0
        else:
            north_count += 1
            assert apex_y > 0.0

    # distribution sanity (5 north-cap, 5 south-cap, 10 alternating)
    assert north_count == 10
    assert south_count == 10


def test_poles_have_five_faces_each():
    _, faces, _ = get_canonicals()
    # count how many faces include vertex 0 and 11
    north_pole = pack_vertex_idx(0, 0b11111, 0)
    south_pole = pack_vertex_idx(0, 0b11111, 11)
    count_v0 = sum(1 for (a, b, c) in faces.values()
                   if north_pole in (a, b, c))
    count_v11 = sum(1 for (a, b, c) in faces.values()
                    if south_pole in (a, b, c))
    assert count_v0 == 5
    assert count_v11 == 5
