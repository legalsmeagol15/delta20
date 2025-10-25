from __future__ import annotations
from typing import Tuple
from .defs import VertexIdx, FaceIdx
from delta20.precomputed.canonical_d20 import CANONICAL_FACES_INDEXED

_lod_mask = 0b11111 << 59
_d20_mask = 0b11111 << 54
_path_mask = ((0b1 << 46) - 1) << 8
_vertex_idx_mask = (1 << 51) - 1
_flag_mask = (0b1 << 8) - 1


def get_pos(path: int, lod: int) -> int:
    '''
    Given a 46-bit path, and a level of detail ("LOD") from [0..22], return the position of the 
    triangle within its parent at that LOD.
    '''
    # Path is a base 4 value tracking the position from the d20 parent (but in base 2, of course),
    # starting at the most significant digit packed left. There are up to 23 levels of detail
    # (LODs).
    # Example: 0b 01 11 10 ... 11
    #             ^  ^  ^      ^
    #         LOD=0  |  |      |
    #            LOD=1  |      |
    #               LOD=2      |
    #                     LOD=22
    # To align the a given LOD's bits into the rightmost position (so we can mask with 0b11 and
    # figure out if we're looking at 0, 1, 2, or 3), we will right-shift ((22 - lod) * 2) bits.
    return (path >> (2 * (22 - lod))) & 0b11


def build_path(*route) -> int:
    '''
    Builds the left-aligned path from the given 0-3 numbers, in a format ready to give to 
    pack_face_idx()

    Call like:  build_path(1,2,0,3)
                    -or-
                build_path(1203)
    '''
    if len(route) == 1:
        r = route[0]
        if isinstance(r, str):
            r = tuple(int(item) for item in list(r))
            return build_path(*r)
        # The values 4..9 can break things
        assert (r >= 0 and r <= 3) or r >= 10
        if r > 3:
            r = tuple(int(item) for item in list(str(r)))
            return build_path(*r)

    result = 0b0
    for r in route:
        assert 0 <= r <= 3
        result <<= 2
        result |= r

    # For example, (1,3,2) will end up left-packed like so:
    # 0q 13 2000000000 0000000000
    result <<= (23 - len(route)) * 2
    return result


def pack_face_idx(lod: int, d20: int, path: int, is_south: bool = None) -> FaceIdx:
    # The 64 bits of a face_idx are packed like so:
    #    lod        d20        path (MSD)     flags
    # (5 bits) | (5 bits) |    (46 bits)   | (8 bits)

    assert 0 <= path < (1 << 46)

    if lod < 0 or lod >= 23:
        raise ValueError(f"LODs outside 0..22 are not permitted ({lod}).")
    if d20 < 0 or d20 >= 20:
        raise ValueError(f"D20 faces outside 0..19 are not permitted ({d20}).")

    if is_south is None:
        # auto-calculate the polarity
        d20_idx = CANONICAL_FACES_INDEXED[d20]
        is_south = (d20_idx & 0b1) == 0b1
        _path = path >> ((23 - lod) * 2)
        _lod = lod
        while _lod > 0:
            if (_path & 0b11) == 0b11:
                is_south = not is_south
            _path >>= 2
            _lod -= 1
    return (lod << 59) | (d20 << 54) | (path << 8) | (0b1 if is_south else 0b0)


def unpack_face_idx(face_idx: FaceIdx) -> Tuple[int, int, int, bool]:
    lod = (face_idx & _lod_mask) >> 59
    d20 = (face_idx & _d20_mask) >> 54
    path = (face_idx & _path_mask) >> 8
    return lod, d20, path, bool(face_idx & 0b1)


def pack_vertex_idx(lod: int, d20: int, index: int) -> VertexIdx:
    if lod < 0 or lod >= 23:
        raise ValueError(f"LODs outside 0..22 are not permitted({lod})")
    if (d20 < 0 or d20 >= 20) and d20 != 0b11111:
        raise ValueError(
            f"D20 faces outside 0..19 are not permitted, except original indices ({d20}).")
    if index < 0 or index >= (1 << 51):
        raise ValueError(
            f"index faces outside 0..(1 << 51) are not permitted({index}).")
    return (lod << 59) | (d20 << 54) | index


def unpack_vertex_idx(vertex_idx: VertexIdx) -> Tuple[int, int, int]:
    lod = (vertex_idx & _lod_mask) >> 59
    d20 = (vertex_idx & _d20_mask) >> 54
    index = vertex_idx & _vertex_idx_mask
    return lod, d20, index


def face_idx_to_str(face_idx: FaceIdx):
    # '0b0100 0000000000 0000000000 0000000000 0000000000'
    #      ^
    #      40
    lod, d20, path, flags = unpack_face_idx(face_idx)
    path_str = ["0"] * 23
    for _l in range(0, lod + 1):
        v = get_pos(path, _l)
        path_str[_l] = str(v)
    return f"lod={lod}, d20={d20}, path={''.join(path_str)}, flags={bin(flags)}"


__all__ = ["build_path", "face_idx_to_str", "get_pos", "pack_face_idx", "pack_vertex_idx",
           "unpack_face_idx", "unpack_vertex_idx"]
