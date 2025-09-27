from __future__ import annotations
from typing import Dict, Tuple, List, Mapping, Sequence

FaceIdx = int
VertexIdx = int

NORTH = 0b0
SOUTH = 0b1


def pack_face_idx(d20: int, lod: int, path: int, flags: int) -> FaceIdx:
    # The 64 bits of a face_idx are packed like so:
    #    d20        lod        path        flags
    # (5 bits) | (5 bits) | (46 bits) | (8 bits)
    return (d20 << 59) | (lod << 54) | (path << 8) | flags
