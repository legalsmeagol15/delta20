from __future__ import annotations
from typing import Dict, Tuple, List, Mapping, Sequence
from math import sqrt, hypot, atan2
from delta20.packing import *
from delta20.precomputed.canonical_d20 import _canonical_neighbors
from delta20.precomputed.raw_d20 import raw_neighbors

NORTH = 0b0
SOUTH = 0b1

FaceIdx = int
VertexIdx = int


def find_neighbor(face_idx: FaceIdx, edge: int) -> Tuple[FaceIdx, int]:
    '''
    Returns the neighbor's face_idx across the given edge, plus the edge that the neigbor would 
    use to return.    
    '''

    if edge < 0 or edge > 2:
        raise ValueError("edge must be from (0,1,2)")

    # TODO: this code is hot and could benefit from Cython or mypyc or some other means of optimization.
    orig_lod, orig_d20, orig_path, is_south = unpack_face_idx(face_idx)

    lod, d20, path = orig_lod, orig_d20, orig_path

    # Step #1 - we don't really care about LODs higher than the given face_idx's LOD.
    path >>= (22 - lod) * 2

    # Step #2 - descend until we find the ancestor of the neighbor.
    nbr_path = nbr_is_south = nbr_d20 = None
    return_edge = None
    while True:
        pos = path & 0b11

        # The descent lands within a central triangle, the neighbor would be hop to a corner.
        if pos == 3:
            nbr_d20 = d20
            nbr_path = path & ~0b11
            nbr_is_south = not is_south
            return_edge = edge
            break

        # If edge is 0, it always means a hop directly north or directly south.
        elif edge == 0:
            nbr_d20 = d20
            nbr_path = path & ~0b11
            nbr_is_south = not is_south
            return_edge = 0
            break

        # If the pos number and the edge match, it means a hop into the central triangle.
        elif pos == edge:
            nbr_d20 = d20
            nbr_path = path & ~0b11
            nbr_is_south = not is_south
            return_edge = edge
            break

        # Maybe we can't go any lower, and we're at the D20 faces. Hop to the d20 neighbor.
        elif lod == 0:
            nbr_d20 = raw_neighbors[d20][edge]
            nbr_path = 0
            nbr_is_south = is_south
            if d20 >= 5 and d20 <= 14:
                nbr_is_south = not nbr_is_south
                return_edge = edge
            else:
                return_edge = 2 if edge == 1 else 1
            break

        # Otherwise, we have dropped into a corner triangle. Carry on in the descent.
        lod -= 1
        path >>= 2

    assert return_edge is not None
    assert lod >= 0

    # Step #3 - at this point, we know what the neigbor's d20 face and is_south bits are, and the
    # neighbor's path is all the ancestor bits of the original path but waiting for the neighbor's
    # position at this LOD.  Ascend, mirroring the orig_path to find the path within the neighbor.
    while lod < orig_lod:
        nbr_pos = None
        if edge == 0:
            # Counter-polar neighbors, by definition
            assert is_south != nbr_is_south
            nbr_pos = 2 if pos == 1 else 1
        elif is_south == nbr_is_south:
            # Co-polar neighbors
            assert pos != 3
            nbr_pos = 0 if pos == 0 else edge
        # All others are counter-polar lateral neighbors
        elif edge == 1:
            nbr_pos = 0 if pos == 2 else 2
        elif edge == 2:
            nbr_pos = 0 if pos == 1 else 1

        nbr_path |= nbr_pos
        nbr_path <<= 2
        lod += 1
        pos = get_pos(orig_path, lod)

    # Step #4 - restore the path in its proper position regarding unused LOD levels. We'll have
    # one extra shift to the left, so subtract the LOD from 21 instead of 22.
    nbr_path <<= (22 - orig_lod) * 2

    # Done.
    return pack_face_idx(orig_lod, nbr_d20, nbr_path, nbr_is_south), return_edge
