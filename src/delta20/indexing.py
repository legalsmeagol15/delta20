from __future__ import annotations
from typing import Dict, Tuple, List, Mapping, Sequence, Callable

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


# <--------------------path-finding--------------------->
TChoiceHeuristic = Callable[[FaceIdx, FaceIdx], Tuple[int, int, int]]
TWeightHeuristic = Callable[[FaceIdx, FaceIdx], float]


def _default_choice_heuristic(start: FaceIdx, target: FaceIdx) -> Tuple[int, int, int]:
    # TODO: a dumb dijksta's algorithm can find a path, but we can be more clever by looking at
    # ancestor triangles.
    return (0, 1, 2)


def _default_weight_finder(start: FaceIdx, target: FaceIdx) -> float:
    return 1.0


def find_path(
        start: FaceIdx,
        target: FaceIdx,
        choice_heuristic: TChoiceHeuristic = _default_choice_heuristic,
        weight_heuristic: TWeightHeuristic = _default_weight_finder):
    '''
    Returns a path list starting from 'start' and going to 'target'. The path list will be accompanied 
    '''
    result = []

    raise NotImplementedError()


if __name__ == '__main__':

    pass
