from __future__ import annotations
from typing import Dict, Tuple, List, Mapping, Sequence
from math import sqrt, hypot, atan2
from delta20.packing import *
from delta20.precomputed.raw_d20 import raw_faces, raw_vertices

NORTH = 0b0
SOUTH = 0b1

FaceIdx = int
VertexIdx = int


def _get_d20_neighbor(d20: int) -> FaceIdx:
    pass


def find_neighbor(face_idx: FaceIdx, edge) -> FaceIdx:
    orig_lod, orig_d20, orig_path, south = unpack_face_idx(face_idx)
    lod, d20, path = orig_lod, orig_d20, orig_path

    # Step #1 - descend until we find the ancestor of the neighbor.

    # the cw_crumbs marks whether we land in the cw or ccw face, given an edge. cw is a 1
    bread_crumbs_cw = 0b0
    path >>= (23 - lod) * 2
    pos_within_parent = path & 0b11
    n_idx = None
    while True:
        if pos_within_parent == 3:
            # The descent lands within a central triangle. The neighbor would therefore be within
            # a corner.
            n_idx = pack_face_idx(lod, d20, (path & ~0b11) | edge, not south)
            break
        elif pos_within_parent == edge:
            # If the pos number and the edge match, it means a hop into the central triangle.
            n_idx = pack_face_idx(lod, d20, path | 0b11, not south)
            break

        # Or maybe we're at the d20 faces.
        lod -= 1
        if lod <= 0:
            n_idx = _get_d20_neighbor(d20)
            break

        # Otherwise, we have dropped into a corner triangle. That corner might be a pos 0, 1, or 2.
        # Depending on what edge we're looking at and the corner we landed on, we went to either
        # the CW or CCW end of the edge. We'll mark that in our breadcrumbs.
        last_pos = pos_within_parent
        path >>= 2
        pos_within_parent = path & 0b11
        if last_pos == pos_within_parent:
            # We're in the same corner still. Keep going down. The same breadcrumbs repeat.
            bread_crumbs_cw = (bread_crumbs_cw << 1) | (bread_crumbs_cw & 0b1)
            continue
        bread_crumbs_cw <<= 1
        if pos_within_parent + 1 == edge or (pos_within_parent == 2 and edge == 0):
            # This happens when we're on the ccw end of an edge
            continue
        bread_crumbs_cw |= 1

    assert n_idx is not None

    while lod < orig_lod:
        pass


canonical_verts: Dict[VertexIdx, Tuple[float, float, float]] = {}
canonical_faces: Dict[FaceIdx, Tuple[VertexIdx, VertexIdx, VertexIdx]] = {}
canonical_neighbors: Dict[FaceIdx, Tuple[FaceIdx, FaceIdx, FaceIdx]] = {}


# This code is not intended for running in the main app. I just used it to generate the canonical
# geometry, which is then hard-coded. I'm keeping the code in case I ever decide to change the way
# the indices are packed, or something like that.
def _set_normalized_and_indexed():

    # Step #1: identify the normalized and indexed vertices. Make a mapping between tuple index
    # and FaceIdx
    vert_to_vertex_idx: Tuple[VertexIdx] = (0,) * 12
    for idx, vert in enumerate(raw_vertices):

        # Step #1a: Find the canonical indexing. The original vertices are not "owned" by any face
        # in the way the later vertices are, so we give arbitrary value 0b11111 for the d20 face.
        packed_vi = pack_vertex_idx(lod=0, d20=0b11111, index=0)
        vert_to_vertex_idx[idx] = packed_vi

        # Step #1b: normalize the vertex.
        x, y, z = vert
        mag = sqrt((x**2) + (y**2) + (z**2))
        x /= mag
        y /= mag
        z /= mag
        canonical_verts[packed_vi] = (x, y, z)

    # Step #2: these have been written a million times, but the chief way to import them is numpy
    # and I don't want to do that. I'll just write them again the 1,000,001st time.
    def vec_sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross_prod(u, v):
        return (u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    def dot(cross, summed):
        return cross[0] * summed[0] + cross[1] * summed[1] + cross[2] * summed[2]

    def is_ccw(face):
        v0, v1, v2 = (canonical_verts[vert_to_vertex_idx[vi]] for vi in face)
        summed = (v0[0] + v1[0] + v2[0],
                  v0[1] + v1[1] + v2[1],
                  v0[2] + v1[2] + v2[2])
        cross = cross_prod(vec_sub(v1, v0), vec_sub(v2, v0))
        # Technically, cross and summed would be normalized, but we're only looking for the sign
        dot = cross[0] * summed[0] + \
            cross[1] * summed[1] + \
            cross[2] * summed[2]

        return dot > 0

    # Step #3: assure canonicity of the faces. This means that north-oriented faces are
    # distinguished from south-oriented faces, the vertices are listed with polar vertex first and
    # then go CCW thereafter, and there is a mapping between tuple index and FaceIdx.
    face_to_face_idx: Tuple[FaceIdx] = (0,) * 20
    for idx, face in enumerate(raw_faces):

        # Step 3a: assure CCW. Honestly, I'm pretty sure they're already CCW, but just in case
        assert is_ccw(face)
        # if not is_ccw(face):
        #     face[1], face[2] = face[2], face[1]

        # Step 3b: assure 0th vertex is the polar orientation vertex.
        verts = (raw_vertices[vi] for vi in face)
        assert verts[1][2] == verts[2][2]
        # while verts[1][2] != verts[2][2]:
        #     face = (face[1], face[2], face[0])
        #     verts = (raw_vertices[vi] for vi in face)

        # Step 3c: determine polarity
        south = verts[0][2] < verts[1][2]

        # Step #3d: pack the face_idx
        fi = pack_face_idx(lod=0, d20=idx, path=0, south=south)
        face_to_face_idx[idx] = fi
        canonical_faces[fi] = tuple(vert_to_vertex_idx[vi] for vi in face)

    # Step #4: use the raw faces to find the neighbors, and store with canonical indexing.
    neighbors: Tuple[List[int, int, int]] = ([],) * 20
    for f_idx, face in enumerate(raw_faces):
        v0, v1, v2 = face
        focus_list = neighbors(face)
