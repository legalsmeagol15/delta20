from __future__ import annotations
from delta20.packing import pack_face_idx, pack_vertex_idx
from delta20.defs import VertexIdx, FaceIdx
from delta20.precomputed.raw_d20 import raw_vertices, raw_faces
from typing import Dict, Tuple, List
from math import sqrt


# This code is not intended for running in the main app. I just used it to generate the canonical
# geometry, which is then hard-coded in the "precomputed" directory. I'm keeping the code in case
# I ever decide to change the way the indices are packed, or something like that.
def get_canonicals():
    canonical_verts: Dict[VertexIdx, Tuple[float, float, float]] = {}
    canonical_faces: Dict[FaceIdx, Tuple[VertexIdx, VertexIdx, VertexIdx]] = {}
    canonical_neighbors: Dict[FaceIdx, Tuple[FaceIdx, FaceIdx, FaceIdx]] = {}

    # Step #1: identify the normalized and indexed vertices. Make a mapping between tuple index
    # and FaceIdx
    vert_to_vertex_idx: List[VertexIdx] = [0] * 12
    for idx, vert in enumerate(raw_vertices):

        # Step #1a: Find the canonical indexing. The original vertices are not "owned" by any face
        # in the way the later vertices are, so we give arbitrary value 0b11111 for the d20 face.
        packed_vi = pack_vertex_idx(lod=0, d20=0, index=idx)
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
    face_to_face_idx: List[FaceIdx] = [0] * 20
    for d20_idx, face in enumerate(raw_faces):

        # Step 3a: assure CCW. Honestly, I'm pretty sure they're already CCW, but just in case
        assert is_ccw(face), face
        # if not is_ccw(face):
        #     face[1], face[2] = face[2], face[1]

        # Step 3b: assure 0th vertex is the polar orientation vertex.
        verts = tuple(raw_vertices[vi] for vi in face)
        assert verts[1][2] == verts[2][2]
        # while verts[1][2] != verts[2][2]:
        #     face = (face[1], face[2], face[0])
        #     verts = tuple(raw_vertices[vi] for vi in face)

        # Step 3c: determine polarity
        south = verts[0][2] < verts[1][2]

        # Step #3d: pack the face_idx.
        packed_fi = pack_face_idx(lod=0, d20=d20_idx, path=0, south=south)
        face_to_face_idx[d20_idx] = packed_fi
        canonical_faces[packed_fi] = tuple(
            vert_to_vertex_idx[vi] for vi in face)

    # Step #4: use the raw faces to find the neighbors, and store with canonical indexing. A
    # neighbor is a face that will feature vertices in the reverse order.

    # Step 4a: store what edges are associated with what faces
    edges: Dict[Tuple[int, int], int] = {}
    for f_idx, face in enumerate(raw_faces):
        edges[(face[0], face[1])] = f_idx
        edges[(face[1], face[2])] = f_idx
        edges[(face[2], face[0])] = f_idx

    # Step 4b: store the non-canonical neighbors by finding the mirror reverse of each edge. The
    # edge associated with a vertex will be the edge OPPOSITE that vertex. Ie, the edge 0 will be
    # the one between vertices 1 and 2, etc.
    neighbors: List[Tuple[int, int, int]] = [()] * 20
    for f_idx, face in enumerate(raw_faces):
        neighbors[f_idx] = (
            edges[(face[2], face[1])],
            edges[(face[0], face[2])],
            edges[(face[1], face[0])],
        )

    # Step 4c: Convert neighbors to canonical neighbors
    for idx, n in enumerate(neighbors):
        n_idx = face_to_face_idx[idx]
        canonical_neighbors[n_idx] = tuple(face_to_face_idx[fi] for fi in n)

    # Done.
    return canonical_verts, canonical_faces, canonical_neighbors


if __name__ == '__main__':
    verts, faces, neighbors = get_canonicals()
    print(f"_canonical_verts = {verts}")
    print(f"_canonical_faces = {faces}")
    print(f"_canonical_neighbors = {neighbors}")
