# Delta20 Conventions (quick reference)

## Base
- d20 faces indexed 0..19.
- Each face has corners (0,1,2) CCW, edges E0/E1/E2.

## Indexing
- A given triangle index or FaceIdx is 64 packed bits:
        d20        lod        path (MSD)     flags
     (5 bits) | (5 bits) |    (46 bits)   | (8 bits)
- A vertex index or VertexIdx is 64 packed bits:
     (TBD)

## Subdivision
- Children: 0,1,2 = corner children; 3 = center. Corners children are numbered along with the vertex.
- Orientation: 
    For north-oriented children, child 0 is NORTH, 1 is SW, and 2 is SE.
    For south-oriented children, child 0 is SOUTH, 1 is NE, and 2 is NW
- Path is base-4 digits (MSB-deep): finest digit in the (lod*2)th bits.

## Orientation ("flip")
- flip = base_flip[d20] XOR (count of digit==3 in path) % 2
- Edges DO NOT rotate with flip (E0/E1/E2 are stable). Edges are numbered opposite the child corners.
- Corner digits rotate by +1 when flip=1:
  - digit_to_parent_corner (flip=1): {0→1, 1→2, 2→0}
  - inverse: {0→2, 1→0, 2→1}

## NORTH-oriented children:
                        V0
                        /\
                       /  \
                      /    \
              ___    /      \    ___
             /      /        \      \
            /      /    C0    \      \
           /      /            \      \
         E2      /              \      E1
         /      /________________\      \
        /      /\                /\      \
       /      /  \              /  \      \
      /      /    \            /    \      \
      \     /      \    C3    /      \     /
       \   /        \        /        \   /
          /    C1    \      /    C2    \
         /            \    /            \
        /              \  /              \
       /________________\/________________\
      V1                                  V2
                  \_____E0_____/


## SOUTH-oriented children:

                   _____E0_____
                  /            \
    V2 ____________________________________ V1
       \                /\                /
        \              /  \              /
         \            /    \            /
          \    C2    /      \    C1    /
        /  \        /        \        /   \
       /    \      /    C3    \      /     \
       \     \    /            \    /      /
        \     \  /              \  /      /
         \     \/________________\/      /
          E1    \                /      E2
           \     \              /      /
            \     \            /      /
             \___  \    C0    /   ___/
                    \        /
                     \      /
                      \    /
                       \  /
                        \/
                        V0
