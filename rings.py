from monoids import *

class Ring:
    def __init__(self, mul:NDArray[int], add:NDArray[int]):
        assert len(add) == len(mul)
        assert all(mul[0] == 0) and all(mul.T[0] == 0)
        assert is_abelian(add) and is_group(add)
        assert is_monoid(mul)
        assert all(is_monoid_hom(add, muli[i]) for i in range(len(mult)))
        self.commutative = is_abelian(mul)
        self.add = add
        self.mul = mul
    @property
    def unit_group(self) -> NDArray[int]:
        pass

def right_proj_plane(R:Ring) -> dict[tuple[int,int,int], tuple[int,int,int]]:
    card = len(R.add)
    plane = {(0,0,0):None}
    for ijk in product(range(len(R.add)), repeat = 3):
        if ijk in plane: continue
        i, j, k = ijk
        lid = set(R.mul[i])
        for idx in (j,k):
            lid |= magma_direct_image(R.add, lid, set(R.mul[idx]))
        if len(lid) != card: continue
        plane[ijk] = ijk
        for arr in R.unit_group[list(ijk)].T:
            assert (t := tuple(arr)) not in plane
            plane[t] = ijk
    
            
            





