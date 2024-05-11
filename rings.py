from functools import lru_cache
from itertools import product, combinations

from monoids import *
from maps import preimage

Coord = tuple[int,int,int]
Points = dict[Coord, set[Coord]] # points -> equiv classes of vectors
Lines = dict[Coord, set[frozenset[Coord]]] # points -> lines (sets of points)

ZERO = (0,0,0)

class NARng:
    def __init__(self, add:NDArray[int], mul:NDArray[int]):    
        assert (n := len(add)) == len(mul)
        assert is_abelian(add) and is_group(add)
        assert is_magma(mul)
        assert all(mul[0] == 0) and all(mul.T[0] == 0)
        assert all(is_homomorphism(add, mul[i], add) for i in range(n))
        self.commutative = is_abelian(mul)
        if not self.commutative:
            assert all(is_homomorphism(add, mul.T[i], add) for i in range(n))
        self.add = add
        self.mul = mul
    def add_vectors(self, p:Coord, q:Coord) -> Coord:
        return tuple(self.add[pp,qq] for pp,qq in zip(p,q))
    def __len__(self) -> int:
        return len(self.add)

class Ring(NARng):
    def __init__(self, add:NDArray[int], mul:NDArray[int]):
        assert (n := len(add)) == len(mul)
        assert is_monoid(mul)
        for i in range(1, n):
            if all(mul[i] == np.arange(n)) and all(mul.T[i] == np.arange(n)):
                if i == 1:
                    orig = np.arange(n)
                else:
                    perm = np.array([0] + list(range(i,n)) + list(range(1,i)))
                    mul = mul[perm].T[perm].T
                    orig = invert(perm)
                break
        else: raise ValueError('Ring must have a 2-sided identity; none found')
        super().__init__(add, mul)
        self.recovery_index = orig
    @property
    def unit_group_with_tr(self) -> tuple[NDArray[int], list[int]]:
        #groupprops.subwiki.org/wiki/Equality_of_left_and_right_inverses_in_monoid
        idx, jdx = (self.mul == 1).nonzero()
        gelem = list(set(idx) & set(jdx))
        return submagma(self.mul, gelem)

class Field(Ring):
    def __init__(self, add:NDArray[int], mul:NDArray[int]):
        assert is_group(mul[1:].T[1:] - 1)
        super().__init__(add, mul)
    @property
    def unit_group_with_tr(self) -> tuple[NDArray[int], list[int]]:
        return self.mul[1:,1:], list(range(1, len(self.mul)))
        
def linear_2d_subspace(R:Ring,
                       p:Coord,
                       q:Coord,
                       mod:dict[Coord,Coord] = None) -> frozenset[Coord]:
    modfcn = (lambda x:x) if mod is None else (lambda x: mod[x])
    seen = {ZERO}
    new = {p,q}
    while new:
        seen.add(n := new.pop())
        addn = lambda x: modfcn(R.add_vectors(x, n))
        new |= set(map(addn, seen)) - seen
    seen.discard(ZERO)
    return frozenset(seen)

def right_proj_plane(R:Ring) -> tuple[Points, Lines]:
    '''
    F2 = Ring(np.array([[0,1],[1,0]]), np.array([[0,0],[0,1]]))
    FanoPlane = right_proj_plane(F2)
    '''
    card = len(R.add)

    # form the partition of arrays into points
    partition = {ZERO:ZERO}
    for ijk in product(range(len(R.add)), repeat = 3):
        if ijk in partition: continue
        i, j, k = ijk
        ideal = set(R.mul[i])
        for idx in (j,k):
            ideal |= magma_direct_image(R.add, ideal, set(R.mul[idx]))
        if len(ideal) != card: continue
        idx, jdx = (R.mul == 1).nonzero()
        units = list(set(idx) & set(jdx))
        multiples = R.mul[list(ijk)].T[units]
        for arr in multiples:
            assert (t := tuple(arr)) not in partition
            partition[t] = ijk
    points = preimage(partition)
    points.pop(ZERO)
    n_pts = len(points)
    
    # join part reps to construct lines
    lines = {}
    for p, q in combinations(points, 2):
        if len(lines.get(p, set())) == n_pts or \
           len(lines.get(q, set())) == n_pts:
            continue
        line = linear_2d_subspace(R, p, q, partition)
        for point in line:
            lines.setdefault(point, set()).add(line)

    return points, lines
