from functools import lru_cache
from itertools import product, combinations

from monoids import *
from maps import preimage

Coord = tuple[int,int,int]
Points = dict[Coord, set[Coord]]
Lines = dict[Coord, set[frozenset[Coord]]]

ZERO = (0,0,0)

class Ring:
    def __init__(self, add:NDArray[int], mul:NDArray[int]):
        assert (n := len(add)) == len(mul)
        assert all(mul[0] == 0) and all(mul.T[0] == 0)
        assert all(mul[1] == np.arange(n)) and all(mul.T[1] == np.arange(n)) 
        assert is_abelian(add) and is_group(add)
        assert is_monoid(mul)
        assert all(is_monoid_hom(add, mul[i], add) for i in range(len(mul)))
        self.commutative = is_abelian(mul)
        if not self.commutative:
            assert all(is_monoid_hom(add, mul.T[i], add) for i in range(len(mul)))
        self.add = add
        self.mul = mul
    @property
    def unit_group_with_tr(self) -> tuple[NDArray[int], list[int]]:
        #groupprops.subwiki.org/wiki/Equality_of_left_and_right_inverses_in_monoid
        idx, jdx = (self.mul == 1).nonzero()
        gelem = list(set(idx) & set(jdx))
        return submagma(self.mul, gelem)
    def add_vectors(self, p:Coord, q:Coord) -> Coord:
        return tuple(self.add[pp,qq] for pp,qq in zip(p,q))

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
