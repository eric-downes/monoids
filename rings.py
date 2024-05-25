from itertools import product, combinations, batched
from functools import lru_cache
from math import sqrt
import pickle

from maps import preimage
from monoids import *

Coord = tuple[int,...]
Vec = Coord | NDArray[int]
ProjLine = frozenset[Coord] # lines contain points (pt = equivalence of vectors)

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
    def sum(self, arr:Vec) -> int:
        # for big arrays ~30x faster than reduce; 130 us x 3 ms on macmini
        while len(arr) > 1:
            n, pad = divmod(len(arr), 2)
            n += pad
            new = np.empty(n, dtype=int)
            new[0:pad] = arr[0]
            new[pad:] = self.add[arr[pad:n] , arr[n:]]
            arr = new
        return arr[0]
    def matmul(self, M:NDArray[int], N:NDArray[int]) -> NDArray[int]:
        # mem vs speed tradeoff... probably should broadcast ...
        assert max(M.shape + N.shape) <= len(self.add)
        assert M.shape[1] == N.shape[0]
        NT = N.T
        MN = np.ndarray(shape = (M.shape[0], N.shape[1]), dtype = int)
        for i in range(MN.shape[0]):
            Mi = M[i]
            for j in range(MN.shape[1]):
                MN[i,j] = self.sum(self.mul[Mi, NT[j]])
        return MN
    def __len__(self) -> int:
        return len(self.add)
    def __str__(self) -> str:
        return f'Addition:\n{self.add}\nMultiplication:\n{self.mul}'

class Ring(NARng):
    def __init__(self, add:NDArray[int], mul:NDArray[int]):
        assert (n := len(add)) == len(mul)
        assert all(mul[1] == np.arange(n)) and all(mul.T[1] == np.arange(n))
        assert is_monoid(mul)        
        super().__init__(add, mul)
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

def flat_array(xs: set[Coord]) -> np.array:
    l = []
    for x in xs:
        l.extend(x)
    return np.array(l)
    
def linear_2d_subspace(R:Ring,
                       p:Coord,
                       q:Coord) -> set[Coord]:
    seen = {ZERO}
    new = {p, q}
    l = len(p)
    while new:
        seen.add(n := new.pop())
        ns = np.array(n * len(seen))
        xs = flat_array(seen)
        for vec in batched(R.add[ns, xs], l):
            if (tup := tuple(vec)) not in seen:
                new.add(tup)
    return seen

def right_proj_plane(R:Ring) -> dict[Coord, set[ProjLine]]:
    '''
    F2 = Ring(np.array([[0,1],[1,0]]), np.array([[0,0],[0,1]]))
    FanoPlane = right_proj_plane(F2)
    '''
    card = len(R.add)
    idx, jdx = (R.mul == 1).nonzero()
    units = np.array(list(set(idx) & set(jdx)))

    # form the partition of arrays into points
    partition = {ZERO:ZERO}
    for ijk in product(range(len(R.add)), repeat = 3):
        if ijk in partition: continue
        idx = np.unique(R.add[R.mul[ijk[0]]].T[R.mul[ijk[1]]].ravel())
        ideal = np.unique(R.add[idx].T[R.mul[ijk[2]]].ravel())
        if len(ideal) != card: continue # only want ideals as large as R itself
        multiples = R.mul[np.array(ijk)].T[units]
        for arr in multiples:
            assert (t := tuple(arr)) not in partition
            partition[t] = ijk
    partition.pop(ZERO)
    n_pts = len(points := set(partition.values()) - {ZERO})
    pts_per_line = (1 + sqrt(4 * n_pts - 3)) / 2
    if pts_per_line != round(pts_per_line):
        with open((fil := 'err.pkl'), 'wb') as f:
            pickle.dump(partition, f)
        raise ValueError(
            f'Cannot create plane of order {pts_per_line - 1}; state in {fil}')
    
    # join part reps to construct lines
    lines = {}
    for p, q in combinations(points, 2):
        if len(lines.setdefault(p, set())) < pts_per_line and \
           len(lines.setdefault(q, set())) < pts_per_line:
            subspace = linear_2d_subspace(R, p, q) & partition.keys()
            pts = frozenset({partition[vec] for vec in subspace})
            lines[p].add(pts)
            lines[q].add(pts)
    return lines
