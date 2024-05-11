from itertools import product
from functools import reduce

from rings import *

SpaceIndex = int
FieldIndex = int
Coord = tuple[int, ...]
Vector = NDArray[int] | Coord

class Algebra:
    def __init__(self, field:Field, magma:NDArray[int]):
        assert is_magma(magma)
        self.associative = is_monoid(magma)
        self.field = field
        self.mul = magma
        self.dim = len(magma)
        self.order = len(field)
        self._add_fcn = lambda i,j: self.field.add[i,j]
    def add_vectors(self, u:Vector, v:Vector) -> Vector:
        return self.field.add[ (u, v) ]
    @property
    def preimg(self) -> dict[int, tuple[Vector, NDArray[int]]]:
        # ~800 ms for 1k x 1k mul; 3x faster than looping over np.nonzero
        # written to optimize speed of mul_vectors
        out = {}
        for i in range(self.dim):
            for j in range(self.dim):
                k = self.mul[i,j]
                iz, jz = out.setdefault(k, (list(), list()))
                iz.append(i)
                jz.append(j)
        return out
    def mul_vectors(self, u:Vector, v:Vector) -> Vector:
        # fibered dot product; fibration from mul, (+,*) from field 
        assert self.dim == len(u) == len(v)
        out = []
        for k, (iz, jz) in self.preimg.items():
            uv = self.field.mul[ (u[iz], v[jz]) ]
            out.append(reduce(self._add_fcn, uv, 0))
        return out
    @property
    def elements(self) -> dict[Coord, int]:
        ivecs = enumerate(product(range(self.order), repeat = self.dim))
        return {vec:idx for idx, vec in ivecs}
    def ring(self) -> NARng:
        # should rewrite taking advantage of distributive property?
        n = len(vecs := self.elements)
        add, mul = [], []
        for u in map(np.array, vecs):
            addi, muli = [], []
            for v in map(np.array, vecs):
                addi += [vecs[tuple(self.add_vectors(u, v))]]
                muli += [vecs[tuple(self.mul_vectors(u, v))]]
            add += [addi]
            mul += [muli]
        ring_cons = Ring if self.associative else NARng
        return ring_cons(np.array(add), np.array(mul))
    def _add_fcn(self, i:int, j:int) -> int:
        return self.field.add[i,j]
    
        
