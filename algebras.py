from itertools import product

from rings import Field

SpaceIndex = int
FieldIndex = int
Coord = tuple[FieldIndex, ...]

class NAAlg:
    def __init__(self, field:Field, magma:NDArray[int]):
        assert is_magma(magma)
        self.field = field
        self.mul = magma
        self.dim = len(magma)
        self.order = len(field)
    def add_vectors(self, u:Coord, v:Coord) -> Coord:
        return tuple(self.field.add[pp,qq] for pp,qq in zip(p,q))
    def mul_vectors(self, u:Coord, v:Coord) -> Coord:
        assert self.dim == len(self.mul) == len(u) == len(v)
        out : dict[SpaceIndex, FieldIndex] = {}
        for d, u_elem in enumerate(u):
            u_mul = self.field.mul[d]
            for prod_idx, v_elem in zip(self.mul[d], v):
                out[prod_idx] = self.field.add[
                    out.get(prod_idx,0), u_mul[v_elem]]
        return tuple(out.pop(i, 0) for i in range(self.dim))
    @property
    def elements(self) -> dict[Coord, int]:
        elems = {}
        for idx, vec in enumerate(
                product(range(self.order), repeat = self.dim)):
            elems[vec] = idx
        return elems
    def ring(self) - > NARing:
        card = len(self.elements) # need to init
        add = np.zeros((card,card))
        mul = np.zeros((card,card))
        for u, i in self.elements.items():
            for v, j in self.elements.items():
                mul[i,j] = self.elements[self.mul_vectors(u, v)]
                add[i,j] = self.elements[self.add_vectors(u, v)]
        return NARing(add, mul)
            
        
        
        
