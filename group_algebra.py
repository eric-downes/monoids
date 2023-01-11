from monoids import *

class GroupAlgebra:
    def __init__(self, G:NDArray[int], scalars:type):
        assert is_group(G)
        self.group = G
        self.maps = [np.nonzero(G == i) for i in range(len(G))]
        self.zero = np.zeros(len(G), dtype=scalars)

class GroupAlgebraElement:
    def __init__(self, parent:GroupAlgebra, vals:np.array):
        self.array = vals.T if len(vals) > len(parent.group) else vals
        assert self.array.shape[0] == len(parent.group)
        self.module = parent
        
    def __mul__(self, other:GroupAlgebraElement):
        assert other.module == self.module
        order = len(self.array)
        slf = np.reshape(self.array, (order, 1)) if len(self.array.shape) == 1 else self.array.T
        othr = np.reshape(self.array, (1, order)) if len(other.array.shape) == 1 else other.array
        prod = slf @ othr
        new = np.array([prod[self.module.maps[i]].sum() for i in range(order)])
        return GroupAlgebraElement(self.module, new)
    
