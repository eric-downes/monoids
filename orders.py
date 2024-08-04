from __future__ import annotations
from collections import ChainMap

class PreOrderElem:
    def __init__(self, order:PreOrder, elem:Any):
        self.order = order
        self.value = elem
    def eq(self, y) -> None:
        if not (leqs := self.order.leqset(y)).setdefault(x, ChainMap()):
            leqs.maps.append()
        self.order.leqset(x).add(y)
    def lt(self, y) -> None: pass
    def __eq__(self, y) -> bool: pass
    def __lt__(self, y) -> bool: pass
    

class PreOrder:
    def __init__(self, members:Container):
        self._data_ = {x:{'maxima':{x}, 'leqset':{x}} for x in members}
    def __call__(self, x:Any) -> PreOrderElem:
        return PreOrderElem(self, x) if x in self._data_ else raise KeyError(x)
    def members(self) -> set: return self._data_.keys()
    def maxima(self, x) -> set: return self._data_[x]['maxima']
    def leqset(self, x) -> set: return self._data_[x]['lowerset']
    
class Lattice(PreOrder): pass
    def __or__(self, a, b): pass
    def __and__(self, a, b): pass

class ModLatt(Lattice): pass

class CompLatt(Lattice): pass

class UnqCompLatt(CompLatt): pass
