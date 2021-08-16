from __future__ import annotations
from typing import *

def ident(x): return x

class Endo(list):
    def __init__(self, l:List[int], mod:bool = False):
        self.fcn = (lambda x: x % len(l)) if mod else ident
        super().__init__(l)
        self.is_id = self == list(range(len(self)))
        
    def __mul__(self, other:Union[int, List[int]]) -> Endo:
        return self(other)

    def __call__(self, other:Union[int, List[int]], strict:bool = True) -> Union[Endo, list, int]:
        if isinstance(other, int): return self[self.fcn(other)]
        if strict: assert len(self) == len(other)
        if self.is_id: return other
        if isinstance(other, Endo) and other.is_id: return self
        x = [self[self.fcn(i)] for i in other]
        if isinstance(other, list): return x
        if isinstance(other, Endo): return Endo(x)
        
    def __rmul__(self, other:List[int]) -> Endo:
        if isinstance(other, list): return Endo(other)(self)
        if isinstance(other, Endo): return other(self)

    
class Coset(set):
    def __init__(self, *args, op:Binop = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = op
        
    def __mul__(self, x:Any):
        return {self.op(x,y) for y in self}
        
