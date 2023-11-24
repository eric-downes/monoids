from __future__ import annotations
from collections import Counter

class PosMultiSet(Counter):
    def __eq__(self, other:PosMultiSet) -> bool:
        return hash(self) == hash(other)
    def __mul__(self, other:PosMultiSet) -> bool:
        return self + other

class Terms:
    def __init__(self, *args, **kwargs):
        m = PosMultiSet(*args, **kwargs)
        self.pos = +m
        self.neg = -m
    def __mul__(self, other:Terms) -> Terms:
        pos = Terms()
        for a in self.pos.values():
            for b in other.pos.values():
                ab = a * b
                pos
                pos *= (a * b)
                

            , self.neg.items())
        
    def __sum__(self, other:Terms) -> Terms:
        
        for hc in self.pos & other.neg
