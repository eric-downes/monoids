import math
from typing import *
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from magmas import submagma, invert
from monoids import group_orbit, is_group

'types'

Order = int
Elem = int

@dataclass
class Rule:
    order: int
    passes: Callable[[int, ...], bool]

class Searcher:
    def __init__(self, G : NDArray[int], rules : list[Rule]):
        assert is_group(G)
        assert 0 < len(rules) <= len(G) // 2
        self.G = G
        self.rules = rules
        self.rank = len(rules)
        self.look = {}
        self.orbs = {}
        for g in range(len(G)):
            orb = tuple(group_orbit(g, G))
            self.orbs[g] = orb
            self.look.setdefault(len(orb), set()).add(g)
        self.gens = []
        self.subs = []
        self.excl = [{0}]
        self.disq = set()

    def generated(self, adjoining:int = None) -> tuple[NDArray[int], list[int]]:
        adj = {0, adjoining} if adjoining else {0}
        return submagma(self.G, set(self.gens).union(adj))
        
    def excluded(self, g:int, rule:Rule) -> bool|None:
        sig = tuple(self.gens) + (g,)
        if (g in self.excl[-1] or
            sig in self.disq or
            not rule.passes(* sig)):
            return True

    def search(self) -> bool:
        if len(self.gens) > self.rank:
            return True
        depth = len(self.gens)
        print(f'at depth = {depth}')
        rule = self.rules[len(self.gens)]
        order = rule.order
        candidates = self.look[order]
        for g in candidates:
            
            if self.excluded(g, rule):
                continue
            print(f'{len(self.gens) + 1}:{g} not excluded outright')
            
            H, helems = self.generated(adjoining = g)
            inh = invert(helems)
            if len(group_orbit(inh[g], H)) != order:
                self.disq.add(tuple(self.gens) + (g,))
                continue
            print(f'{len(self.gens)}:|{g}| correct in subgroup')
            
            self.gens.append(g)
            self.subs.append(set(helems))
            self.excl.append(self.excl[-1] | self.subs[-1])
            if self.search():
                return True
            
            self.disq.add(tuple(self.gens))
            self.gens.pop()
            self.subs.pop()
            self.excl.pop()
            print(f'{len(self.gens) + 1}:{g} didnt lead anywhere')
            
        return False

    
'extra-special-128 specific stuff'

def pto4(a:int) -> bool:
    asq = G[a,a]
    return G[asq,asq] == 0

def a_passes(*gs:int) -> bool:
    return pto4(gs[0])

def b_passes(*gs:int) -> bool:
    a,b = gs[:2]
    for x in (a,b):
        if not pto4(x):
            return False
    asq = G[a,a]
    ab = G[a,b]
    if (G[b,b]   == asq and
        G[ab,ab] == asq):
        return True

def c_passes(*gs:int) -> bool:
    a,b,c = gs[:3]
    for x in (a,b,c):
        if not pto4(x):
            return False
    if (G[c,c] == G[a,a] and 
        G[a,c] == G[c,a] and
        G[b,c] == G[c,b]):
        return True

def d_passes(*gs:int) -> bool:
    a,b,c,d = gs[:4]
    for x in (a,b,c,d):
        if not pto4(x):
            return False
    csq = G[c,c]
    cd = G[c,d]
    if (G[cd,cd] == csq and
        G[d,d]   == csq and
        G[a,d]   == G[d,a]):
        return True
    
def e_passes(*gs:int) -> bool:
    a,b,c,d,e = gs[:5]
    for x in (a,b,c,d):
        if not pto4(x):
            return False
        if G[e,x] != G[x,e]:
            return False
    return True

def f_passes(*gs:int) -> bool:
    a,b,c,d,e,f = gs[:6]
    for x in (a,b,c,d):
        if not pto4(x):
            return False       
        if G[e,x] != G[x,e]:
            return False
    ef = G[e,f]
    return G[a,a] == G[ef,ef]


if __name__ == '__main__':
    G = pd.read_csv('oct_monoid.csv', header = None).to_numpy()
    rules = [Rule(4, a_passes),
             Rule(4, b_passes),
             Rule(4, c_passes),
             Rule(4, d_passes),
             Rule(2, e_passes),
             Rule(2, f_passes)]
    searcher = Searcher(G, rules)
    searcher.search()
    H, hmap = searcher.generated()

    
'''
@dataclass
class Point:
    value: int
    group: NDArray[int]
    def __mul__(self, other) -> Point:
        return Point(self.group[self.value, other.value], self.group)
    def __eq__(self, other) -> bool:
        return self.value == other.value

def in_group(G: NDArray[int], g: int) -> bool:
    return 0 < g < len(G)

def order_n(G: NDArray[int], g: int, n:int) -> bool:
    return len(group_orbit(g, G)) == n

def xx_eq(G: NDArray[int], i:int, isq:int) -> bool:
    return G[i, i] == isq

def xx_eq_yy(G: NDArray[int], i:int, j:int) -> bool:
    return xx_eq(G, i, G[j,j])

'''
