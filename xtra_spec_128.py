import math
from typing import *
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from magmas import submagma
from monoids import group_orbit

'types'

Order = int
Elem = int
G : NDArray[int]
DISQUAL : set[tuple[None|Elem, ...]] = set()
LOOKUP : dict[Order, set[Elem]] = {}
ORBITS : dict[Elem, tuple[Elem]] = {}

@dataclass
class Rule:
    order: int
    passes: Callable[[int, ...], bool]

'general recursive search'

def recurse(generators:list[int],
            known_subgroups:list[set[int]],
            disqual:set[tuple[int]]) -> bool:
    depth = len(generators)
    if depth == len(RULES):
        return True
    rule = RULES[depth]
    for g in LOOKUP[rule.order]:
        # attempt to disqualify candidate point
        sig = tuple(generators + [g])
        if g not in known_subgroups and \
           sig not in disqual and \
           rule.passes(*sig):
            # found a candidate!
            # add point to the generators & update excluded
            disqual.add(sig) # havent failed yet but will never visit this again...
            generators += [g]
            known_subgroups += [known_subgroups[-1].union(submagma(G, generators)[1])]
            if recurse(generators, known_subgroups, disqual):
                return True # got to the end
        # else, e.g. you popped back up, undo last step
        generators.pop()
        known_subgroups.pop()
    # for loop exhausted; fail up
    print(f'failed with {sig}')
    return False

def validate_presentation():
    while True:
        stack = []
        subs = [set()]
        disqual = set()
        if recurse(stack, subs, disqual):
            assert submagma(G, subs[-1])[0] == G
            print(f'Success!!! generators = ({stack})')
        else:
            print(f'Failure :(')

            
'extra-special-128 specific stuff'

def a_passes(*gs:int) -> bool:
    return True

def b_passes(*gs:int) -> bool:
    a,b = gs[:2]
    asq = G[a,a]
    ab = G[a,b]
    if (G[b,b]   == asq and
        G[ab,ab] == asq): return True

def c_passes(*gs:int) -> bool:
    a,b,c = gs[:3]
    if (G[c,c] == G[a,a] and 
        G[a,c] == G[c,a] and
        G[b,c] == G[c,b]): return True

def d_passes(*gs:int) -> bool:
    a,b,c,d = gs[:4]
    csq = G[c,c]
    cd = G[c,d]
    if (G[cd,cd] == csq and
        G[d,d]   == csq and
        G[a,d]   == G[d,a]): return True
    
def e_passes(*gs:int) -> bool:
    a,b,c,d,e = gs[:5]
    for x in (a,b,c,d):
        if G[e,x] != G[x,e]: return False
    return True

def f_passes(*gs:int) -> bool:
    a,b,c,d,e,f = gs[:6]
    for x in (a,b,c,d):
        if G[e,x] != G[x,e]: return False
    ef = G[e,f]
    return G[a,a] == G[ef,ef]

def xs128_setup(filename:str = 'oct_monoid.csv'):
    global G, ORBITS, LOOKUP, RULES
    G = pd.read_csv(filename, index = None, header = None).to_numpy()
    RULES = [Rule(4, a_passes),
             Rule(4, b_passes),
             Rule(4, c_passes),
             Rule(4, d_passes),
             Rule(2, e_passes),
             Rule(2, f_passes)]
    for g in range(len(G)):
        orb = tuple(group_orbit(g, G))
        ORBITS[g] = orb
        LOOKUP.setdefault(len(orb), set()).add(g)

        
    
    
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
