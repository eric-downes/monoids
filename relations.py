from collections import defaultdict
from functools import reduce
from typing import *
import warnings


'faster intersections'

class Empty(Exception): pass

def fcap(*args) -> frozenset:
    return frozenset(cap(*args))

def cap(*args) -> set:
    try: return reduce(_intersect, args[1:], args[0].copy()) 
    except Empty: return set()

def _intersect(x:set, y:set) -> set:
    x.intersection_update(y)
    if x: return x
    raise Empty() 
    



'Tai-Danae Bradley'

# this is TDB's system w/ same syntax
# https://arxiv.org/abs/2004.05631
# if we inherit from nx.DiGraph can -> graph & use connected_components to find formal concepts
# https://networkx.github.io/documentation/networkx-1.9.1/reference/generated/networkx.algorithms.components.connected.connected_components.html

class PreConcept:
    def __init__(self, rels: Set[Tuple[Any, Any]]):
        a = defaultdict(lambda : set())
        b = defaultdict(lambda : set())
        for l, r in rels:
            a[l].add(r)
            b[r].add(l)
        self.a = a
        self.b = b
    def ext(self, sub:set, to_right:bool = True, U:bool=False) -> set:
        supdom = self.a if to_right else self.b
        assert(sub.issubset(supdom.keys()))
        f = set.union if U else cap
        return f( *{supdom[x] for x in sub} )
    def f(self, A:set, U:bool=False) -> set:
        return self.ext(A, U=U)
    def g(self, B:set, U:bool=False) -> set:
        return self.ext(B, to_right=False, U=U)
    def formal_concepts(self):
        fc = set()
        for A in powerlist(self.a.keys()):
            B = f(A)
            if not B: continue
            if A == g(B): fc.add(frozenset({A,B}))
        return fc    
    
class PwrSetRelation:
    # a possible extension of TDB Thesis Ch 1
    # PwrSetRel : 2^X --> 2 think hyperedges...
    def __init__(self, rels:Iterable[set]):
        objs = set()
        emaps = defaultdict(set)
        dmaps = defaultdict(set)
        imaps = dict() #{id(None): set()} # used in .image() below
        for s in map(frozenset, rels):
            d = len(s)
            if d <= 1: continue
            sid = id(s)
            imaps[sid] = s
            objs.update(s)
            for e in s:
                emaps[e].add(sid)
                dmaps[d].add(sid)
        self.objects = frozenset(objs)
        self._emaps = emaps # object -> set of sid's
        self._dmaps = dmaps # rank -> set of sid's
        self._imaps = imaps # sid -> set of objects
        
    # TDB is using set-valued functions so ea image is also a set
    # each element of these is the image of a morphism in Set
    # empty sets commonly result from fiber and preimage with U=False,
    # so using cap() instead of set.intersection() for speed improvement
    def images(self, e, rm_self:bool = True):
        rm = {e} if rm_self else set()
        return {self._imaps[sid] - rm for sid in self._emaps[e]}
        
    def fiber(self, e, U:bool = True, rm_self:bool = True):
        f = frozenset.union if U else fcap
        rm = {e} if rm_self else set()
        return f( *self.images(e, rm_self=False) ) - rm

    def preimage(self, sub:set, U:bool = True, rm_self:bool = True): 
        sub = sub & self.objects # otherwise if U risk false {} preimage
        f = frozenset.union if U else fcap
        rm = sub if rm_self else set()
        return f( *{self.fiber(e, U, False) for e in sub} ) - rm

    # TDB's "extension f(A) of a(x)" is preimage(A, U = False)
    def extension(self, sub:set, rm_self:bool = True): 
        return self.preimage(sub, U=False, rm_self = rm_self)

    def relations(self):
        return self._imaps.values()

    def export_as(self, typ):
        if typ == pd.DataFrame: pass # export table / biadjacency matrix
        if typ == nx.Graph: pass # export hypergraph as bigraph




def powerlist(seq:list):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for sub in powerlist(seq[1:]):
            yield [seq[0]] + sub
            yield sub

def concept_generator(self, seq:list):
    if len(seq) > 1:
        for sub in concept_search(seq[1:]):
            x = [seq[0]] + sub
            if self.extension(set(x))
            yield [seq[0]] + sub
            yield sub

'KKM binary relations'
# 

            
class BinRel(NDRelation): pass
class SelfBinRel(BinRel): pass
class Map(BinRel):pass

def is_reflexive(r:SelfBinRel) -> bool: pass
def is_symmetric(r:SelfBinRel) -> bool: pass
def is_antisymmetric(r:SelfBinRel) -> bool: pass
def is_transitive(r:SelfBinRel) -> bool: pass
def is_right_unique(r:BinRel) -> bool: pass
def is_left_unique(r:BinRel) -> bool: pass
def is_right_total(r:BinRel) -> bool: pass
def is_left_total(r:BinRel) -> bool: pass
def is_map(r:BinRel) -> bool: pass
def is_inj(r:Map) -> bool: pass
def is_surj(r:Map) -> bool: pass
def is_constant(r:Map) -> bool: pass

def restriction(r:Map, sub:set) -> Map: pass




    
