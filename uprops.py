'''

Goal
0. consider using relations to define as per Freyd; bootstrap from Rel class (tdb)\
1. specify category in termns of C0, C1, C2
   ? `assert ...` properties about commuting diagrams manually
   ? iterator for commutative dias using C2, C1 & note brute force
   ?? assert properties of commuting diagrams using a Diagram class
2. use `assert ...` to verify properties:
   ? left adjoints preserve epis (?? colimits)
   ? right adjoints preserve monos (?? limits)
   ? is_subctageory, with wide, etc. detectors for subsets of C0 x C1 x C2
3. calculate things
   ? calc. comma category over subcat

Qs:

-- are commuting diagrams isos or equalities? assume eq until shown diff


use pydantic infra to validate the uprop
- make a generic class
- class has public and under methods that user must provide


'''
from typing import TypeVar, Callable

from pydantic import BaseModel
from annotated_types import *

# basic types

class Object(BaseModel):
    def __init__(self, cat:Category): pass

D, C = TypeVar('D', 'C', bound = Object)
MorphismType = Callable[[D], C]

class Morphism(BaseModel):
    def __init__(self, cat:Category, dom:Object, cod:Object): pass
    @composable
    def __call__(self, *args, **kwargs): pass

class Category(BaseModel):
    def __init__(self,
                 objects:Iterator[Objects],
                 arrows:Iterator[Morphism],
                 paths:Mapping[tuple[Morphhism], Morphism]): pass
    def hom(self, dom:Object, cod:Object) -> set[Morphism]: pass

'''
typing.Mapping is an object which defines __getitem__,__len__,__iter__
another method needs to return it
'''        
    
# derived types

def composable(f:Mthd) -> Wrpt:
    @ft.wraps(f)
    def wrapped(self, ) -> None:
            self._comp_(f, *args, **kwargs)
        wrapped._protocol_step_ = True
        return wrapped
    return inner_wrap

def mm_callback(do_next:str = '') -> Callable[[Mthd], Wrpt]:
    def inner_wrap(f:Mthd) -> Wrpt:
        @ft.wraps(f)
        def wrapped(self:Protocol, pyld:Payload = {}) -> None:
            self._wrap_protocol_step_(f, pyld, do_next)
        wrapped._protocol_step_ = True
        return wrapped
    return inner_wrap

    
        
def mm_callback(do_next:str = '') -> Callable[[Mthd], Wrpt]:

    return inner_wrap



class ExpObj(Object):
    def __init__(self, base:Object, exp:Object) -> Object: pass
    @composable
    def __callable__(self,
        
    
    

