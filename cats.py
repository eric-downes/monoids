from itertools import product, permutations
from typing import Any, Map

CatObj = Any
Cat = Map
CatHom = Callable[[CatObj], CatObj]

def general_aut(obj:CatObj, endo_test:Callable[[Cat, CatHom, CatObj], bool]) -> CatHom:
    assert endo_test(obj.cat, obj.id, obj)
