from typing import Collection, TypeVar, Callable

Object = object
X = TypeVar('X', bound = Object)
Y = TypeVar('Y', bound = Object)
Morphism = Callable
ExpObj = Collection[Morphism[[X], Y]]
EvalMap = Morphism[[ExpObj[X,Y], X], Y]
ExpObjUProp = Mapping[tuple[EvalMap[X,Y],
                            Morphism[[Z,Y], Y]],
                      Morphism[[Z,Y], [ExpObj[X,Y], X], Y]] 

def evalmap(fcn:Morphism[X,Y], val:X) -> Y:
    return fcn(val)

'''
ev:Morphism
f:Morphism

ZxY = f.dom()
X = f.cod()
X2YxY = v.dom()
assert X == v.cod(), f'cod(eval) {v} =/= cod(f) {f}'
assert X2YxY.proj(-1) == ZxY.proj(-1), f'products inconsistent'
X2Y = X2YxY.proj(0)
'''
