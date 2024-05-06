from typing import Hashable, TypeVar, Callable
import functools
import weakref

H = TypeVar('H', bound = Hashable)
K = TypeVar('K', bound = Hashable)
Binop = Callable[[H,H], H]
GenDict = dict[H, tuple[H,...]]

def preimage(d:dict[H,K]) -> dict[K, set[H]]:
    p = {}
    for k,v in d.items():
        p.setdefault(v, set()).add(k)
    return p

def weak_lru(maxsize:int = 128, typed:bool = False):
    'LRU Cache decorator that keeps a weak reference to "self"'
    def wrapper(func: Callable):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)
        return inner
    return wrapper


'''
def attempt_closure(genereators:GenDict[H],
                    binop:Binop[H],
                    max_time:int = 22 * 60) -> tuple[GenDict[H], bool]:
    t0 = arrow.now().timestamp()
    closure : GenDict()
    stack = generators.copy()
    while stack and arrow.now().timestamp() - t0 < max_time:
        elem, expnd = stack.popitem()
        closure[elem] = expnd
        for elem, expnd in closure.items():
            new = binop(elem, elem)
            nexpnd = expnd
            while new not in closure | stack:
                stack[new] = (nexpnd := expnd + nexpnd)
                newk = 
'''                
            
        
