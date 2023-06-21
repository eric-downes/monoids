from typing import TypeVar, Callable, Iterator
from dataclasses import dataclass
from hashlib import blake2b
from typing import *

from numpy.typing import NDArray
from pprint import pprint
import pandas as pd
import numpy as np

T = TypeVar('T')
DoK = dict[tuple[T, T], T]

class Applicator:
    def __init__(self, fcn : Callable[[NDArray[T], int, int], NDArray[T]]):
        self.f = fcn
        self.lo = 0
    def square(self, a: NDArray[T], until: int) -> NDArray[T]:
        for i,j in iprod(range(self.lo, until), range(self.lo, until)):
            a = self.f(a, i, j)
        self.lo = until
        return a
    def extend(self, a: NDArray[T], until: int) -> NDArray[T]:
        for i,j in iprod(range(0, self.lo), range(self.lo, until)):
            a = self.f(a, i, j)
            a = self.f(a, j, i)
        return self.square(a, until)

def ident(x:Any) -> Any:
    return x
    
def iprod(itera:Iterator[T], iterb:Iterator[T]) -> Iterator[tuple[T,T]]:
    for a in itera:
        for b in iterb:
            yield a,b

def row_hash(r : np.array) -> tuple[int,...]|bytes:
    return tuple(r)
    # only works so long as row is C-contiguous...
    # return blake2b(r).digest()

def fingerprint(x):
    return hash(tuple(x))

