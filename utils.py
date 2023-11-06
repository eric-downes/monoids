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

def first_true_idx(x:np.array) -> None|int:
    i = x.view(bool).argmax() // x.itemsize
    return i if i or x[i] else None

def findall(x:np.array) -> list[int]:
    return np.nonzero(x)

def iprod(itera:Iterator[T], iterb:Iterator[T]) -> Iterator[tuple[T,T]]:
    for a in itera:
        for b in iterb:
            yield a,b

def row_hash(r : np.array) -> tuple[int,...]|bytes:
    return tuple(r)    
    # works so long as row is C-contiguous; otherwise consider tuple(r)
    # return blake2b(r).digest()

    
