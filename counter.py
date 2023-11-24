from __future__ import annotations
from typing import TypeVar, Callable
from collections import Counter
from collections.abc import Mapping, Hashable

T = TypeVar('T')
Binop = Callable[[T,T], T]

class ZCounter(Counter):
    '''
    Hashable Counter that maintains all nonzero counts through + and - ops
    &,|,.subtract() undef; for original see:
    https://github.com/python/cpython/blob/3.11/Lib/collections/__init__.py
    '''
    
    def _pm_(self, other:ZCounter, theop:Binop[int],
             result:ZCounter = None) -> ZCounter:
        if result is None:
            result = ZCounter()
        for elem in self.keys() | other.keys():
            if count := theop(self.get(elem, 0), other.get(elem, 0)):
                result[elem] = count
        return result

    def __hash__(self) -> int:
        return hash(tuple(self.items()))

    def __eq__(self, other):
        'True if all counts agree.'
        if hash(self) != hash(other): return False
        return super().__eq__(other)

    def __add__(self, other):
        '''Add counts from two counters.
        >>> ZCounter('abbb') + ZCounter('bcc')
        ZCounter({'b': 4, 'c': 2, 'a': 1})'''
        return self._pm_(other, op.add)

    def __sub__(self, other):
        ''' Subtract count.
        >>> ZCounter('abbbc') - ZCounter('bccd')
        ZCounter({'b': 2, 'a': 1, 'c': -1, 'd': -1})'''
        return self._pm_(other, op.sub)

    def __iadd__(self, other) -> ZCounter:
        '''Inplace add from another counter
        >>> c = Counter('abbb')
        >>> c += Counter('bcc')
        Counter({'b': 4, 'c': 2, 'a': 1})'''
        return self._pm_(other, op.add, result = self)

    def __isub__(self, other) -> ZCounter:
        '''Inplace subtract counter
        >>> c = Counter('abbbc')
        >>> c -= Counter('bccd')
        Counter({'b': 2, 'a': 1})'''
        return self._pm_(other, op.sub, result = self)
    
    def __or__(self, other):
        return NotImplemented

    def __and__(self, other):
        return NotImplemented

    def __ior__(self, other):
        return NotImplemented

    def __iand__(self, other):
        return NotImplemented


