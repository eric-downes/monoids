import pytest

from relations import *

def test_ndrel():
    R = {frozenset(s) for s in [{1,2}, {1,2,3}, {1,3}, {3,4}]}
    ndrel = NDRelation(R)
    assert ndrel.objects == {1,2,3,4}, "objects"
    assert ndrel.images(1) == {frozenset(s) for s in [{2},{2,3},{3}]}, "images"
    assert ndrel.fiber(1) == {2,3}, "fiber"
    assert ndrel.preimage({3,4}) == {1,2}, "preimage"
    assert ndrel.extension({2,3}) == set(), "empty extension"
    assert ndrel.extension({2}) == {1}, "non-empty ext"

    R = {frozenset(s) for s in [{'orange','fruit'},
                                {'green','fruit'},
                                {'purple','vegetable'}]}
    ndrel = NDRelation(R)
    assert ndrel.images('orange') == {frozenset({'fruit'})}
    assert ndrel.images('fruit') == {frozenset(s) for s in [{'orange'}, {'green'}]}
    assert ndrel.extension({'orange','green'}) == {'fruit'}
    assert ndrel.extension({'purple'}) == {'vegetable'}
    assert ndrel.extension({'orange','purple'}) == set()
