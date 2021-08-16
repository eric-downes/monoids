from row_compose import *

def test_composition():
    assert Endo([1,0,2]) * Endo([2,1,0]) == Endo([2,0,1])
    assert Endo([1,0,3,2]) * Endo([3,0,1,2]) * Endo([1,1,2,3]) == Endo([1,1,0,3])
    
