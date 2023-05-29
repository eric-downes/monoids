from monoids import *

def test_dihedral():
    C2 = cyclic_group(2)
    C3 = cyclic_group(3)
    D6 = np.array([[0,1,2,3,4,5],
                   [1,2,0,5,3,4],
                   [2,0,1,4,5,3],
                   [3,4,5,0,1,2],
                   [4,5,3,2,0,1],
                   [5,3,4,1,2,0]])
    phi = np.array([[0,1,2],[0,2,1]])
    C23 = semidirect_product(C2, C3, phi)
    assert C23 == D6



    
