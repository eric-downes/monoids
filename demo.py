from presentations import *
from monoids import *

VERBOSE = '--verbose' in sys.argv if __name__ == '__main__' else False
OCTOS = '--octonions' in sys.argv if __name__ == '__main__' else False

def octonion_check(verbose = VERBOSE) -> RowMonoid:
    # https://ncatlab.org/nlab/show/octonion
    octos = adjoin_negatives(
        np.array([[0,1,2,3,4,5,6,7], \
                  [1,8,3,-2,5,-4,-7,6],\
                  [2,-3,8,1,6,7,-4,-5],\
                  [3,2,-1,8,7,-6,5,-4],\
                  [4,-5,-6,-7,8,1,2,3],\
                  [5,4,-7,6,-1,8,-3,2],\
                  [6,7,4,-5,-2,3,8,-1],\
                  [7,-6,5,4,-3,-2,1,8]]) )
    fil = 'oct_monoid.csv'
    print(f'row_monoid(magma) demo using octonion magma; saving to {fil}')
    labels = ['1','i','j','ij','l','il','jl','(ij)l'] + \
        ['-1','-i','-j','-ij','-l','-il','-jl','-(ij)l']
    data = row_monoid(octos, labels = labels, verbose = verbose)
    is_group_pres_valid(data.monoid_table, pres_Q128)
    is_group_pres_valid(data.monoid_table, pres_xs128)    
    return data

def rps_check(verbose = VERBOSE) -> RowMonoid:
    fil = 'rps_monoid.csv'
    print(f'row_monoid(magma) demo using RPS magma; saving to {fil}')
    rps_magma = np.array([[0,1,0], [1,1,2], [0,2,2]])
    return row_monoid(rps_magma, verbose=verbose)

if __name__ == '__main__':
    if OCTOS:
        data = octonion_check()
    else:
        data = rps_check()
    pd.DataFrame(data.monoid_table).to_csv(fil, index=False, header=False)
    print('\n\n\nresults!')
    print(f'\n\noriginal magma:\n{data.row_closure[:data.magma_order]}')
    print(f'\n\nrow monoid:\n{data.monoid_table}')
    for r,i in data.row_map.items():
        n = data.labels[i]
        name = n.replace('-','')
        if n.count('-') % 2:
            name = '-' + name
        s = f'[{r}]'.replace(',','')
        print(f'g({i}) = {name}: {s}')

'''sanity check:
def conj(x:int)->int:
    if x % 8:
                 return A[8,x]
             else:
                 return x
         quaternions = [0,1,2,3,8,9,10,11]
         A = octos
         for p in quaternions:
             for q in quaternions:
                assert A[q, A[l,p]] == A[l, A[conj(q),p]], f'1: (p,q,l)=({p},{q},{l})'
                assert A[A[q,l], p] == A[A[q,conj(p)], l], f'2: (p,q,l)=({p},{q},{l})'
                assert A[A[l,q],A[p,l]] == A[8, conj(A[q,p])], f'3: (p,q,l)=({p},{q},{l})'
'''
