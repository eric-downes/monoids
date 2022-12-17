from presentations import *
from monoids import *

if __name__ == '__main__':

    if '--octonions' in sys.argv:
        # https://en.wikipedia.org/wiki/Octonion#Definition # with -e_0 -> 8
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
        data = row_monoid(octos)
        is_group_rep_valid(data.monoid_table, rep_Q128)
    else:
        fil = 'rps_monoid.csv'
        print(f'row_monoid(magma) demo using RPS magma; saving to {fil}')
        rps_magma = np.array([[0,1,0], [1,1,2], [0,2,2]])
        data = row_monoid(rps_magma)
    DataFrame(data.monoid_table).to_csv(fil, index=False, header=False)
    print('\n\n\nresults!')
    print(f'\n\noriginal magma:\n{data.row_closure[:data.magma_order]}')
    print(f'\n\nrow monoid:\n{data.monoid_table}')
    print('\n\nmapping: \n')
    pprint(data.row_map)
