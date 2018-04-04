import os
import sys
sys.path.append('../lib')
from SETTINGS import DATA_DIR

if __name__ == '__main__':

    suffixes_path = os.path.join(DATA_DIR, 'suffixes.txt')
    suffixes = open(suffixes_path).read().split('\n')
    pruned_suffixes_path = os.path.join(DATA_DIR, 'suffixes_pruned.txt')
    pruned_suffixes_f = open(pruned_suffixes_path, 'w')

    pruned_suffixes = set(suffixes)

    for i in range(len(suffixes)):
        for j in range(i+1, len(suffixes)):
            suff_i, suff_j = suffixes[i], suffixes[j]
            if suff_i.endswith(suff_j):
                print 'removing %s because already have %s' % (suff_i, suff_j)
                pruned_suffixes.remove(suff_i)
            elif suff_j.endswith(suff_i):
                print 'removing %s because already have %s' % (suff_j, suff_i)
                pruned_suffixes.remove(suff_j)

    pruned_suffixes_f.write('\n'.join(pruned_suffixes))
    

    


   
