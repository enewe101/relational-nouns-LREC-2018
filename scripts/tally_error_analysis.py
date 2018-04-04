import t4k
import os
import itertools
from collections import defaultdict, Counter
import sys
sys.path.append('../lib')
from SETTINGS import DATA_DIR



err_fnames = [
    'analyze-errors-rand-1.tsv',
    'analyze-errors-rand-2.tsv',
    'analyze-errors-top-1.tsv',
    'analyze-errors-top-2.tsv',
]
err_paths = [os.path.join(DATA_DIR, fname) for fname in err_fnames]
err_tally_path = os.path.join(DATA_DIR, 'tallied-errors.tsv')

def tally_errors():

    lines = []
    for err_path in err_paths:
        lines += open(err_path).readlines()

    counts = Counter()
    for line in t4k.skip_blank(lines):
        line = line.strip()
        try:
            category, word, was_correct_str = line.split('\t')
        except ValueError:
            print line
            raise
        was_correct = check_was_correct(was_correct_str)
        counts[category, was_correct] += 1
    print counts
    tally_file = open(err_tally_path, 'w')

    buckets = itertools.product(['k','s','o','r','x'], [True, False])
    for category, was_correct in buckets:
        count = counts[category, was_correct]
        tally_file.write('%s\t%s\t%d\n' % (category, str(was_correct), count))
    return counts



def check_was_correct(was_correct_str):
    if was_correct_str == '1':
        return True
    elif was_correct_str == '-1':
        return False
    else:
        raise ValueError


if __name__ == '__main__':
    tally_errors()
