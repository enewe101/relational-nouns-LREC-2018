import sys
sys.path.append('../lib')
import utils
from SETTINGS import DATA_DIR
import os
import random
import csv
import t4k

CANDIDATES_DIR = os.path.join(DATA_DIR, 'relational-nouns', 'candidates')
CROWDFLOWER_DIR = os.path.join(DATA_DIR, 'crowdflower')

def make_crowdflower_csv(iteration=2):

    # Seed randomness for reproducibility
    random.seed(0)

    # Open a file at which to write the csv file
    t4k.ensure_exists(CROWDFLOWER_DIR)
    task_fname = 'task%d.csv' % iteration
    csv_path = os.path.join(CROWDFLOWER_DIR, task_fname)
    csv_f = open(csv_path, 'w')

    # First read the scored candidates
    pos_common_candidates = []
    neg_common_candidates = []
    neut_common_candidates = []
    candidates_fname = 'candidates%d.txt' % iteration
    for line in open(os.path.join(CANDIDATES_DIR, candidates_fname)):
        token, class_ = line.split('\t')[:2]
        if class_ == '+':
            pos_common_candidates.append(token)
        elif class_ == '-':
            neg_common_candidates.append(token)
        elif class_ == '0':
            neut_common_candidates.append(token)
        else:
            raise ValueError('Unexpected classification character: %s' % class_)

    # We'll only keep the first 1000 negatives.
    positives = pos_common_candidates[:1000]
    neutrals = neut_common_candidates[:1000]
    negatives = neg_common_candidates[:1000]

    #num_neut = min(250, len(neut_common_candidates))
    #neg_common_candidates = neg_common_candidates[:500-num_neut]
    #neut_common_candidates = neut_common_candidates[:num_neut]

    # Next read the random candidates
    random_candidates_fname = 'random_candidates%d.txt' % iteration
    random_candidates_path = os.path.join(
        CANDIDATES_DIR, random_candidates_fname)
    random_candidates = open(random_candidates_path).read().strip().split('\n')
    random_candidates[:2000]

    # Collect all the candidate words together and elminate dupes
    all_candidates = set(positives + negatives + neutrals + random_candidates)

    # Now keep track of why each word was included (i.e. was it a word labelled
    # by the classifier-to-date as positive? negative? or was it randomly 
    # sampled?  Note that a word could be both randomly drawn and labelled.
    positives = set(positives)
    negatives = set(negatives)
    neutrals = set(neutrals)
    random_candidates = set(random_candidates)
    sourced_candidates = []
    for candidate in all_candidates:
        sources = []
        if candidate in pos_common_candidates:
            sources.append('pos2')
        if candidate in neg_common_candidates:
            sources.append('neg2')
        if candidate in neut_common_candidates:
            sources.append('neut2')
        if candidate in random_candidates:
            sources.append('rand2')
        sourced_candidates.append((candidate, ':'.join(sources)))

    # randomize the ordering
    random.shuffle(sourced_candidates)

    # Write a csv file with the candidate words in it
    writer = csv.writer(csv_f)
    writer.writerow(['token', 'source'])
    writer.writerows(sourced_candidates)



if __name__ == '__main__':
    #make_crowdflower_csv(1)
    make_crowdflower_csv(2)
