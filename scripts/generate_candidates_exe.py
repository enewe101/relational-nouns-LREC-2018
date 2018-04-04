import sys
sys.path.append('../lib')
import os
import t4k
from SETTINGS import DATA_DIR
import generate_candidates
import utils
from crowdflower.interpret_results import transcribe_labels
import random
import csv

CROWDFLOWER_DIR = os.path.join(DATA_DIR, 'crowdflower')
CANDIDATES_DIR = os.path.join(DATA_DIR, 'relational-nouns', 'candidates')
BEST_FEATURES_DIR = os.path.join(
    DATA_DIR, 'relational-noun-features-wordnet-only', 
    'accumulated-pruned-5000')

def do_generate_candidates1():

    # Decide the output path and the number of positive candidates to find
    t4k.ensure_exists(CANDIDATES_DIR)
    out_path = os.path.join(CANDIDATES_DIR, 'candidates1.txt')
    num_to_generate = 1000

    # Read in the seed set, which is the basis for the model that selects new 
    # candidates
    pos, neg, neut = utils.get_full_seed_set()

    # Don't keep any candidates that were already in the seed set
    exclude = pos | neg | neut

    generate_candidates.generate_candidates(
        num_to_generate, out_path, pos, neg, exclude)


def generate_uniform_random_candidates1():

    # Open a path that we want to write to 
    out_path = os.path.join(CANDIDATES_DIR, 'random_candidates1.txt')

    # Don't keep any candidates that were already in the seed set
    pos, neg, neut = utils.get_full_seed_set()
    exclude = pos | neg | neut

    generate_random_candidates(500, out_path, exclude)



def do_generate_candidates_iteration(iteration=2, kernel=None, features=None):

    # Work out the file names
    candidates_fname = 'candidates%d.txt' % iteration
    random_candidates_fname = 'random_candidates%d.txt' % iteration
    previous_results_fname = 'results%d.json' % (iteration-1)
    previous_labels_dirname = 'results%d' % (iteration-1)
    previous_task_fnames = ['task%d.csv' % j for j in range(1,iteration)]

    # Decide the output path and the number of positive candidates to find
    t4k.ensure_exists(CANDIDATES_DIR)
    out_path = os.path.join(CANDIDATES_DIR, candidates_fname)
    random_out_path = os.path.join(CANDIDATES_DIR, random_candidates_fname)
    num_to_generate = 1000

    # Read in the seed set, which is the basis for the model that selects new 
    # candidates
    pos, neg, neut = utils.get_full_seed_set()
    exclude = pos | neg | neut

    # Read in the labelled data inside the first set of results
    transcribe_labels(previous_results_fname)
    add_pos, add_neg, add_neut = utils.read_all_labels(os.path.join(
        DATA_DIR, 'relational-nouns', previous_labels_dirname))

    # Add in these nouns to the seeds
    pos.update(add_pos)
    neg.update(add_neg)
    neut.update(add_neut)

    # Don't keep any candidates that were already in the seed set or previously
    # loaded questions
    for task_fname in previous_task_fnames:
        task_path = os.path.join(CROWDFLOWER_DIR, task_fname)
        reader = csv.DictReader(open(task_path))
        exclude.update([row['token'] for row in reader])

    ## Generate the non-random candidates, enabling enrichment of positives
    #generate_candidates.generate_candidates_ordinal(
    #    num_to_generate, out_path, pos, neg, neut, exclude, kernel, features)

    # Generate random candidates, enabling exploration and model testing.
    random_candidates_path = os.path.join(
        CANDIDATES_DIR, random_candidates_fname)
    generate_candidates.generate_random_candidates(2000, random_out_path)


if __name__ == '__main__':
    # do_generate_candidates1()
    #generate_uniform_random_candidates1()
    do_generate_candidates_iteration(2)
