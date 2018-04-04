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

RESULTS_PATHS = [
    os.path.join(DATA_DIR, 'crowdflower', 'results1.json'),
    os.path.join(DATA_DIR, 'crowdflower', 'results2.json'),
]

DICTIONARY_PATH = os.path.join(
    DATA_DIR, 'relational-noun-features-wordnet-only', 
    'accumulated-pruned-5000', 'dictionary'
)


def generate_top_random_task(out_path, num_top, num_rand):

    # Open the file that we'll write to
    out_file = open(out_path, 'w')

    # First, figure out what words have already been annotated
    results = t4k.CrowdflowerResults(RESULTS_PATHS)
    already_annotated = set([row['data']['token'] for row in results])

    # Now get a sorted dictionary for gigaword
    dictionary = t4k.UnigramDictionary()
    dictionary.load(DICTIONARY_PATH)

    # Get the top ``num_top`` words that haven't yet been annotated
    top_words = set()
    for token in dictionary.get_token_list():

        # Skip the UNK token
        if token is 'UNK':
            continue

        # Add words that haven't been annotated before
        if token not in already_annotated:
            top_words.add(token)

        # Stop once we have enough words
        if len(top_words) >= num_top:
            break

    # Now, get ``num_rand`` uniformly randomly selected words that have not 
    # been annotated.  Candidates include any non 'UNK' word that hasn't been 
    # annotated before.
    candidates = set(list(dictionary.get_token_list())[1:]) - already_annotated
    rand_words = set(random.sample(candidates, num_rand))

    # We're almost ready to start writing out to file.  Let's make a list of 
    # all the rows so that we can randomly shuffle them before writing to file.
    rows = []
    for word in rand_words | top_words:
        if word in rand_words and word in top_words:
            rows.append((word, 'top:rand'))
        elif word in rand_words:
            rows.append((word, 'rand'))
        else:
            rows.append((word, 'top'))
    random.shuffle(rows)

    # Write out the headings, then write the rows
    writer = csv.writer(out_file)
    writer.writerow(('token', 'source'))
    writer.writerows(rows)


def generate_top_random_to_fill_out_annotations():
    out_fname = 'task3.csv'
    out_path = os.path.join(DATA_DIR, 'crowdflower', out_fname)
    generate_top_random_task(out_path, num_top=1000, num_rand=1000)
    
    
    
if __name__ == '__main__':
    generate_top_random_to_fill_out_annotations()
