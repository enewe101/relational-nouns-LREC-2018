import sys
sys.path.append('..')
import os
from collections import defaultdict
import t4k
from SETTINGS import DATA_DIR

USUALLY_RELATIONAL = '+'
OCCASIONALLY_RELATIONAL = '0'
NEVER_RELATIONAL = '-'

CROWDFLOWER_DIR = os.path.join(DATA_DIR, 'crowdflower')
PARTICIPANT_RESULTS_PATHS = [
    os.path.join(CROWDFLOWER_DIR, 'results1.json'),
    os.path.join(CROWDFLOWER_DIR, 'results2.json')
]
TRIPLE_EXPERT_RESULTS_PATH = os.path.join(
    CROWDFLOWER_DIR, 'results-expert.json')
EXPERT_RESULTS_PATH = os.path.join(CROWDFLOWER_DIR, 'results3-expert-annot.tsv')
EXPERT_TASK_PATH = os.path.join(CROWDFLOWER_DIR, 'results3-expert.csv')
FEATURES_DIR = os.path.join(
    DATA_DIR, 'relational-noun-features-wordnet-only', 'accumulated-pruned-5000'
)
DICTIONARY_DIR = os.path.join(FEATURES_DIR, 'dictionary')


def interpret_annotations(crowdflower_results, policy='majority'):
    """
    Given a CrowdflowerResults object (obtained from passing a crowdflower 
    json report into t4k.CrowdFlowerResult), interpret the label for given 
    words (either "Usually Relational", "Occasionally Relational", or "Almost
    Never Relational".
    """

    word_labels = {}

    for result in crowdflower_results:

        # Work out the word, its sampling source(s), and its label
        word = result['data']['token']
        label = get_label(result)
        word_labels[word] = label

    return word_labels



def finalize_noncrowdflower_data_rows():
    """
    This formats the annotations that I did directly in a file.  The "sources"
    are stored in a different file, so both files are read and the info is 
    merged.
    """
    non_cf_results_path = os.path.join(
        CROWDFLOWER_DIR, 'results3-expert-annot.tsv')
    non_cf_task_path = os.path.join(
        CROWDFLOWER_DIR, 'results3-expert.csv')

    top_words = set(get_top_words())
    sources = dict([line.strip().split(',') for line in open(non_cf_task_path)])
    labels =  dict([
        line.strip().split('\t') for line in open(non_cf_results_path)])

    finalized_rows = {}
    for token in labels:
        finalized_rows[token] = (
            token, sources[token], '1-expert', labels[token])

    return finalized_rows



def finalize_crowdflower_data_rows(crowdflower_results, annotator):
    """
    Given a CrowdflowerResults object (obtained from passing a crowdflower 
    json report into t4k.CrowdFlowerResult), interpret the label for given 
    words (either "Usually Relational", "Occasionally Relational", or "Almost
    Never Relational".
    """

    top_words = set(get_top_words())
    finalized_rows = {}
    for result in crowdflower_results:

        # Work out the word, its sampling source(s), and its label
        token = result['data']['token']
        label = get_label(result)

        # Work out the sources
        sources = result['data']['source']
        if token in top_words:
            sources += ':top'

        finalized_rows[token] = (token, sources, annotator, label)

    return finalized_rows


def get_top_words():
    """
    Get the k most common words in gigawords for which all words were annotated
    and k is as big as it can be.
    """
    all_annotated_words = get_all_annotated_words()
    dictionary = t4k.UnigramDictionary()
    dictionary.load(DICTIONARY_DIR)
    top_words = []
    for i, token in enumerate(dictionary.get_token_list()):
        if token == 'UNK':
            continue
        if token in all_annotated_words:
            top_words.append(token)
        else:
            break

    return top_words


def get_all_annotated_words():

    # First open the participant results
    results = t4k.CrowdflowerResults(PARTICIPANT_RESULTS_PATHS)
    participant_words = set([row['data']['token'] for row in results])

    # Next get the expert crowdflower results
    results = t4k.CrowdflowerResults(TRIPLE_EXPERT_RESULTS_PATH)
    expert_crowdflower_words = set([row['data']['token'] for row in results])

    # Finally get the expert results not annotated in crowdflower
    expert_words = set([
        line.split(',')[0] for line in open(EXPERT_TASK_PATH)])

    return participant_words | expert_words | expert_crowdflower_words



def interpret_annotations_by_source(crowdflower_results):
    """
    Given a CrowdflowerResults object (obtained from passing a crowdflower 
    json report into t4k.CrowdFlowerResult), interpret the label for given 
    words (either "Usually Relational", "Occasionally Relational", or "Almost
    Never Relational".
    """

    word_labels = {}

    for result in crowdflower_results:

        # Work out the word, its sampling source(s), and its label
        word = result['data']['token']
        sources =  result['data']['source'].split(':')

        # Check if the word is in the top X sources
        label = get_label(result)

        # Store accordingly
        for source in sources:
            word_labels[source][word] = label

    return word_labels



def transcribe_labels(results_fname):
    """
    Read the results stored under the crowdflower subdir of the data dir
    named by results_fname, and interpret the annotations into labels.  Then,
    write those labels out into the relational-nouns subdir of the data dir
    within a subsubdir that has the same name as the results_fname.  Record teh
    results in three separate files -- based on what source the word was drawn
    from.
    """

    # Work out paths
    results_path = os.path.join(DATA_DIR, 'crowdflower', results_fname)
    result_fname_no_ext = results_fname.rsplit('.', 1)[0]
    labels_dir = os.path.join(DATA_DIR, 'relational-nouns', result_fname_no_ext)
    t4k.ensure_exists(labels_dir)

    # Read in the results, and interpret labels
    crowdflower_results = t4k.CrowdflowerResults(
        results_path, lambda x:x['data']['token'])
    word_labels = interpret_annotations_by_source(crowdflower_results)

    # Write labels to disk, with words coming from different sources put into
    # different files.
    for source in word_labels:
        source_label_file = open(os.path.join(labels_dir, source + '.tsv'), 'w')
        for word, label in word_labels[source].iteritems():
            source_label_file.write(word + '\t' + label + '\n')


def get_label(result):
    """
    Gets the correct label based on the annotations of a single word.  If a
    strict majority of annotations indicate that the word is either "usually
    relational" or "almost never relational", then the corresponding label is
    taken.  Otherwise, for ambiguous cases, "occasionally relational" is taken.
    """
    # If there is a clear majority for "usually relational" or "almost
    # never relational" then take the correponding label, otherwise we
    # default to "occasionally relational".
    label = OCCASIONALLY_RELATIONAL

    try:
        mode = result['mode']['response']
    except KeyError:
        mode = result['mode']['is_relational']

    if len(mode) < 2:
        if mode[0] == 'usually relational':
            label = USUALLY_RELATIONAL
        elif mode[0] == 'almost never relational':
            label = NEVER_RELATIONAL

    return label


