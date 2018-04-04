import random
import os
import sys
sys.path.append('..')
import cjson
import numpy as np
import t4k
from collections import defaultdict
from SETTINGS import (
    WORDNET_INDEX_PATH,
    DATA_DIR, SEED_PATH
    #TRAIN_PATH, TEST_PATH, SEED_PATH,
)


def find_threshold(classifier, X, Y, positive=set([1]), negative=set([-1])):
    scored = zip(classifier.score(X), Y)
    return calculate_best_score(scored, positive, negative)


def calculate_best_score(scored_typed, positive=set([1]), negative=set([-1])):
    """
    This function helps to convert from a scoring function to a classification
    function.  Given some function which provides scores to items that are
    either "positive" or "negative", find the best threshold score that gives
    the highest f1 score, when used to label any items whose score is higher as
    "positive" and lower as "negative"

    INPUTS
        ``scored_typed`` should be a list of tuples of scored items, where the
        first element of the tuple is the score, and the second element is the
        true class of the item, which should be 'pos' or 'neg'

    OUTPUTS 
        ``(best_f1, threshold)`` where best_metric is the best value for
        the chosen metric, achieved when threshold is used to label items
        according to their assigned scores.
    """

    # Tally up the number of positive and negative examples having given scores
    # This simplifies finding the best threshold if there are repeated scores.
    labels_by_score = defaultdict(lambda: {1:0, -1:0})
    for score, label in scored_typed:
        binary_label = binarize(label, positive, negative)
        labels_by_score[score][binary_label] += 1

    # Get a sorted list of the *unique* scores
    sorted_scores = sorted(labels_by_score.keys())

    # Start with the threshold lower than the minimum score, then gradually 
    # raise it, keeping track of the performance metric
    num_pos = sum([v[1] for v in labels_by_score.values()])
    num_neg = sum([v[-1] for v in labels_by_score.values()])

    # We start with the threshold below the minimum score, so that all items
    # are considered '1's. The initial number correct then is just the number
    # of positives in total
    true_pos = num_pos
    false_pos = num_neg
    initial_f1 = f1(true_pos, false_pos, num_pos)
    initial_pointer = -1

    maximum = t4k.Max()
    maximum.add(initial_f1, initial_pointer)
    for pointer, score in enumerate(sorted_scores):

        # Determine the effect of moving the threshold just *above* this score
        # Any positives at this score are now mis-labelled as negatives
        true_pos -= labels_by_score[score][1]

        # Any negatives at this score are now correctly labelled as negatives
        false_pos -= labels_by_score[score][-1]

        # Recalculate the F1 score now.
        this_f1 = f1(true_pos, false_pos, num_pos)

        # If this is an improvement over the previous best value, keep it
        maximum.add(this_f1, pointer)

    best_f1, best_pointer = maximum.get()
    if best_pointer == -1:
        threshold = min(sorted_scores) - 1
    elif best_pointer == len(sorted_scores)-1:
        threshold = max(sorted_scores) + 1
    else:
        threshold = (
            sorted_scores[best_pointer] + sorted_scores[best_pointer+1]) / 2.0

    return best_f1, threshold


def binarize(label, positive=set([1]), negative=set([-1])):
    """
    Interprets ``label`` as either positive or negative, based on which set
    it is found in.  If the label is positive, return 1, if the label is
    negative, return -1.  If the label is not found in either set, it's a
    ValueError.
    """
    if label in positive:
        return 1
    if label in negative:
        return -1
    raise ValueError('Unexpected label: %s' % label)


def f1(true_pos, false_pos, num_pos):

    # Calculate recall (handle case where denominator is zero)
    if num_pos == 0:
        recall = 1.0
    else:
        recall = true_pos / float(num_pos)

    # Calculate precision (handle case where denominator is zero)
    if true_pos + false_pos == 0:
        precision = 1.0
    else:
        precision = true_pos / float(true_pos + false_pos)

    # Calculate F1 (handle the case where denominator is zero)
    if precision + recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)


def read_wordnet_index():
    return set(open(WORDNET_INDEX_PATH).read().split('\n'))


def read_word(line):
    word, typestring = line.strip().split()
    original_typestring = typestring
    partial = False

    # Is the first character an "m", which stands for "mainly"
    if typestring.startswith('m'):
        partial = True
        typestring = typestring[1:]

    # Is the next character an "n" or a "p" (negative or positive)
    if typestring.startswith('p'):
        is_relational = True
    elif typestring.startswith('n'):
        is_relational = False
    else:
        raise ValueError(
            'No relational indicator: %s.' % original_typestring)

    typestring = typestring[1:]
    if len(typestring) == 0:
        subtype = None
    elif len(typestring) == 1:
        if typestring.startswith('b'):
            subtype = 'body-part'
        elif typestring.startswith('p'):
            subtype = 'portion'
        elif typestring.startswith('j'):
            subtype = 'adjectival'
        elif typestring.startswith('v'):
            subtype = 'deverbal'
        elif typestring.startswith('a'):
            subtype = 'aspectual'
        elif typestring.startswith('f'):
            subtype = 'functional'
        elif typestring.startswith('r'):
            subtype = 'link'
        else:
            raise ValueError(
                'Unrecognized noun subtype: %s.' % original_typestring
            )

    else:
        raise ValueError(
                'Trailing characters on typestring: %s.' 
                % original_typestring
            )

    return {
        'word':word,
        'is_relational':is_relational,
        'subtype':subtype
    }


def get_seed_set(path, dictionary=None):
    '''
    Get a set of positive (relational) words and a set of negative 
    (non-relational) words, to be used as a training set
    '''
    positives, negatives, neutrals = set(), set(), set()
    for line in open(path):
        token, classification = line.strip().split('\t')[:2]
        if dictionary is not None and token not in dictionary:
            continue
        if classification == '+':
            positives.add(token)
        elif classification == '0':
            neutrals.add(token)
        elif classification == '-':
            negatives.add(token)
        elif classification == '?':
            continue
        else:
            raise ValueError(
                'Unexpected classification for token "%s": %s' 
                % (token, classification)
            )

    return positives, negatives, neutrals


def read_all_labels(path):
	"""
	Read all the label files within the directory given by ``path``.
	"""
	positives = set()
	negatives = set()
	neutrals = set()
	for file_path in t4k.ls(path, dirs=False):
		pos, neg, neut = get_seed_set(file_path)
		positives.update(pos)
		negatives.update(neg)
		neutrals.update(neut)

	return positives, negatives, neutrals


def get_full_seed_set(dictionary=None):
    """
    Reads in the small initial set of expert-labelled examples that were used
    to design the task and help select new words for inclusion in the task.

    If a dictionary is provided, then only words in the dictionary will be
    included.
    """
    pos, neg, neut = get_seed_set(SEED_PATH, dictionary)
    return pos, neg, neut


def get_train_test_seed_split(dictionary=None):
    random.seed(0)
    split_ratio = 0.33
    pos, neg, neut = get_seed_set(SEED_PATH, dictionary)

    train = {}
    test = {}

    test['pos'] = set(random.sample(pos, int(len(pos)*split_ratio)))
    train['pos'] = pos - test['pos']

    test['neg'] = set(random.sample(neg, int(len(neg)*split_ratio)))
    train['neg'] = neg - test['neg']

    test['neut'] = set(random.sample(neut, int(len(neut)*split_ratio)))
    train['neut'] = neut - test['neut']

    return train, test


def make_vectors(
    dataset, features, count_based_features, non_count_features,
    count_feature_mode, whiten=False, threshold=0.5
):

    Q = list(dataset['pos']) + list(dataset['neut']) + list(dataset['neg'])

    X = features.as_sparse_matrix(
        Q,
        count_based_features, non_count_features, count_feature_mode, whiten,
        threshold
    )

    Y = np.array(
        [1] * len(dataset['pos']) 
        + [0] * len(dataset['neut'])
        + [-1] * len(dataset['neg'])
    )
    return Q, X, Y


def make_kernel_vector(dataset, features):
    # Make the training set.  Each "row" in the training set has a 
    # single "feature" -- it's the id which identifies the token.  
    # This will let us lookup the non-numeric features in kernel 
    # functions
    X = (
        [[features.get_id(s)] for s in dataset['pos']]
        + [[features.get_id(s)] for s in dataset['neut']]
        + [[features.get_id(s)] for s in dataset['neg']]
    )
    Y = (
        [1] * len(dataset['pos']) 
        + [0] * len(dataset['neut'])
        + [-1] * len(dataset['neg'])
    )
    return X, Y


def get_dictionary(path):
    dictionary = t4k.UnigramDictionary()
    dictionary.load(path)
    return dictionary


#def load_feature_file(path):
#    return cjson.decode(open(path).read())


#def get_features(path):
#    return {
#        'dep_tree': load_feature_file(os.path.join(
#            path, 'dependency.json')),
#        'baseline': load_feature_file(os.path.join(
#            path, 'baseline.json')),
#        'hand_picked': load_feature_file(os.path.join(
#            path, 'hand_picked.json')),
#        'dictionary': get_dictionary(os.path.join(
#            path, 'lemmatized-noun-dictionary'))
#    }


def filter_seeds(words, dictionary):
    '''
    Filters out any words in the list `words` that are not found in 
    the dictionary (and are hence mapped to UNK, which has id 0
    '''
    return [w for w in words if w in dictionary]


def ensure_unicode(s):
    try:
        return s.decode('utf8')
    except UnicodeEncodeError:
        return s

def normalize_token(s):
    """Ensures the token is unicode and lower-cased."""
    return ensure_unicode(s).lower()
