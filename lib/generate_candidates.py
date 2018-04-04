import random
import sys
sys.path.append('../lib')
import extract_features
import os
import t4k
from SETTINGS import DATA_DIR
import classifier
import utils

BEST_CLASSIFIER_CONFIG = {
    'on_unk': False,
    'C': 0.01,
    'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
    'semantic_similarity': 'res',
    'include_suffix' :  True,
    'syntactic_multiplier': 10.0,
    'semantic_multiplier': 2.0,
    'suffix_multiplier': 0.2
}
BEST_WORDNET_ONLY_FEATURES_PATH = os.path.join(
    DATA_DIR, 'relational-noun-features-wordnet-only', 'accumulated-pruned-5000'
)

def generate_candidates(num_to_generate, out_path, pos, neg, exclude):

    # Open the file to which we will write candidates
    out_f = open(out_path, 'w')

    # Read in the extracted features, which we'll also need for a couple things
    features = extract_features.FeatureAccumulator(
        load=BEST_WORDNET_ONLY_FEATURES_PATH)

    # Make the best performing classifier.  This is what we'll use to score the
    # "relationalness" of new words.
    clf = classifier.make_classifier(
        kind='svm',
        features=features,
        positives=pos,
        negatives=neg,
        **BEST_CLASSIFIER_CONFIG
    )

    # Now generate the candidates.  We only keep track of the number of 
    # positives generated, because there are always more negatives
    num_generated = 0
    for token in features.dictionary.get_token_list():
        if token in exclude:
            print '\t\tx\t%s' % token
            continue
        score = clf.score(token)[0]
        if score > clf.threshold:
            print '%s\t+' % token
            out_f.write('%s\t+\t%f\n' % (token, score))
            num_generated += 1
            if num_generated == num_to_generate:
                break
        else:
            print '\t-\t%s' % token
            out_f.write('%s\t-\t%f\n' % (token, score))


def generate_random_candidates(num_to_generate, out_path, exclude=set()):

    # Open a path that we want to write to 
    out_f = open(out_path, 'w')

    # Open the dictionary of words seen in the corpus
    dictionary_path = os.path.join(BEST_WORDNET_ONLY_FEATURES_PATH, 'dictionary')
    dictionary = t4k.UnigramDictionary()
    dictionary.load(dictionary_path)

    # Uniformly randomly sample from it
    samples = set()
    while len(samples) < num_to_generate:
        token = random.choice(dictionary.token_map.tokens)
        if token != 'UNK' and token not in exclude and token not in samples:
            samples.add(token)

    out_f.write('\n'.join(samples))


def generate_candidates_ordinal(
    num_to_generate, out_path, pos, neg, neut, exclude, 
    kernel=None, features=None
):

    # Open the file to which we will write candidates
    out_f = open(out_path, 'w')

    # Read in the extracted features, which we'll also need for a couple things
    if features is None:
        features = extract_features.FeatureAccumulator(
            load=BEST_WORDNET_ONLY_FEATURES_PATH)

    # Make the best performing classifier.  This is what we'll use to score the
    # "relationalness" of new words.
    clf = classifier.make_classifier(
        kind='osvm',
        kernel=kernel,
        features=features,
        positives=pos,
        negatives=neg,
        neutrals=neut,
        **BEST_CLASSIFIER_CONFIG
    )

    # Now generate the candidates.  We only keep track of the number of 
    # positives generated, because there are always more negatives
    num_generated = 0
    filtered_tokens = [
        t for t in features.dictionary.get_token_list() if t not in exclude]

    for token, score in clf.score_parallel(filtered_tokens):
        if score >= 1:
            print '%s\t+' % token
            out_f.write('%s\t+\t%f\n' % (token, score))
            num_generated += 1
            if num_generated == num_to_generate:
                break
        elif score > -1:
            print '\t0\t%s' % token
            out_f.write('%s\t0\t%f\n' % (token, score))
        else:
            print '\t-\t%s' % token
            out_f.write('%s\t-\t%f\n' % (token, score))


