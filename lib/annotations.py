import t4k
import os
import sys
from SETTINGS import DATA_DIR, SEED_PATH, ANNOTATIONS_PATH
import numpy as np
import random
import utils

ALL_SOURCES = {'rand', 'top', 'guess', 'seed'}


class UnknownLabelError(Exception):
    pass


class Annotations(object):

    def __init__(self, vocabulary=None):

        # All tokens and their classification are stored here, according to 
        # their source
        self.examples_by_source = {
            'rand':set(),
            'top':set(),
            'guess':set(),
            'seed':set()
        }
        self.guessed_examples_by_source = {
            'pos': set(),
            'neut': set(),
            'neg': set()
        }
        self.example_source_lookup = {}

        self.examples = {}

        for line in open(ANNOTATIONS_PATH):

            # Clean trailing whitespace and skip blank lines
            line = line.strip()
            if line == '':
                continue

            # Parse and store this example
            token, sources, annotator, label = line.split('\t')
            if vocabulary is not None and token not in vocabulary:
                continue
            self.examples[token] = get_label(label)
            self.example_source_lookup[token] = (
                get_sources(sources) + list(get_guessed_sources(sources)))
            for source in get_sources(sources):
                self.examples_by_source[source].add(token)
            for source in get_guessed_sources(sources):
                self.guessed_examples_by_source[source].add(token)

        for line in open(SEED_PATH):

            # Clean trailing whitespace and skip blank lines
            line = line.strip()
            if line == '':
                continue

            # Parse the line
            fields = line.split('\t')
            token, label = fields[0], fields[1]

            # Don't add seeds that are duplicates
            if token in self.examples:
                continue

            # Don't add seeds that are outside the vocabulary
            if vocabulary is not None and token not in vocabulary:
                continue

            # Store this token, it's label, and its source ('seed').
            try:
                self.examples[token] = get_label(label)
            except UnknownLabelError:
                print token
                continue
            self.example_source_lookup[token] = ['seed']
            self.examples_by_source['seed'].add(token)



    def get_source_tokens(self, sources, exclude_sources=[]):

        # Handle the special case of getting all sources.  This overrides 
        # the exclude_sources setting
        if sources == 'all':
            return set(self.examples.keys())


        # Sources can be a single source or list of sources. Normalize to list.
        if isinstance(sources, basestring):
            sources = [sources]

        # Same for exclude sources
        if isinstance(exclude_sources, basestring):
            exclude_sources = [exclude_sources]

        # Get all the tokens for the requested sources
        tokens = set()
        for source in sources:
            if source in self.guessed_examples_by_source:
                tokens |= self.guessed_examples_by_source[source]
            else:
                tokens |= self.examples_by_source[source]

        # Get all the tokens for the excluded sources
        exclude_tokens = set()
        for source in exclude_sources:
            if source in self.guessed_examples_by_source:
                exclude_tokens |= self.guessed_examples_by_source[source]
            else:
                exclude_tokens |= self.examples_by_source[source]

        tokens = tokens - exclude_tokens

        return tokens

    def get_train_test(self, source, num_test=500, seed=0):
        """
        Get items split into testing and training sets, drawing `num_test` 
        testing items from ``source``, while putting the remainder into the
        training set along with all elements from sources other than `source`.
        """

        print 'seeding with', seed
        random.seed(seed)

        # The test set will consist of 500 examples from top and 500 examples
        # from rand.
        focal_pos, focal_neut, focal_neg = self.get_as_tokens(source)
        total = float(len(focal_pos) + len(focal_neut) + len(focal_neg))

        # Take proportionately from each
        num_test_pos = int(np.round(num_test*len(focal_pos)/total))
        test_pos = set(random.sample(focal_pos, num_test_pos))
        num_test_neut = int(np.round(num_test*len(focal_neut)/total))
        test_neut = set(random.sample(focal_neut, num_test_neut))
        num_test_neg =  int(np.round(num_test*len(focal_neg)/total))
        test_neg = set(random.sample(focal_neg, num_test_neg))

        # Everything else will be part of the training set.
        add_train_pos = focal_pos - test_pos
        add_train_neut = focal_neut - test_neut
        add_train_neg = focal_neg - test_neg

        # Get the training set from the other parts of the dataset
        non_focal_sources = set(ALL_SOURCES) - set([source])
        train_pos, train_neut, train_neg = self.get_as_tokens(
            non_focal_sources, source)

        # take items from focal sets not used for testing and add to training
        train_pos |= focal_pos - test_pos
        train_neut |= focal_neut - test_neut
        train_neg |= focal_neg - test_neg

        train = {'pos': train_pos, 'neut': train_neut, 'neg': train_neg}
        test = {'pos': test_pos, 'neut': test_neut, 'neg': test_neg}

        return train, test


    def get_train_dev(self, mode=None, num_dev=500, num_test=500):
        """
        Get the training set split into a training and dev set (excludes items
        that would be served in the test set for get_train_test('top') and
        get_train_test('rand').
        """

        # begin by getting both relevant test sets: one based on items in 'top'
        # and one based on items from 'rand'.  Then, eliminate these items
        # before making the dev and train sets.  Split what remains into dev
        # and train.

        # First, get both relevant test sets
        train_top, test_top = self.get_train_test('top', num_test)
        train_rand, test_rand = self.get_train_test('rand', num_test)

        # Now get the items remaining once the test items have been removed.
        # train_rand already doesn't have rand items, so only need to remove top
        dev_train = {}
        dev_train['pos'] = train_rand['pos'] - test_top['pos']
        dev_train['neut'] = train_rand['neut'] - test_top['neut']
        dev_train['neg'] = train_rand['neg'] - test_top['neg']

        # Now split what remains into a training and dev set
        dev = {}
        train = {}
        total = float(
            len(dev_train['pos'] | dev_train['neut'] | dev_train['neg']))
        num_dev_pos = int(np.round(num_dev * len(dev_train['pos']) / total))
        dev['pos'] = set(random.sample(dev_train['pos'], num_dev_pos))
        train['pos'] = dev_train['pos'] - dev['pos']

        num_dev_neut = int(np.round(num_dev * len(dev_train['neut']) / total))
        dev['neut'] = set(random.sample(dev_train['neut'], num_dev_neut))
        train['neut'] = dev_train['neut'] - dev['neut']

        num_dev_neg = int(np.round(num_dev * len(dev_train['neg']) / total))
        dev['neg'] = set(random.sample(dev_train['neg'], num_dev_neg))
        train['neg'] = dev_train['neg'] - dev['neg']

        # If we want to test only on top items in the dev set, then 
        # we need to intersect dev with top.
        if mode == 'top':
            top_pos, top_neut, top_neg = self.get_as_tokens('top')
            dev['pos'] = dev['pos'] & top_pos
            dev['neut'] = dev['neut'] & top_neut
            dev['neg'] = dev['neg'] & top_neg
        elif mode == 'rand':
            rand_pos, rand_neut, rand_neg  = self.get_as_tokens('rand')
            dev['pos'] = dev['pos'] & rand_pos
            dev['neut'] = dev['neut'] & rand_neut
            dev['neg'] = dev['neg'] & rand_neg

        return train, dev


    def get_as_tokens(self, sources, exclude_sources=[]):
        """
        Return token sets corresponding to the positives, negatives, and
        neutrals found within the given sources but not found within exclude
        sources.
        """
        tokens = self.get_source_tokens(sources, exclude_sources)

        # convert tokens to an array of token-ids and provide labels as 
        # separate array
        positive, negative, neutral = set(), set(), set()
        for token in tokens:
            if self.examples[token] == 2:
                positive.add(token)
            elif self.examples[token] == 1:
                neutral.add(token)
            elif self.examples[token] == 0:
                negative.add(token)

        return positive, neutral, negative


    #def get(self, sources, exclude_sources=[]):
    #    """
    #    Get the feature and label arrays, in a format suitable for scikit 
    #    classifiers.
    #    """

    #    tokens = self.get_source_tokens(sources, exclude_sources)

    #    # convert tokens to an array of token-ids and provide labels as 
    #    # separate array
    #    X, Y = [], []
    #    X = [[self.dictionary.get_id(token)] for token in tokens]
    #    Y = [self.examples[token] for token in tokens]

    #    return X, Y

                


# This converts the sources from which an example was drawn into a 
# smaller set of standardized sources.
SOURCE_MAP = {
    'rand': 'rand',
    'rand2': 'rand',
    'top': 'top',
    'pos': 'guess',
    'neg': 'guess',
    'neut': 'guess',
    'pos2': 'guess',
    'neg2': 'guess',
    'neut2': 'guess'
}
def get_sources(sources):
    return [SOURCE_MAP[s] for s in  sources.split(':')]

GUESSED_SOURCE_MAP = {
    'pos': 'pos',
    'neg': 'neg',
    'neut': 'neut',
    'pos2': 'pos',
    'neg2': 'neg',
    'neut2': 'neut'
}


def get_guessed_sources(sources):
    for source in sources.split(':'):
        if source in GUESSED_SOURCE_MAP:
            yield GUESSED_SOURCE_MAP[source]


# This converts labels to integers
LABEL_MAP = {
    '-': 0, # never relational
    '0': 1, # occasionally relational
    '+': 2  # usually relational
}
def get_label(label):
    label = label[0]
    try:
        return LABEL_MAP[label]
    except KeyError:
        raise UnknownLabelError(KeyError.message)
