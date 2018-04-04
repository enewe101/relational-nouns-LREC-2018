#!/usr/bin/env python
import sklearn
from nltk.corpus import wordnet
import numpy as np
import utils
import shutil
import t4k
import subprocess
import tarfile
from t4k import UnigramDictionary, UNK
import json
from iterable_queue import IterableQueue
from multiprocessing import Process
import time
import os
from corenlp_xml_reader import AnnotatedText, Token, Sentence
import sys
from collections import defaultdict, Counter, deque
from SETTINGS import (
    DATA_DIR, RELATIONAL_NOUN_FEATURES_DIR, 
    #ACCUMULATED_FEATURES_PATH,
    SUFFIX_PATH, GOOGLE_VECTORS_PATH, GAZETTEER_DIR
)
from nltk.corpus import wordnet
from scipy.sparse import csr_matrix, coo_matrix
import itertools


LEN_GOOGLE_VECTORS = 300
NUM_ARTICLE_LOADING_PROCESSES = 12
GAZETTEER_FILES = [
    'country', 'city', 'us-state', 'continent', 'subcontinent'
]

# Average dot product between two randomly chosen vectors.  Should be used
# in the kernel when one or both of the token's vectors are missing
AVERAGE_ABSOLUTE_DOT = 0.068708472297183756

@np.vectorize
def safe_log2(x):
    return 0.0 if x==0 or np.isnan(x) else np.log2(x)


def coalesce_features(
    out_dir, 
    min_token_occurrences=5,
    min_feature_occurrences=5,
    vocabulary=None,
    feature_dirs=t4k.ls(
        RELATIONAL_NOUN_FEATURES_DIR, 
        absolute=True, match='/[0-9a-f]{3,3}$', files=False
    )
):

    # Make the feature accumulator in which to aggregate results
    feature_accumulator = make_feature_accumulator(vocabulary)

    # Read in the features for each feature_dir and accumulate
    for feature_dir in feature_dirs:
        print 'processing %s...' % os.path.basename(feature_dir)
        try:
            feature_accumulator.merge_load(feature_dir)

        # Tolerate errors reading or json parsing
        except (IOError, ValueError), e:
            print 'problem with feature_dir %s: %s' % (feature_dir, str(e))
            pass

    # Prune the features
    print 'prunning...'
    feature_accumulator.prune_dictionary(min_token_occurrences)
    feature_accumulator.prune_features(min_feature_occurrences)

    coalesced_path = out_dir
    feature_accumulator.write(coalesced_path)


def load_gazetteers():
    # Read a the gazetteer files, and save each as a set.
    # We have place names and demonyms for coutries, cities, etc.
    gazetteer = {'names': set(), 'demonyms': set()}
    for gazetteer_fname_prefix in GAZETTEER_FILES:
        for pos in ['names', 'demonyms']:

            # Work out the path
            gazetteer_type = gazetteer_fname_prefix + '-' + pos
            gazetteer_fname = gazetteer_type + '.txt'
            gazetteer_path = os.path.join(GAZETTEER_DIR, gazetteer_fname)

            # Read the file into a set
            gazetteer[gazetteer_type] = set([
                line.strip() for line in open(gazetteer_path)
            ])

            # Pool all names and separately pool all demonyms
            gazetteer[pos] |= gazetteer[gazetteer_type]

    return gazetteer

# Keep a global gazetteer for easy access
GAZETTEERS = load_gazetteers()


#def get_accumulated_features():
#    """
#    Factor for loading a feature accumulator that contains all features for
#    non-cmpound nouns in wordnet extracted across gigaword.
#    """
#    return FeatureAccumulator(
#        utils.read_wordnet_index(), load=ACCUMULATED_FEATURES_PATH)



# TODO: deprecate.  Either have defaults or don't.
def make_feature_accumulator(vocabulary=None, load=None):
    """
    Factory for making feature accumulators.  This de-parametrizes the
    dictionary of words for which we should accumulate features, making it
    always equal to the single-word noun lemma entries in wordnet.
    """
    return FeatureAccumulator(vocabulary, load=load)


def extract_all_features(
    article_archive_dir, out_dir, untar=True, limit=None, vocabulary=None
):

    start = time.time()
    t4k.ensure_exists(out_dir)

    # If ``untar`` is true, then untar the path first
    if untar:
        untarred_article_dir = article_archive_dir[:-len('.tgz')]
        dirname = os.path.basename(untarred_article_dir)
        print 'untarring %s' % (dirname)
        containing_dir = os.path.dirname(article_archive_dir)
        subprocess.check_output([
            'tar', '-zxf', article_archive_dir, '-C', containing_dir])
    else:
        untarred_article_dir = article_archive_dir

    # First, make an iterable queue and load all the article fnames onto it
    fnames_q = IterableQueue()
    fnames_producer = fnames_q.get_producer()
    for fname in get_fnames(untarred_article_dir)[:limit]:
        fnames_producer.put(fname)
    fnames_producer.close()

    # Make a queue to hold feature stats (results), and a consumer to 
    # receive completed feature stats objects from workers
    features_q = IterableQueue()
    features_consumer = features_q.get_consumer()

    # Create workers that consume filenames and produce feature counts.
    for p in range(NUM_ARTICLE_LOADING_PROCESSES):
        fnames_consumer = fnames_q.get_consumer()
        features_producer = features_q.get_producer()
        process = Process(
            target=extract_features_worker,
            args=(fnames_consumer, features_producer, vocabulary, False)
        )
        process.start()

    # Close the iterable queues
    fnames_q.close()
    features_q.close()

    # Accumulate the results.  This blocks until workers are finished
    feature_accumulator = make_feature_accumulator(vocabulary=vocabulary)
    for accumulator in features_consumer:
        feature_accumulator.merge(accumulator)

    # Write the features to disk
    dirname = os.path.basename(untarred_article_dir)
    write_path = os.path.join(out_dir, dirname)
    feature_accumulator.write(write_path)

    # If we untarred the file, then delete the untarred copy
    if untar:
        shutil.rmtree(untarred_article_dir)

    elapsed = time.time() - start
    print 'elapsed', elapsed


def extract_features_worker(
    files_consumer, features_producer, vocabulary, has_content='True'
):
    '''
    Extracts features from articles on `files_consumer`, and puts 
    featres onto `features_producer`.  If `has_content` is 'true', then
    each item is a tuple containing path and a string representing the
    file contents.  Otherwise, only the path is provided, and the file
    will be opened and read here.
    '''

    # The dictionary determines which words we aggregate features for.
    # If a word isn't in the dictionary, we don't aggregate features for it.
    feature_accumulator = make_feature_accumulator(vocabulary=vocabulary)

    # Read articles named in files_consumer, accumulate features from them
    for item in files_consumer:

        if has_content:
            fname, content = item
        else:
            fname = item
            content = open(fname).read()

        # Get features from this article
        t4k.out('.')
        feature_accumulator.extract(AnnotatedText(content))

    # Put the accumulated features onto the producer queue then close it
    features_producer.put(feature_accumulator)
    features_producer.close()


# Wordnet parts of speech are encoded using the following characters.
# (Apparently 'r' also occurs, but it wasn't observed in this dataset.)
WORDNET_POS = 'asnv'
COUNT_BASED_FEATURES = [
    'baseline', 'dependency', 'hand_picked',
    'lemma_unigram', 'lemma_bigram', 'pos_unigram', 'pos_bigram', 
    'surface_unigram', 'surface_bigram'
]
NON_COUNT_FEATURES = ['derivational', 'google_vectors', 'suffix']
ALL_FEATURES = COUNT_BASED_FEATURES + NON_COUNT_FEATURES


FEATURES = {}
def get_features(features_dirname='accumulated450-min_token_5-min_feat1000'):
    global FEATURES
    if features_dirname not in FEATURES:
        print 'loading features from %s...' % features_dirname
        features_path = os.path.join(
            RELATIONAL_NOUN_FEATURES_DIR, features_dirname)
        FEATURES[features_dirname] = FeatureAccumulator(
            utils.read_wordnet_index(), load=features_path)

    return FEATURES[features_dirname]



class FeatureAccumulator(object):
    """
    Extracts and accumulates features associated to nouns by reading
    CoreNLP-annotated articles.
    """

    def __init__(
        self,
        vocabulary=None,
        load=None, 
        google_vectors_path=GOOGLE_VECTORS_PATH,
        lexical_feature_window=5
    ):
        """
        ``vocabulary`` is a set of words for which we will collect features.
        words not in ``vocabulary`` will be skipped.
        """
        self.vocabulary = vocabulary
        self.google_vectors_path = google_vectors_path
        self._initialize_accumulators()
        self.lexical_feature_window = lexical_feature_window
        self.feature_map_contents = None

        if load is not None:
            self.merge_load(load)


    def get_id(self, token):
        return self.dictionary.get_id(token)


    def get_token(self, idx):
        return self.dictionary.get_token(idx)


    def _initialize_accumulators(self):
        # This operation makes features stale
        self.fresh = set()
        self.threshold_features = defaultdict(Counter)
        self.dictionary = UnigramDictionary()
        for feature_type in COUNT_BASED_FEATURES:
            setattr(self, feature_type, defaultdict(Counter))


    def accumulate_sorted_feature_values(self):
        self.all_feature_values = defaultdict(lambda:defaultdict(list))
        for feature_type in COUNT_BASED_FEATURES:
            # We will first need to determine what the (100*p)th percentile is
            # for each feature.  To do that, we need to gather up all the
            # values for each feature, sort them, then select the middle value.
            print (
                'accumulating feature values for %s' 
                % feature_type
            )
            feature_counts = getattr(self, feature_type)
            for lemma in feature_counts:
                for feature_name, val in feature_counts[lemma].iteritems():
                    self.all_feature_values[feature_type][feature_name].append(
                        val)

        # Sort all of the feature values
        for feature_type in COUNT_BASED_FEATURES:
            print 'sorting feature values for %s' % feature_type
            for feature_name in self.all_feature_values[feature_type]:
                self.all_feature_values[feature_type][feature_name].sort()

        self.fresh.add('sorted-features')


    def read_google_vectors(self):
        print 'reading google vectors'
        self.vectors = {}
        for line in open(self.google_vectors_path):

            # Skip blank lines
            line = line.strip()
            if line == '':
                continue

            token = line.split(' ', 1)[0]
            if token not in self.dictionary:
                continue
            self.vectors[token] = [float(v) for v in line.split(' ')[1:]]

        print 'done reading google vectors'
        self.fresh.add('vectors')


    def calculate_derivational_features(self):
        """ 
        Pre-computes features that indicate whether a given noun has a
        derivationally related verb or adjective.  This is precomputed from
        wordnet and stored on disk as a tsv which is faster to read in than
        computing from wordnet as needed.  
        """
        print 'computing derivational features'
        all_pos = set()

        self.derivational_features = {}
        for i, token in enumerate(self.dictionary.get_token_list()):
            related_forms = t4k.flatten([
                lemma.derivationally_related_forms() 
                for lemma in wordnet.lemmas(token)
            ])
            poss = set([rf.synset().pos() for rf in related_forms])
            self.derivational_features[token] = [
                int(pos in poss) for pos in WORDNET_POS]

        print 'done calculating derivational features'
        self.fresh.add('derivational')


    def calculate_threshold_features(self, p=0.5):
        """
        Calculate a thresholded version of the features.  The thresholded
        feature is 1 if the value of the feature is greater than the 50th
        percentile for that feature, otherwise it's zero.
        """

        print 'calculating thresholded features'

        # If necessary accumulate all feature values, sorted.  This allows us
        # to determine the (p*100)th percentile, to be used as the threshold
        if 'sorted-features' not in self.fresh:
            self.accumulate_sorted_feature_values()

        # calculate the thresholds for each feature based on this p-value
        thresholds = defaultdict(dict)
        for feature_type in COUNT_BASED_FEATURES:
            print 'calculating thresholds for %s' % feature_type
            this_feature_type_set = self.all_feature_values[feature_type]
            for feature_name in this_feature_type_set:

                # Get the threshold value for this particular feature
                this_feature_set = this_feature_type_set[feature_name]
                num_feats = len(this_feature_set)
                threshold_idx = int(np.floor(num_feats * p))
                threshold = this_feature_set[threshold_idx]
                thresholds[feature_type][feature_name] = threshold

        # Calculate thresholded features now.  These are binary features where
        # the value is 1 if the feature is above the threshold, 0 otherwise.
        threshold_features = defaultdict(lambda:defaultdict(dict))
        self.threshold_features[p] = threshold_features
        for feature_type in COUNT_BASED_FEATURES:
            print 'calculating threshold features for %s' % feature_type
            this_feature_type_counts = getattr(self, feature_type)
            this_thresholded_type = threshold_features[feature_type]
            for lemma in this_feature_type_counts:
                feature_counts_for_lemma = this_feature_type_counts[lemma]
                this_thresholded_lemma = this_thresholded_type[lemma]
                for feature_name in feature_counts_for_lemma:
                    val = feature_counts_for_lemma[feature_name]
                    thresholded = val > thresholds[feature_type][feature_name]
                    this_thresholded_lemma[feature_name] = thresholded

        print 'done thresholding'


    def calculate_log_features(self):
        """
        Calculate log-transformed version of normalized feature counts.
        """

        # First, we'll need up-to-date normalized features
        if 'normalize' not in self.fresh:
            self.normalize_features()

        self.log_features = defaultdict(lambda:defaultdict(dict))
        for feature_type in COUNT_BASED_FEATURES:
            print 'calculating log features %s' % feature_type
            normalized_type = getattr(self, '%s_normalized' % feature_type)
            for lemma in normalized_type:
                normalized_lemma = normalized_type[lemma]
                self.log_features[feature_type][lemma] = {
                    feat_name : np.log(normalized_lemma[feat_name])
                    for feat_name in 
                    getattr(self, '%s_normalized' % feature_type)[lemma]
                }

        print 'done calculating log features'
        self.fresh.add('log')


    def calculate_suffix_features(self):
        """
        Calculate suffix-based features.
        """
        print 'calculating suffix features'
        # Determine the suffix for every lemma
        self.suffix = {}
        for lemma in self.dictionary.get_token_list():
            self.suffix[lemma] = get_suffix(lemma)

        print 'done calculating suffix features'
        self.fresh.add('suffix')


    def normalize_features(self):
        """
        Divides a lemma's feature counts by the number of times the lemma 
        appeared in the corpus.

        Calculate suffix-based features.
        """
        for feature_type in COUNT_BASED_FEATURES:
            print 'normalizing %s' % feature_type

            # Initialize an normalized version of this feature_type
            feature_norm = defaultdict(Counter)
            setattr(self, '%s_normalized' % feature_type, feature_norm)

            # Get this feature_type
            feature_counts = getattr(self, feature_type)

            # Normalize the feature counts for each lemma
            lemmas_to_remove = set()
            for lemma in feature_counts:

                lemma_counts = self.dictionary.get_token_frequency(lemma)
                if lemma_counts == 0:
                    lemmas_to_remove.add(lemma)
                    continue

                for feature in feature_counts[lemma]:
                    feature_norm[lemma][feature] = (
                        feature_counts[lemma][feature] / float(lemma_counts))

            # Remove any entries in the main feature counters for lemmas that
            # were pruned from the dictionary previously (there must be a bug
            # in somewhere for these to be left behind...)
            for lemma in lemmas_to_remove:
                print 'DELETING', lemma
                del feature_counts[lemma]

        # Determine the suffix for every lemma
        self.suffix = {}
        for lemma in self.dictionary.get_token_list():
            self.suffix[lemma] = get_suffix(lemma)

        print 'done normalizing'
        self.fresh.add('normalize')


    def get_suffix_idx(self, lemma_idx):
        lemma = self.get_token(lemma_idx)
        return self.get_suffix(lemma)


    def get_suffix(self, lemma):
        if 'suffix' not in self.fresh:
            self.calculate_suffix_features()

        # Ensure the lemma is unicode and lowercased
        lemma = utils.normalize_token(lemma)

        try:
            return self.suffix[lemma]
        except KeyError:
            return None

            
    def as_sparse_matrix(
        self,
        tokens=None,
        count_based_features=COUNT_BASED_FEATURES,
        non_count_features=NON_COUNT_FEATURES,
        mode='raw', # 'log' | 'normalized' | threshold,
        whiten=False,
        threshold=0.5
    ):
        """
        Reformats the storage of the features into a pandas data frame, so that
        we can easily slice out sparse, vectorized versions of the feature
        data.
        """

        # Get a mapping from feature (names) to column indices.
        self.get_feature_map(count_based_features, non_count_features)

        # For which tokens are we building the feature vectors?  If nothing
        # was provided, make it for all tokens in the dictionary.
        if tokens is None:
            tokens = self.dictionary.get_token_list()

        # Get a feature map, which maps each feature (by name) to an integer
        # correponding to it's (column) index in the sparse matrix
        # representation of tokens' features.
        coo_val, coo_i, coo_j = [], [], []
        for token_num, lemma in enumerate(tokens):

            t4k.progress(token_num, len(tokens))

            for feature_type in count_based_features:
                count_features = self.get_count_based_features(
                    lemma, [feature_type], mode, threshold
                )

                # Add the count feature to the coo_matrix construction lists
                coo_val.extend(count_features.values())
                coo_i.extend([token_num]*len(count_features))
                coo_j.extend([
                    self.feature_map['%s:%s' % (feature_type, feature_name)] 
                    for feature_name in count_features
                ])

            # Next get google embedding features (if desired)
            if 'google_vectors' in non_count_features:
                vec = self.get_vector(lemma)
                coo_val.extend(vec)
                coo_i.extend([token_num]*LEN_GOOGLE_VECTORS)
                coo_j.extend([
                    self.feature_map['google_vectors:%d'%f] 
                    for f in range(LEN_GOOGLE_VECTORS)
                ]) 

            # Next get derivational features (if desired)
            if 'derivational' in non_count_features:
                deriv_feats = self.get_derivational_features(lemma)
                coo_val.extend(deriv_feats)
                coo_i.extend([token_num]*len(WORDNET_POS))
                coo_j.extend([
                    self.feature_map['derivational:%s'%d] 
                    for d in WORDNET_POS
                ])

            # Next get suffix features (if desired)
            if 'suffix' in non_count_features:
                suffix = self.get_suffix(lemma)
                if suffix is not None:
                    coo_val.append(1)
                    coo_i.append(token_num)
                    coo_j.append(self.feature_map['suffix:%s' % suffix])

        num_tokens = token_num + 1

        # Build a sparse row-compressed matrix
        sparse_matrix = csr_matrix(
            coo_matrix(
                (coo_val, (coo_i, coo_j)),
                shape=(num_tokens, len(self.feature_map))
            )
        )

        # Center and scale the data
        if whiten:
            print 'whitening...'
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
            sparse_matrix = scaler.fit_transform(sparse_matrix)

        return sparse_matrix


    def get_feature_map(
        self, 
        count_based_features=COUNT_BASED_FEATURES, 
        non_count_features=NON_COUNT_FEATURES
    ):
        """
        Given the subset of features that have been chosen, create a mapping
        from each individual feature to a component in a sparse feature vector.
        This let's us vectorize the features in a consistent way.
        """

        # Check to see if we have already assembled this feature map
        desired_map_contents = set(count_based_features + non_count_features)
        if self.feature_map_contents == desired_map_contents:
            return

        # First accumulate all of the keys for all features.  We will make a
        # mapping from a specific feature (key) to incrementing integers
        # representing that features's component in the feature vector.
        #for token in self.hand_picked for feature in self.hand_picked[token]
        self.feature_map = t4k.IncrementingMap()

        # Add keys for the count-based features
        for feature_type in count_based_features:
            #if feature_type in include_feature_types:
            feature_names = set([
                '%s:%s' % (feature_type, feature)
                for lemma in getattr(self, feature_type)
                for feature in getattr(self, feature_type)[lemma]
            ])
            self.feature_map.add_many(feature_names)

        # Add keys for other features
        if 'derivational' in non_count_features:
            feature_names = ['derivational:%s' % pos for pos in WORDNET_POS]
            self.feature_map.add_many(feature_names)

        # Add keys for google_vector features
        if 'google_vectors' in non_count_features:
            feature_names = [
                'google_vectors:%d' % i for i in range(LEN_GOOGLE_VECTORS)]
            self.feature_map.add_many(feature_names)

        # Add keys for suffix features
        if 'suffix' in non_count_features:
            feature_names = ['suffix:%s' % s for s in SUFFIXES]
            self.feature_map.add_many(feature_names)

        self.feature_map_contents = desired_map_contents



    def get_vector(self, lemma):
        """
        Get the google_vector embedding associated to a given token
        """
        if 'vectors' not in self.fresh:
            self.read_google_vectors()

        # Ensure the lemma is unicode and lowercased
        lemma = utils.normalize_token(lemma)

        # If there is no entry for this lemma, then return the average vector
        if lemma not in self.vectors:

            # Check if we've already calculated the average vector
            if 'avg-google-vec' not in self.fresh:
                self.avg_google_vec = np.mean(self.vectors.values(), axis=0)
                self.fresh.add('avg-google-vec')

            # Return the average vector (because no vector was found for lemma)
            return self.avg_google_vec

        # Return this lemma's vector embedding
        return self.vectors[lemma]


    def get_count_based_features(
        self, 
        lemma,
        feature_types=COUNT_BASED_FEATURES,
        mode='normalized',
        threshold=0.5
    ):

        if mode == 'raw':
            return self.get_feature_counts(lemma, feature_types)
        elif mode == 'normalized':
            return self.get_normalized_features(lemma, feature_types)
        elif mode == 'log':
            return self.get_log_features(lemma, feature_types)
        elif mode == 'threshold':
            return self.get_threshold_features(lemma, feature_types, threshold)
        else:
            raise ValueError('Unexpected mode: %s' % mode)


    def get_feature_counts(
        self, lemma, feature_types=COUNT_BASED_FEATURES
    ):
        """
        Get the features associated to a given token.  Features of all types
        requested are merged into one dictionary and returned.
        """
        # Ensure the lemma is unicode and lowercased
        lemma = utils.normalize_token(lemma)

        # Get all the feature types requested for this lemma
        features =  t4k.merge_dicts(*[
            getattr(self, feature_type)[lemma]
            for feature_type in feature_types
        ])

        return features


    def get_derivational_features(self, lemma):
        """
        Get the derivational features for the provided word.
        """
        if 'derivational' not in self.fresh:
            self.calculate_derivational_features()

        # Ensure the lemma is unicode and lowercased
        lemma = utils.normalize_token(lemma)

        # Return the derivational features for this lemma
        return self.derivational_features.get(lemma, [0,0,0,0])


    def get_normalized_features(
        self, lemma, feature_types=COUNT_BASED_FEATURES
    ):
        """
        Get the normalized features associated to a given token.  Features of
        all types requested are merged into one dictionary and returned.
        Ensures that the normalized features returned are in sync with latest
        updates by calling ``normalize_features()`` in case recent changes have
        altered feature counts.
        """
        if 'normalize' not in self.fresh:
            self.normalize_features()

        # Ensure the lemma is unicode and lowercased
        lemma = utils.normalize_token(lemma)

        # Get all the feature types requested for this lemma
        features =  t4k.merge_dicts(*[
            getattr(self, '%s_normalized' % feature_type)[lemma]
            for feature_type in feature_types
        ])

        return features


    def get_log_features(self, lemma, feature_types=COUNT_BASED_FEATURES):
        """
        Get the features associated to a given token under log transformation.
        Features of all types requested are merged into one dictionary and
        returned.  Ensures that the log features returned are in sync
        with latest updates by calling ``calculate_log_features()`` in case
        recent changes have altered feature counts.
        """
        if 'log' not in self.fresh:
            self.calculate_log_features()

        # Ensure the lemma is unicode and lowercased
        lemma = utils.normalize_token(lemma)

        # Get all the feature types requested for this lemma
        features =  t4k.merge_dicts(*[
            self.log_features[feature_type][lemma]
            for feature_type in feature_types
        ])

        return features


    def get_threshold_features(
        self, lemma, feature_types=COUNT_BASED_FEATURES, threshold=0.5
    ):
        """
        Get the thresholded features associated to a given token.  Features of
        all types requested are merged into one dictionary and returned.
        Ensures that the thresholded features returned are in sync with latest
        updates by calling ``calculate_log_features()`` in case recent changes
        have altered feature counts.
        """
        if threshold not in self.threshold_features:
            self.calculate_threshold_features(threshold)

        # Ensure the lemma is unicode and lowercased
        lemma = utils.normalize_token(lemma)

        # Get all the feature types requested for this lemma
        features =  t4k.merge_dicts(*[
            self.threshold_features[threshold][feature_type][lemma]
            for feature_type in feature_types
        ])

        return features


    def get_log_features_idx(
        self, lemma_idx, feature_types=COUNT_BASED_FEATURES
    ):
        """
        Similar to ``get_log_features()``, but look up the features by a
        token's integer id.  Useful when tokens need to be encoded as ints.
        """
        lemma = self.dictionary.get_token(lemma_idx)
        return self.get_log_features(lemma, feature_types)


    def get_features_idx(
        self, lemma_idx, feature_types=COUNT_BASED_FEATURES
    ):
        """
        Similar to ``get_features()``, but look up the features by a token's 
        integer id.  Useful when tokens need to be encoded as ints.
        """
        lemma = self.dictionary.get_token(lemma_idx)
        return self.get_features(lemma, feature_types)


    def merge(self, other):
        """
        Add counts from ``other`` to self.
        """
        # This operation makes normalized features stale
        self.fresh = set()
        self.threshold_features = {}

        self.dictionary.add_dictionary(other.dictionary)
        for feature_type in COUNT_BASED_FEATURES:
            own_features = getattr(self, feature_type)
            other_features = getattr(other, feature_type)
            for key in other_features:
                own_features[key] += other_features[key]


    def merge_load(self, path):
        print 'reading features from disc...'
        # This operation makes normalized features stale
        self.fresh = set()
        self.threshold_features = {}

        # Read the dictionary from disc and merge it with the one in memory.
        add_dictionary = UnigramDictionary()
        add_dictionary.load(os.path.join(path, 'dictionary'))

        # Filter out any tokens that are out-of-vocabulary (if vocab was given)
        if self.vocabulary is not None:
            for token in add_dictionary.get_token_list():
                if token not in self.vocabulary and token != 'UNK':
                    add_dictionary.remove(token)
            add_dictionary.compact()

        # Merge the loaded and filtered dictionary with the one in memory.
        self.dictionary.add_dictionary(add_dictionary)

        # Read each of the feature types from disc and merge with those in mem.
        for feature_type in COUNT_BASED_FEATURES:
            feature_accumulator = getattr(self, feature_type)
            feature_path = os.path.join(path, feature_type + '.json')
            as_dict = json.loads(open(feature_path).read())
            for key in as_dict:
                if self.vocabulary is not None and key not in self.vocabulary:
                    continue
                feature_accumulator[key].update(as_dict[key])


    def load(self, path):
        self._initialize_accumulators()
        self.merge_load(path)


    def extract(self, article):
        # This operation makes normalized features stale
        self.fresh = set()
        self.threshold_features = {}

        for sentence in article.sentences:
            for token in sentence['tokens']:

                # Ensure the lemma is unicode and lower-cased
                lemma = utils.normalize_token(token['lemma'])

                # Skip words that aren't in the vocabulary.
                if self.vocabulary is not None:
                    if lemma not in self.vocabulary:
                        continue

                # Skip words that aren't a common noun.
                pos = token['pos']
                if pos != 'NN' and pos != 'NNS':
                    continue

                # Record the occurrence of the word.
                self.dictionary.add(lemma)

                # Extract features seen for this word.
                self.get_dep_tree_features(token)
                self.get_baseline_features(token)
                self.get_hand_picked_features(token)
                self.get_lexical_features(token, sentence)


    def calculate_mutual_information(self,
        annotations,
        out_path, 
        count_based_features=COUNT_BASED_FEATURES,
        non_count_features=['derivational', 'suffix'] # exclude goog-vec
    ):
        """
        Calculate the mutual information for all features, sort features in
        descending order of mutual information, write the results to disc,
        and return the results.
        """

        # Open a file for writing
        if out_path is not None:
            out_file = open(out_path, 'w')

        # We will compute the mutual information based on the features 
        # thresholded at their fiftieth percentile.
        tokens = list(annotations.examples.keys())
        dense_features = self.as_sparse_matrix(
            tokens=tokens,
            count_based_features=count_based_features,
            non_count_features=non_count_features, 
            mode='threshold',
            threshold=0.5
        ).toarray()

        relational_nouns = np.array([
            int(annotations.examples[t]>0)  for t in tokens
        ])

        print 'calculating mutual information...'
        # Calculate the mutual information for all features
        mutual_inf = self.mutual_inf(dense_features, relational_nouns)

        print 'sorting featrues by mutual information...'
        # Get the indices that sorts features in descending order of mutual
        # information.  The slice indexing provides descending order.
        sort_indices = mutual_inf.argsort()[::-1]
        mutual_inf = mutual_inf[sort_indices]
        feature_names = np.array(
            [t4k.ensure_unicode(s) for s in self.feature_map._keys]
        )[sort_indices]

        # Write the mutual informations to disc
        if out_path is not None:
            print 'writing feature mutual information to disk..'
            for feature, m_inf in itertools.izip(feature_names, mutual_inf):
                out_file.write('%s\t%f\n' % (feature, m_inf))

        print 'returning...'
        return feature_names, mutual_inf


    def mutual_inf(self, features, classes):

        N = float(len(features))

        # Determine the frequency that given features "fire".
        print 'summing feature occurrences...'
        f_sum = features.sum(axis=0)/N

        # Determine the frequency of relational nouns
        print 'summing class occurrences...'
        c_sum = classes.sum(axis=0)/N

        # Calculate the concordance of presence/absence of features with nouns
        # being relational/non-relational.  There are four cases.  When we 
        # consider the absence of features, we take 1-features, and similarly
        # when considering non-relational nouns, we take 1-classes.
        mutual_information = np.zeros(features.shape[1])
        for f_inv, c_inv in itertools.product([True, False], repeat=2):

            # Possibly invert the features / classes
            print 'inverting occurrences...'
            f_sum_ = 1-f_sum if f_inv else f_sum
            c_sum_ = 1-c_sum if c_inv else c_sum
            f = 1-features if f_inv else features
            c = 1-classes if c_inv else classes

            # Calculate joint feature and class frequency
            print 'calculating concordance...'
            fc = c.dot(f)/N

            # Add the contribution for this concordance to the mutual inf.
            print 'accumulating mutual information...'
            mutual_information += self.mutual_inf_term(fc,f_sum_,c_sum_)

        return mutual_information
            

    def mutual_inf_term(self, joint_ab, marginal_a, marginal_b):
        """
        Calculate one term in the mutual information sum.
        """
        return joint_ab * safe_log2(joint_ab/(marginal_a*marginal_b))


    def prune_features_keep_only(self, keep_features):
        """
        Eliminates any features not listed in ``keep_features``.
        """

        # Look at each count feature type, and prune any features that aren't
        # in ``keep_features``
        for feature_type in COUNT_BASED_FEATURES:
            feature_set = getattr(self, feature_type)

            # Do pruning on features stored for each lemma in the feature set
            for lemma in feature_set:
                for feature in feature_set[lemma].keys():
                    if feature not in keep_features:
                        del feature_set[lemma][feature]


    def prune_features(self, min_frequencey=5):
        """
        Eliminates features that occur in fewer than ``min_frequencey``
        examples.
        """
        # This operation makes normalized features stale
        self.fresh = set()
        self.threshold_features = {}

        # First we need to count how many times each feature occurs accross all
        # tokens.  We'll do this separately for the different feature types.
        # And then, for each, we'll remove any features.
        for feature_type in COUNT_BASED_FEATURES:

            # First count the occurrences of each feature for this feature type
            features = getattr(self, feature_type)
            feature_occurrences = Counter()
            for token in features:
                feature_occurrences.update(features[token].iterkeys())

            # Make a set of all features that occured too few times
            features_to_eliminate = {
                feature for feature in feature_occurrences 
                if feature_occurrences[feature] < min_frequencey
            }

            # Now eliminate those features
            for token in features:
                for feature in features[token].keys():
                    if feature in features_to_eliminate:
                        del features[token][feature]


    def prune_dictionary(self, min_frequencey=5):
        """
        Eliminiates words in the vocabulary (and the associated features we 
        collected) that occur less than 5 times.
        """
        # This operation makes normalized features stale
        self.fresh = set()
        self.threshold_features = {}

        # First, delegate to the underlying unigram dictionary, which will
        # prune itself and return the set of tokens that were removed (so we
        # can use it to prune the feature Counters).
        eliminated_tokens = self.dictionary.prune(min_frequencey)

        # Now go through each feature type and eliminate entries for the
        # removed words
        for feature_type in COUNT_BASED_FEATURES:
            features = getattr(self, feature_type)
            for token in eliminated_tokens:
                try:
                    del features[token]
                except KeyError:
                    pass


    def write(self, path):

        t4k.ensure_exists(path)

        # Save the dictionary
        self.dictionary.save(os.path.join(path, 'dictionary'))

        # Save each of the features
        for feature_type in COUNT_BASED_FEATURES:
            features = getattr(self, feature_type)
            open(os.path.join(path, feature_type + '.json'), 'w').write(
                json.dumps(features))


    def get_lexical_features(self, token, sentence):
        """
        Get's lexical features such as the occurrence and position of 
        neighboring tokens and bigrams, POSs, and lemmas.
        """

        # Establish a window around the token, but bounded by the sentence
        window = self.lexical_feature_window
        start = max(0, token['id'] - window)
        end = min(len(sentence['tokens']), token['id'] + window + 1)

        # Record the position-based lexical features
        lemma = token['lemma']
        for i in range(start, end):

            # Skip the token itself.
            if i == token['id']:
                continue

            # Get the relative position to the token
            rel_pos = i - token['id']

            # Get unigram features
            neighbor_token = sentence['tokens'][i]
            lemma_feature = '%s-(%d)' % (neighbor_token['lemma'], rel_pos)
            self.lemma_unigram[lemma][lemma_feature] += 1
            pos_feature = '%s-(%d)' % (neighbor_token['pos'], rel_pos)
            self.pos_unigram[lemma][pos_feature] += 1
            word_lowercase = neighbor_token['word'].lower()
            surface_feature = '%s-(%d)' % (word_lowercase, rel_pos)
            self.surface_unigram[lemma][surface_feature] += 1

            # If we can create bigram features at this location, do so.
            # This is possible either if we're looking at a token ahead of
            # the focal token separated by at least one token, or if we're
            # looking at a token after the focal token with at least one more
            # token left before the end of the sentnece.
            has_space_before = rel_pos < -1
            has_space_after = rel_pos > 0 and i < len(sentence['tokens'])-1
            if has_space_before or has_space_after:
                next_neighbor_token = sentence['tokens'][i+1]

                # Add lemma bigram features
                lemma_feature = '%s-%s-(%d)' % (
                    neighbor_token['lemma'], next_neighbor_token['lemma'],
                    rel_pos
                )
                self.lemma_bigram[lemma][lemma_feature] += 1

                # Add pos bigram features
                pos_feature = '%s-%s-(%d)' % (
                    neighbor_token['pos'], next_neighbor_token['pos'],
                    rel_pos
                )
                self.pos_bigram[lemma][pos_feature] += 1

                # Add surface bigram features
                next_word_lowercase = next_neighbor_token['word'].lower()
                surface_feature = '%s-%s-(%d)' % (
                    word_lowercase, next_word_lowercase, rel_pos)
                self.surface_bigram[lemma][surface_feature] += 1


    def get_dep_tree_features(self, token, depth=3):
        '''
        Extracts a set of features based on the dependency tree relations
        for the given token.  Each feature describes one dependency tree 
        relation in terms of a "signature".  The signature of a dependency
        tree relation consists of whether it is a parent or child, what the 
        relation type is (e.g. nsubj, prep:for, etc), and what the pos of the 
        target is.
        '''
        self.seen_tokens = set()
        add_features = self.get_dep_tree_features_recurse(token, depth)
        lemma = utils.normalize_token(token['lemma'])
        self.dependency[lemma].update(add_features)


    def get_dep_tree_features_recurse(self, token, depth=3):
        """
        Recursively construct dependency tree features.  A feature is a
        specific path along the dependency tree, characterized by the type of
        relation, whether it's a parent node or child node, and the part of
        speach of a node.

        Paths starting from the given ``token`` are recorded for up to
        ``depth`` hops.  ``self.seen_tokens`` ensures that a path doesn't go
        back on itself during recursive calls.
        """

        features = []
        self.seen_tokens.add(token['id'])

        # The call generates features so long as depth is greater than 0
        if depth < 1:
            return features

        # Record all parent signatures
        for relation, parent_token in token['parents']:

            # Don't back-track in traversal of dep-tree
            if parent_token['id'] in self.seen_tokens:
                continue

            feature = '%s:%s:%s' % ('parent', relation, parent_token['pos'])
            features.append(feature)

            recurse_features = self.get_dep_tree_features_recurse(
                parent_token, depth=depth-1)
            features.extend([
                '%s-%s' % (feature, rfeature) 
                for rfeature in recurse_features
            ])

        # Record all child signatures
        for relation, child_token in token['children']:

            # Don't back-track in traversal of dep-tree
            if child_token['id'] in self.seen_tokens:
                continue

            feature = '%s:%s:%s' % ('child', relation, child_token['pos'])
            features.append(feature)

            recurse_features = self.get_dep_tree_features_recurse(
                child_token, depth=depth-1)
            features.extend([
                '%s-%s' % (feature, rfeature) 
                for rfeature in recurse_features
            ])

        return features



    def get_baseline_features(self, token):
        '''
        Looks for specific syntactic relationships that are indicative of
        relational nouns.  Keeps track of number of occurrences of such
        relationships and total number of occurrences of the token.
        '''

        # Record all parent signatures
        lemma = utils.normalize_token(token['lemma'])
        for relation, child_token in token['children']:
            if relation == 'nmod:of' and child_token['pos'].startswith('NN'):
                self.baseline[lemma]['nmod:of:NNX'] += 1
            if relation == 'nmod:poss':
                self.baseline[lemma]['nmod:poss'] += 1

    
    # IDEA: look at mention property of tokens to see if it in a coref chain
    #     with named entity, demonym, placename, etc.
    #
    # IDEA: ability to infer in cases where conj:and e.g. '9fd7d0a5189bb351 : 2'
    def get_hand_picked_features(self, token):
        '''
        Looks for a specific set of syntactic relationships that are
        indicative of relational nouns.  It's a lot more thourough than 
        the baseline features.
        '''

        # Get features that are in the same noun phrase as the token
        lemma = utils.normalize_token(token['lemma'])
        NP_tokens = get_constituent_tokens(token['c_parent'], recursive=False)
        focal_idx = NP_tokens.index(token)
        for i, sibling_token in enumerate(NP_tokens):

            # Don't consider the token itself
            if sibling_token is token:
                continue

            # Get the position of this token relative to the focal token
            rel_idx = i - focal_idx

            # Note the sibling's POS
            key = 'sibling(%d):pos(%s)' % (rel_idx, sibling_token['pos'])
            self.hand_picked[lemma][key] += 1

            # Note if the sibling is a noun of some kind
            if sibling_token['pos'].startswith('NN'):
                self.hand_picked[lemma]['sibling(%d):pos(NNX)' % rel_idx] += 1

            # Note the sibling's named entity type
            key = 'sibling(%d):ner(%s)' % (rel_idx, sibling_token['ner'])
            self.hand_picked[lemma][key] += 1

            # Note if the sibling is a named entity of any type
            if sibling_token['ner'] is not None:
                self.hand_picked[lemma]['sibling(%d):ner(x)' % rel_idx] += 1

            # Note if the sibling is a demonym
            if sibling_token['word'] in GAZETTEERS['demonyms']:
                self.hand_picked[lemma]['sibling(%d):demonym' % rel_idx] += 1

            # Note if the sibling is a place name
            if sibling_token['word'] in GAZETTEERS['names']:
                self.hand_picked[lemma][
                    'sibling(%d):place-name' % rel_idx
                ] += 1

        # Note if the noun is plural
        if token['pos'] == 'NNS':
            self.hand_picked[lemma]['plural'] += 1

        # Detect construction "is a <noun> of"
        children = {
            relation:child for relation, child 
            in reversed(token['children'])
        }
        cop = children['cop']['lemma'] if 'cop' in children else None
        det = children['det']['lemma'] if 'det' in children else None
        nmod = (
            'of' if 'nmod:of' in children 
            else 'to' if 'nmod:to' in children 
            else None
        )
        poss = 'nmod:poss' in children 

        # In this section we accumulate various combinations of having
        # a copula, a prepositional phrase, a posessive, and a determiner.
        if nmod:
            self.hand_picked[lemma]['<noun>-prp'] += 1
            self.hand_picked[lemma]['<noun>-%s' % nmod] += 1

        if poss:
            self.hand_picked[lemma]['poss-<noun>'] += 1

        if cop and nmod:
            self.hand_picked[lemma]['is-<noun>-prp'] += 1
            self.hand_picked[lemma]['is-<noun>-%s' % nmod] += 1
            if det:
                self.hand_picked[lemma]['is-%s-<noun>-prp' % det] += 1

        if det and nmod:
            self.hand_picked[lemma]['%s-<noun>-prp' % det] += 1
            self.hand_picked[lemma]['%s-<noun>-%s' % (det, nmod)] += 1

        if cop and poss:
            self.hand_picked[lemma]['is-poss-<noun>'] += 1

        if det and poss:
            self.hand_picked[lemma]['%s-poss-<noun>' % det] += 1

        if det and not nmod and not poss:
            self.hand_picked[lemma]['%s-<noun>' % det] += 1
        
        if cop and det and poss:
            self.hand_picked[lemma]['is-det-poss-<noun>'] += 1

        if cop and det and nmod:
            self.hand_picked[lemma]['is-det-<noun>-prp'] += 1

        # Next we consider whether the propositional phrase has a named
        # entity, demonym, or place name in it
        if nmod:

            for prep_type in ['of', 'to', 'for']:

                # See if there is a prepositional noun phrase of this type, and
                # get it's head.  If not, continue to the next type
                NP_head = get_first_matching_child(
                    token, 'nmod:%s' % prep_type)
                if NP_head is None:
                    continue

                # Get all the tokens that are part of the noun phrase
                NP_constituent = NP_head['c_parent']
                NP_tokens = get_constituent_tokens(NP_constituent)

                # Add feature counts for ner types in the NP tokens
                ner_types = set([t['ner'] for t in NP_tokens])
                for ner_type in ner_types:
                    self.hand_picked[lemma][
                        'prp(%s)-ner(%s)' % (prep_type, ner_type)
                    ] += 1

                # Add feature counts for demonyms 
                lemmas = [t['lemma'] for t in NP_tokens]
                if any([l in GAZETTEERS['demonyms'] for l in lemmas]):
                    self.hand_picked[lemma][
                        'prp(%s)-demonyms' % prep_type
                    ] += 1

                # Add feature counts for place names 
                if any([l in GAZETTEERS['names'] for l in lemmas]):
                    self.hand_picked[lemma]['prp(%s)-place' % prep_type] += 1 
        
        # Next we consider whether the posessor noun phrase has a named
        # entity, demonym, or place name in it
        if poss:
            NP_head = get_first_matching_child(token, 'nmod:poss')
            NP_constituent = NP_head['c_parent']
            NP_tokens = get_constituent_tokens(NP_constituent)

            # Add feature counts for ner types in the NP tokens
            ner_types = set([t['ner'] for t in NP_tokens])
            for ner_type in ner_types:
                self.hand_picked[lemma]['poss-ner(%s)' % ner_type] += 1

            # Add feature counts for demonyms 
            lemmas = [t['lemma'] for t in NP_tokens]
            if any([l in GAZETTEERS['demonyms'] for l in lemmas]):
                self.hand_picked[lemma]['poss-demonyms'] += 1

            # Add feature counts for place names 
            if any([l in GAZETTEERS['names'] for l in lemmas]):
                self.hand_picked[lemma]['poss-place'] += 1 


def read_suffixes():
    suffixes = set(open(SUFFIX_PATH).read().split('\n'))
    suffixes.remove('')
    return suffixes


SUFFIXES = read_suffixes()
def get_suffix(lemma):
    suffixes = read_suffixes()
    for i in range(11,0,-1):
        potential_suffix = lemma[-i:]
        if potential_suffix in SUFFIXES:
            return potential_suffix
    return None



def get_first_matching_child(token, relation):
    '''
    Finds the first child of `token` in the dependency tree related by
    `relation`.
    '''
    try:
        return [
            child for rel, child in token['children'] if rel == relation
        ][0]

    except IndexError:
        return None


def get_constituent_tokens(constituent, recursive=True):

    tokens = []
    for child in constituent['c_children']:
        if isinstance(child, Token):
            tokens.append(child)
        elif recursive:
            tokens.extend(get_constituent_tokens(child, recursive))

    return tokens
 

def get_fnames(path):
    corenlp_path = os.path.join(path, 'CoreNLP')
    return t4k.ls(corenlp_path, absolute=True)
    return fnames


if __name__ == '__main__':

    # Accept the number of articles to process for feature extraction
    limit = int(sys.argv[1])

    # Extract and save the features
    extract_all_features(limit)

