import iterable_queue as iq
import multiprocessing
import os
import random
import t4k
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import json
import sys
sys.path.append('..')
from t4k import UnigramDictionary, UNK, SILENT
from collections import Counter, deque, defaultdict
from SETTINGS import RELATIONAL_NOUN_FEATURES_DIR
from kernels import bind_kernel, bind_dist
import utils
from utils import (
    ensure_unicode, get_dictionary, 
    #filter_seeds, 
    #get_features
)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as w
import extract_features
import itertools as it
import annotations


BEST_SETTINGS = {
    'on_unk': False,
    'C': 1.0,
    'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
    'semantic_similarity': 'res',
    'include_suffix' :  True,
    'syntactic_multiplier': 10.0,
    'semantic_multiplier': 2.0,
    'suffix_multiplier': 0.2
}


def make_classifier(

    kind,                   # 'osvm','svm','knn','wordnet', 'basic_syntax'
    X_train, Y_train,
    features,
    classifier_definition,

    on_unk=False,
    kernel=None,

    positives=None,
    negatives=None,
    neutrals=None,

    # SVM options
    C=1.0,
    min_feature_frequency=None,
    syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
    semantic_similarity='res',
    include_suffix = True,

    # How to weight the different types of features
    syntactic_multiplier=10.0,
    semantic_multiplier=2.0,
    suffix_multiplier=0.2,

    # KNN options
    k=3,
):
    """
    Convenience method to create a RelationalNounClassifier using
    default seed data and feature data

    # FeatureAccumulator or path load features from
    """

    # Use the provided features or load from the provided path
    if isinstance(features, basestring):
        features = extract_features.make_feature_accumulator(load=features)

    if min_feature_frequency is not None:
        features.prune_features(min_feature_frequency)

    # We can only use words that we actually have features for
    #positives = filter_seeds(positives, features.dictionary)
    #negatives = filter_seeds(negatives, features.dictionary)
    #neutrals = filter_seeds(neutrals, features.dictionary)

    # The proposed most-performant classifier
    if kind == 'osvm':
        print 'building an OrdinalSvmNounClassifier'
        return OrdinalSvmNounClassifier(
            positive_seeds=positives,
            neutral_seeds=neutrals,
            negative_seeds=negatives,

            # SVM options
            kernel=kernel,
            features=features,
            on_unk=False,

            C=C,
            syntax_feature_types=syntax_feature_types,
            semantic_similarity=semantic_similarity,
            include_suffix=include_suffix,

            syntactic_multiplier=syntactic_multiplier,
            semantic_multiplier=semantic_multiplier,
            suffix_multiplier=suffix_multiplier
        )

    # The proposed most-performant classifier
    elif kind == 'svm':
        print 'building SimplerSvmClassifier'
        # Pull out the expected options (tolerate unexpected options)
        options = t4k.select(
            classifier_definition, 
            [
                'kernel', 'C', 'gamma', 'class_weight', 'use_threshold', 
                'cache_size', 'count_based_features', 'non_count_features',
                'count_feature_mode', 'whiten', 'feature_threshold'
            ], 
            require=False
        )
        return SimplerSvmClassifier(X_train, Y_train, features, options)

    # A using logistic regression as the learner
    elif kind == 'logistic':
        print 'building LogisticNounClassifier'
        options = t4k.select(
            classifier_definition,
            ['C', 'solver', 'threshold', 'penalty', 'class_weight', 
                'multi_class', 'n_jobs',],
            require=False
        )
        return LogisticNounClassifier(
            X_train,
            Y_train,
            options
        )

    # Using naive bayes as the learner
    elif kind == 'NB':
        print 'building NaiveBayesNounClassifier'
        options = t4k.select(
            classifier_definition, ['alpha', 'threshold'], require=False
        )
        return NaiveBayesNounClassifier(X_train, Y_train, options)

    # Using RandomForest as the learner
    elif kind == 'RF':
        print 'building RandomForestNounClassifier'
        options = t4k.select(
            classifier_definition,
            ['n_estimators', 'criterion', 'max_features', 'max_depth', 
                'min_samples_split', 'min_samples_leaf', 
                'min_weight_fraction_leaf', 'max_leaf_nodes', 
                'min_impurity_split', 'n_jobs', 'oob_score', 'bootstrap', 
                'class_weight'],
            require=False
        )
        return RandomForestNounClassifier(X_train, Y_train, options)

    # Simple rule: returns true if query is hyponym of known relational noun
    elif kind == 'wordnet':
        print 'building a WordnetClassifier'
        return WordnetClassifier(positives, negatives)

    # Classifier using basic syntax cues
    elif kind == 'basic_syntax':
        print 'building a BalancedLogisticClassifier'
        get_features_func = arm_get_basic_syntax_features(
            features['baseline'])
        return BalancedLogisticClassifier(
            positives,
            negatives,
            get_features=get_features_func
        )

    else:
        raise ValueError('Unrecognized kind: %s' % kind)

    #elif kind == 'svm':
    #    print 'building an SvmNounClassifier'
    #    return SvmNounClassifier(
    #        positive_seeds=positives,
    #        negative_seeds=negatives,

    #        # SVM options
    #        features=features,
    #        on_unk=False,

    #        C=C,
    #        syntax_feature_types=syntax_feature_types,
    #        semantic_similarity=semantic_similarity,
    #        include_suffix=include_suffix,

    #        syntactic_multiplier=syntactic_multiplier,
    #        semantic_multiplier=semantic_multiplier,
    #        suffix_multiplier=suffix_multiplier
    #    )


#    # A runner up, using KNN as the learner
#    elif kind == 'knn':
#        print 'building a knn'
#        return OldKnnNounClassifier(
#            positives,
#            negatives,
#            # KNN options
#            features['dep_tree'],
#            features['dictionary'],
#            on_unk=False,
#            k=3
#        )






def balance_samples(populations, target='largest'):

    # A target of largest means adjust all samples to be as big as the
    # largest.  Figure out the largest sample size
    largest = 0
    if target == 'largest':
        target = max([len(pop) for pop in populations])

    print 'target:', target

    resampled_populations = []
    for pop in populations:

        # If the sample size is off target, adjust it
        if len(pop) != target:
            new_pop = resample(pop, target)

        # If the sample size is on target, we still want to make our own 
        # copy so we don't encourage side effects downstream
        else:
            new_pop = list(pop)

        resampled_populations.append(new_pop)

    return resampled_populations


def resample(population, sample_size):

    # If the sample needs to be smaller, just subsample without replacement
    if len(population) > sample_size:
        return random.sample(population, sample_size)

    # If the sample needs to be bigger, add additional samples with 
    # replacement
    else:

        # Use a list (we don't know what kind of iterable population is)
        population = list(population)

        # Add samples drawn randomly with replacement
        population.extend([
            random.choice(population) 
            for i in range(sample_size - len(population))
        ])
        return population


def arm_get_basic_syntax_features(features):
    '''
    Bind the features dictionary to a function that retrieves basic
    syntax features when given a token.  Suitable to be passed in as
    the `get_features` function for a BalancedLogisticClassifier
    '''

    def get_features(lemma):

        if lemma not in features:
            return [0, 0]

        count = features[lemma]['count']
        f1 = (
            features[lemma]['nmod:of:NNX'] #/ float(count)
            if 'nmod:of:NNX' in features[lemma] else 0
        )
        f2 = (
            features[lemma]['nmod:poss'] #/ float(count)
            if 'nmod:poss' in features[lemma] else 0
        )
        return [f1, f2]

    return get_features



class RandomForestNounClassifier(object):

    def __init__(self, X, Y, options={}):
        self.threshold = options.pop('threshold', None)
        self.classifier = RandomForestClassifier(**options)
        self.classifier.fit(X,Y)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def predict(self, X):
        if self.threshold is None:
            return self.classifier.predict(X)
        else:
            return np.array([
                1 if x > self.threshold else -1 for x in self.score(X)
            ])

    def score(self, X):
        classes = self.classifier.classes_
        if len(classes) == 2:
            return self.classifier.predict_proba(X)[:,1]
        else:
            return [
                (classes[np.argmax(x)], max(x))
                for x in self.classifier.predict_proba(X)
            ]


class NaiveBayesNounClassifier(object):

    def __init__(self, X, Y, options={}):
        self.threshold = options.pop('threshold', None)
        self.classifier = MultinomialNB(**options)
        self.classifier.fit(X,Y)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def predict(self, X):
        if self.threshold is None:
            return self.classifier.predict(X)
        else:
            return np.array([
                1 if x > self.threshold else -1 for x in self.score(X)
            ])

    def score(self, X):
        classes = self.classifier.classes_
        if len(classes) == 2:
            return self.classifier.predict_proba(X)[:,1]
        else:
            return [
                (classes[np.argmax(x)], max(x))
                for x in self.classifier.predict_proba(X)
            ]


class LogisticNounClassifier(object):
    '''
    Wraps a logistic regression classifier, making it able to operate
    on an artificially balanced dataset, but is adjusts the decision
    function based on the true prior.

    Assumes that the features are available for objects to be 
    classified using the get_features function.
    '''

    def __init__(self, X, Y, options={}):
        self.threshold = options.pop('threshold', None)
        self.classifier = LogisticRegression(**options)
        self.classifier.fit(X, Y)

#        adjustment = -np.log(
#            (1-self.prior)/self.prior
#            * training_prior/(1 - training_prior)
#        )
#        # Make the adjustment
#        self.classifier.intercept_ += adjustment

    def set_threshold(self, threshold):
        self.threshold = threshold


    def find_threshold(self, X, Y):
        """
        Adjusts the position(s) of the decision surfaces to optimize f1.
        """
        classes = self.classifier.classes_
        if len(classes) == 2:
            best_f1, threshold = utils.calculate_best_score(
                zip(self.classifier.decision_function(X), Y))
            self.threshold = np.array([threshold])
            return self.threshold

        else:
            scores = self.classifier.decision_function(X)
            label_scores = defaultdict(list)
            for correct_label, row_scores in zip(Y, scores):
                for label, score in zip(classes, row_scores):
                    if label == correct_label:
                        label_scores[label].append((score, 1))
                    else:
                        label_scores[label].append((score, -1))

            thresholds = []
            for label in classes:
                best_f1, threshold = utils.calculate_best_score(
                    label_scores[label])
                thresholds.append(threshold)

            self.threshold = np.array([thresholds])
            return self.threshold


    def adjusted_score(self, X):
        scores = self.classifier.decision_function(X)
        if self.threshold is None:
            return scores

        else:
            return scores - self.threshold


    def predict(self, X):
        if self.threshold is None:
            return self.classifier.predict(X)

        elif len(self.classifier.classes_) == 2:
            return np.array([
                1 if x > 0 else -1 for x in self.adjusted_score(X)
            ])

        else:
            return np.array([
                self.classifier.classes_[np.argmax(row_scores)] 
                for row_scores in self.adjusted_score(X)
            ])


    def score(self, X):
        classes = self.classifier.classes_
        if len(classes) == 2:
            return self.classifier.decision_function(X)

        else:
            return [
                (classes[np.argmax(x)], max(x))
                for x in self.classifier.decision_function(X)
            ]




class BasicSyntaxNounClassifier(object):
    '''
    Uses a logistic regression classifier to classify nouns based on basic
    syntax statistics.
    '''

    def __init__(
        self, 
        positive_seeds,
        negative_seeds,
        features,
        dictionary,
        prior=None,
        balance=True,
        do_adjust=True
    ):
        '''
        Note that if the prevalence of positives and negatives in the 
        dataset is not representative of their prior probabilities, you
        need to explicitly specifiy the prior probability of positive
        examples.

        If no prior is given, it will be assumed that the prior is

            len(positive_seens) / float(
                len(positive_seeds) + len(negative_seeds))

        Logistic regression is sensitive to data imbalances.  A better
        model can be achieved by training on a balanced dataset, and then
        adjusting the model intercept based on the true class prevalences.
        if `balance` is True, this is done automatically.
        '''
        self.positive_seeds = positive_seeds
        self.negative_seeds = list(negative_seeds)
        self.features = features
        self.dictionary = dictionary
        self.prior = prior
        self.balance = balance
        self.do_adjust = do_adjust

        # Calculate the prior as it exists in the data.  The meaning and
        # usefulness of this value will depend on the settings of `balance`
        # and `prior`
        total = float(len(positive_seeds) + len(negative_seeds))
        self.data_prior = len(positive_seeds) / total

        # If prior is not given, we assume that it is given by the data
        # prior
        if self.prior is None:
            self.prior = self.data_prior

        self.classifier = LogisticRegression(
            solver='newton-cg',
        )
        self.fit()


    def fit(self):
        X,Y = self.make_training_set()
        print 'balance:', sum(Y) / float(len(Y))
        self.classifier.fit(X,Y)

        # We'll need to adjust the model's intercept if it was not trained
        # on data distributed according to it's natural prior.  This
        # can happen if an explicit value for the prior was given or if
        # we have artificially balanced the dataset during training
        intercept_needs_adjustment = self.prior is not None or self.balance
        if intercept_needs_adjustment and self.do_adjust:
            self.adjust_intercept()


    def adjust_intercept(self):
        print 'adjusting...'

        # Determine what the apparent prior (in the training data) was
        if self.balance:
            training_prior = 0.5
        else:
            training_prior = self.data_prior

        # Determine the adjustment needed based on what the real prior is
        adjustment = -np.log(
            (1-self.prior)/self.prior
            * training_prior/(1 - training_prior)
        )

        # Make the adjustment
        self.classifier.intercept_ += adjustment


    def predict(self, tokens):
        tokens = maybe_list_wrap(tokens)
        lemmas = lemmatize_many(tokens)
        features = [self.get_features(lemma) for lemma in lemmas]
        return self.classifier.predict(features)


    def score(self, tokens):
        tokens = maybe_list_wrap(tokens)
        lemmas = lemmatize_many(tokens)
        features = [self.get_features(l) for l in lemmas]
        return self.classifier.predict_proba(features)[:,1]

    def get_features(self, lemma):

        if lemma not in self.features:
            return [0, 0]

        count = self.features[lemma]['count']
        f1 = (
            self.features[lemma]['nmod:of:NNX'] / float(count)
            if 'nmod:of:NNX' in self.features[lemma] else 0
        )
        f2 = (
            self.features[lemma]['nmod:poss'] / float(count)
            if 'nmod:poss' in self.features[lemma] else 0
        )
        return [f1, f2]


    def make_training_set(self):

        if self.balance:
            print 'balancing...'
            positive_seeds, negative_seeds = balance_samples(
                [self.positive_seeds, self.negative_seeds]
            )
        else:
            positive_seeds = self.positive_seeds
            negative_seeds = self.negative_seeds

        X = np.array(
            [self.get_features(s) for s in positive_seeds]
            + [self.get_features(s) for s in negative_seeds]
        )

        Y = np.array(
            [1]*len(positive_seeds) + [0]*len(negative_seeds)
        )

        return X, Y


class WordnetClassifier(object):

    def __init__(self, positive_seeds, negative_seeds):
        # Register the arguments locally
        self.positive_seeds = positive_seeds
        self.negative_seeds = negative_seeds
        self.fit()


    def fit(self):
        '''
        Get all the synsets that correspond to the positive and negative
        seeds
        '''
        self.positive_synsets = get_all_synsets(self.positive_seeds)
        self.negative_synsets = get_all_synsets(self.negative_seeds)


    def predict(self, tokens):

        # We expect an iterable of strings, but we can also accept a single
        # string.  If we got a single string, put it in a list.
        tokens = maybe_list_wrap(tokens)

        # Get the lemmas for the word to predict
        predictions = []
        for token in tokens:
            synset_deque = deque(w.synsets(token))
            predictions.append(self.predict_one(synset_deque))

        return predictions


    def predict_one(self, synset_deque):
        '''
        Do a breadth-first search, following the hypernyms of `lemma`
        in Wordnet, until one of the following conditionsn is met: 
            1) an synset corresponding to a positive seed is reached
            2) a synset corersponding to a negative seed is reached
            3) the hypernym root ("entity") is reached.
        if 1) occurs, return `True`, otherwise return `False`
        '''
        while len(synset_deque) > 0:

            # Pull the next synset of the queue (or in this case, deque)
            next_synset = synset_deque.popleft()

            # If we encountered a positive seed, return True
            if next_synset in self.positive_synsets:
                return True

            # If we encountered a negative seed, return True
            elif next_synset in self.negative_synsets:
                return False

            # Otherwise, add the parents (hypernyms), to be searched later
            else:
                synset_deque.extend(next_synset.hypernyms())

        # If we hit the hypernym root without finding any seeds, then
        # assume that the word is not relational (statistically most aren't)
        return False

    def score(self, tokens):
        '''
        The wordnet classifier inherently produces scores that are either
        1 or 0, so we just call to predict here.
        '''
        return self.predict(tokens)

            
class OrdinalSvmNounClassifier(object):

    def __init__(
        self,

        # Training data
        positive_seeds,
        neutral_seeds,
        negative_seeds,

        # Features to be used in the kernel
        kernel=None,
        features=None,

        # SVM options
        on_unk=False,

        # Which features to include
        C=1.0,
        syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
        semantic_similarity='res',
        include_suffix = True,

        # How to weight the different types of features
        syntactic_multiplier=10.0,
        semantic_multiplier=2.0,
        suffix_multiplier=0.2
    ):

        # We create two classifiers: one to distinguish positive from '
        # (neutral + negative) and the other to distinguish (positive + neutral)
        # from negative.

        # Our classifiers will share a kernel function that caches values so 
        # that they don't both need to calculate all kernel evaluations 
        # multiple times and so that the kernel evaluations on the training set 
        # can be multiprocessed

        # Make classifier for distinguishing negative from (neutral + positive)
        binary_negative = negative_seeds
        binary_positive = neutral_seeds + positive_seeds
        print 'training subclassifier 1 of 2'
        self.classifier_0 = SvmNounClassifier(
            binary_positive,
            binary_negative,
            kernel,
            features, on_unk, C,
            syntax_feature_types,
            semantic_similarity,
            include_suffix,
            syntactic_multiplier,
            semantic_multiplier,
            suffix_multiplier,
        )

        # Make classifier for distinguishing (negative + neutral) from positive
        binary_negative = negative_seeds + neutral_seeds
        binary_positive = positive_seeds
        print 'training subclassifier 2 of 2'
        self.classifier_1 = SvmNounClassifier(
            binary_positive,
            binary_negative,
            kernel,
            features, on_unk, C,
            syntax_feature_types,
            semantic_similarity,
            include_suffix,
            syntactic_multiplier,
            semantic_multiplier,
            suffix_multiplier,
        )

    def score_parallel(self, tokens, num_processes=16):

        # Make queues to parallelize work
        work_queue = iq.IterableQueue()
        result_queue = iq.IterableQueue()

        # put work
        work_producer = work_queue.get_producer()
        for token in tokens:
            work_producer.put(token)
        work_producer.close()

        # Start workers
        for p in range(num_processes):
            proc = multiprocessing.Process(
                target=self.score_worker,
                args=(work_queue.get_consumer(), result_queue.get_producer())
            )
            proc.start()

        # Collect results.  Close queues.
        result_consumer = result_queue.get_consumer()
        result_queue.close()
        work_queue.close()

        for token, score in result_consumer:
            yield token, score[0]


    def score_worker(self, work_consumer, result_producer):
        for token in work_consumer:
            score = self.score(token)
            result_producer.put((token, score))
        result_producer.close()


    def score(self, tokens):

        scores_0 = self.classifier_0.score(tokens)
        scores_1 = self.classifier_1.score(tokens)

        overall_scores = []
        for s0, s1 in zip(scores_0, scores_1):
            if s0 < 0 and s1 < 0:
                overall_scores.append(s0-1)
                print 'neg'

            elif s0 > 0 and s1 > 0:
                overall_scores.append(s1+1)
                print 'pos'

            elif s1 < 0 and s0 > 1:
                overall_scores.append((s0+s1)/(s0-s1))
                print 'neut'

            else:
                overall_scores.append((s0+s1)/(s1-s0))
                print 'ambig'

        return overall_scores


    def predict(self, tokens):
        scores = self.score(tokens)
        return [-1 if s <= -1 else 0 if s < 1 else 1 for s in scores]

        
SMALL_SETTINGS = {
    'features_dirname': 'features-small',
    'count_feature_mode':'normalized',
    'whiten': False,
    'C': 100.,
    'gamma': 0.001,
    'kind':'svm',
    'count_based_features': extract_features.COUNT_BASED_FEATURES,
    'non_count_features':[
        f for f in extract_features.NON_COUNT_FEATURES if f!='suffix'],
    'use_threshold': -0.23498053631221705,
}
BEST_SETTINGS = {
    'features_dirname': 'accumulated450-min_token_5-min_feat1000',
    'count_feature_mode':'normalized',
    'whiten': False,
    'C': 100.,
    'gamma': 0.001,
    'kind':'svm',
    'count_based_features': extract_features.COUNT_BASED_FEATURES,
    'non_count_features':[
        f for f in extract_features.NON_COUNT_FEATURES if f!='suffix'],
    'use_threshold': -0.23498053631221705,
}
def build_classifier(classifier_definition=BEST_SETTINGS):
    """
    Build the classifier specified by the given classifier_definition.
    By default, build the best classifier.
    """

    # Load the features
    features_dirname = classifier_definition['features_dirname']
    print 'loading features from %s...' % features_dirname
    features_path = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, features_dirname)
    features = extract_features.FeatureAccumulator(
        utils.read_wordnet_index(), load=features_path)

    # Load the training dataset
    dataset = annotations.Annotations(features.dictionary)
    tokens = dataset.examples.keys()
    pos = set([t for t in tokens if dataset.examples[t] > 0])
    neg = set([t for t in tokens if dataset.examples[t] == 0])
    train = {'pos':pos, 'neg':neg, 'neut':set()}

    # Convert the training and test sets into a numpy sparse matrix format
    count_based_features = classifier_definition.get(
        'count_based_features')
    non_count_features = classifier_definition.get('non_count_features')
    count_feature_mode = classifier_definition.get('count_feature_mode')
    whiten = classifier_definition.get('whiten', False)
    feature_threshold = classifier_definition.get('feature_threshold', 0.5)
    Q_train, X_train, Y_train = utils.make_vectors(
        train, features, count_based_features, non_count_features, 
        count_feature_mode, whiten=whiten, threshold=feature_threshold
    )

    # Make the classifier
    kind = classifier_definition.get('kind')
    clf = make_classifier(
        kind=kind,
        X_train=X_train, 
        Y_train=Y_train,
        features=features,
        classifier_definition=classifier_definition
    )

    ## Set the decision threshold to account for imbalanced dataset
    #clf.set_threshold(classifier_definition['use_threshold'])

    # TODO: Move this to an outer function that wants a classifier
    #Y_predicted = clf.predict(X_test)

    return clf


DEFAULT_SVM_OPTIONS = {
    'classifier': {
        'C': 1.0,
    },
    'kernel': {
        'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
        'semantic_similarity': 'res',
        'include_suffix': True,
        'syntactic_multiplier': 10.0,
        'semantic_multiplier': 2.0,
        'suffix_multiplier': 0.2
    }
}


class SimplerSvmClassifier(object):
    """
    Options
    - C=1.0: Constant affecting the SVM fitting routine
    - pre-bound-kernel: include a ready-bound kernel function to be used.
    - kernel: Subdictionary of options for binding the kernel
      -  syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
      -  semantic_similarity='res',
      -  include_suffix = True,
      -  syntactic_multiplier=0.33,
      -  semantic_multiplier=0.33,
      -  suffix_multiplier=0.33
            
    """

    # Make the init function take a default for options
    def __init__(self, X, Y, features=None, options={}):

        # Pop options related to features onto self namespace
        self.count_based_features = options.pop('count_based_features')
        self.non_count_features = options.pop('non_count_features')
        self.count_feature_mode = options.pop('count_feature_mode')
        self.whiten = options.pop('whiten', False)
        self.feature_threshold = options.pop('feature_threshold', None)

        # Pop option related to offset of the decision surface.
        self.threshold = options.pop('use_threshold', None)

        # If we have a "pre-bound-kernel", then pop it out of the options, 
        # and bind it's kernel function to the 'kernel' option, where the
        # SVC constructor expects it
        if 'pre-bound-kernel' in options:
            options['kernel'] = options.pop('pre-bound-kernel').eval_pair

        if 'class_weight' in options:
            print options['class_weight']

        self.features = features
        self.classifier = svm.SVC(**options)
        self.classifier.fit(X,Y)


    def set_threshold(self, threshold):
        self.threshold = threshold


    def find_threshold(self, X, Y):
        """
        Adjusts the position(s) of the decision surfaces to optimize f1.
        """

        # The approach to finding the threshold is a bit different depending
        # on whether we're doing a binary or ternary classification.
        # Binary is the simpler case, there's just one decision surface.
        if len(self.classifier.classes_) == 2:

            # Make a pairing of scores given by the classifier and correct 
            # labels
            scored_typed = zip(self.classifier.decision_function(X), Y)

            # Use a utility function that finds the cutoff score which, if
            # used as the classification criterion, yields the best f1
            best_f1, threshold= utils.calculate_best_score(
                scored_typed, positive=set([self.classifier.classes_[1]]),
                negative=set([self.classifier.classes_[0]])
            )

            # Set that best cutoff as the threshold used in classification
            self.threshold = np.array([threshold])
            return self.threshold

        elif len(self.classifier.classes_) > 2:

            # The first step is to collect the individual scores given for 
            # the classifications made between each label pair.  So, whenever
            # an example arises in the test set that matches one of the labels
            # in a pair, we record the score given from the decision surface
            # between those to labels, and what the correct result is.  
            # This means we accumulate a bunch of binary classification 
            # scores, along with correct answers, on a per-pair basis.  We 
            # can then use this to adjust the decision surface for each label
            # pair to optimize f1 for classifications between those labels.
            scores = self.classifier.decision_function(X)
            scores_per_label_pair = defaultdict(list)
            classes = self.classifier.classes_
            class_pairs = list(it.combinations(classes, 2))
            for label, row_scores in zip(Y, scores):

                for p,(i,j) in enumerate(class_pairs):

                    # If label i wins, then in the context of the
                    # sub-classification between i and j, the correct result is
                    # positive (1)
                    if i == label:
                        scores_per_label_pair[p].append((row_scores[p], 1))

                    # If label j wins, then the correct result is negative (-1)
                    elif j == label:
                        scores_per_label_pair[p].append((row_scores[p], -1))

            # Now that we have accumulated scores on a per-label-pair basis,
            # we consider each label pair in turn, and find the best cutoff
            # value for the decision surface between them.
            thresholds = []
            for p, score_labels in scores_per_label_pair.iteritems():
                best_f1, threshold = utils.calculate_best_score(score_labels)
                thresholds.append(threshold)

            # Save the threshold, and return it too.
            self.threshold = np.array([thresholds])
            return self.threshold



    def vectorize(self, tokens):
        """
        To classify tokens, we need to get their feature vectors.  But the
        particular form of the vectors needs to be the same as was used to
        train the classifier.  So, to simplify the classifier's API, it
        remembers its feature database, and knows how to convert new tokens
        into feature vectors.
        """

        # Won't work if bypassed adding features during classifier building.
        if self.features is None:
            raise ValueError(
                'No database of ``features`` was provided when building the '
                'classifier, so it cannot classify tokens by name: you must '
                'provide the feature vectors for the tokens you wish to '
                'classify, or provide a ``features`` database when building '
                'the classifier.'
            )

        return self.features.as_sparse_matrix(
            tokens,
            self.count_based_features,
            self.non_count_features,
            self.count_feature_mode,
            self.whiten,
            self.feature_threshold
        )


    def predict_tokens(self, tokens):
        X = self.vectorize(tokens)
        return self.predict(X)


    def predict(self, X):

        if self.threshold is None:
            return self.classifier.predict(X)

        else:

            if len(self.classifier.classes_) == 2:
                # If this is a binary classification, then we only have one
                # threshold to test, and the result is either positive or
                # negative.
                return np.array([
                    self.classifier.classes_[1] if x > 0
                    else self.classifier.classes_[0] 
                    for x in self.adjusted_score(X)
                ])

            elif len(self.classifier.classes_) == 3:

                return ternary_predict(
                    self.classifier.classes_, self.adjusted_score(X)) 

            else:
                NotImplementedError(
                    'Applying thresholds for more than 3 classes is '
                    'not supported.'
                )


    def adjusted_score(self, X):
        scores = self.classifier.decision_function(X)
        if self.threshold is None:
            return scores

        else:
            return scores - self.threshold


    def score(self, X):

        # We need to handle this in a special way if there are more than
        # two classes
        classes = self.classifier.classes_
        scores = self.classifier.decision_function(X)
        if len(classes) == 2:

            # When classification is binary, there is only one score to
            # consider.  And, because internally the classes are stored in
            # lexicographic order a positive score always corresponds to class
            # 1 beating class -1.  So we can return the scores unmodified
            return scores

        else:

            # The scores we will return are 2-tuples instead of floats.  The 
            # first position gives the label having the greatest score (whic
            # for our purposes is 1, 0, or -1, and the second label gives what
            # that score actually was.  That way, when scores are sorted,
            # they will primarily be sorted based on which label had the 
            # greatest score, and then secondarily, within each label, they 
            # will be sorted by which score was the greatest.  Now, since
            # we sort from the perspective that label -1 is a "negative" label
            # the higher it's score, the lower should be its sort order.
            # so, for examples where the -1 label is largest, we take the 
            # negative of the -1 label's score as the second position in the 
            # tuple
            real_scores = []
            predictions = self.classifier.predict(X)
            classes = self.classifier.classes_
            class_pairs = list(it.combinations(classes, 2))
            for prediction, row_scores in zip(predictions, scores):
                this_score = 0
                for p,(i,j) in enumerate(class_pairs):
                    if i == prediction:
                        prediction, 
                        this_score += row_scores[p]
                    elif j == prediction:
                        this_score -= row_scores[p]
                if prediction > -1:
                    real_scores.append((prediction, this_score))
                elif prediction == -1:
                    real_scores.append((prediction, -this_score))
                else:
                    raise ValueError(
                        'Unexpected predicted label: %s' % prediction)

            return real_scores


def ternary_predict(classes, scores):

    # If this is a ternary classification, we need to get the 
    # adjusted scores and then judge each item on its own.
    predictions = []
    class_pairs = list(it.combinations(classes, 2))
    for row_scores in scores:

        # For each row of scores, keep track of how many times each
        # label wins within face-offs that it is involved in.
        # Also, keep track of the sum of each label's scores which
        # is used to break ties.
        votes = Counter()
        scores = Counter()

        # Look at the result for each label pair face-off
        for p, (i,j) in enumerate(class_pairs):

            # If the result is greater than zero, it means the
            # first label won, else the second won.
            if row_scores[p] > 0:
                votes[i] += 1
            else:
                votes[j] += 1

            # Accumulate the scores.  The second label gets a
            # negated score because negative numbers mean a
            # preference for second label.
            scores[i] += row_scores[p]
            scores[j] -= row_scores[p]

        # Now find the label that won the most times.  At the same
        # time, find the label with the highest total score.
        max_votes = t4k.Max()
        max_scores = t4k.Max()
        for label in classes:
            max_votes.add(votes[label], label)
            max_scores.add(scores[label], label)

        # As long as there's no draw, the label that won the most
        # times wins overall.
        if not max_votes.is_draw():
            votes, prediction = max_votes.get()

        # If there's a draw, then take the label with the highest
        # total score.
        else:
            score, prediction = max_scores.get()

        # Store the winner for this example
        predictions.append(prediction)

    # Return all the "winners"
    return np.array(predictions)


class SvmNounClassifier(object):

    def __init__(
        self,

        # Training data
        positive_seeds,
        negative_seeds,

        # Features to be used in the kernel
        kernel=None,
        features=None,

        # SVM options
        on_unk=False,

        # Which features to include
        C=1.0,
        syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
        semantic_similarity='res',
        include_suffix = True,

        # How to weight the different types of features
        syntactic_multiplier=0.33,
        semantic_multiplier=0.33,
        suffix_multiplier=0.33
    ):
        '''
        on_unk [-1 || any]: (Stands for "on unknown token").  This
            controls behavior when a prediction is requested for a token 
            that did not appear in the features.dictionary:
                * -1 -- raise ValueError
                * anything else -- return that as the predicted class
        '''

        # Register parameters locally
        self.positive_seeds = positive_seeds
        self.negative_seeds = negative_seeds

        # Register SVM options
        self.kernel = kernel
        self.features = features
        self.on_unk = on_unk
        self.C = C
        self.syntax_feature_types = syntax_feature_types 
        self.semantic_similarity = semantic_similarity  
        self.include_suffix = include_suffix
        self.syntactic_multiplier = syntactic_multiplier 
        self.semantic_multiplier = semantic_multiplier  
        self.suffix_multiplier = suffix_multiplier

        # Make the underlying classifier
        self.classifier = self.make_svm_classifier()
        self.adjust_threshold()


    def make_svm_classifier(self):
        '''
        Make an SVM classifier
        '''
        X, Y = self.make_training_set()
        if self.kernel is None:
            
            kernel_func = bind_kernel(
                self.features, 

                syntax_feature_types=self.syntax_feature_types,
                semantic_similarity=self.semantic_similarity,
                include_suffix=self.include_suffix,

                syntactic_multiplier=self.syntactic_multiplier,
                semantic_multiplier=self.semantic_multiplier,
                suffix_multiplier=self.suffix_multiplier
            )

        else:
            kernel_func = self.kernel.eval

        classifier = svm.SVC(kernel=kernel_func, C=self.C)
        classifier.fit(X,Y)
        return classifier


    def adjust_threshold(self):

        # We'll find the best threshold.  First calculate scores for all
        # the training data.
        pos_scores = self.score(self.positive_seeds)
        neg_scores = self.score(self.negative_seeds)
        scored_typed = (
            [(s, 'pos') for s in pos_scores] 
            + [(s, 'neg') for s in neg_scores]
        )

        # Next, find the best threshold that optimizes f1 on training data.
        metric, threshold = utils.calculate_best_score(
            scored_typed, metric='f1')
        self.threshold = threshold



    def handle_unk(self, func, ids, lemmas):

        try:
            return func(ids)

        # If we get a ValueError, try to report the offending word
        except UnicodeDecodeError:
            raise
        except ValueError as e:
            try:
                offending_lemma = lemmas[e.offending_idx]
            except AttributeError:
                raise
            else:
                raise ValueError('Unrecognized word: %s' % offending_lemma)


    def predict_adjusted_threshold(self, tokens):
        # and need special handling of UNK tokens
        ids, lemmas = self.convert_tokens(tokens)
        return self.handle_unk(self.predict_adjusted_threshold_id, ids, lemmas)


    def predict_adjusted_threshold_id(self, token_ids):
        return [s > self.threshold for s in self.score_id(ids)]


    def predict(self, tokens):

        # and need special handling of UNK tokens
        ids, lemmas = self.convert_tokens(tokens)
        return self.handle_unk(self.predict_id, ids, lemmas)


    def convert_tokens(self, tokens):
        # Expects a list of lemmas, but can accept a single lemma too:
        # if that's what we got, put it in a list.
        if isinstance(tokens, basestring):
            tokens = [tokens]
        # Ensure tokens are unicode, lemmatize, and lowercased
        lemmas = lemmatize_many(tokens)

        # Convert lemma(s) to token_ids
        return self.features.dictionary.get_ids(lemmas), lemmas


    def score(self, tokens):
        ids, lemmas = self.convert_tokens(tokens)
        return self.handle_unk(self.score_id, ids, lemmas)
        

    def score_id(self, token_ids):
        scores = []
        for i, token_id in enumerate(token_ids):

            # Handle cases where the token_id corresponds to unknown token.
            if token_id == UNK:

                # Raise an error on unrecognized token_id (if desired)
                if self.on_unk < 0:
                    e = ValueError('Unrecognized token_id: %d' % token_id)
                    e.offending_idx = i
                    raise e

                # if desired behavior for unrecognized tokens is to 
                # classify as False, then return a score of negative infinty
                elif self.on_unk == 0:
                    scores.append(-np.inf)

                # if desired behavior for unrecognized tokens is to classify
                # as True, then return the smallest positive value.
                else:
                    scores.append(np.finfo(float).eps)

                continue

            scores.append(
                self.classifier.decision_function([[token_id]])[0]
            )

        return scores


    def predict_id(self, token_ids):
        predictions = []
        for i, token_id in enumerate(token_ids):

            # Handle cases where the token_id corresponds to unknown token.
            if token_id == UNK:

                # Raise an error on unrecognized token_id (if desired)
                if self.on_unk < 0:
                    e = ValueError('Unrecognized token_id: %d' % token_id)
                    e.offending_idx = i
                    raise e

                # Otherwise return the value assigned to on_unk 
                # as the class (default False)
                else:
                    predictions.append(self.on_unk)

                continue

            predictions.append(self.classifier.predict([[token_id]])[0])

        return predictions


    def make_training_set(self):
        # Make the training set.  Each "row" in the training set has a 
        # single "feature" -- it's the id which identifies the token.  
        # This will let us lookup the non-numeric features in kernel 
        # functions
        X = (
            [ [self.features.get_id(s)] for s in self.positive_seeds]
            + [ [self.features.get_id(s)] for s in self.negative_seeds]
        )
        Y = (
            [True] * len(self.positive_seeds) 
            + [False] * len(self.negative_seeds)
        )

        return X, Y




class OldKnnNounClassifier(object):
    '''
    Class that wraps underlying classifiers and handles training, testing,
    and prediction logic that is specific to making a relational noun 
    classifier.
    '''

    def __init__(
        self,
        positive_seeds,
        negative_seeds,

        # KNN options
        features,
        dictionary,
        on_unk=False,
        k=3,
    ):
        '''
        on_unk [-1 || any]: (Stands for "on unknown token").  This
            controls behavior when a prediction is requested for a token 
            that did not appear in the dictionary:
                * -1 -- raise ValueError
                * anything else -- return that as the predicted class
        '''

        # Register parameters locally
        self.positive_seeds = positive_seeds
        self.negative_seeds = negative_seeds

        # Register KNN options
        self.features = features
        self.dictionary = dictionary
        self.on_unk = on_unk
        self.k = k

        # Make the underlying classifier
        self.classifier = self.make_knn_classifier()


    def make_knn_classifier(self):
        '''
        Make a kNN classifier
        '''
        X, Y = make_training_set(
            self.positive_seeds, self.negative_seeds, self.dictionary)
        mydist = bind_dist(self.features, self.dictionary)
        classifier = KNN(metric=mydist, k=k)
        classifier.fit(X,Y)
        return classifier


    def handle_unk(self, func, ids, lemmas):

        try:
            return func(ids)

        # If we get a ValueError, try to report the offending word
        except ValueError as e:
            try:
                offending_lemma = lemmas[e.offending_idx]
            except AttributeError:
                raise
            else:
                raise ValueError('Unrecognized word: %s' % offending_lemma)


    def predict(self, tokens):
        ids, lemmas = self.convert_tokens(tokens)
        return self.handle_unk(self.predict_id, ids, lemmas)


    def convert_tokens(self, tokens):
        # Expects a list of lemmas, but can accept a single lemma too:
        # if that's what we got, put it in a list.
        if isinstance(tokens, basestring):
            tokens = [tokens]
        # Ensure tokens are unicode, lemmatize, and lowercased
        lemmas = lemmatize_many(tokens)

        # Convert lemma(s) to token_ids
        return self.dictionary.get_ids(lemmas), lemmas


    def score(self, tokens):
        ids, lemmas = self.convert_tokens(tokens)
        return self.handle_unk(self.score_id, ids, lemmas)
        

    def score_id(self, token_ids):
        scores = []
        for i, token_id in enumerate(token_ids):

            # Handle cases where the token_id corresponds to unknown token.
            if token_id == UNK:

                # Raise an error on unrecognized token_id (if desired)
                if self.on_unk < 0:
                    e = ValueError('Unrecognized token_id: %d' % token_id)
                    e.offending_idx = i
                    raise e

                # if desired behavior for unrecognized tokens is to 
                # classify as False, then return a score of negative infinty
                elif self.on_unk == 0:
                    scores.append(-np.inf)

                # if desired behavior for unrecognized tokens is to classify
                # as True, then return the smallest positive value.
                else:
                    scores.append(np.finfo(float).eps)

                continue

            scores.append(
                self.classifier.decision_function([[token_id]])[0]
            )

        return scores


    def predict_id(self, token_ids):
        predictions = []
        for i, token_id in enumerate(token_ids):

            # Handle cases where the token_id corresponds to unknown token.
            if token_id == UNK:

                # Raise an error on unrecognized token_id (if desired)
                if self.on_unk < 0:
                    e = ValueError('Unrecognized token_id: %d' % token_id)
                    e.offending_idx = i
                    raise e

                # Otherwise return the value assigned to on_unk 
                # as the class (default False)
                else:
                    predictions.append(self.on_unk)

                continue

            predictions.append(self.classifier.predict([[token_id]])[0])

        return predictions



class KNN(object):
    '''
    K-Nearest Neighbors classifier which accepts a custom distance function.
    I wrote this because the scikit KNN doesn't work well with custom
    distance functions that aren't true metrics, e.g. those based on 
    cosine distance
    '''

    def __init__(self, k=3, metric=None):
        self.k = k
        self.metric = metric
        if metric is None:
            self.metric = self.euclidean

    def euclidean(self, z,x):
        return np.linalg.norm(z-x)


    def fit(self, X, Y):
        self.X = X
        self.Y = Y


    def predict(self, z):

        if not isinstance(z, np.ndarray):
            z = np.array(z)
        if len(z.shape) == 0:
            z = np.array([[z]])
        elif len(z.shape) == 1:
            z = np.array([z])
        elif len(z.shape) > 2:
            raise ValueError(
                'Predict accepts a list of examples whose labels are '
                'to be predicted, or a single example.  Each examples '
                'should be a feature vector (or the type accepted by '
                'distance metric).'
            )

        # Make predictions for each example
        predictions = []
        for row in z:
            predictions.append(self._predict(row))

        return predictions


    def _predict(self, z):

        distances = [np.inf]*self.k
        labels = [None]*self.k

        # Iterate over all stored examples.  If the distance beats the
        # stored examples, keep it and the associated label
        for idx in range(len(self.X)):
            x, l = self.X[idx], self.Y[idx]
            d = self.metric(z,x)
            distances, labels = self.compare(d, l, distances, labels)

        # Now return the majority vote on the label
        return Counter(labels).most_common(1)[0][0]


    def compare(self, d, l, distances, labels):
        '''
        compare the distance d to the sorted (ascending) list of distances,
        starting from the end.  If d is less than any of the items, put it
        in its proper location (preserving sortedness), then truncate the
        list back to its original length.  Put l in the corresponding
        location in labels, and truncate it too.
        '''

        # First compare to the least element
        ptr = self.k
        while ptr > 0 and d < distances[ptr-1]:
            ptr -= 1

        if ptr < self.k:
            distances.insert(ptr, d)
            distances.pop()
            labels.insert(ptr, l)
            labels.pop()

        return distances, labels



###    Helper functions

def get_all_synsets(tokens):
    return set(t4k.flatten([
        w.synsets(lemma) for lemma in lemmatize_many(tokens)
    ]))


LEMMATIZER = WordNetLemmatizer()
def lemmatize(token):
    return LEMMATIZER.lemmatize(ensure_unicode(token).lower())


def lemmatize_many(tokens):
    return [lemmatize(token) for token in tokens ]


def maybe_list_wrap(tokens):
    '''
    Checks to see if `tokens` actually corresponds to a single token, in
    which case it gets put into a list.  Helps to overload methods so that
    they can accept a single token even though they are designed to accept
    an iterable of tokens.
    '''
    if isinstance(tokens, basestring):
        return [tokens]
    return tokens


def make_training_set(positive_seeds, negative_seeds, dictionary):
    '''
    Make the training set in the format expected by scikit's svm classifier,
    and my KNN classifier, based on positive and negative examples and
    a dictionary.  
    
    Each "row" in the training set has a 
    single "feature" -- it's the id which identifies the token.  
    This will let us lookup the non-numeric features in kernel 
    functions
    '''
    X = (
        [ [dictionary.get_id(s)] for s in positive_seeds]
        + [ [dictionary.get_id(s)] for s in negative_seeds]
    )
    Y = (
        [True] * len(positive_seeds) 
        + [False] * len(negative_seeds)
    )

    return X, Y
