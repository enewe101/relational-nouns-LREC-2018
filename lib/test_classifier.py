import t4k
import numpy as np
import multiprocessing
import iterable_queue as iq
import itertools
from functools import partial
import json
import os
import sys
sys.path.append('..')
from SETTINGS import DATA_DIR, RELATIONAL_NOUN_FEATURES_DIR
from t4k import UnigramDictionary, SILENT
import classifier as c
DICTIONARY_DIR = os.path.join(DATA_DIR, 'dictionary')
import utils
from utils import filter_seeds, ensure_unicode
from nltk.stem import WordNetLemmatizer
import annotations
import extract_features
import kernels
from collections import defaultdict, Counter

UNRECOGNIZED_TOKENS_PATH = os.path.join(DATA_DIR, 'unrecognized-tokens.txt')


def test_svm_ternary_fract_nowhite():
   feature_accumulator = extract_features.get_accumulated_features()
   feature_accumulator.normalize_features()
   kernel = 0


all_sets = set(['rand', 'guess', 'top'])
def evaluate_simple_classifier(
    annotations, features, kernel, test_sets, num_folds=3
):
    """
    Assesses the performance of the classifier defined by the dictionary
    ``classifier_definitions`` using the provided data.
    """
    # For the first test, we'll take a random split of the top words
    test_sets = set(test_sets)
    training_sets = all_sets - test_sets
    results = []
    for fold in range(num_folds):

        print '\nstarting fold %d\n' % fold
        # Get the test set folds.
        X_test_folds, Y_test_folds = annotations.get(test_sets)
        X_train_fold, X_test = t4k.get_fold(X_test_folds, num_folds, fold)
        Y_train_fold, Y_test = t4k.get_fold(Y_test_folds, num_folds, fold)

        # Initially the training set consists of data not in the test set, but 
        # we also add in test_set data not being tested in this fold.
        X_train, Y_train = annotations.get(training_sets, test_sets)
        X_train = np.concatenate([X_train, X_train_fold])
        Y_train = np.concatenate([Y_train, Y_train_fold])

        # Train the classifier
        options = {'pre-bound-kernel': kernel.eval}
        clf = c.SimplerSvmClassifier(X_train, Y_train, options)

        # Run classifier on test set
        prediction = clf.predict(X_test)

        # Adjust the predictions / labels to be in the correct range
        prediction = prediction / 2.0
        Y_test = np.array(Y_test) / 2.0

        precision, recall, f1 = get_ordinal_f1(prediction, Y_test)
        result = {
            'test': ','.join(test_sets),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print result
        results.append(result)

    return results



all_sets = set(['rand', 'guess', 'top'])
def evaluate_ordinal_classifier_(
    annotations, features, kernel, test_sets, num_folds=3
):
    """
    Assesses the performance of the classifier defined by the dictionary
    ``classifier_definitions`` using the provided data.
    """
    # For the first test, we'll take a random split of the top words
    test_sets = set(test_sets)
    training_sets = all_sets - test_sets
    results = []
    for fold in range(num_folds):

        print '\nstarting fold %d\n' % fold
        # Get the test set folds.
        test_pos, test_neut, test_neg = annotations.get_as_tokens(test_sets)
        pos_train_fold, pos_test_fold = t4k.get_fold(
            list(test_pos), num_folds, fold)
        neut_train_fold,neut_test_fold = t4k.get_fold(
            list(test_neut), num_folds, fold)
        neg_train_fold, neg_test_fold = t4k.get_fold(
            list(test_neg), num_folds, fold)

        train_pos, train_neut, train_neg = annotations.get_as_tokens(
            training_sets, test_sets)

        train_pos = list(train_pos) + list(pos_train_fold)
        train_neut = list(train_neut) + list(neut_train_fold)
        train_neg = list(train_neg) + list(neg_train_fold)

        # Train the classifier
        options = {'pre-bound-kernel': kernel.eval}
        # Train the classifier
        clf = c.OrdinalSvmNounClassifier(
            list(train_pos), list(train_neut), list(train_neg), kernel, features
        )

        # Run classifier on test set
        test_X = list(pos_test_fold) + list(neut_test_fold) + list(neg_test_fold)
        test_Y = (
            [1]*len(pos_test_fold) 
            + [0.5]*len(neut_test_fold) 
            + [0]*len(neg_test_fold)
        )

        prediction = np.array(clf.predict(test_X))
        prediction = (prediction + 1) / 2.0

        precision, recall, f1 = get_ordinal_f1(prediction, Y_test)
        result = {
            'test': ','.join(test_sets),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print result
        results.append(result)

    return results



    ## For the first test, we'll take a random split of all the annotated data.
    #print 'training on all, testing on holdout'
    ## Get the test set
    #X, Y = annotations.get(['top', 'rand', 'guess'])
    #X_train, X_test = t4k.get_fold(X, 5, 0)
    #Y_train, Y_test = t4k.get_fold(Y, 5, 0)

    ## Train the classifier
    #options = {'pre-bound-kernel': kernel.eval}
    #clf = c.SimplerSvmClassifier(X_train, Y_train, features, options)
    #
    ## Run classifier on test set
    #prediction = clf.predict(X_test)

    ## Adjust the predictions / labels to be in the correct range
    #prediction = prediction / 2.0
    #Y_test = np.array(Y_test) / 2.0

    #precision, recall, f1 = get_ordinal_f1(prediction, Y_test)
    #result = {
    #    'test': 'top,rand,guess (holdout)', 
    #    'train': 'top,rand,guess',
    #    'precision': precision,
    #    'recall': recall,
    #    'f1': f1
    #}
    #print result
    #results.append(result)


    tests = [
        #('top', ['rand', 'guess']),
        #('rand', ['top', 'guess']),
        #('guess', ['rand', 'top']),
        ('top', ['guess']),
        ('guess', ['top']),
        #('top', ['rand']),
    ]
    for test, train in tests:

        print 'training on %s; testing on %s.' % (str(train), str(test))
        # Get the test set
        X_train, Y_train = annotations.get(train)
        X_test, Y_test = annotations.get(test, train)

        # Train the classifier
        options = {'pre-bound-kernel': kernel.eval}
        clf = c.SimplerSvmClassifier(X_train, Y_train, features, options)
        
        # Run classifier on test set
        prediction = clf.predict(X_test)

        # Adjust the predictions / labels to be in the correct range
        prediction = prediction / 2.0
        Y_test = np.array(Y_test) / 2.0

        precision, recall, f1 = get_ordinal_f1(prediction, Y_test)
        result = {
            'test': test, 
            'train': ','.join(train), 
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print result
        results.append(result)

    return results



data_sources = ['guess', 'rand', 'top']
def evaluate_ordinal_classifier(annots=None, features=None, kernel=None):

    # Get the labels, features, and kernel
    if annots is None:
        annots = annotations.Annotations()
    if features is None:
        features = extract_features.get_accumulated_features()
    if kernel is None:
        kernel = kernels.PrecomputedKernel(features)
        # Precalculate the kernel for all the data in parallel
        kernel.precompute_parallel(annots.examples.keys())

    # Do tests with an ordinal svm
    results = {}
    results['ordinal'] = []
    for i in range(len(data_sources)):

        # Use one of the sets as the training set
        train = data_sources[:i] + data_sources[i+1:]
        test = data_sources[i]

        print 'training on %s; testing on %s.' % (str(train), str(test))
        # Get the test set
        train_pos, train_neut, train_neg = annots.get_as_tokens(train)
        test_pos, test_neut, test_neg = annots.get_as_tokens(test, train)

        # Train the classifier
        clf = c.OrdinalSvmNounClassifier(
            list(train_pos), list(train_neut), list(train_neg), kernel, features
        )
        
        # Set up the training set and evaluate the classifier
        x_test = list(test_pos) + list(test_neut) + list(test_neg)
        y_test = np.array(
            [1.0]*len(test_pos) + [0.5]*len(test_neut) + [0.0]*len(test_neg))
        prediction = np.array(clf.predict(x_test))

        # Adjust the predictions to be in the correct range / scale
        prediction = (prediction + 1) / 2.0
        precision, recall, f1 = get_ordinal_f1(prediction, y_test)
        result = {
            'test': test[0], 
            'train': ','.join(train), 
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print result
        results['ordinal'].append(result)

    return results


        
def get_ordinal_f1(predictions, actual):

    # Rescale the class coding.  Now a relational noun is 1.0
    # (for fully relational) and an occasionally relational is 0.5 (for 
    # halfway relational).
    positives = 0
    true_positives = 0
    false_positives = 0
    for predicted, actual in zip(predictions, actual):

        positives += actual
        if predicted == 0:
                pass
        elif predicted == 0.5:
            if actual == 0:
                false_positives += 0.5
            elif actual == 0.5:
                true_positives += 0.5
            elif actual == 1.0:
                true_positives += 0.5

        elif predicted == 1.0:
            if actual == 0:
                false_positives += 1.0
            elif actual == 0.5:
                true_positives += 0.5
                false_positives += 0.5
            elif actual == 1.0:
                true_positives += 1.0

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if positives == 0:
        recall = 0
    else:
        recall = true_positives / positives

    if precision+recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision , recall, f1


def get_MAP(n, ranks):
    '''
    Get the Mean Average Precision for the first n relevant documents.
    `ranks` should give the rank for each of the first n documents, 
    zero-indexed.
    '''

    # Define the MAP for zero results to be 1
    if n == 0:
        return 1.0

    # Look at the rank for the first n relevant documents (or as many are
    # available, if less than n) and accumulate the contributions each 
    # makes to the MAP
    numerator = 0
    precision_sum = 0
    average_precision_sum = 0
    for k in range(n):

        # Calculate the contribution of the kth relevant document 
        # to the MAP numerator.  If the k'th relevant doc didn't exist,
        # then it contributes nothing
        try:
            precision_sum += (k+1) / float(ranks[k]+1)
        except IndexError:
            pass

        average_precision_sum += precision_sum / float(k+1)

    mean_average_precision = average_precision_sum / float(n)

    return mean_average_precision


def get_top(
    n=50,
    kind='svm',
    on_unk=False,

    # SVM options
    syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
    semantic_similarity=None,
    syntactic_multiplier=1.0,
    semantic_multiplier=1.0,

    # KNN options
    k=3,
):
    '''
    Build a classifier based on the given settings, and then return the
    n highest-scoring words
    '''
    evaluator = get_map_evaluator(
        kind=kind,
        on_unk=on_unk,
        syntax_feature_types=syntax_feature_types,
        semantic_similarity=semantic_similarity,
        syntactic_multiplier=syntactic_multiplier,
        semantic_multiplier=semantic_multiplier,
        k=k,
    )
    evaluator.get_scores()
    print '\n'.join([s[1] for s in evaluator.scores[:n]])


def generate_classifier_definitions(
    parameter_ranges,
    constants={}
):
    tagged_param_values = []
    for param in parameter_ranges:
        this_param_tagged_values = [
            (param, val) for val in parameter_ranges[param]]
        tagged_param_values.append(this_param_tagged_values)

    definitions = []
    for combo in itertools.product(*tagged_param_values):
        new_def = dict(constants)
        new_def.update(dict(combo))
        definitions.append(new_def)

    return definitions


class SearchValues(list):
    """
    This is just a special kind of list, which, because it has a different
    Name, can be told appart from normal lists using `isinstance`.  No new
    functionality!
    """
    pass


def expand_run_specification(spec):
    search_keys = [k for k in spec if isinstance(spec[k], SearchValues)]
    specs = recurse_expand_specifications(spec, search_keys)
    return specs


def recurse_expand_specifications(spec, search_keys):

    # If there are no search keys, just return the spec unchanged.
    # This is how recursion bottoms out.  Wrap the spec in a list for
    # consistency with non-leaf calls.
    if len(search_keys) == 0:
        return [spec]

    # Pop one of the search keys, and make copies of the spec with each 
    # value of the search key.  Recurse with each copy so that other search
    # keys get expanded combinatorially.
    specs = []
    search_key = search_keys.pop()
    for val in spec[search_key]:

        # Copy the spec, setting the value for one of the search keys
        new_spec = t4k.merge_dicts({search_key:val}, spec)

        # Copy search_keys to prevent side effect in recursive call.
        new_search_keys = list(search_keys) 

        # Recurse, and accumulate the expanded specs
        specs.extend(
            # Here is the recursive call, which expands the keys other 
            # than `search_key`.
            recurse_expand_specifications(new_spec, new_search_keys)
        )

    # Return the expansions to the previous level
    return specs

   

def optimize_classifier(
    classifier_definitions,
    features,
    train_pos, train_neg,
    test_pos, test_neg,
    out_path,
    num_procs=12
):

    # Open the file where we'll write the results
    out_f = open(out_path, 'w')

    # Open queues to spread work and collect results
    work_queue = iq.IterableQueue()
    results_queue = iq.IterableQueue()

    # Load all of the classifier definitions onto the work queue, then close it
    work_producer = work_queue.get_producer()
    for clf_def in classifier_definitions:
        work_producer.put(clf_def)
    work_producer.close()

    # Start a bunch of workers, give them iterable queue endpoints.
    for proc in range(num_procs):
        p = multiprocessing.Process(
            target=evaluate_classifiers,
            args=(
                work_queue.get_consumer(),
                results_queue.get_producer(), 
                features, 
                train_pos, train_neg,
                test_pos, test_neg
            )
        )
        p.start()

    # Get an endpoint for collecting the results
    results_consumer = results_queue.get_consumer()

    # We're done making queue endpoints
    work_queue.close()
    results_queue.close()

    # Collect the results, and write them to disc
    best_score, best_threshold, best_clf_def = None, None, None
    for score, threshold, clf_def in results_consumer:

        # Write the result to stdout and to disc
        performance_record = '%f\t%f\t%s\n' % (score, threshold, str(clf_def))
        print performance_record
        out_f.write(performance_record)

        # Keep track of the best classifier definition and its performance
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_clf_def = clf_def

    # Write the best performance out to sdout and disc
    best_performance_record = '%f\t%f\t%s\n' % (
        best_score, best_threshold, str(best_clf_def))
    print best_performance_record
    out_f.write('\nBest:\n')
    out_f.write(best_performance_record)


def evaluate_classifiers(
    classifier_definitions, results_queue, 
    features, 
    #train_pos, train_neg,
    #test_pos, test_neg
):

    # Evaluate performance of classifier for each classifier definition, and
    # put the results onto the result queue.
    for clf_def in classifier_definitions:
        f1, evaluation_details = evaluate_classifier(clf_def, features)
        results_queue.put((f1, evaluation_details, clf_def))

    # Close the results queue when no more work will be added
    results_queue.close()
    




#def vectorize_tokens(tokens, features, classifier_definition=BEST_SETTINGS):
#    """
#    Build the classifier specified by the given classifier_definition.
#    By default, build the best classifier.
#    """
#
#    # Convert the training and test sets into a numpy sparse matrix format
#    count_based_features = classifier_definition.get(
#        'count_based_features')
#    non_count_features = classifier_definition.get('non_count_features')
#    count_feature_mode = classifier_definition.get('count_feature_mode')
#    whiten = classifier_definition.get('whiten', False)
#    threshold = classifier_definition.get('feature_threshold', 0.5)
#
#    return features.as_sparse_matrix(
#        tokens,
#        count_based_features, non_count_features, count_feature_mode, whiten,
#        threshold
#    )


def evaluate_classifier(name, classifier_definition, features, out_path=None):
    """
    Assesses the performance of the classifier defined by the dictionary
    ``classifier_definitions``.  That dictionary should provide the arguments
    needed to construct the classifier when provided to the function
    classifier.make_classifier.
    """

    print 'evaluating:', json.dumps(classifier_definition, indent=2)
    print 'writing result to:%s' % out_path

    errors_path = classifier_definition.get(
        'error-analysis-path', 'analyze-errors.tsv')
    print 'errors_path:', errors_path
    analyze_errors_f = open(os.path.join(DATA_DIR, errors_path), 'w')

    if out_path is not None:
        out_file = open(out_path, 'a')

    # Some of the "classifier definition" settings control the train / test
    # data and features supplied to the classifier, rather than the classifiers
    # config.  Handle those settings now.

    # Get the desired datset
    data_source = classifier_definition.get('data_source', None)
    seed = classifier_definition.get('seed', 0)
    print 'seed:', seed
    if data_source == 'seed':
        train, test = utils.get_train_test_seed_split(features.dictionary)
    elif data_source == 'crowdflower-annotated-top':
        train, test = annotations.Annotations(
            features.dictionary).get_train_test('top', seed=seed)
    elif data_source == 'crowdflower-annotated-rand':
        train, test = annotations.Annotations(
            features.dictionary).get_train_test('rand', seed=seed)
    elif data_source == 'crowdflower-dev-top':
        train, test = annotations.Annotations(
            features.dictionary).get_train_dev('top')
    elif data_source == 'crowdflower-dev-rand':
        train, test = annotations.Annotations(
            features.dictionary).get_train_dev('rand')
    elif data_source == 'crowdflower-dev':
        train, test = annotations.Annotations(
            features.dictionary).get_train_dev()
    else:
        raise ValueError('Unexpected data_source: %s' % data_source)

    # Binarize the dataset if desired by converting items labelled `neutral`
    # into either positive or negative.
    binarize_mode = classifier_definition.get('binarize_mode', None)
    if binarize_mode is not None:
        if binarize_mode == '+/0-':
            train['neg'] = train['neut'] | train['neg']
            test['neg'] = test['neut'] | test['neg']
        elif binarize_mode == '+0/-':
            train['pos'] = train['pos'] | train['neut']
            test['pos'] = test['pos'] | test['neut']
        else:
            raise ValueError('Unexpected binarize_mode: %s' % binarize_mode)
        train['neut'] = set()
        test['neut'] = set()

    # Convert the training and test sets into a vectorized format. In the
    # "kernel" format, the feature vectors are just token ids (the kernel
    # function is "smart" and knows how to compute the kernels given ids).
    data_format = classifier_definition.get('data_format')
    if data_format == 'kernel':
        X_train, Y_train = utils.make_kernel_vector(train, features)
        X_test, Y_test = utils.make_kernel_vector(test, features)
        classifier_definition['verbose'] = 1
        precomputed_kernel = kernels.PrecomputedKernel(
            features, classifier_definition)

        # Trigger the necessary various lazy calculates on the features class 
        # by doing one kernel calculation
        precomputed_kernel.eval_pair_token('car', 'tree')
        # Now precompute values of the kernel for all example pairs
        num_processes = classifier_definition.get('kernel_processes', 4)
        precomputed_kernel.precompute_parallel(
            examples=X_train + X_test, num_processes=num_processes
        )
        classifier_definition['pre-bound-kernel'] = precomputed_kernel

    # Or convert the training and test sets into a numpy sparse matrix format
    elif data_format == 'vector':
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
        Q_test, X_test, Y_test = utils.make_vectors(
            test, features, count_based_features, non_count_features, 
            count_feature_mode, whiten=whiten, threshold=feature_threshold
        )

    else:
        raise ValueError('Unexpected data_format: %s' % data_format)

    # Allow automatically re-weighting the class to help with unbalanced
    # classifications.
    if 'class_weight' in classifier_definition:
        if classifier_definition['class_weight'] == 'auto':
            classifier_definition['class_weight'] = get_class_weights(Y_train)

    # Make the classifier
    kind = classifier_definition.get('kind')
    clf = c.make_classifier(
        kind=kind,
        X_train=X_train, 
        Y_train=Y_train,
        features=features,
        classifier_definition=classifier_definition
    )

    # We can either tune the decision threshold
    if classifier_definition.get('find_threshold', False):
        decision_threshold = clf.find_threshold(X_test, Y_test).tolist()

    # Or set it based on a prior fitted value
    elif classifier_definition.get('use_threshold', None) is not None:
        decision_threshold = classifier_definition['use_threshold']
        clf.set_threshold(decision_threshold)

    # Or stick with the default decision threshold built into the classifier
    else:
        decision_threshold = None

        ## If binarize mode has been set, then the classifier is already 
        ## configured to find treat the problem as a binary classification,
        ## so finding the classification threshold is straightforward.
        #if binarize_mode is '+0/-'
        #    best_loose_f1, loose_threshold = utils.find_threshold(
        #        clf, X_test, Y_test)
        #    best_strict_f1, strict_threshold = None, None
        #    clf.set_threshold(threshold)

        #elif binarize_mode is '+/0-'
        #    best_strict_f1, strict_threshold = utils.find_threshold(
        #        clf, X_test, Y_test)
        #    best_loose_f1, loose_threshold = None, None
        #    clf.set_threshold(threshold)

        #else:
        #    best_strict_f1, strict_threshold = utils.find_threshold(
        #        clf, X_test, Y_test_strict, positive=set([1,0]), 
        #        negative=set([-1])
        #    )
        #    best_loose_f1, loose_threshold = utils.find_threshold(
        #        clf, X_test, Y_test_strict, positive=set([1]), 
        #        negative=set([0,-1])
        #    )

    # Test it on the test set, generating a confusion matrix

    Y_predicted = clf.predict(X_test)
    for word, pred, actual in zip(Q_test, Y_predicted, Y_test):
        if actual == 1:
            print '%s : %d' % (word, pred)
            analyze_errors_f.write('%s\t%d\n' % (word, pred))

    confusion_matrix = generate_confusion_matrix(clf, X_test, Y_test)

    # Calculate the F1 relative to each class, and the macro-average
    if binarize_mode is None:
        f1s, macro_f1 = calculate_f1(confusion_matrix)
        tight_f1 = f1s[1]
        loose_f1 = calculate_f1_loose(confusion_matrix)

    elif binarize_mode == '+0/-':
        loose_f1 = calculate_simple_f1(confusion_matrix)
        tight_f1 = None
        macro_f1 = None

    elif binarize_mode == '+/0-':
        tight_f1 = calculate_simple_f1(confusion_matrix)
        loose_f1 = None
        macro_f1 = None

    # Calculate the MAP
    AP = calculate_classifier_MAP(clf, X_test, Y_test, [1,0])

    results = {
        'confusion_matrix':confusion_matrix,
        'precision':loose_f1[0],
        'recall':loose_f1[1],
        'f1':loose_f1[1],
        'AP':AP,
        'threshold':decision_threshold
    }

    performance_record = (
        '{"name":"%s",\n\n' % name
        + '"run-specification":' 
        + json.dumps(classifier_definition, indent=2) + ','
        + '\n\n'
        + '"results":'
        + json.dumps(results) + '}\n\n\n'
    )

    print performance_record
    if out_path is not None:
        out_file.write(performance_record)

    return clf, results


def get_class_weights(Y):
    classes = set(Y)
    class_amounts = {c:sum([y==c for y in Y]) for c in classes}
    max_class_amount = max(class_amounts.values())
    return {c:max_class_amount/float(class_amounts[c]) for c in classes}


def calculate_classifier_MAP(classifier, X, Y, relevant_labels):
    """
    Given a ``classifier``, a set of examples `X` with known labels ``Y``,
    and the set of labels that should be considered relevant 
    (``relevant_labels``), determine the average precision for predictions on 
    the set X.

    The classifier must implement a method called ``score``, which returns
    higher values for the relevant labels and lower values for the non-relevant
    labels.  X should  be a numpy 2-D array of features (each row is a feature
    vector for a single example), and Y should be a 1-D array of labels in
    corresponding order to X.
    """
    relevant_labels = set(relevant_labels)
    relevancies = [y in relevant_labels for y in Y]
    scores = classifier.score(X)
    return calculate_MAP(scores, relevancies)


def calculate_MAP(scores, relevancies):

    # Sort the items based on scores, keeping the relevencies along for the
    # ride.  Settle ties randomly.
    scores = sorted(
        zip(scores, relevancies), 
        key=lambda x: (x[0], np.random.random()),
        reverse=True
    )

    # Sum the precisions at each true positive 
    precision_sums = 0.0
    num_relevant = 0.0
    num_checked = 0.0
    precisions = []
    for score, is_relevant in scores:
        num_checked += 1
        if is_relevant:
            num_relevant += 1
            precision_sums += num_relevant / num_checked
            precisions.append(num_relevant / num_checked)

        #print '\n\n' + '-'*24 + '\n\n'
        #print 'is relevant:', is_relevant
        #if len(precisions) > 0:
        #    print 'this precision:', precisions[-1]
        #    print 'average precision:', np.mean(precisions)

    return precision_sums / num_relevant


def calculate_f1(confusion_matrix):
    """
    Returns F1 relative to each class, as well as the macro-average of the 
    F1 scores.
    """

    # First calculate the individual and macro f1
    f1s = {}
    labels = confusion_matrix.keys()
    for y in labels:

        positives_predicted = float(
            sum([confusion_matrix[y_][y] for y_ in labels]))
        if positives_predicted == 0:
            precision = 1
        else:
            precision = confusion_matrix[y][y] / positives_predicted

        actual_positives = float(
            sum([confusion_matrix[y][y_] for y_ in labels]))
        if actual_positives == 0:
            recall = 1
        else:
            recall = confusion_matrix[y][y] / actual_positives

        if precision + recall == 0:
            f1s[y] = 0
        else:
            f1s[y] = 2 * precision * recall / (precision + recall)

    macro_f1 = sum(f1s.values()) / float(len(f1s))

    return f1s, macro_f1


def calculate_simple_f1(confusion):
    positives_predicted = sum([confusion[y][1] for y in confusion])
    if positives_predicted == 0:
        precision = 1
    else:
        precision = confusion[1][1] / float(positives_predicted)

    actual_positives = sum([confusion[1][y] for y in confusion])
    if actual_positives == 0:
        recall = 1
    else:
        recall = confusion[1][1] / float(actual_positives)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def calculate_f1_loose(confusion):

    positives_predicted = float(
        sum([confusion[y][1] for y in confusion])
        + sum([confusion[y][0] for y in confusion])
    )
    print positives_predicted

    true_positives = (
        confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1])
    if positives_predicted == 0:
        precision = 1
    else:
        precision = true_positives / positives_predicted

    actual_positives = (
        sum([confusion[1][y] for y in confusion])
        + sum([confusion[0][y] for y in confusion])
    )
    if actual_positives == 0:
        recall = 1
    else:
        recall = true_positives / float(actual_positives)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def generate_confusion_matrix(classifier, X_test, Y_test):
    confusion_matrix = defaultdict(Counter)
    Y_predicted = classifier.predict(X_test)
    for y, y_pred in zip(Y_test, Y_predicted):
        confusion_matrix[y][y_pred] += 1

    return confusion_matrix


def diagnose_map_evaluators(
    classifier_definitions=[],
    map_evaluators=None,
    out_path=UNRECOGNIZED_TOKENS_PATH,
    n=100
):
    '''
    Given a set of classifier definitions (which should be a list of
    dictionaries containing keyword arguments for the function 
    get_map_evaluator), create the classifier for each, and find the
    unrecognized tokens (i.e. not in the test set) for each of the 
    classifier's n top-scoring tokens
    '''
    out_file = open(UNRECOGNIZED_TOKENS_PATH, 'w')

    # Create the map evaluators, if they weren't supplied
    if map_evaluators is None:
        map_evaluators = [
            (cdef, get_map_evaluator(**cdef)) 
            for cdef in classifier_definitions
        ]

    # For each classifier, find the unrecognized tokens among the top n
    # scoring tokens, and write them to file
    for cdef, map_evaluator in map_evaluators:
        unrecognized_tokens = map_evaluator.diagnose_MAP(n)
        out_file.write(str(cdef)+'\n')
        out_file.write('\n'.join(unrecognized_tokens) + '\n\n')

    return map_evaluators


def get_map_evaluator(
    kind='svm',
    on_unk=False,
    syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
    semantic_similarity=None,
    syntactic_multiplier=1.0,
    semantic_multiplier=1.0,
    k=3,
):
    classifier = c.make_classifier(
        kind=kind,
        on_unk=on_unk,
        syntax_feature_types=syntax_feature_types,
        semantic_similarity=semantic_similarity,
        syntactic_multiplier=syntactic_multiplier,
        semantic_multiplier=semantic_multiplier,
        k=k,
    )
    train_positives, train_negatives, train_neutrals = get_train_sets()
    test_positives, test_negatives, test_neutrals = get_test_sets()
    evaluator = RelationalNounMapEvaluator(
        classifier, train_positives, train_negatives, test_positives,
        test_negatives
    )

    return evaluator


def get_descriminant_func(test_positives=None, test_negatives=None):

    if test_positives is None or test_negatives is None:
        test_positives, test_negatives = get_test_sets()
    
    def descriminant_func(result):
        try:
            if result in test_positives:
                return True
            elif result in test_negatives:
                return False
            else:
                raise ValueError(
                    'Cannot decide if %s is relevant' % str(result)
                )
        except TypeError:
            print type(result), repr(result)

    return descriminant_func



class RelationalNounMapEvaluator(object):

    def __init__(
        self, 
        classifier,
        train_positives, 
        train_negatives,
        test_positives,
        test_negatives,
    ):

        # Register parameters
        self.classifier = classifier
        self.training_data = train_positives | train_negatives
        self.testing_data = test_positives | test_negatives
        self.test_positives = test_positives
        self.test_negatives = test_negatives
        self.vocabulary = utils.read_wordnet_index()

        # Make a MAP evaluator
        descriminant_func = get_descriminant_func(
            test_positives, test_negatives
        )
        self.evaluator = MapEvaluator(descriminant_func)

        # This will hold the "relational-nounishness-scores" to rank the
        # nouns that seem most relational
        self.scores = None


    def diagnose_MAP(self, n):
        # Get classifier's scores for all tokens (if not already done)
        if self.scores is None:
            self.get_scores()

        # Go through the top n tokens and collect all unrecognized ones
        print 'checking...'
        num_seen = 0
        unrecognized_tokens = []
        top_tokens = (s[1] for s in self.scores)
        for token in top_tokens:

            # Check if this token is recognized
            if token not in self.testing_data:
                unrecognized_tokens.append(token)

            # Only look at the top n tokens
            num_seen += 1
            if num_seen >= n:
                break

        return unrecognized_tokens


    def get_MAP(self, n):
        if self.scores is None:
            self.get_scores()

        top_tokens = (s[1] for s in self.scores)
        return self.evaluator.get_MAP(n, top_tokens)


    def get_scores(self):
        # Get the scores for each token in the vocabulary
        # Skip any tokens that were in the training_data!
        print 'scoring...'
        self.scores = [
            (self.classifier.score(token), token)
            for token in self.vocabulary
            if token not in self.training_data
        ]
        print 'sorting...'
        self.scores.sort(reverse=True)



class MapEvaluator(object):
    '''
    Class for evaluating Mean Average Precision for some result-iterator.
    It requires a `descriminant_funct` -- a function that can distinguish 
    which results are relevant and which are irrelevant.
    '''

    def __init__(self, descriminant_func):
        self.descriminant_func = descriminant_func


    def get_MAP(self, n, results_iterator):
        '''
        Given some iterator that yields results (`results_iterator`), 
        calculate it's Mean Average Precision. 
        '''
        ranks = self.get_ranks(n, results_iterator)
        return get_MAP(n, ranks)


    def get_ranks(self, n, results_iterator):
        '''
        Given some iterator that yields results (`results_iterator`), 
        find the ranks for the first n relevant results.  
        '''

        # If n is 0, we don't have to report any ranks at all
        if n == 0:
            return []

        i = 0        # indexes relevant results
        ranks = []    # stores ranks

        # Get the ranks for the first `n` relevant
        for rank, result in enumerate(results_iterator):

            # If a result is positive, save it's rank
            if self.descriminant_func(result):
                ranks.append(rank)
                i += 1

            # Stop if we've found `n` relevant results
            if i >= n:
                break

        return ranks


def cross_val_positives(classifier='svm', clf_kwargs={}, use_wordnet=False):
    positive_seeds, negative_seeds = get_train_sets()
    features = load_features()
    dictionary = get_dictionary(features)
    positive_seeds = filter_seeds(positive_seeds, dictionary)
    negative_seeds = filter_seeds(negative_seeds, dictionary)

    num_correct = 0
    num_tested = 0
    for test_item in positive_seeds:
        positive_seeds_filtered = [
            p for p in positive_seeds if p is not test_item
        ]

        args = (features,dictionary,positive_seeds_filtered,negative_seeds)
        if classifier == 'svm':
            clf = make_svm_classifier(
                *args, clf_kwargs=clf_kwargs, use_wordnet=use_wordnet)
        elif classifier == 'knn':
            clf = make_knn_classifier(*args, clf_kwargs=clf_kwargs)
        else:
            raise ValueError('Unexpected classifier type: %s.' % classifier)

        num_tested += 1
        prediction = clf.predict(test_item)[0]
        if prediction:
            num_correct += 1

        padding = 40 - len(test_item)
        print test_item, ' ' * padding, 'correct' if prediction else '-'

    print (
        '\n' + '-'*70 + '\n\n' +
        'true positives / positives = %f' 
        % (num_correct / float(num_tested))
    )




def cross_val_negatives(classifier='svm', clf_kwargs={}, use_wordnet=False):
    positive_seeds, negative_seeds = get_train_sets()
    features = load_features()
    dictionary = get_dictionary(features)
    positive_seeds = filter_seeds(positive_seeds, dictionary)
    negative_seeds = filter_seeds(negative_seeds, dictionary)

    num_correct = 0
    num_tested = 0
    for test_item in negative_seeds:
        negative_seeds_filtered = [
            n for n in negative_seeds if n is not test_item
        ]

        args = (features,dictionary,positive_seeds,negative_seeds_filtered)
        if classifier == 'svm':
            clf = make_svm_classifier(
                *args, clf_kwargs=clf_kwargs, use_wordnet=use_wordnet)
        elif classifier == 'knn':
            clf = make_knn_classifier(*args, clf_kwargs=clf_kwargs)
        else:
            raise ValueError('Unexpected classifier type: %s.' % classifier)

        num_tested += 1
        prediction = clf.predict(test_item)[0]
        if not prediction:
            num_correct += 1

        padding = 40 - len(test_item)
        print test_item, ' ' * padding, '-' if prediction else 'correct'

    print (
        '\n' + '-'*70 + '\n\n' +
        'true negatives / negatives = %f' 
        % (num_correct / float(num_tested))
    )

if __name__ == '__main__':
    diagnose_map_evaluators([{}])
