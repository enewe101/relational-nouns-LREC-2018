import sys
sys.path.append('../lib')
from SETTINGS import DATA_DIR
import test_classifier
import utils
import extract_features
import os

HYPERPARAMETER_TUNING_DIR = os.path.join(DATA_DIR, 'hyperparameters')

def optimize_pruning():

    # We'll write results for this hyperparameter optimization here:
    out_path = os.path.join(HYPERPARAMETER_TUNING_DIR, 'optimize_pruning.tsv')

    # Read in the training set splits and the features
    train, test = utils.get_train_test_split()
    features = extract_features.FeatureAccumulator(load=os.path.join(
        DATA_DIR, 'relational-noun-features', 'accumulated'))

    # Define the ranges over which parameters should be varied
    parameter_ranges = {
        'min_feature_frequency':[5, 10, 20, 40, 80, 160]
    }

    # Define the values of parameters to be held constant
    constants = {
        'kind': 'svm',
        'on_unk': False,
        'C': 0.01,
        'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
        'semantic_similarity': 'res',
        'include_suffix' :  True,
        'syntactic_multiplier': 0.33,
        'semantic_multiplier': 0.33,
        'suffix_multiplier': 0.33,
    }

    # Generate all combinations of variable parameters, while including
    # constant paramteres.
    classifier_definitions = test_classifier.generate_classifier_definitions(
        parameter_ranges, constants)

    # Evaluate the classifier when running for all classifier definitions
    test_classifier.optimize_classifier(
        classifier_definitions, features, 
        train['pos'], train['neg'],
        test['pos'], test['neg'],
        out_path, num_procs=12
    )


def optimize_pruning2():

    # We'll write results for this hyperparameter optimization here:
    out_path = os.path.join(HYPERPARAMETER_TUNING_DIR, 'optimize_pruning2.tsv')

    # Read in the training set splits and the features
    train, test = utils.get_train_test_split()
    features = extract_features.FeatureAccumulator(load=os.path.join(
        DATA_DIR, 'relational-noun-features-wordnet-only', 'accumulated'))

    # Define the ranges over which parameters should be varied
    parameter_ranges = {
        'min_feature_frequency':[
            200, 500, 1000, 2000, 5000, 10000,
            #20000, 50000, 100000, 200000, 500000, 1000000,
        ]
    }

    # Define the values of parameters to be held constant
    constants = {
        'kind': 'svm',
        'on_unk': False,
        'C': 0.01,
        'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
        'semantic_similarity': 'res',
        'include_suffix' :  True,
        'syntactic_multiplier': 0.33,
        'semantic_multiplier': 0.33,
        'suffix_multiplier': 0.33,
    }

    # Generate all combinations of variable parameters, while including
    # constant paramteres.
    classifier_definitions = test_classifier.generate_classifier_definitions(
        parameter_ranges, constants)

    # Evaluate the classifier when running for all classifier definitions
    test_classifier.optimize_classifier(
        classifier_definitions, features, 
        train['pos'], train['neg'],
        test['pos'], test['neg'],
        out_path, num_procs=12
    )


def optimize_C():

    # We'll write results for this hyperparameter optimization here:
    out_path = os.path.join(HYPERPARAMETER_TUNING_DIR, 'optimize_C.tsv')

    # Read in the training set splits and the features
    train, test = utils.get_train_test_split()
    features = extract_features.FeatureAccumulator(load=os.path.join(
        DATA_DIR, 'relational-noun-features-wordnet-only', 'test-coalesce'))
    features.normalize_features()   # Do this here else every worker will do it

    # Define the ranges over which parameters should be varied
    parameter_ranges = {'C':[0.01, 0.1, 1.0, 10., 100., 1000.]}

    # Define the values of parameters to be held constant
    constants = {
        'kind': 'svm',
        'on_unk': False,
        'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
        'semantic_similarity': 'res',
        'include_suffix' :  True,
        'syntactic_multiplier': 0.33,
        'semantic_multiplier': 0.33,
        'suffix_multiplier': 0.33,
    }

    # Generate all combinations of variable parameters, while including
    # constant paramteres.
    classifier_definitions = test_classifier.generate_classifier_definitions(
        parameter_ranges, constants)

    # Evaluate the classifier when running for all classifier definitions
    test_classifier.optimize_classifier(
        classifier_definitions, features, 
        train['pos'], train['neg'],
        test['pos'], test['neg'],
        out_path, num_procs=12
    )


def optimize_feature_weight():

    # We'll write results for this hyperparameter optimization here:
    out_path = os.path.join(
        HYPERPARAMETER_TUNING_DIR, 'optimize_feature_weights.tsv')

    # Read in the training set splits and the features
    train, test = utils.get_train_test_split()
    features = extract_features.FeatureAccumulator(load=os.path.join(
        DATA_DIR, 'relational-noun-features-wordnet-only',
        'accumulated-pruned-5000'))

    # Define the ranges over which parameters should be varied
    parameter_ranges = {
        'syntactic_multiplier': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        'semantic_multiplier': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        'suffix_multiplier': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    }

    # Define the values of parameters to be held constant
    constants = {
        'kind': 'svm',
        'on_unk': False,
        'C': 0.01,
        'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
        'semantic_similarity': 'res',
        'include_suffix' :  True,
    }

    # Generate all combinations of variable parameters, while including
    # constant paramteres.
    classifier_definitions = test_classifier.generate_classifier_definitions(
        parameter_ranges, constants)

    # Evaluate the classifier when running for all classifier definitions
    test_classifier.optimize_classifier(
        classifier_definitions, features, 
        train['pos'], train['neg'],
        test['pos'], test['neg'],
        out_path, num_procs=12
    )



def optimize_syntactic_feature_sets():

    # We'll write results for this hyperparameter optimization here:
    out_path = os.path.join(
        HYPERPARAMETER_TUNING_DIR, 'optimize_syntactic_feature_sets.tsv')

    # Read in the training set splits and the features
    train, test = utils.get_train_test_split()
    features = extract_features.FeatureAccumulator(load=os.path.join(
        DATA_DIR, 'relational-noun-features-wordnet-only',
        'accumulated-pruned-5000'))

    # Define the ranges over which parameters should be varied
    parameter_ranges = {
        'syntax_feature_types': [
            #[],
            #['baseline'], 
            #['dependency'], 
            #['hand_picked'],
            ['pos_unigram'],
            ['pos_unigram', 'pos_bigram'],
            ['lemma_unigram'],
            ['lemma_unigram', 'lemma_bigram'],
            ['surface_unigram', 'surface_bigram'],
            #['dependency', 'hand_picked'],
            #['baseline', 'hand_picked'],
            #['baseline', 'dependency'],
            #['baseline', 'dependency', 'hand_picked'],
        ]
    }

    # Define the values of parameters to be held constant
    constants = {
        'kind': 'svm',
        'on_unk': False,
        'C': 0.01,
        'semantic_similarity': 'res',
        'include_suffix' :  True,
        'syntactic_multiplier': 10.0,
        'semantic_multiplier': 2.0,
        'suffix_multiplier': 0.2
    }

    # Generate all combinations of variable parameters, while including
    # constant paramteres.
    classifier_definitions = test_classifier.generate_classifier_definitions(
        parameter_ranges, constants)

    # Evaluate the classifier when running for all classifier definitions
    test_classifier.optimize_classifier(
        classifier_definitions, features, 
        train['pos'], train['neg'],
        test['pos'], test['neg'],
        out_path, num_procs=12
    )


def optimize_C2():

    # We'll write results for this hyperparameter optimization here:
    out_path = os.path.join(
        HYPERPARAMETER_TUNING_DIR, 'optimize_C2.tsv')

    # Read in the training set splits and the features
    train, test = utils.get_train_test_split()
    features = extract_features.FeatureAccumulator(load=os.path.join(
        DATA_DIR, 'relational-noun-features-wordnet-only',
        'accumulated-pruned-5000'))

    # Define the ranges over which parameters should be varied
    parameter_ranges = {
        'C': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    }

    # Define the values of parameters to be held constant
    constants = {
        'kind': 'svm',
        'on_unk': False,
        'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
        'semantic_similarity': 'res',
        'include_suffix' :  True,
        'syntactic_multiplier': 10.0,
        'semantic_multiplier': 2.0,
        'suffix_multiplier': 0.2
    }

    # Generate all combinations of variable parameters, while including
    # constant paramteres.
    classifier_definitions = test_classifier.generate_classifier_definitions(
        parameter_ranges, constants)

    # Evaluate the classifier when running for all classifier definitions
    test_classifier.optimize_classifier(
        classifier_definitions, features, 
        train['pos'], train['neg'],
        test['pos'], test['neg'],
        out_path, num_procs=12
    )


def optimize_pruning3():

    # We'll write results for this hyperparameter optimization here:
    out_path = os.path.join(
        HYPERPARAMETER_TUNING_DIR, 'optimize_pruning3.tsv')

    # Read in the training set splits and the features
    train, test = utils.get_train_test_split()
    features = extract_features.FeatureAccumulator(load=os.path.join(
        DATA_DIR, 'relational-noun-features-wordnet-only', 'accumulated'))

    # Define the ranges over which parameters should be varied
    parameter_ranges = {
        'min_feature_frequency':[
            200, 500, 1000, 2000, 5000, 10000,
        ]
    }

    # Define the values of parameters to be held constant
    constants = {
        'kind': 'svm',
        'on_unk': False,
        'C': 0.01,
        'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
        'semantic_similarity': 'res',
        'include_suffix' :  True,
        'syntactic_multiplier': 10.0,
        'semantic_multiplier': 2.0,
        'suffix_multiplier': 0.2
    }

    # Generate all combinations of variable parameters, while including
    # constant paramteres.
    classifier_definitions = test_classifier.generate_classifier_definitions(
        parameter_ranges, constants)

    # Evaluate the classifier when running for all classifier definitions
    test_classifier.optimize_classifier(
        classifier_definitions, features, 
        train['pos'], train['neg'],
        test['pos'], test['neg'],
        out_path, num_procs=12
    )


def optimize_syntactic_feature_sets2():

    # We'll write results for this hyperparameter optimization here:
    out_path = os.path.join(
        HYPERPARAMETER_TUNING_DIR, 'optimize_syntactic_feature_sets2.tsv')

    # Read in the training set splits and the features
    train, test = utils.get_train_test_split()
    features_path = os.path.join(
        DATA_DIR, 'relational-noun-features-lexical-wordnet', '0ba')
    features = extract_features.FeatureAccumulator(
        vocabulary=utils.read_wordnet_index(), load=features_path)

    # Define the ranges over which parameters should be varied
    parameter_ranges = {
        'syntax_feature_types': [
            #[],
            #['baseline'], 
            #['dependency'], 
            #['hand_picked'],
            ['pos_unigram'],
            ['pos_unigram', 'pos_bigram'],
            ['lemma_unigram'],
            ['lemma_unigram', 'lemma_bigram'],
            ['surface_unigram', 'surface_bigram'],
            #['dependency', 'hand_picked'],
            #['baseline', 'hand_picked'],
            #['baseline', 'dependency'],
            #['baseline', 'dependency', 'hand_picked'],
        ]
    }

    # Define the values of parameters to be held constant
    constants = {
        'kind': 'svm',
        'on_unk': False,
        'C': 0.01,
        'semantic_similarity': 'res',
        'include_suffix' :  True,
        'syntactic_multiplier': 10.0,
        'semantic_multiplier': 2.0,
        'suffix_multiplier': 0.2
    }

    # Generate all combinations of variable parameters, while including
    # constant paramteres.
    classifier_definitions = test_classifier.generate_classifier_definitions(
        parameter_ranges, constants)

    # Evaluate the classifier when running for all classifier definitions
    test_classifier.optimize_classifier(
        classifier_definitions, features, 
        train['pos'], train['neg'],
        test['pos'], test['neg'],
        out_path, num_procs=1
    )


if __name__ == '__main__':
    #optimize_pruning()
    #optimize_pruning2()
    #optimize_feature_weight()
    #optimize_syntactic_feature_sets()
    #optimize_C2()
    #optimize_pruning3()
    optimize_syntactic_feature_sets2()
