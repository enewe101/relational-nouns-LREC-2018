import sys
import os
from lib.SETTINGS import RELATIONAL_NOUN_FEATURES_DIR, DATA_DIR
import t4k
import lib.extract_features as ef
import lib.test_classifier as tc
import lib.utils as utils
import time


def run_classifier(names_of_runs, out_fname=None):

    # Accept being given a single run name -- in which case, wrap it in a list.
    if isinstance(names_of_runs, basestring):
        names_of_runs = [names_of_runs]

    # Define the path at which to write.  If no fname was given, then name
    # the file after the first element of names_of_runs
    if out_fname is None:
        out_fname = names_of_runs[0]
    out_path = os.path.join(DATA_DIR, 'classifier-performance', out_fname)

    # Regrieve the definitions for the desired runs
    these_runs = t4k.select(runs, names_of_runs)

    features_dirname = 'accumulated450-min_token_5-min_feat5000'
    features = None

    # Do the runs
    for name, run_specification in these_runs.iteritems():

        # Parameter optimization is performed by running at various values
        # for settings specified as type `SearchValues`.
        expanded_specs = tc.expand_run_specification(run_specification)

        # Run the expanded specification (searches all values of `SearchValues`
        # fields).
        for i, spec in enumerate(expanded_specs):

            # If different features path is given, load features from there.
            if 'features_dirname' in spec:
                new_fpath = spec['features_dirname']
                if new_fpath != features_dirname:
                    features_dirname = new_fpath
                    features = load_features(new_fpath)

            # And in any case, load the features if we haven't yet.
            if features is None:
                features = load_features(features_dirname)

            start = time.time()
            tc.evaluate_classifier('%s__%d'%(name,i), spec, features, out_path)
            print 'time to run model: %s' % (time.time() - start)


def load_features(features_dirname):

    print 'loading features from %s...' % features_dirname
    start = time.time()

    features_path = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, features_dirname)
    features = ef.FeatureAccumulator(
        utils.read_wordnet_index(), load=features_path)

    print 'time to read features elapsed: %s' % (time.time() - start)
    return features


runs = {
    'small':{
        'seed': 1,
        'error-analysis-path': 'error-analysis-small.tsv',
        'features_dirname': 'features-small',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': 10.,
        'gamma': 0.01,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'optimal-test-rand':{
        'seed': 1,
        'error-analysis-path': 'analyze-errors-rand-2.tsv',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': 100.,
        'gamma': 0.001,
        'kind':'svm',
        'data_source': 'crowdflower-annotated-rand',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'optimal-test-top':{
        'seed': 1,
        'error-analysis-path': 'analyze-errors-top-2.tsv',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': 100.,
        'gamma': 0.001,
        'kind':'svm',
        'data_source': 'crowdflower-annotated-top',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'optimal-dev':{
        'error-analysis-path': 'analyze-errors-optimal-dev.tsv',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'optimal-dev-optFeatRep':{
        'error-analysis-path': 'error-analysis-optFeatRep.tsv',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode': tc.SearchValues(['normalized', 'log', 'raw']),
        'whiten': False,
        'C': 100.,
        'gamma': 0.001,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'optimal-dev-optFeatRepThresh':{
        'error-analysis-path': 'error-analysis-optFeatRepThresh.tsv',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode': 'threshold',
        'feature_threshold': tc.SearchValues([0.25, 0.5, 0.75]),
        'whiten': False,
        'C': 100.,
        'gamma': 0.001,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    # -- SVM -- 
    # First, using smaller features, decide whether whitening is better,
    # and which feature representation is better.
    'svm-(+0|-)-normalized-whiten-C_1': {
        'count_feature_mode':'normalized',
        'whiten':True,

        'C': 1.0,
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # DONE
    'svm-(+0|-)-normalized-nowhiten-C_1': {
        'count_feature_mode':'normalized',
        'whiten':False,

        'C': 1.0,
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # DONE
    'svm-(+0|-)-raw-whiten-C_1': {
        'count_feature_mode':'raw',
        'whiten':True,

        'C': 1.0,
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # DONE
    'svm-(+0|-)-raw-nowhiten-C_1': {
        'count_feature_mode':'raw',
        'whiten':False,

        'C': 1.0,
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # DONE
    'svm-(+0|-)-log-whiten-C_1': {
        'count_feature_mode':'log',
        'whiten':True,

        'C': 1.0,
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # DONE
    'svm-(+0|-)-log-nowhiten-C_1': {
        'count_feature_mode':'log',
        'whiten':False,

        'C': 1.0,
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # DONE
    'svm-(+0|-)-threshold-whiten-C_1': {
        'count_feature_mode':'threshold',
        'whiten':True,

        'C': 1.0,
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # DONE
    'svm-(+0|-)-threshold-nowhiten-C_1': {
        'count_feature_mode':'threshold',
        'whiten':False,

        'C': 1.0,
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # DONE

    # Next, choose the best feature representation to optimize C
    'svm-(+0|-)-normalized-nowhite-optimize-C': {
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 'auto',

        'count_feature_mode':'normalized',
        'whiten':False,

        'kind':'svm',
        'data_source':'crowdflower-dev',
        'binarize_mode':'+0/-',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # CHECKED

    # Try using the strict-positive formulation
    'svm-(+|0-)-best':{
        'binarize_mode':'+/0-',

        'count_feature_mode':'best?',
        'whiten': 'best?',
        'C': '?',
        'gamma': '?',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    }, # cross-validate C, gamma

    'svm-(+|1|-)-normalized-nowhite-optimize_C':{
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 'auto',
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+|1|-)-normalized-nowhite-optimize_C-gamma0.1':{
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+|1|-)-normalized-nowhite-optimize_C-gamma0.01':{
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.01,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+|0-)-normalized-nowhite-optimize_C-gamma0.1':{
        'binarize_mode':'+/0-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+|0-)-normalized-nowhite-optimize_C-gamma0.01':{
        'binarize_mode':'+/0-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.01,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma


    'svm-(+1|-)-norm-nowhite-optC-gamma1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 1.,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-norm-nowhite-optC-gamma10':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 10.,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-norm-nowhite-optC-gamma100':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 100.,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-norm-nowhite-optC-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-norm-nowhite-optC-gamma0.01':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.01,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+|0-)-[14a]-normalized-nowhite-C10-gamma0.01':{
        'features_dirname': '14a',
        'binarize_mode':'+/0-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': 10.,
        'gamma': 0.01,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-normalized-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-normalized-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<dep>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='dependency'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<dep>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='dependency'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<baseline>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='baseline'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<baseline>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='baseline'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<hand_picked>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='hand_picked'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<hand_picked>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='hand_picked'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<lemma>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('lemma')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<lemma>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('lemma')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<surface>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('surface')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<surface>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('surface')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<pos>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('pos')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<pos>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('pos')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<derivational>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='derivational'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<derivational>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='derivational'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<google>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='google'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<google>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='google'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<suffix>-nowhite-optC1000-100000-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1000,10000,100000]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<suffix>-nowhite-optC-optGamma0.001-0.00001':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([0.001, 0.0001]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-normalized-nowhite-C10-optGamma-devTopRand':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': 10.,
        'gamma': tc.SearchValues([0.01, 0.1, 1., 10., 100.]),
        'kind':'svm',
        'data_source': tc.SearchValues([
           'crowdflower-dev-rand', 'crowdflower-dev-top']),
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-normalized-nowhite-optC-Gamma0.01-devTopRand':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([0.01, 0.1, 1., 10., 100.]),
        'gamma': 0.01,
        'kind':'svm',
        'data_source': tc.SearchValues([
           'crowdflower-dev-rand', 'crowdflower-dev-top']),
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-normalized-nowhite-C10-gamma0.01-devTopRand':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': 10.,
        'gamma': 0.01,
        'kind':'svm',
        'data_source': tc.SearchValues([
           'crowdflower-dev-rand', 'crowdflower-dev-top']),
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<baseline>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='baseline'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<dep>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='dependency'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<dep>-normalized-nowhite-C100-gamma0.001-testing':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten':False,
        'C': 100.,
        'gamma': 0.001,
        'kind':'svm',
        'data_source': tc.SearchValues([
            'crowdflower-annotated-rand', 'crowdflower-annotated-top']),
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='dependency'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':False,
        'use_threshold': -0.23498053631221705,
        'cache_size':8000
    }, # Done


    'svm-(+0|-)-[min1000]-<hand>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='hand_picked'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<lemma>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f!='hand_picked'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<lemma>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('lemma')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<surface>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('surface')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<pos>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('pos')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<pos>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('pos')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<derivational>-normalized-nowhite-optC-gamma0.01':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.01,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='derivational'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<derivational>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='derivational'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<google>-normalized-nowhite-optC-gamma0.01':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([1., 0.1, 0.01]),
        'gamma': 0.01,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='google_vectors'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<google>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='google_vectors'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<suffix>-normalized-nowhite-C100-gamma0.001-test':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': 100.,
        'gamma': 0.001,
        'kind':'svm',
        'data_source': tc.SearchValues([
            'crowdflower-annotated-rand', 'crowdflower-annotated-top']),
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<suffix>-normalized-nowhite-optC-gamma0.01':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.01,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[min1000]-<suffix>-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features':[
            f for f in ef.NON_COUNT_FEATURES if f!='suffix'],
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[top10000]-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-top_10000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[top20000]-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-top_20000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+0|-)-[top40000]-normalized-nowhite-optC-optGamma':{
        'features_dirname': 'accumulated450-min_token_5-top_40000',
        'binarize_mode':'+0/-',
        'count_feature_mode':'normalized',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
        'cache_size':8000
    },  # cross-validate C, gamma

    'svm-(+1|-)-log-nowhite-optC-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'log',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-log-nowhite-C0.1-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'log',
        'whiten': False,
        'C': 0.1,
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-log-nowhite-C0.01-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'log',
        'whiten': False,
        'C': 0.01,
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-thresh-nowhite-optC-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'threshold',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-thresh-nowhite-C0.1-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'threshold',
        'whiten': False,
        'C': 0.1,
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-thresh-nowhite-C0.01-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'threshold',
        'whiten': False,
        'C': 0.01,
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-raw-nowhite-optC-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'raw',
        'whiten': False,
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-raw-nowhite-C0.1-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'raw',
        'whiten': False,
        'C': 0.1,
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+1|-)-raw-nowhite-C0.01-gamma0.1':{
        'binarize_mode':'+0/-',
        'count_feature_mode':'raw',
        'whiten': False,
        'C': 0.01,
        'gamma': 0.1,
        'kind':'svm',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
        'cache_size':8000
    },

    'svm-(+0|-)-baseline-raw-nowhite-C1-gamma10-testing':{
        'kind':'svm',
        'C': 1.,
        'gamma': 10.,
        'data_source': tc.SearchValues([
            'crowdflower-annotated-rand', 'crowdflower-annotated-top']),
        'count_feature_mode': 'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
        'cache_size':8000
    }, # Done


    'svm-(+0|-)-baseline-optRep-nowhite-optC-optGamma':{
        'kind':'svm',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode': tc.SearchValues(['raw', 'log', 'normalized']),
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
        'cache_size':8000
    }, # Done

    'svm-(+0|-)-baseline-optThresh-nowhite-optC-optGamma':{
        'kind':'svm',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'gamma': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode': 'threshold',
        'feature_threshold': tc.SearchValues([0.25, 0.5, 0.75]),
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
        'cache_size':8000
    }, # Done



    # -- Logistic -- 
    # Use logistic with (+0/-) formulation and best feature representation to
    # find best type of penalty.  For each penalty, fit C.  Use the sag solver,
    # except for the L1 penalty, since it is only available for liblinear.
    # Then run against different classification formultations.  Note that the
    # 'multinomial' (+/0/-) may be better when not paired with liblinear
    # solver...
    'logistic-(+0|-)-norm-white-L1':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C':1.0, #xval
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':True,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # DONE

    'logistic-(+0|-)-norm-white-L2':{
        'kind':'logistic',
        'solver': 'newton-cg',
        'penalty': 'l2',
        'C':1.0,
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':True,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # DONE

    'logistic-(+0|-)-norm-nowhite-L1-C1':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C':1.0, #xval
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # DONE

    'logistic-(+0|-)-norm-nowhite-L2-C100':{
        'kind':'logistic',
        'solver': 'newton-cg',
        'penalty': 'l2',
        'C': 100.,
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # DONE

    'logistic-(+0|-)-norm-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([10., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # DONE

    'logistic-(+0|-)-norm-nowhite-L1-C100':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': 100.,
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # DONE

    'logistic-(+0|-)-norm-nowhite-L2-C1':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C':1.0, #xval
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-norm-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([10., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-raw-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l2', 'l1']),
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-raw-nowhite-L2-C0.01':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': 0.01,
        'data_source':'crowdflower-dev',
        'count_feature_mode':'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-raw-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-raw-nowhite-L1-C10':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': 10.,
        'data_source':'crowdflower-dev',
        'count_feature_mode':'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-raw-nowhite-L1-C1':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': 1.,
        'data_source':'crowdflower-dev',
        'count_feature_mode':'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-raw-nowhite-L1-C0.1':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': 0.1,
        'data_source':'crowdflower-dev',
        'count_feature_mode':'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-thresh-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'threshold',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-thresh25-nowhite-L1-optC':{
        'kind':'logistic',
        'feature_threshold': 0.25,
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'threshold',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-thresh75-nowhite-L1-optC':{
        'kind':'logistic',
        'feature_threshold': 0.75,
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'threshold',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-thresh-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'threshold',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l2', 'l1']),
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-log-nowhite-L1-optC1000-100000':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([1000,10000,100000]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<baseline>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f != 'baseline'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<baseline>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f != 'baseline'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<dependency>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f != 'dependency'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<dependency>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f != 'dependency'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<hand>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f != 'hand_picked'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<hand>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if f != 'hand_picked'],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<lemma>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('lemma')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<lemma>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('lemma')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<surface>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('surface')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<surface>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('surface')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<pos>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('pos')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<pos>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':[
            f for f in ef.COUNT_BASED_FEATURES if not f.startswith('pos')],
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<derivational>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features': [
            f for f in ef.NON_COUNT_FEATURES if f != 'derivational'],
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<derivational>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features': [
            f for f in ef.NON_COUNT_FEATURES if f != 'derivational'],
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<google>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features': [
            f for f in ef.NON_COUNT_FEATURES if f != 'google_vectors'],
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<google>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features': [
            f for f in ef.NON_COUNT_FEATURES if f != 'google_vectors'],
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<suffix>-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features': [
            f for f in ef.NON_COUNT_FEATURES if f != 'suffix'],
        'find_threshold':True,
    }, # Done

    'logistic-[min1000]-(+0|-)-<suffix>-log-nowhite-optPenalty-optC1e-5':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'features_dirname': 'accumulated450-min_token_5-min_feat1000',
        'C': tc.SearchValues([0.001, 0.0001, 0.00001]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ef.COUNT_BASED_FEATURES,
        'non_count_features': [
            f for f in ef.NON_COUNT_FEATURES if f != 'suffix'],
        'find_threshold':True,
    }, # Done

    'logistic-[top10000]-(+0|-)-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-top_10000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[top20000]-(+0|-)-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-top_20000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-[top40000]-(+0|-)-log-nowhite-optPenalty-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': tc.SearchValues(['l1','l2']),
        'features_dirname': 'accumulated450-min_token_5-top_40000',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source':'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done


    'logistic-(+0|-)-log-nowhite-L2-0.1-testTop-setThresh':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': 0.1,
        'use_threshold': 0.16678778479873391,
        'data_source': 'crowdflower-annotated-top',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':False,
    }, # Done

    'logistic-(+0|-)-log-nowhite-L2-0.1-testRand-setThresh':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': 0.1,
        'use_threshold': 0.16678778479873391,
        'data_source': 'crowdflower-annotated-rand',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':False,
    }, # Done

    'logistic-(+|0|-)-log-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([100., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        #'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+|0|-)-log-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        #'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-log-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-log-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'log',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-thresh25-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'threshold',
        'feature_threshold': 0.25,
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-thresh25-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'threshold',
        'feature_threshold': 0.25,
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-thresh5-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'threshold',
        'feature_threshold': 0.5,
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-thresh5-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'threshold',
        'feature_threshold': 0.5,
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-thresh75-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'threshold',
        'feature_threshold': 0.75,
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-thresh75-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'threshold',
        'feature_threshold': 0.75,
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-raw-nowhite-L1-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    'logistic-(+0|-)-baseline-raw-nowhite-L2-optC':{
        'kind':'logistic',
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': tc.SearchValues([100., 10., 1., 0.1, 0.01]),
        'data_source': 'crowdflower-dev',
        'count_feature_mode':'raw',
        'data_format':'vector',
        'whiten':False,
        'binarize_mode':'+0/-',
        'count_based_features': ['baseline'],
        'non_count_features': [],
        'find_threshold':True,
    }, # Done

    # -- NaiveBayes -- 
    # Use NB with (+0/-) formulation and best feature representation to fit the
    # best alpha.  Note that we're using the multinomial NB formulation, so
    # that we can't include google vectors (Unless we shift and quantize them).
    # After fitting alpha, we also run the other (+/0/-) formulations.
    'NB-(+0|-)-opt_rep-white': { # cross-validate alpha
        'kind':'NB',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':['derivational', 'suffix'],
        'count_feature_mode': tc.SearchValues([
            'raw', 'normalized', 'threshold']),
        'whiten':True,
        'binarize_mode':'+0/-',
        'find_threshold':True,
    },
    'NB-(+0|-)-threshold-opt_white': { # cross-validate alpha
        'kind':'NB',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':['derivational', 'suffix'],
        'count_feature_mode': 'threshold',
        'whiten': tc.SearchValues([True, False]),
        'binarize_mode':'+0/-',
        'find_threshold':True,
    },
    'NB-(+0|-)-opt_rep-nowhite': { # cross-validate alpha
        'kind':'NB',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':['derivational', 'suffix'],
        'count_feature_mode': tc.SearchValues([
            'raw', 'normalized', 'threshold']),
        'whiten':False,
        'binarize_mode':'+0/-',
        'find_threshold':True,
    },

    'NB-(+0|-)-raw-nowhite-optAlpha': { # cross-validate alpha
        'alpha': tc.SearchValues([1, 0.1, 0.01, 0.001, 0.0001, 0.00001]),
        'kind':'NB',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':['derivational', 'suffix'],
        'count_feature_mode': 'raw',
        'whiten':False,
        'binarize_mode':'+0/-',
        'find_threshold':True,
    },

    'NB-(+0|-)-thresh-white-optAlpha': { # cross-validate alpha
        'alpha': tc.SearchValues([1, 0.1, 0.01, 0.001, 0.0001, 0.00001]),
        'kind':'NB',
        'data_source':'crowdflower-dev',
        'data_format':'vector',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':['derivational', 'suffix'],
        'count_feature_mode': 'threshold',
        'whiten':True,
        'binarize_mode':'+0/-',
        'find_threshold':True,
    },



    # -- Random Foreest -- 
    # We need to test different numbers for Max Features.  Once we find the
    # best value under the (+0/-) formulation, we then run it for the other
    # formulations.
    'RF-(+0|-)-norm-optwhite-optm':{  

        'whiten': tc.SearchValues([True, False]),
        'max_features': tc.SearchValues([23, 80, 272, 931, 3185]),
        'n_estimators':100,
        'criterion':'gini',
        'kind':'RF',
        'data_source':'crowdflower-dev',
        'count_feature_mode':'normalized',
        'data_format':'vector',
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
    }, # DONE

    'RF-(+0|-)-optrep-optwhite-m80':{  
        'n_estimators':100,
        'criterion':'gini',
        'max_features':80,
        'kind':'RF',
        'data_source':'crowdflower-dev',
        'count_feature_mode': tc.SearchValues([
            'normalized', 'log', 'threshold', 'raw']),
        'data_format':'vector',
        'whiten': tc.SearchValues([True, False]),
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
    }, # DONE

    'RF-(+0|-)-norm-noWhite-m80':{  
        'n_estimators':100,
        'criterion':'gini',
        'max_features':80,
        'kind':'RF',
        'data_source':'crowdflower-dev',
        'count_feature_mode': 'normalized',
        'data_format':'vector',
        'whiten': False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
    }, # DONE

    'RF-(+0|-)-optrep-noWhite-optM':{  
        'n_estimators':100,
        'criterion':'gini',
        'max_features': tc.SearchValues([272, 931, 3185, 10926]),
        'kind':'RF',
        'data_source':'crowdflower-dev',
        'count_feature_mode': tc.SearchValues(['raw', 'log', 'threshold']),
        'data_format':'vector',
        'whiten': False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
    }, # 

    'RF-(+0|-)-norm-noWhite-m10926':{  
        'n_estimators':100,
        'criterion':'gini',
        'max_features': 10926,
        'kind':'RF',
        'data_source':'crowdflower-dev',
        'count_feature_mode': 'normalized',
        'data_format':'vector',
        'whiten': False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
    }, # 

    'RF-(+0|-)-norm-noWhite-extendOptM':{  
        'n_estimators':100,
        'criterion':'gini',
        'max_features': tc.SearchValues([37481, 128574, 441060]),
        'kind':'RF',
        'data_source':'crowdflower-dev',
        'count_feature_mode': 'normalized',
        'data_format':'vector',
        'whiten': False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
    }, # 

    'RF-(+0|-)-thresh-noWhite-extendOptM':{  
        'n_estimators':100,
        'criterion':'gini',
        'max_features': tc.SearchValues([37481, 128574, 441060]),
        'kind':'RF',
        'data_source':'crowdflower-dev',
        'count_feature_mode': 'threshold',
        'data_format':'vector',
        'whiten': False,
        'binarize_mode':'+0/-',
        'count_based_features':ef.COUNT_BASED_FEATURES,
        'non_count_features':ef.NON_COUNT_FEATURES,
        'find_threshold': True,
    }, # 

}

if __name__ == '__main__':
    names_of_runs = sys.argv[2:]
    out_fname = sys.argv[1]
    run_classifier(names_of_runs, out_fname)
