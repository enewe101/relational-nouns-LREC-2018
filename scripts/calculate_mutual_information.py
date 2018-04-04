import sys
sys.path.append('../lib')
import os
from SETTINGS import RELATIONAL_NOUN_FEATURES_DIR
import extract_features as ef
import utils
import time
import annotations

def calculate_mutual_information(feature_sets, out_fname):

    # Tolerate providing a single feature set.  Make into a proper set.
    if isinstance(feature_sets, basestring):
        feature_sets = set([feature_sets])
    else:
        feature_sets = set(feature_sets)

    # Separate count based features and non-count features
    count_based_features = list(feature_sets & set(ef.COUNT_BASED_FEATURES))
    non_count_features = list(feature_sets & set(ef.NON_COUNT_FEATURES))

    # Validation: ensure no unexpected features were provided
    unexpected_features = (
        feature_sets - set(ef.COUNT_BASED_FEATURES) - set(ef.NON_COUNT_FEATURES)
    )
    # Make sure no misspelled features were included
    if len(unexpected_features):
        raise ValueError(
            'Unexpected feature(s): %s' % ', '.join(unexpected_features)
        )

    # Define the path at which to write.  If no fname was given, then name
    # the file after the first element of names_of_runs
    out_path = os.path.join(RELATIONAL_NOUN_FEATURES_DIR, out_fname)

    # Load the features if not provided
    wni = utils.read_wordnet_index()
    features_path = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, 
        'accumulated450-min_token_5-min_feat1000')
    start = time.time()
    features = ef.FeatureAccumulator(wni, load=features_path)
    print 'time to read features elapsed: %s' % (time.time() - start)

    # Load relational noun annotations
    annots = annotations.Annotations(features.dictionary)

    features.calculate_mutual_information(
        annots, out_path, count_based_features, non_count_features)


if __name__ == '__main__':
   out_fname = sys.argv[1]
   feature_sets = sys.argv[2:]
   calculate_mutual_information(feature_sets, out_fname)
