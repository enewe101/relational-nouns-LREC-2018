import sys
import t4k
sys.path.append('../lib')
import extract_features as ef
from SETTINGS import DATA_DIR, RELATIONAL_NOUN_FEATURES_DIR
import os
import utils


def coalesce_wordnet_only():
    ef.coalesce_features(
        'accumulated',
        os.path.join(DATA_DIR, 'relational-noun-features-wordnet-only')
    )


def coalesce():
    ef.coalesce_features(
        'accumulated',
        os.path.join(DATA_DIR, 'relational-noun-features')
    )


def coalesce_batch(batch_num):
    """
    Coalesce some of the feature extracts (a batch of 50 of them).  
    batch_num determines which 50 extracts will be coalesced.
    Do some light pruning.
    """

    in_dir = os.path.join(DATA_DIR, 'relational-noun-features-lexical-wordnet')
    start = 50*batch_num
    stop = 50*(batch_num+1)
    feature_dirs = t4k.ls(
        in_dir, absolute=True, match='/[0-9a-f]{3,3}$', files=False
    )[start:stop]
    out_dir = os.path.join(
        in_dir, 'accumulated50-min_token_5-min_feat100-%d'%batch_num)

    ef.coalesce_features(
        out_dir=out_dir,
        min_token_occurrences=2,
        min_feature_occurrences=100,
        vocabulary=utils.read_wordnet_index(),
        feature_dirs=feature_dirs
    )

def combine_batches():
    # Note, when run, pruning to features having at least 5000 occurrences
    # didn't work.  So I ran ``prune_features_more()`` somewhat later after
    # noticing that.
    in_dir = os.path.join(DATA_DIR, 'relational-noun-features-lexical-wordnet')
    feature_dirs = t4k.ls(
        in_dir, absolute=True, match='.*accumulated50-', files=False
    )
    out_dir = os.path.join(
        in_dir, 'accumulated450-min_token_5-min_feat5000')

    ef.coalesce_features(
        out_dir=out_dir,
        min_token_occurrences=5,
        min_feature_occurrences=5000,
        vocabulary=utils.read_wordnet_index(),
        feature_dirs=feature_dirs
    )


def prune_features_more():
    in_path = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, 'accumulated450-min_token_5-min_feat5000')
    out_path = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, 'accumulated450-min_token_5-min_feat1000')
    features = ef.FeatureAccumulator(
        vocabulary=utils.read_wordnet_index(), load=in_path)
    features.prune_features(1000)
    features.write(out_path)


def prune_to_top_k_features(k):
    """
    Prunes the features to only the k features having highest mutual 
    information.  Only the features listed in
    extract_features.COUNT_BASED_FEATURES were subjected to this filtering.
    """

    in_path = os.path.join( 
        RELATIONAL_NOUN_FEATURES_DIR, 'accumulated450-min_token_5-min_feat1000')
        
    out_path = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, 'accumulated450-min_token_5-top_%d' % k)

    # Get the top k features to be kept
    print 'getting the top k features'
    keep_features = get_top_k_features(k)

    # Load the base set of features that we'll be pruning
    features = ef.FeatureAccumulator(
        vocabulary=utils.read_wordnet_index(), load=in_path)

    # Do the pruning
    print 'pruning...'
    features.prune_features_keep_only(keep_features)

    # Save the pruned features
    print 'writing pruned features to disc...'
    features.write(out_path)


def get_top_k_features(k):

    mutual_information_path = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, 'mutual_info.tsv')

    # Read in the top k features from the sorted mutual information score file
    keep_features = set()
    for i, line in enumerate(open(mutual_information_path)):

        # Only take the top k features
        if i == k: break

        # Parse out the feature name.  Need to remove it's prefix.
        feature_with_prefix = line.split('\t')[0]
        feature_no_prefix = feature_with_prefix.split(':', 1)[1]
        keep_features.add(feature_no_prefix)

    return keep_features


if __name__ == '__main__':
    #coalesce_wordnet_only()
    #coalesce()
    #batch_num = int(sys.argv[1])
    #combine_batches()
    #prune_features_more()

    k = int(sys.argv[1])
    prune_to_top_k_features(k)


