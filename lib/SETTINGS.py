import os
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RELATIONAL_WORDS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data','relational-nouns'
)
SEED_PATH = os.path.join(RELATIONAL_WORDS_PATH, 'seed.tsv')
ANNOTATIONS_PATH = os.path.join(RELATIONAL_WORDS_PATH, 'annotated.tsv')

RELATIONAL_NOUN_FEATURES_DIR = os.path.join(DATA_DIR, 'features')

ACCUMULATED_FEATURES_PATH = os.path.join(RELATIONAL_NOUN_FEATURES_DIR, '000')

SUFFIX_PATH = os.path.join(RELATIONAL_NOUN_FEATURES_DIR, 'suffixes.txt')
GOOGLE_VECTORS_PATH = os.path.join(
    RELATIONAL_NOUN_FEATURES_DIR, 'google-vectors-negative-300.txt')

GAZETTEER_DIR = os.path.join(DATA_DIR, 'gazetteers')

WORDNET_INDEX_PATH = os.path.join(DATA_DIR, 'wordnet_index.txt')

#GIGAWORD_DIR = '/home/ndg/dataset/gigaword-corenlp/'


#SCRATCH_DIR = '/gs/scratch/enewel3/relational-nouns'
#COOCCURRENCE_DIR = '/home/ndg/users/enewel3/relational-nouns/data/cooccurrence'
#TRAIN_PATH = os.path.join(RELATIONAL_WORDS_PATH, 'train', 'all.tsv')
#TEST_PATH = os.path.join(RELATIONAL_WORDS_PATH, 'test', 'all.tsv')
#
#
#
#TRAIN_POSITIVE_PATH = os.path.join(
#	RELATIONAL_WORDS_PATH, 'train-positive.txt'
#)
#TRAIN_NEGATIVE_PATH = os.path.join(
#	RELATIONAL_WORDS_PATH, 'train-negative.txt'
#)
#TEST_POSITIVE_PATH = os.path.join(
#	RELATIONAL_WORDS_PATH, 'test-positive.txt'
#)
#TEST_NEGATIVE_PATH = os.path.join(
#	RELATIONAL_WORDS_PATH, 'test-negative.txt'
#)
#
##DEPENDENCY_FEATURES_PATH = os.path.join(
##    DATA_DIR, 'relational-noun-dependency-features.json')
##BASELINE_FEATURES_PATH = os.path.join(
##    DATA_DIR, 'relational-noun-baseline-features.json')
##HAND_PICKED_FEATURES_PATH = os.path.join(
##    DATA_DIR, 'relational-noun-hand-picked-features.json')
##DICTIONARY_DIR = os.path.join(DATA_DIR, 'lemmatized-noun-dictionary')
#
#
