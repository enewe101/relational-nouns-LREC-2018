import os
import t4k
import sys
sys.path.append('../lib')
from SETTINGS import GIGAWORD_DIR, DATA_DIR
import extract_features as e
import utils


DIR_PROCESSING_TIMEOUT = 800
KILL_TIMEOUT = 100

def do_extract_all_features(out_dir, vocabulary, test=False, pipe=None):

    # Get paths to the gigaword dataset.  If running in test mode, then yield
    # a small test set.
    if test:
        print 'RUNNING A SMALL TEST EXTRACTION!'
        gigaword_archives = [
            '/home/ndg/dataset/gigaword-corenlp/test-data/000.tgz']
    else:

        # Get the list of dirs to process
        gigaword_archives = set(t4k.ls(
            os.path.join(GIGAWORD_DIR, 'data'),
            match=r'\.tgz',
            basename=True
        ))

        # What dirs have already been processed?
        already_processed_dirs = set([
            dir_name + '.tgz' for dir_name in t4k.ls(out_dir, basename=True)
        ])

        # To-be-processed dirs
        to_be_processed = sorted([
            os.path.join(GIGAWORD_DIR, 'data', dir_name)
            for dir_name in gigaword_archives - already_processed_dirs
        ])


    # Run feature extraction against each of the gigaword archives.
    for path in to_be_processed:
        pipe.send(t4k.NOT_DONE)
        print 'running on', path
        try:
            e.extract_all_features(
                path, untar=True, limit=None, vocabulary=vocabulary, 
                out_dir=out_dir
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print 'problem with %s: %s' % (os.path.basename(path), str(exc))

    pipe.send(t4k.DONE)


def extract_all_featurea_for_wordet_nouns():
    do_extract_all_features(
        out_dir=os.path.join(DATA_DIR, 'relational-noun-features-wordnet-only'),
        vocabulary=utils.read_wordnet_index()
    )


def extract_all_features_for_all_nouns():
    do_extract_all_features(
        out_dir=os.path.join(DATA_DIR, 'relational-noun-features'),
        vocabulary=None
    )


def extract_features_wordnet_lexical():
    """
    Extract all features for words in the wordnet vocabulary including the 
    recently added lexical features.
    """
    out_dir = out_dir=os.path.join(
        DATA_DIR, 'relational-noun-features-lexical-wordnet')

    managed_process = t4k.ManagedProcess(
        target=do_extract_all_features,
        args=(out_dir, None),
        timeout=DIR_PROCESSING_TIMEOUT,
        kill_timeout=KILL_TIMEOUT
    )
    managed_process.start()


    #do_extract_all_features(out_dir, vocabulary=None)


if __name__ == '__main__':
    #extract_all_featurea_for_wordet_nouns()
    #extract_all_features_for_all_nouns()
    extract_features_wordnet_lexical()

