#!/usr/bin/env python
import tarfile
from word2vec import UnigramDictionary, UNK
import json
from iterable_queue import IterableQueue
from multiprocessing import Process
import time
import os
from corenlp_xml_reader import AnnotatedText, Token, Sentence
import sys
from subprocess import check_output
from collections import defaultdict, Counter, deque
from SETTINGS import (
    GIGAWORD_DIR, DICTIONARY_DIR, DATA_DIR, DEPENDENCY_FEATURES_PATH,
    BASELINE_FEATURES_PATH, HAND_PICKED_FEATURES_PATH,
    RELATIONAL_NOUN_FEATURES_DIR, WORDNET_INDEX_PATH
)
from nltk.corpus import wordnet

NUM_ARTICLE_LOADING_PROCESSES = 12
GAZETTEER_DIR = os.path.join(DATA_DIR, 'gazetteers')
GAZETTEER_FILES = [
    'country', 'city', 'us-state', 'continent', 'subcontinent'
]
MIN_FREQUENCY = 5
    

def coalesce_features(out_dir_name, limit=None):

    # Get a list of all the feature_dirs
    feature_dirs = t4k.ls(
        RELATIONAL_NOUN_FEATURES_DIR, absolute=True, exclude='agg', files=False
    )[:limit]

    # Make the feature accumulator in which to aggregate results
    feature_accumulator = FeatureAccumulator()

    # Read in the features for each individual archive and accumulate
    for feature_dir in feature_dirs:
        print 'processing %s...' % os.path.basename(archive)
        try:
            feature_accumulator.merge_load(feature_dir)

        # Tolerate errors reading or json parsing
        except (IOError, ValueError), e:
            print 'problem with feature_dir %d: %s' % (feature_dir, str(e))
            pass

    # Prune the features
    print 'prunning...'
    #prune_features(features)

    coalesced_path = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, out_dir_name)
    feature_accumulator.write(coalesced_path)



def prune_features(features, min_frequency=MIN_FREQUENCY):
    '''
    Prune the features for tokens that occur less than min_frequency
    number of times.  Note, this operates through side effects.
    '''

    # Prune tokens that have a low frequency from the dictionary
    discarded_tokens = features['dictionary'].prune(
        min_frequency=min_frequency)

    # Go through each feature, and prune out features for tokens nolonger
    # in the dictionary
    for feature_type in features:

        # Skip the dictionary, it's already pruned
        if feature_type == 'dictionary':
            continue

        # Remove all the discarded tokens from the feature listings
        for token in discarded_tokens:
            try:
                del features[feature_type][token]
            except KeyError:
                pass


def add_nested_counter_inplace(accumulating_counter, dict_to_add):
    for outer_key in dict_to_add:
        for inner_key in dict_to_add[outer_key]:
            accumulating_counter[outer_key][inner_key] += (
                dict_to_add[outer_key][inner_key])

    return accumulating_counter


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


def extract_and_save_features(path, untar=True, limit=None):

    # If ``untar`` is true, then untar the path first
    if tarred:
        subprocess.call(['tar', '-zxf', path])
        if path.endswith('.tgz'):
            path = path[:-4]

    # Extract all the features and a dictionary
    feature_accumulator = extract_all_features(path, limit)

    dirname = os.path.basename(path)
    write_path = os.path.join(RELATIONAL_NOUN_FEATURES_DIR, dirname)

    # Write the features to disk
    feature_accumulator.write(write_path)

    # If we untarred the file, then delete the untarred copy
    if tarred:
        shutil.rmtree(path)



def extract_and_save_features_from_archive(archive_path):

    # Create a folder for features extracted from this archive
    this_archive = os.path.basename(archive_path)[:-4]
    this_archive_features_dir = os.path.join(
        RELATIONAL_NOUN_FEATURES_DIR, this_archive)
    if not os.path.exists(this_archive_features_dir):
        os.makedirs(this_archive_features_dir)

    # Open a log file where errors will be put in case archive is corrupted
    log_path = os.path.join(this_archive_features_dir, 'log')
    log = open(log_path, 'w')

    # Extract all the features
    extract = extract_all_features_from_archive(archive_path, log)

    # Save each of the features
    write_features(extract, this_archive_features_dir)

    #dependency_features_path = os.path.join(
    #    this_archive_features_dir, 'dependency.json')
    #open(dependency_features_path, 'w').write(json.dumps(
    #    extract['dep_tree_features']))

    #baseline_features_path = os.path.join(
    #    this_archive_features_dir, 'baseline.json')
    #open(baseline_features_path, 'w').write(json.dumps(
    #    extract['baseline_features']))

    #hand_picked_features_path = os.path.join(
    #    this_archive_features_dir, 'hand-picked.json')
    #open(hand_picked_features_path, 'w').write(json.dumps(
    #    extract['hand_picked_features']))

    ## Save the dictionary
    #dictionary_dir = os.path.join(
    #    this_archive_features_dir, 'lemmatized-noun-dictionary')
    #extract['dictionary'].save(dictionary_dir)


def extract_all_features_from_archive(archive_path, log=None):

    start = time.time()

    # First, make an iterable queue.  Extract all the corenlp files from the
    # archive and load them onto it
    fnames_q = IterableQueue()
    fnames_producer = fnames_q.get_producer()
    archive = tarfile.open(archive_path)

    # Extract each member, putting it's path and contents on the queue
    try:
        for member in archive:

            # Extract the contents of the corenlp files, putting the text
            # for each file directly onto the queue
            if member.name.endswith('xml'):
                fnames_producer.put((
                    member.name,
                    archive.extractfile(member).read()
                ))

    # If we encounter corruption in the archive, log or print a warning
    # and proceed with the processing of what was extracted so far.
    except IOError, e:
        message = '%s\tlast file was: %s' % (str(e), member.name)
        if log:
            log.write(message)
        else:
            print message

    # We're done adding files to the queue
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
            target=extract_features_from_articles,
            args=(fnames_consumer, features_producer, 'content')
        )
        process.start()

    # We're done making endpoints for the queues
    fnames_q.close()
    features_q.close()

    # We're going to accumulate the results.  Make some containers for that.
    dep_tree_features = defaultdict(Counter)
    baseline_features = defaultdict(Counter)
    hand_picked_features = defaultdict(Counter)
    dictionary = UnigramDictionary()

    # Accumulate the results.  This blocks until workers are finished
    for extract in features_consumer:
        dictionary.add_dictionary(extract['dictionary'])
        for key in extract['dep_tree_features']:
            dep_tree_features[key] += extract['dep_tree_features'][key]
        for key in extract['baseline_features']:
            baseline_features[key] += extract['baseline_features'][key]
        for key in extract['hand_picked_features']:
            hand_picked_features[key] += (
                extract['hand_picked_features'][key])

    # Print a message about how long it all took
    elapsed = time.time() - start
    print 'elapsed', elapsed

    # Return the accumulated features
    return {
        'dep_tree_features':dep_tree_features, 
        'baseline_features': baseline_features, 
        'hand_picked_features': hand_picked_features,
        'dictionary': dictionary
    }


def extract_all_features(articles_dir, limit=None):

    start = time.time()

    # First, make an iterable queue and load all the article fnames onto it
    fnames_q = IterableQueue()
    fnames_producer = fnames_q.get_producer()
    for fname in get_fnames(articles_dir)[:limit]:
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
            target=extract_features_from_articles,
            args=(fnames_consumer, features_producer, False)
        )
        process.start()

    # Close the iterable queues
    fnames_q.close()
    features_q.close()

    # Accumulate the results.  This blocks until workers are finished
    feature_accumulator = make_feature_accumulator()

    for accumulator in features_consumer:
        feature_accumulator.merge(accumulator)

    elapsed = time.time() - start
    print 'elapsed', elapsed

    return feature_accumulator


def make_feature_accumulator():
    """
    Factory for making feature accumulators.  This de-parametrizes the
    dictionary of words for which we should accumulate features, making it
    always equal to the single-word noun lemma entries in wordnet.
    """
    return FeatureAccumulator(read_wordnet_index())


def read_wordnet_index():
    return set(open(WORDNET_INDEX_PATH).read().split('\n'))



def extract_features_from_articles(
    files_consumer, features_producer, has_content='True'
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
    feature_accumulator = make_feature_accumulator()

    # Read articles named in files_consumer, accumulate features from them
    for item in files_consumer:

        if has_content:
            fname, content = item
        else:
            fname = item
            content = open(fname).read()

        # Get features from this article
        print 'processing', fname, '...'
        feature_accumulator.extract(AnnotatedText(content))

    # Put the accumulated features onto the producer queue then close it
    features_producer.put(feature_accumulator)
    features_producer.close()


class FeatureAccumulator(object):
    """
    Extracts and accumulates features associated to nouns by reading
    CoreNLP-annotated articles.
    """

    FEATURE_TYPES = ['baseline', 'dependency', 'hand_picked.json']

    def __init__(self, dictionary, load=None):
        """
        ``dictionary`` is a set of words for which we will collect features.
        words not in ``dictionary`` will be skipped.
        """
        self.dictionary = dictionary
        self._initialize_accumulators()
        if load is not None:
            self.merge_load(load)


    def _initialize_accumulators(self):
        for feature_type in self.FEATURE_TYPES:
            setattr(self, feature_type, defaultdict(Counter))


    def merge(self, other):
        """
        Add counts from ``other`` to self.
        """
        for key in other.dependency:
            self.dependency[key] += other.dependency[key]
        for key in other.baseline:
            self.baseline[key] += other.baseline[key]
        for key in other.hand_picked:
            self.hand_picked[key] += other.hand_picked[key]


    def merge_load(self, path):

        for feature_type in self.FEATURE_TYPES:
            feature_accumulator = getattr(feature_type)
            feature_path = os.path.join(path, feature_type + '.json')
            as_dict = json.loads(feature_path)
            for key in as_dict:
                feature_accumulator[key].update(as_dict[key])


    def load(self, path):
        self._initialize_accumulators()
        self.merge_load(path)


    def extract(self, article):

        for sentence in article.sentences:
            for token in sentence['tokens']:

                # Skip words that aren't in the dictionary
                lemma = token['lemma']
                if lemma not in self.dictionary:
                    continue

                # Skip words that aren't a common noun
                pos = token['pos']
                if pos != 'NN' and pos != 'NNS':
                    continue

                # Add features seen for this instance.  We use the lemma as
                # the key around which to aggregate features for the same word
                self.get_dep_tree_features(token)
                self.get_baseline_features(token)
                self.get_hand_picked_features(token)


    def write(self, path):

        # Make the directory we want to write to if it doesn't exist
        t4k.ensure_exists(path)

        # Save each of the features
        open(os.path.join(path, 'dependency.json'), 'w').write(json.dumps(
            self.dependency))
        open(os.path.join(path, 'baseline.json'), 'w').write(json.dumps(
            self.baseline))
        open(os.path.join(path, 'hand_picked.json'), 'w').write(json.dumps(
            self.hand_picked))


    # TODO: go to arbitrary depth recursivley
    def get_dep_tree_features(self, token):
        '''
        Extracts a set of features based on the dependency tree relations
        for the given token.  Each feature describes one dependency tree 
        relation in terms of a "signature".  The signature of a dependency
        tree relation consists of whether it is a parent or child, what the 
        relation type is (e.g. nsubj, prep:for, etc), and what the pos of the 
        target is.
        '''

        # Record all parent signatures
        for relation, token in token['parents']:
            signature = '%s:%s:%s' % ('parent', relation, token['pos'])
            #signature = '%s:%s:%s' % ('parent', relation, token['ner'])
            self.dependency[signature] += 1

        # Record all child signatures
        for relation, token in token['children']:
            signature = '%s:%s:%s' % ('child', relation, token['pos'])
            #signature = '%s:%s:%s' % ('child', relation, token['ner'])
            self.dependency[signature] += 1


    def get_baseline_features(self, token):
        '''
        Looks for specific syntactic relationships that are indicative of
        relational nouns.  Keeps track of number of occurrences of such
        relationships and total number of occurrences of the token.
        '''

        # Record all parent signatures
        for relation, child_token in token['children']:
            if relation == 'nmod:of' and child_token['pos'].startswith('NN'):
                self.baseline['nmod:of:NNX'] += 1
            if relation == 'nmod:poss':
                self.baseline['nmod:poss'] += 1

    
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
            self.hand_picked[key] += 1

            # Note if the sibling is a noun of some kind
            if sibling_token['pos'].startswith('NN'):
                self.hand_picked['sibling(%d):pos(NNX)' % rel_idx] += 1

            # Note the sibling's named entity type
            key = 'sibling(%d):ner(%s)' % (rel_idx, sibling_token['ner'])
            self.hand_picked[key] += 1

            # Note if the sibling is a named entity of any type
            if sibling_token['ner'] is not None:
                self.hand_picked['sibling(%d):ner(x)' % rel_idx] += 1

            # Note if the sibling is a demonym
            if sibling_token['word'] in GAZETTEERS['demonyms']:
                self.hand_picked['sibling(%d):demonym' % rel_idx] += 1

            # Note if the sibling is a place name
            if sibling_token['word'] in GAZETTEERS['names']:
                self.hand_picked[
                    'sibling(%d):place-name' % rel_idx
                ] += 1

        # Note if the noun is plural
        if token['pos'] == 'NNS':
            self.hand_picked['plural'] += 1

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
            self.hand_picked['<noun>-prp'] += 1
            self.hand_picked['<noun>-%s' % nmod] += 1

        if poss:
            self.hand_picked['poss-<noun>'] += 1

        if cop and nmod:
            self.hand_picked['is-<noun>-prp'] += 1
            self.hand_picked['is-<noun>-%s' % nmod] += 1
            if det:
                self.hand_picked['is-%s-<noun>-prp' % det] += 1

        if det and nmod:
            self.hand_picked['%s-<noun>-prp' % det] += 1
            self.hand_picked['%s-<noun>-%s' % (det, nmod)] += 1

        if cop and poss:
            self.hand_picked['is-poss-<noun>'] += 1

        if det and poss:
            self.hand_picked['%s-poss-<noun>' % det] += 1

        if det and not nmod and not poss:
            self.hand_picked['%s-<noun>' % det] += 1
        
        if cop and det and poss:
            self.hand_picked['is-det-poss-<noun>'] += 1

        if cop and det and nmod:
            self.hand_picked['is-det-<noun>-prp'] += 1

        # Next we consider whether the propositional phrase has a named
        # entity, demonym, or place name in it
        if nmod:

            for prep_type in ['of', 'to', 'for']:

                # See if there is a prepositional noun phrase of this type, and
                # get it's head.  If not, continue to the next type
                NP_head = get_first_matching_child(token, 'nmod:%s' % prep_type)
                if NP_head is None:
                    continue

                # Get all the tokens that are part of the noun phrase
                NP_constituent = NP_head['c_parent']
                NP_tokens = get_constituent_tokens(NP_constituent)

                # Add feature counts for ner types in the NP tokens
                ner_types = set([t['ner'] for t in NP_tokens])
                for ner_type in ner_types:
                    self.hand_picked[
                        'prp(%s)-ner(%s)' % (prep_type, ner_type)
                    ] += 1

                # Add feature counts for demonyms 
                lemmas = [t['lemma'] for t in NP_tokens]
                if any([l in GAZETTEERS['demonyms'] for l in lemmas]):
                    self.hand_picked[
                        'prp(%s)-demonyms' % prep_type
                    ] += 1

                # Add feature counts for place names 
                if any([l in GAZETTEERS['names'] for l in lemmas]):
                    self.hand_picked['prp(%s)-place' % prep_type] += 1 
        
        # Next we consider whether the posessor noun phrase has a named
        # entity, demonym, or place name in it
        if poss:
            NP_head = get_first_matching_child(token, 'nmod:poss')
            NP_constituent = NP_head['c_parent']
            NP_tokens = get_constituent_tokens(NP_constituent)

            # Add feature counts for ner types in the NP tokens
            ner_types = set([t['ner'] for t in NP_tokens])
            for ner_type in ner_types:
                self.hand_picked['poss-ner(%s)' % ner_type] += 1

            # Add feature counts for demonyms 
            lemmas = [t['lemma'] for t in NP_tokens]
            if any([l in GAZETTEERS['demonyms'] for l in lemmas]):
                self.hand_picked['poss-demonyms'] += 1

            # Add feature counts for place names 
            if any([l in GAZETTEERS['names'] for l in lemmas]):
                self.hand_picked['poss-place'] += 1 



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
    return t4k.ls(absolute=True)
    return fnames


if __name__ == '__main__':

    # Accept the number of articles to process for feature extraction
    limit = int(sys.argv[1])

    # Extract and save the features
    extract_and_save_features(limit)

