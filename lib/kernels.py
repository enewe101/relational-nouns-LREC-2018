from nltk.corpus.reader.wordnet import WordNetError
import multiprocessing
import iterable_queue as iq
import itertools
import numpy as np
import sys
sys.path.append('..')
import t4k
from nltk.corpus import wordnet, wordnet_ic
import utils as u
import extract_features

INFORMATION_CONTENT_FILE = 'ic-treebank-resnik-add1.dat'
LEGAL_SIMILARITIES = [
    'jcn', 'wup', 'res', 'path', 'lin', 'lch' 
]
LEGAL_SYNTACTIC_SIMILARITIES = extract_features.COUNT_BASED_FEATURES


def bind_dist(features, dictionary):
    def dist(X1, X2):
        token1, token2 = dictionary.get_tokens([X1[0], X2[0]])
        features1 = features[token1]
        features2 = features[token2]
        dot = dict_dot(features1, features2)
        if dot == 0:
            return np.inf
        return 1 / float(dot)
    return dist


class PrecomputedKernel(object):

    def __init__(self, features, options):

        self.features = features

        # Pull out and register the relevant options.  This doubles as
        # validation that all the expected options were provided.
        self.count_based_features = options['count_based_features']
        self.non_count_features = options['non_count_features']
        self.count_feature_mode = options['count_feature_mode']
        self.semantic_similarity = options['semantic_similarity']
        self.syntactic_multiplier = options['syntactic_multiplier']
        self.semantic_multiplier = options['semantic_multiplier']
        self.suffix_multiplier = options['suffix_multiplier']

        # Validate values for semantic and syntactic similarity.
        self.validate_semantic_similarity()
        self.validate_syntactic_similarity()

        # Semantic similarity functions need an "information content" file 
        # to calculate similarity values.
        if self.semantic_similarity is not None:
            self.information_content = wordnet_ic.ic(INFORMATION_CONTENT_FILE)

        self.cache = {}
        self.verbose = options.get('verbose', 1)
        self.eval_counter = 0


    def validate_syntactic_similarity(self):
        syntactic_similarity_is_valid = (
            self.count_based_features is None or all(
                feature_type in LEGAL_SYNTACTIC_SIMILARITIES 
                for feature_type in self.count_based_features
        ))
        if not syntactic_similarity_is_valid:
            raise ValueError(
                'count_based_features must be a list with any of the '
                'following: '
                + ', '.join(LEGAL_SYNTACTIC_SIMILARITIES) 
                + '.  Got %s.' % repr(self.count_based_features)
            )


    def validate_semantic_similarity(self):
        semantic_similarity_is_valid = (
            self.semantic_similarity in LEGAL_SIMILARITIES 
            or self.semantic_similarity is None
        )
        if not semantic_similarity_is_valid:
            raise ValueError(
                'semantic_similarity must be one of the following: '
                + ', '.join(LEGAL_SIMILARITIES) 
                + '.  Got %s.' % repr(self.semantic_similarity)
            )

    def precompute(self, examples):
        """
        Precompute the kernel evaluation of all pairs in examples.
        """
        # Add all the example pairs to the work queue
        combinations = list(
            itertools.combinations_with_replacement(examples, 2))
        for i, (ex1, ex2) in enumerate(combinations):
            t4k.progress(i, len(combinations))
            dot = self.eval_pair(ex1, ex2)


    def precompute_parallel(self, examples, num_processes=12):
        """
        Use multiprocessing to precompute the kernel evaluation of all pairs in 
        examples.
        """
        work_queue = iq.IterableQueue()
        result_queue = iq.IterableQueue()

        # Add all the example pairs to the work queue
        print 'loading work onto queue'
        work_producer = work_queue.get_producer()
        num_combinations = len(examples) * (len(examples)-1) / 2
        combinations = itertools.combinations_with_replacement(examples,2)
        for i, (ex1, ex2) in enumerate(combinations):
            t4k.progress(i, num_combinations)
            work_producer.put((ex1, ex2))
        work_producer.close()

        # Start a bunch of workers
        for proc in range(num_processes):
            print 'starting worker %d' % proc
            p = multiprocessing.Process(
                target=self.precompute_worker,
                args=(work_queue.get_consumer(), result_queue.get_producer())
            )
            p.start()

        # Get a result consumer, which is the last endpoint.  No more endpoints 
        # will be made from either queue, so close them
        print 'starting to collect results'
        result_consumer = result_queue.get_consumer()
        result_queue.close()
        work_queue.close()

        # Get all the results and cache them
        for i, (ex1, ex2, dot) in enumerate(result_consumer):
            ex1_token = u.ensure_unicode(self.features.get_token(int(ex1[0])))
            ex2_token = u.ensure_unicode(self.features.get_token(int(ex2[0])))

            t4k.progress(i, num_combinations)
            self.cache[frozenset((ex1_token, ex2_token))] = dot


    def precompute_worker(self, work_consumer, result_producer):
        for ex1, ex2 in work_consumer:
            dot = self.eval_pair(ex1, ex2)
            result_producer.put((ex1, ex2, dot))
        result_producer.close()


    def eval_pair(self, a, b):
        '''
        Custom kernel function that expects token ids.
        '''

        # Convert ids to tokens
        a = u.ensure_unicode(self.features.get_token(int(a[0])))
        b = u.ensure_unicode(self.features.get_token(int(b[0])))

        return self.eval_pair_token(a,b)


    def eval_pair_token(self, a, b):
        '''
        Custom kernel function that expects tokens.
        '''

        # Keep track of calls to this
        self.eval_counter += 1
        #if self.eval_counter % 10000 == 0:
        #    print self.eval_counter

        # Check the cache
        if frozenset((a,b)) in self.cache:
            if self.verbose:
                if self.eval_counter % 10000 == 0:
                    t4k.out('+')
            return self.cache[frozenset((a,b))]

        if self.verbose:
            if self.eval_counter % 10000 == 0:
                t4k.out('.')

        kernel_score = 0

        # Compute the count-based contribution to the kernel
        if self.count_based_features is not None:
            syntax_features_a = self.features.get_count_based_features(
                a, self.count_based_features,
                self.count_feature_mode
            )
            syntax_features_b = self.features.get_count_based_features(
                b, self.count_based_features,
                self.count_feature_mode
            )
            count_based_contribution = self.syntactic_multiplier * dict_dot(
                syntax_features_a, syntax_features_b)            
            if self.verbose > 1:
                print 'count-based contribution:', count_based_contribution
            kernel_score += count_based_contribution

        # Compute derivational feature contribution to kernel
        if 'derivational' in self.non_count_features:
            derivational_contribution = np.dot(
                self.features.get_derivational_features(a),
                self.features.get_derivational_features(b)
            )
            if self.verbose > 1:
                print 'derivational contribution', derivational_contribution
            kernel_score += derivational_contribution

        # Compute embedding feature contribution to kernel
        if 'google-vectors' in self.non_count_features:
            embedding_contribution = np.dot(
                self.features.get_vector(a), self.features.get_vector(b))
            if self.verbose > 1: 
                print 'embedding contribution', embedding_contribution
            kernel_score += embedding_contribution

        # Compute suffix contribution to kernel
        if 'suffix' in self.non_count_features:
            suffix_a = self.features.get_suffix(a)
            suffix_b = self.features.get_suffix(b)
            if suffix_a is not None and suffix_a == suffix_b:
                suffix_contribution = self.suffix_multiplier
            else:
                suffix_contribution = 0
            if self.verbose > 1:
                print 'suffix contribution', suffix_contribution
            kernel_score += suffix_contribution

        # Compute semantic similarity contribution to kernel
        if self.semantic_similarity is not None:
            semantic_features_a, semantic_features_b = None, None
            try:
                semantic_features_a = nouns_only(wordnet.synsets(a))
            except WordNetError:
                print 'WordNetError-a', a

            try:
                semantic_features_b = nouns_only(wordnet.synsets(b))
            except WordNetError:
                print 'WordNetError-b', a

            if semantic_features_a and semantic_features_b:
                try:
                    semantic_contribution = (
                        self.semantic_multiplier * max_similarity(
                            self.semantic_similarity, semantic_features_a, 
                            semantic_features_b, self.information_content
                    ))
                    if self.verbose > 1:
                        print 'semantic contribution', semantic_contribution
                    kernel_score += semantic_contribution
                except WordNetError:
                    print 'WordNetError-ab', a, b

        self.cache[frozenset((a,b))] = kernel_score

        return kernel_score


    def eval(self, A, B):

        """
        Custom kernel function.  This counts how often the links incident on
        two different words within their respective dependency trees are the 
        same, up to the dependency relation and the POS of the neighbour.

        Note that A references a set of words' dependency trees, and B
        references another set.  So that this function end up making
        len(A) * len(B) of such comparisons, and return the result as a 
        len(A) by len(B) matrix.
        """

        result = []
        for a in A:
            result_row = []
            result.append(result_row)
            for b in B:
                result_row.append(self.eval_pair(token_a,token_b))

        return result



def bind_cached_kernel(
    features=None, # Must be provided if syntax_feature_types is True
    syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
    semantic_similarity='res',
    include_suffix=True,
    syntactic_multiplier=0.33,
    semantic_multiplier=0.33,
    suffix_multiplier=0.33
):

    '''
    Returns a kernel function that has a given dictionary and features
    lookup bound to its scope.
    '''

    # Validate that a sensible value for semantic similarity was provided
    semantic_similarity_is_valid = (
        semantic_similarity in LEGAL_SIMILARITIES 
        or semantic_similarity is None
    )
    if not semantic_similarity_is_valid:
        raise ValueError(
            'semantic_similarity must be one of the following: '
            + ', '.join(LEGAL_SIMILARITIES) 
            + '.  Got %s.' % repr(semantic_similarity)
        )

    # Validate that a sensible value for syntactic similarity was provided
    syntactic_similarity_is_valid = syntax_feature_types is None or all(
        feature_type in LEGAL_SYNTACTIC_SIMILARITIES 
        for feature_type in syntax_feature_types
    )
    if not syntactic_similarity_is_valid:
        raise ValueError(
            'syntax_feature_types must be a list with any of the following: '
            + ', '.join(LEGAL_SYNTACTIC_SIMILARITIES) 
            + '.  Got %s.' % repr(syntax_feature_types)
        )

    # Semantic similarity functions need an "information content" file 
    # to calculate similarity values.
    if semantic_similarity is not None:
        information_content = wordnet_ic.ic(INFORMATION_CONTENT_FILE)
        
    cache = {}
    def kernel(A,B):
        '''
        Custom kernel function.  This counts how often the links incident on
        two different words within their respective dependency trees are the 
        same, up to the dependency relation and the POS of the neighbour.

        Note that A references a set of words' dependency trees, and B
        references another set.  So that this function end up making
        len(A) * len(B) of such comparisons, and return the result as a 
        len(A) by len(B) matrix.
        '''

        result = []
        for a in A:

            token_a = u.ensure_unicode(features.get_token(int(a[0])))

            # Get token_a's dependency tree features
            if syntax_feature_types is not None:
                syntax_features_a = features.get_features_idx(
                    int(a[0]), syntax_feature_types
                )

            # Get the token_a's synset if semantic similarity is being used
            if semantic_similarity is not None:
                semantic_features_a = nouns_only(wordnet.synsets(token_a))

            if include_suffix:
                suffix_a = features.get_suffix(token_a)

            result_row = []
            result.append(result_row)
            for b in B:

                # Check the cache
                if frozenset((a,b)) in cache:
                    result_row.append(cache[frozenset((a,b))])
                    continue

                kernel_score = 0
                token_b = u.ensure_unicode(features.get_token(int(b[0])))

                # Calculate the dependency tree kernel
                if syntax_feature_types is not None:
                    syntax_features_b = features.get_features_idx(
                        int(b[0]), syntax_feature_types
                    )
                    kernel_score += syntactic_multiplier * dict_dot(
                        syntax_features_a, syntax_features_b)

                # Calculate semantic similarity is being used
                if semantic_similarity is not None:
                    semantic_features_b = nouns_only(wordnet.synsets(token_b))
                    kernel_score += semantic_multiplier * max_similarity(
                        semantic_similarity, semantic_features_a, 
                        semantic_features_b, information_content
                    )

                # Determine if suffixes match
                if include_suffix:
                    suffix_b = features.get_suffix(token_b)
                    if suffix_a is not None and suffix_a == suffix_b:
                        kernel_score += suffix_multiplier

                cache[frozenset((a,b))] = kernel_score
                result_row.append(kernel_score)

        return result

    return kernel



def bind_kernel(
    features=None, # Must be provided if syntax_feature_types is True
    syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
    semantic_similarity='res',
    include_suffix=True,
    syntactic_multiplier=0.33,
    semantic_multiplier=0.33,
    suffix_multiplier=0.33
):
    '''
    Returns a kernel function that has a given dictionary and features
    lookup bound to its scope.
    '''

    # Validate that a sensible value for semantic similarity was provided
    semantic_similarity_is_valid = (
        semantic_similarity in LEGAL_SIMILARITIES 
        or semantic_similarity is None
    )
    if not semantic_similarity_is_valid:
        raise ValueError(
            'semantic_similarity must be one of the following: '
            + ', '.join(LEGAL_SIMILARITIES) 
            + '.  Got %s.' % repr(semantic_similarity)
        )

    # Validate that a sensible value for syntactic similarity was provided
    syntactic_similarity_is_valid = syntax_feature_types is None or all(
        feature_type in LEGAL_SYNTACTIC_SIMILARITIES 
        for feature_type in syntax_feature_types
    )
    if not syntactic_similarity_is_valid:
        raise ValueError(
            'syntax_feature_types must be a list with any of the following: '
            + ', '.join(LEGAL_SYNTACTIC_SIMILARITIES) 
            + '.  Got %s.' % repr(syntax_feature_types)
        )

    # Semantic similarity functions need an "information content" file 
    # to calculate similarity values.
    if semantic_similarity is not None:
        information_content = wordnet_ic.ic(INFORMATION_CONTENT_FILE)
        
    def kernel(A,B):
        '''
        Custom kernel function.  This counts how often the links incident on
        two different words within their respective dependency trees are the 
        same, up to the dependency relation and the POS of the neighbour.

        Note that A references a set of words' dependency trees, and B
        references another set.  So that this function end up making
        len(A) * len(B) of such comparisons, and return the result as a 
        len(A) by len(B) matrix.
        '''

        result = []
        for a in A:

            token_a = u.ensure_unicode(features.get_token(int(a[0])))

            # Get token_a's dependency tree features
            if syntax_feature_types is not None:
                syntax_features_a = features.get_features_idx(
                    int(a[0]), syntax_feature_types
                )

            # Get the token_a's synset if semantic similarity is being used
            if semantic_similarity is not None:
                semantic_features_a = nouns_only(wordnet.synsets(token_a))

            if include_suffix:
                suffix_a = features.get_suffix(token_a)

            result_row = []
            result.append(result_row)
            for b in B:

                kernel_score = 0
                token_b = u.ensure_unicode(features.get_token(int(b[0])))

                # Calculate the dependency tree kernel
                if syntax_feature_types is not None:
                    syntax_features_b = features.get_features_idx(
                        int(b[0]), syntax_feature_types
                    )
                    kernel_score += syntactic_multiplier * dict_dot(
                        syntax_features_a, syntax_features_b)

                # Calculate semantic similarity is being used
                if semantic_similarity is not None:
                    semantic_features_b = nouns_only(wordnet.synsets(token_b))
                    kernel_score += semantic_multiplier * max_similarity(
                        semantic_similarity, semantic_features_a, 
                        semantic_features_b, information_content
                    )

                # Determine if suffixes match
                if include_suffix:
                    suffix_b = features.get_suffix(token_b)
                    if suffix_a is not None and suffix_a == suffix_b:
                        kernel_score += suffix_multiplier

                result_row.append(kernel_score)

        return result

    return kernel


def nouns_only(synsets):
    '''
    Filters provided synsets keeping only nouns.
    '''
    return [s for s in synsets if s.pos() == 'n']


def max_similarity(
    similarity_type,
    synsets_a,
    synsets_b,
    information_content
):

    similarity_type += '_similarity'
    max_similarity = 0
    for synset_a in synsets_a:
        for synset_b in synsets_b:
            similarity = getattr(synset_a, similarity_type)(
                synset_b, information_content)
            if similarity > max_similarity:
                max_similarity = similarity

    return max_similarity


def dict_dot(a,b):
    result = 0
    for key in a:
        if key in b:
            result += a[key] * b[key]
    return result


class WordnetFeatures(object):

    def __init__(self, dictionary_path=None):

        if dictionary_path is None:
            self.dictionary = None
        else:
            self.dictionary = t4k.UnigramDictionary()
            self.dictionary.load(dictionary_path)

    def get_concept_weight(self, name):
        # Given a concept name, get it's weight.  This takes into account
        # the frequency of occurrence of all lemmas that can disambiguate
        # to that concept.  No attempt is made to figure out how often
        # a term in fact did disambiguate to the lemma
        pass


    def get_wordnet_features(self, lemma):
        synsets = [s for s in wordnet.synsets(lemma) if s.pos() == 'n']
        concepts = set()
        for s in synsets:
            concepts.update(self.get_wordnet_features_recurse(s))
        return concepts

    def get_wordnet_features_recurse(self, synset):
        concepts = set([synset.name()])
        parents = synset.hypernyms()
        parents = [p for p in parents if p.pos() == 'n']
        for p in parents:
            concepts.update(self.get_wordnet_features_recurse(p))
        return concepts

    def wordnet_kernel(self, lemma0, lemma1):
        concepts0 = self.get_wordnet_features(lemma0)
        concepts1 = self.get_wordnet_features(lemma1)
        concepts_in_common = concepts0 & concepts1

        kernel_score = 0
        for c in concepts_in_common:
            kernel_score += self.concept_weighting[c]

        return kernel_score
