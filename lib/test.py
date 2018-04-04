import os
import json
import numpy as np
import kernels as k
import t4k
from nltk.corpus import wordnet, wordnet_ic
import classifier
import extract_features
from unittest import TestCase, main
import corenlp_xml_reader


class TestFeatureAccumulator(TestCase):

    def test_extract_(self):
        feature_accumulator = extract_features.make_feature_accumulator()

        article = corenlp_xml_reader.AnnotatedText(
            open('test/corenlp1.xml').read()
        )
        feature_accumulator.extract(article)

        article = corenlp_xml_reader.AnnotatedText(
            open('test/corenlp2.xml').read()
        )
        feature_accumulator.extract(article)

        feature_accumulator.write('test/merged-extracted')


    def test_extract(self):
        article = corenlp_xml_reader.AnnotatedText(
            open('test/corenlp1.xml').read()
        )
        feature_accumulator = extract_features.make_feature_accumulator()
        feature_accumulator.extract(article)

        # Test that the features extracted are the ones expected
        self.assert_feature_like_on_disc(feature_accumulator, 'test/extracted1')


    def assert_feature_like_on_disc(self, feature_accumulator, path):

        # Test that the dictionary extracted is the same as the ones on disk
        expected_dictionary = t4k.UnigramDictionary()
        expected_dictionary.load(os.path.join(path, 'dictionary'))
        self.assertEqual(
            dict(feature_accumulator.dictionary.get_frequency_list()),
            dict(expected_dictionary.get_frequency_list())
        )

        # Test that the features extracted are the same as the ones on disk
        for feature_type in ['dependency', 'baseline', 'hand_picked']:
            expected = json.loads(
                open(os.path.join(path, feature_type + '.json')).read())
            self.assertDictEqual(
                getattr(feature_accumulator, feature_type), expected
            )


    def test_load(self):

        feature_accumulator = extract_features.make_feature_accumulator()
        feature_accumulator.load('test/extracted1')
        self.assert_feature_like_on_disc(feature_accumulator, 'test/extracted1')

        feature_accumulator.load('test/extracted2')
        self.assert_feature_like_on_disc(feature_accumulator, 'test/extracted2')


    def test_merge_load(self):

        article = corenlp_xml_reader.AnnotatedText(
            open('test/corenlp1.xml').read()
        )
        feature_accumulator = extract_features.make_feature_accumulator()
        feature_accumulator.extract(article)

        feature_accumulator.merge_load('test/extracted2')
        self.assert_feature_like_on_disc(
            feature_accumulator, 'test/merged-extracted')


    def test_normalized(self):
        feature_accumulator = extract_features.make_feature_accumulator()
        article = corenlp_xml_reader.AnnotatedText(
            open('test/corenlp1.xml').read()
        )
        feature_accumulator.extract(article)

        normalized_features = {
            token : feature_accumulator.get_features(token, ['dependency'])
            for token in feature_accumulator.dictionary.get_token_list()
        }
        
        open('test/normalized1.json', 'w').write(
            json.dumps(normalized_features)
        )



    def test_get_dep_tree_features(self):
        # Make a mock (empty) dictionary (does not affect test, but needed to 
        # create the feature accumulator).
        dictionary = set()

        # Make a mock dependency tree
        F = {
            'parents':[],
            'children':[],
            'pos':'pos_F'
        }
        E = {
            'parents':[('rel_F', F)],
            'children':[],
            'pos':'pos_E'
        }
        D = {
            'parents':[],
            'children':[],
            'pos':'pos_D'
        }
        C = {
            'parents':[('rel_E', E)],
            'children':[('rel_D', D)],
            'pos':'pos_C'
        }
        B = {
            'parents':[],
            'children':[],
            'pos':'pos_B'
        }
        BB = {
            'parents':[],
            'children':[],
            'pos':'pos_BB'
        }
        A = {
            'parents':[('rel_C', C)],
            'children':[('rel_B', B), ('rel_BB', BB)],
            'pos':'pos_A'
        }

        accumulator = extract_features.FeatureAccumulator(dictionary)
        features = accumulator.get_dep_tree_features_recurse(A, depth=2)

        # Note that because we called it with depth=2, no feature is made for 
        # token F
        expected_features = [
            'parent:rel_C:pos_C', 'parent:rel_C:pos_C-parent:rel_E:pos_E',
            'parent:rel_C:pos_C-child:rel_D:pos_D', 'child:rel_B:pos_B',
            'child:rel_BB:pos_BB'
        ]

        self.assertItemsEqual(features, expected_features)




class TestKernel(TestCase):

    def test_syntax_kernel(self):

        test_tokens = ['ceo','coach', 'manager','boss', 'brother','sister']
        feature_accumulator = extract_features.make_feature_accumulator(
            load='test/000')
        test_ids = [[feature_accumulator.get_id(token)] for token in test_tokens]

        syntax_feature_types = ['baseline', 'dependency', 'hand_picked']
        kernel = k.bind_kernel(
            features=feature_accumulator,
            syntax_feature_types=syntax_feature_types,
            semantic_similarity=None,
            syntactic_multiplier=1.0, semantic_multiplier=1.0,
        )

        found_results = kernel(test_ids, test_ids)
        expected_results = self.get_expected_results(
            test_tokens,
            lambda x,y: 1.0 * k.dict_dot(
                feature_accumulator.get_features(x, syntax_feature_types), 
                feature_accumulator.get_features(y, syntax_feature_types)
            )
        )
        print np.round(np.array(expected_results), 3)
        self.assertEqual(found_results, expected_results)


    def test_semantic_kernel(self):

        information_content = wordnet_ic.ic('ic-treebank-resnik-add1.dat')
        test_tokens = ['ceo','coach', 'manager','boss', 'brother','sister']
        feature_accumulator = extract_features.make_feature_accumulator(
            load='test/000')
        test_ids = [[feature_accumulator.get_id(token)] for token in test_tokens]


        # Next test each of the semantic similarities
        for similarity_type in k.LEGAL_SIMILARITIES:

            kernel = k.bind_kernel(
                features=feature_accumulator,
                syntax_feature_types=None,
                semantic_similarity=similarity_type,
                semantic_multiplier=1.0
            )
            found_results = kernel(test_ids, test_ids)
            expected_results = self.get_expected_results(
                test_tokens, 
                lambda x,y: 1.0 * k.max_similarity(
                    similarity_type,
                    k.nouns_only(wordnet.synsets(x)),
                    k.nouns_only(wordnet.synsets(y)), 
                    information_content
                )
            )
            #print '\n' + '-'*70 + '\n'
            #print similarity_type
            #print np.round(np.array(expected_results), 3)
            self.assertEqual(found_results, expected_results)


    def test_syntax_kernel(self):

        information_content = wordnet_ic.ic('ic-treebank-resnik-add1.dat')
        test_tokens = ['ceo','coach', 'manager','boss', 'brother','sister']
        feature_accumulator = extract_features.make_feature_accumulator(
            load='test/000')
        test_ids = [[feature_accumulator.get_id(token)] for token in test_tokens]


        # Next test each of the semantic similarities
        suffix_multiplier=2.0
        kernel = k.bind_kernel(
            features=feature_accumulator,
            syntax_feature_types=None,
            semantic_similarity=None,
            include_suffix=True,
            suffix_multiplier=suffix_multiplier
        )
        found_results = kernel(test_ids, test_ids)
        expected_results = self.get_expected_results(
            test_tokens, 
            lambda x,y: suffix_multiplier * (
                feature_accumulator.get_suffix(x) is not None 
                and feature_accumulator.get_suffix(x) 
                == feature_accumulator.get_suffix(y)
            )
        )

        self.assertEqual(found_results, expected_results)



    def get_expected_results(self, token_list, func):

        expected_results = []
        for token_a in token_list:
            expected_result_row = []
            expected_results.append(expected_result_row)
            for token_b in token_list:
                expected_result_row.append(
                    func(token_a, token_b)
                )

        return expected_results




if __name__ == '__main__':
    main()
