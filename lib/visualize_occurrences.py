import sys
sys.path.append('..')
from SETTINGS import DATA_DIR, GIGAWORD_DIR
from subprocess import check_output
from corenlp_xml_reader import AnnotatedText 
import os
import utils

GREP_RESULT_PATH = os.path.join(DATA_DIR, 'grep-result.html')

def grep_giga_for_relational_nouns(limit=100, path=GREP_RESULT_PATH):
	positives, negatives = utils.get_training_sets()
	grep_giga(positives, limit=limit, path=path)


def load_article(article_id):
	subdir = article_id[:3]
	path = os.path.join(
		GIGAWORD_DIR, 'data', subdir, 'CoreNLP', '%s.txt.xml' % article_id
	)
	return AnnotatedText(open(path).read())


def gigaword(limit=100):
	giga_path = os.path.join(GIGAWORD_DIR, 'data', '9fd', 'CoreNLP')
	fnames = check_output(['ls', giga_path]).split()
	fnames = [os.path.join(giga_path, f) for f in fnames][:limit]

	return [(fname, AnnotatedText(open(fname).read())) for fname in fnames]


def load_from_def(article_def):
	article_id, sentence_id = [s.strip() for s in article_def.split(':')]
	sentence_id = int(sentence_id)
	article = load_article(article_id)
	sentence = article.sentences[sentence_id]
	return article, sentence


def show(article_def):
	article, sentence = load_from_def(article_def)
	print sentence


def ctree(article_def):
	'''
	Print a representation of the constituency parse tree.
	'''
	article, sentence = load_from_def(article_def)
	article.print_tree(sentence['c_root'])


def dtree(article_def):
	'''
	Print a representation of the dependency parse tree.
	'''
	article, sentence = load_from_def(article_def)
	print sentence.dep_tree_str()


def get_article_id(path):
	fname = path.split('/')[-1]
	article_id = fname.split('.')[0]
	return article_id


def grep_giga(target_lemmas, limit=100, path=GREP_RESULT_PATH):
	'''
	Search through `limit` number of gigaword articles, finding sentences
	that match the lemmas listed in `target_lemmas` (a set of strings), 
	and create an html page that displays the matched senteces with
	matched text highlighted
	'''

	if path is not None:
		out_file = open(path, 'w')

	markup = ''
	for fname, article in gigaword(limit=limit):

		# Get the markup for matched sentences
		match_markups = grep_article(target_lemmas, article)

		# Wrap it in additional markup, and accumulate the markup
		for sentence_id, match_markup in match_markups:
			markup += '<div class="sentence">'
			markup += '<div class="sentence_id">'
			markup += '%s : %d' % (get_article_id(fname), sentence_id)
			markup += '</div>'
			markup += match_markup
			markup += '</div>'

	# Wrap the markup in an html page with styling
	markup = '<html>%s<body>%s</body></html>' % (get_html_head(), markup)

	# Write markup to file (if given)
	if path is not None:
		out_file.write(markup)

	# Return the markup
	return markup


def grep_article(target_lemmas, annotated_text):
	'''
	Find sentences in `annotated_text` that have lemmas that match
	elements in `target_lemmas`, then output html markup for such sentences
	so as to highlight occurrences of elements in `target_lemmas` and 
	indicate parts of speech.
	- `target_lemmas` should be a set of strings representing lemmas
	- `annotated_text` a corenlp_xml_reader.AnnotatedText instance
	'''

	for sentence_id, sentence in enumerate(annotated_text.sentences):

		# If this sentence matches any of the lemmas, add markup for it
		marked_up_sentences = []
		lemmas = set([t['lemma'] for t in sentence['tokens']])
		if lemmas & target_lemmas:

			markup = ''
			for token in sentence['tokens']:
				token_markup = '<span class="token">'

				# Add the word, highlight if its lemma was a match
				if token['lemma'] in target_lemmas:
					token_markup += (
						'<span class="match">%s</span>' % token['word'])
				else:
					token_markup += token['word']

				# Add the pos tag then close the token tag
				token_markup += '<span class="pos">'
				token_markup += '<span class="pos-inner">'
				token_markup += token['pos'] + '</span></span></span> '

				# Accumulate the markup for each token
				markup += token_markup

			marked_up_sentences.append((sentence_id, markup))

	return marked_up_sentences

'ignore occurrences of verbs'

patterns = {
	'PRP$ nounphrase-with-head(NNS?)': [],
	'WP$ noun': ['9fd68017af26898c : 20'],
	'demonym nounphrase-with-head(NNS?)': ['9fd0a4ee9f097d72 : 11',
		'9fd637e46710f28e : 14'],

	'is a <noun> of NP': ['9fd22ab41e0e5ae9 : 3'],
	'<noun> of NP': [
		'9fd0a4ba93b96249 : 9', '9fd0e375e8355175 : 66'],
	'as [a] <noun>': ['9fd0af9b8725e34c : 52', '9fd3717d0477a5b7 : 13',
		'9fd57c371f384ec4 : 14'],
	'a noun to noun (e.g. a brother to me)': [],
	'<noun> to NNP': ['9fd500920273bea6 : 0'],
	'at the end of a noun catenary': ['9fd0a6aae6576477 : 13'],
	'at the end of a noun catenary, preceeded by NNP': [
		'9fd1017387fe1274 : 42'],
	'at beginning of noun phrase, preceeding NNP': [
		'9fd387a122e7d0c8 : 17', '9fd425bc42a69688 : 3', 
		'9fd583fd4a3586ad : 19'],
	'x has a noun': [],
	'candidate for': ['9fd297b4c671fe0f : 9', '9fd706a241e8995a : 14'],
	'other': ['9fd26f6f33891f2f : 10'],
	'use of "the" determiner, but never "a" except before "of" or appos': [
		'9fd2d2494cea4d73 : 12'],
	'having an appositive': [
		'9fd3d817389fee39 : 14', '9fd4101bfdccba4d : 7', 
		'9fd4a5506e1c5bca : 49'],
	'compound': ['9fd4d43d76bd7805 : 2'],
	'preceeded by NNP that is country or company, preceeding NNP that is person': [],
}

def get_html_head():
	return ' '.join([
		'<head><style>',
		'.pos {position: absolute; font-size: 0.6em;',
			'top: 6px; left: 50%; font-weight: normal;',
			'font-style: normal}',
		'.pos-inner {position: relative; left:-50%}',
		'body {line-height: 40px;}',
		'p {margin-bottom: 30px; margin-top: 0}',
		'.match {color: blue; font-weight: bold;}',
		'.token {position: relative}',
		'.attribution-id {display: block; font-size:0.6em;',
			'margin-bottom:-20px;}',
		'</style></head>'
	])


if __name__ == '__main__':
	grep_giga_for_relational_nouns(
		limit=100, path='../../data/grep-relational-nouns100.html')
