
def make_wordnet_index(in_file, out_file, exclude_compound=True):

	in_file = open(in_file)
	out_file = open(out_file, 'w')

	for line in in_file:

		# Skip comments at beginning of file.  Comment lines start with a
		# space.
		if line.startswith(' '):
			continue

		lemma = line.split(' ', 1)[0]

		# Compound words are joined by '_'.  Exclude them if that's what we
		# want to do.
		if exclude_compound and '_' in lemma:
			continue

		out_file.write(lemma + '\n')


if __name__ == '__main__':
	in_path = '/Users/enewel3/nltk_data/corpora/wordnet/index.noun'
	out_path = '../../../data/wordnet_index.txt'
	make_wordnet_index(in_path, out_path)
