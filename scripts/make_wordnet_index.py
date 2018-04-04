import sys
sys.path.append('../lib')
import make_wordnet_index as mwi


def make_wordnet_index():
	in_path = '/Users/enewel3/nltk_data/corpora/wordnet/index.noun'
	out_path = '../../../data/wordnet_index.txt'
	mwi.make_wordnet_index(in_path, out_path)


if __name__=='__main__': 
	make_wordnet_index()

