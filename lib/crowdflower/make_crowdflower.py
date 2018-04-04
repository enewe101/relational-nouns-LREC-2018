import sys
sys.path.append('..')
sys.path.append('../..')
import os
from SETTINGS import DATA_DIR, TRAIN_DIR
from utils import get_train_sets, get_test_sets
import csv
import t4k
import random
from labeled_words import LabeledWords

OUT_PATH = os.path.join(DATA_DIR, 'relational-noun-crowdflower.csv')
TRAIN_SET_PATH = os.path.join(TRAIN_DIR, 'all.tsv')

def make_crowdflower_task(out_path=OUT_PATH):

	random.seed(0)

	# Read in positives and negatives
	labeled_words = LabeledWords(TRAIN_SET_PATH)
	random.shuffle(labeled_words)

	# Start the tsv file
	writer = csv.writer(open(out_path, 'wb'))
	headings = [
		'question_id', 'word', '_golden', 'response_gold',
		'response_gold_reason'
	]
	writer.writerow(headings)

	# Generate all the rows first
	rows = []
	for i, word in enumerate(labeled_words):

		# Make 1 in 10 questions gold
		is_golden = (i % 10) == 0

		if word['subtype-code'] == 'r':
			gold_answer = 'non-relational'
			reason = (
				'This stands for the a type of relationship rather than '
				'standing for one of the members of the relationship, '
				'so it is not a relational noun.'
			)

		if word['subtype-code'] in 'bafvpj':
			gold_answer = '\r\n'.join([
				'non-relational', 'mainly non-relational'
			])
			reason = '%s is non-relational.' % word['word']


		if word['is_relational']:
			if word['is_partial']:
				gold_answer = '\r\n'.join([
					'relational', 'mainly relational', 
					'mainly nonrelational'
				])
				reason = '%s is mainly relational.' % word['word']
			else:
				gold_answer = '\r\n'.join([
					'relational', 'mainly-relational'
				])
				reason = '%s is relational.' % word['word']

		else:
			if word['is_partial']:
				gold_answer = '\r\n'.join([
					'non-relational', 'mainly non-relational',
					'mainly relational'
				])
				reason = '%s is mainly non-relational.' % word['word']
			else:
				gold_answer = '\r\n'.join([
					'non-relational', 'mainly non-relational'
				])
				reason = '%s is non-relational.' % word['word']

		rows.append(
			(t4k.get_id(), word['word'], is_golden, gold_answer, reason)
		)

	# Write the csv file
	writer.writerows(rows)
	

