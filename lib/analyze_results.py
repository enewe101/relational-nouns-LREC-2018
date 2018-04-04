from collections import Default
import json
import sys
sys.path.join('..')
from SETTINGS import DATA_DIR

RESULTS_PATH = os.path.join(
		DATA_DIR, 'crowdflower', 'results-binary-comprehensive.json')

def read_raw_results(results_path=RESULTS_PATH):
	d = [json.loads(l) for l in open(results_path)]


def results_by_contributor():
	raw_results = read_raw_results()
	contributor_results = 
	for result in raw_results:
		for result in raw_results:
			for judgment in result['results']['judgments']:
				user = judgment['worker_id']
				contributor_results[user].append()


