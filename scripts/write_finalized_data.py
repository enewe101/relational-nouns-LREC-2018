import os
import t4k
import sys
sys.path.append('../lib')
from SETTINGS import DATA_DIR
import crowdflower.interpret_results as interp

CROWDFLOWER_DIR = os.path.join(DATA_DIR, 'crowdflower')
PARTICIPANT_RESULTS_PATHS = [
    os.path.join(CROWDFLOWER_DIR, 'results1.json'),
    os.path.join(CROWDFLOWER_DIR, 'results2.json')
]
TRIPLE_EXPERT_RESULTS_PATH = os.path.join(
    CROWDFLOWER_DIR, 'results-expert.json')
EXPERT_RESULTS_PATH = os.path.join(CROWDFLOWER_DIR, 'results3-expert-annot.tsv')
EXPERT_TASK_PATH = os.path.join(CROWDFLOWER_DIR, 'results3-expert.csv')
FEATURES_DIR = os.path.join(
    DATA_DIR, 'relational-noun-features-wordnet-only', 'accumulated-pruned-5000'
)
DICTIONARY_DIR = os.path.join(FEATURES_DIR, 'dictionary')
FINALIZED_ANNOTATIONS_PATH = os.path.join(
    DATA_DIR, 'relational-nouns', 'all-annotations.tsv')


def write_finalized_data():

    # Open the file that we'll be writing to
    out_file = open(FINALIZED_ANNOTATIONS_PATH, 'w')

    # Get the triple-annotated expert results
    triple_expert_results = t4k.CrowdflowerResults(TRIPLE_EXPERT_RESULTS_PATH)
    finalized_triple_expert = interp.finalize_crowdflower_data_rows(
        triple_expert_results, '3-experts')

    # Get the single-annotated expert results
    finalized_single_expert = interp.finalize_noncrowdflower_data_rows()

    # Get the participant results
    participant_results = t4k.CrowdflowerResults(PARTICIPANT_RESULTS_PATHS)
    finalized_participant = interp.finalize_crowdflower_data_rows(
        participant_results, 'participants')

    # Merge these together, earlier ones take precedence
    finalized_results = t4k.merge_dicts(
        finalized_triple_expert, finalized_participant, finalized_single_expert)

    # Write the results out to file as a tsv
    out_file.write(
        '\n'.join(['\t'.join(row) for row in finalized_results.values()])
    )



if __name__ == '__main__':
    write_finalized_data()

