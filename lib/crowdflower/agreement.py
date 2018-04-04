"""
Calculate the agreement between annotators
"""

import os
import sys
sys.path.append('..')
import t4k
from SETTINGS import DATA_DIR
from collections import defaultdict
import krippendorff_alpha as k


def as_number_string(response):
    return {
        'usually relational': '2',
        'occasionally relational': '1',
        'almost never relational': '0',
        '+':'2',
        '0':'1',
        '-':'0'
    }[response]


def convert_interpeted_annotations(expert_annotations, participant_annotations):
    all_words = set()
    expert_converted = []
    participant_converted = []
    for word in expert_annotations:
        expert_converted.append(as_number_string(expert_annotations[word]))
        participant_converted.append(as_number_string(
            participant_annotations[word]))

    return [expert_converted, participant_converted]


def load_agreement_set(annotators="both"): # both, experts, nonexperts
    """
    Loads the 250 nouns which were annotated both by experts and non-experts.
    """
    expert_results_path = os.path.join(
        DATA_DIR, 'crowdflower', 'results-expert.json')
    nonexpert_results_path = os.path.join(
        DATA_DIR, 'crowdflower', 'results1.json')

    # First load only the expert results, which includes only the 250 nouns we
    # are interested in.
    results_exp = t4k.CrowdflowerResults(expert_results_path)

    if annotators == 'experts':
        return results_exp
    elif annotators == 'nonexperts':
        results = t4k.CrowdflowerResults(nonexpert_results_path)
    elif annotators == 'both':
        results = t4k.CrowdflowerResults(
            [nonexpert_results_path, expert_results_path], 
            merge_key=lambda r: r['data']['token']
        )
    else:
        raise ValueError(
            '`annotators` must be "experts", "nonexperts" or "both"')

    # Now among the merged results, retain only those that are part of the 250
    # nouns of interest
    nouns_of_interest = {r['data']['token'] for r in results_exp}
    results = [r for r in results if r['data']['token'] in nouns_of_interest]

    return results


def calculate_agreement(annotators='both'): # both, experts, nonexperts
    results = load_agreement_set(annotators)
    converted = convert_data(results)
    agreement = calculate_krippendorf(converted, metric='iterval')
    return agreement


def convert_data(results):
    """
    Convert the data from the format given from analyze_results.get_results
    to the format expected by the Krippendorph alpha calculation package.
    This involves aggregating the responses on a per-worker basis.
    """

    # Each worker is a key, and the values are a question:response dictionary
    # summarizing that worker's responses
    question_ids = set()
    workers_responses = defaultdict(dict)
    for result in results:
        question_id = result['data']['token']
        question_ids.add(question_id)
        judgments = result['results']['judgments']
        for judgment in judgments:
            worker = judgment['worker_id']

            # Get the response from one of two locations
            try:
                response = judgment['data']['response']
            except KeyError:
                response = judgment['data']['is_relational']

            # Convert it into a number
            response = as_number_string(response)

            workers_responses[worker][question_id] = response

    # Now convert the dictionary of worker responses to a list, one row per
    # worker.  (Using a dict just helped to accumulate responses on a
    # per-worker basis, whereas the agreement calculator just expects workers
    # as "rows")
    worker_responses_as_list = []
    ordered_questions = list(question_ids)
    for response_set in workers_responses.values():
        ordered_responses = [
            response_set[x] if x in response_set else '*'
            for x in ordered_questions
        ]
        worker_responses_as_list.append(ordered_responses)

    return worker_responses_as_list


def calculate_krippendorf(data, metric='nominal'):

    # Calculate agreement
    if metric == 'nominal':
        print k.krippendorff_alpha(
            data,
            k.nominal_metric,
            missing_items='*'
        )

    elif metric == 'interval':
        print k.krippendorff_alpha(
            data,
            k.interval_metric,
            missing_items='*'
        )

if __name__ == '__main__': 
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

    data = ( 
        "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3", # coder A
        "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *", # coder B
        "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4", # coder C
    )   

    missing = '*' # indicator for missing values
    array = [d.split() for d in data]  # convert to 2D list of string items


    print k.krippendorff_alpha(
        array,
        k.nominal_metric,
        missing_items='*'
    )
    
    #print("nominal metric: %.3f" % k.krippendorff_alpha(array, k.nominal_metric, missing_items=missing))
    #print("interval metric: %.3f" % k.krippendorff_alpha(array, k.interval_metric, missing_items=missing))
