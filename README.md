# relational-nouns-LREC-2018
Public repo for code accompanying the LREC 2018 publication

# Usage

## Command line
To invoke the classifier on a list of nouns, do this:
one noun per line, and then do:

    python relational_nouns.py classify <nouns-to-classify>

where <nouns-to-classify> is theh path to a list of nouns, one per line.
the output is a list of integers, -1 means non-relational, and 1 means
relational or occasionally relational.

The classifier takes a long time to load.  To load a classifier that performs
less well but loads more quickly, e.g. for debugging purposes, do this:

    python relational_nouns.py classify-small <nouns-to-classify>



## As an import in a python script
To use the classifier in your own python scripts, do this:

    from relational_nouns import build_classifier
    classifier = build_classifier()

Building the classifier takes about 15 minutes.  To build a classifier that
has performs less well but loads more quickly, e.g. for debugging purposes,
do this:

    from relational_nouns import build_classifier, SMALL_SETTINGS
    classifier = build_classifier(SMALL_SETTINGS)

To classify new nouns, provide them as a list:

    classifier.predict_tokens(['tomato', 'amigo', 'kiwi', 'compadre'])

the output is a list of integers, -1 means non-relational, and 1 means
relational or occasionally relational.



## Replicating tests in paper
To replicate the testing run of our best performing classifier, do:

    python relational_nouns.py test <out-file> <run-name> [<run-name>, ...]

Where <out-file> is the filename, under `data/performance/`, at which the
results should be written, and where run-name should be either
'optimal-test-rand' or 'optimal-test-top', to respectively test the best
classifier words in the test set drawn from the most common words in
Gigaword, or words in the test set drawn at random.

