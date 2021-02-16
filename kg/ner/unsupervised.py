"""
This module implements unsupervised noun phrase extraction, entity detection, and
entity type detection.

Implementation notes:
1) Extract noun phrases. 
    - Evaluate noun phrase detection against CoNLL-2000 test data.
    Custom RegExpParser:
        IOB Accuracy:  81.5%%
        Precision:     71.9%%
        Recall:        63.4%%
        F-Measure:     67.4%%
2) Score the noun phrases using TFIDF and TextRank to yield untyped entities.
    - Empirically determine an appropriate cutoff threshold using the CoNLL-2003 validation set.
    - Score overlap with ground truth CoNLL-2003 untyped entities.
3) Assign types to entities.
    - Cluster phrases using pre-trained language models (or hierarchically).
    - Need a metric to determine the appropriate number of clusters.
    - May be able to tune this process using the validation set.
    - Also, how to label clusters and map them to the ground truth labels?
    - Score predictions against ground truth CoNLL-2003 typed entities.

Examples:
    $ python unsupervised.py \
        --data-directory /Users/tmorrill002/Documents/datasets/conll/transformed
"""

import argparse

import nltk
nltk.download('punkt')  # word tokenizer
nltk.download('averaged_perceptron_tagger')  # pos tagger
nltk.download('conll2000')  # noun phrase evaluation
from nltk.corpus import conll2000

import kg.ner.utils as utils


class NounPhraseDetection(nltk.RegexpParser):
    def __init__(self, grammar=r'NP: {<[CDJNP].*>+}'):
        super().__init__(grammar)


class EntityDetection():
    # implement TF-IDF/TextRank filters
    pass


class EntityTypeDetection():
    # implement type detection (cluster based?)
    pass


def main(args):
    # df_dict = utils.load_train_data(args.data_directory)
    # noun_phrase_patterns = [
    #     '(<DT|PRP\$|RBR>?<JJ.*>*<NN.*>+)', '(<CD><NN><TO><CD><NN>)',
    #     '(<JJR>?<IN>?<CD>?<NN>+)', '(<POS><NN>)'
    # ]
    # grammar = f'NP:  {{{"|".join(noun_phrase_patterns)}}}'
    grammar = r'NP: {<[CDJNP].*>+}'
    chunk_parser = NounPhraseDetection(grammar)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)