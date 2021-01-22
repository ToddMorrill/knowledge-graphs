"""
This module will implement some unsupervised NER experiments.

Implementation notes:
1) Extract noun phrases.
2) Score the phrases using:
    - TFIDF
    - TextRank
3) Empirically determine an appropriate cutoff threshold using the validation set.
4) Score overlap with entities.
5) Cluster phrases using pre-trained language models (or hierarchically).
    - Need a metric to determine the appropriate number of clusters.
    - May be able to tune this process using the validation set.
    - Also, how to label clusters and map them to the ground truth labels?
6) Score predictions against ground truth typed entities.

Examples:
    $ python unsupervised.py \
        --data-directory /Users/tmorrill002/Documents/datasets/conll/transformed
"""

import argparse

import nltk

import kg.ner.utils as utils


def tag_pos(sentence):
    # sentences = [nltk.pos_tag(sent) for sent in sentences]
    pass


def extract_noun_phrases():
    pass


def main(args):
    df_dict = utils.load_train_data(args.data_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)