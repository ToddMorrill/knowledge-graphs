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
import string

import nltk
from nltk.corpus.reader.rte import norm
from numpy.lib.function_base import vectorize
nltk.download('conll2000')  # noun phrase evaluation
nltk.download('stopwords')  # stopwords
from nltk.corpus import conll2000
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import kg.ner.utils as utils


class NounPhraseDetection(nltk.RegexpParser):
    def __init__(self, grammar=r'NP: {<[CDJNP].*>+}'):
        super().__init__(grammar)

    def extract(self, text, preprocess=True):
        # exclude candidates that are stop words or entirely punctuation
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))

        # optionally preprocess (tokenize sentences/words, tag POS)
        if preprocess:
            preprocessed_sentences = utils.preprocess(text)
        else:
            preprocessed_sentences = text

        # extract
        noun_phrases = []
        for sentence in preprocessed_sentences:
            tree = self.parse(sentence)
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                temp_phrase = subtree.leaves()
                noun_phrase_text = ' '.join(
                    [token for token, pos in temp_phrase])
                noun_phrases.append(noun_phrase_text)

        # remove stop words
        noun_phrases = [
            phrase for phrase in noun_phrases if phrase not in stop_words
        ]

        # remove punction (maybe)
        noun_phrases = [
            phrase for phrase in noun_phrases
            if phrase is not all(char in punct for char in phrase)
        ]
        return noun_phrases


class EntityDetection():
    # implement TF-IDF/TextRank filters
    def __init__(self, parser) -> None:
        self.parser = parser

    def candidates(self, text, preprocess=True):
        noun_phrases = self.parser.extract(text, preprocess=preprocess)
        return noun_phrases

    def fit_tfidf(self, documents, preprocess=True):
        # TODO: determine the best tokenizer/casing to use
        # # extract noun phrases from documents
        # phrases = []
        # for doc in documents:
        #     for candidate in self.candidates(doc, preprocess=preprocess):
        #         phrases.append(candidate)

        # fit global idf scores
        self.vectorizer = TfidfVectorizer(norm=None)
        self.vectorizer.fit(documents)

    def score_phrases(self, phrases):
        tfidf_vectors = self.vectorizer.transform(phrases)

        # average non-zero entries
        # row sums
        sums = np.squeeze(np.asarray(tfidf_vectors.sum(axis=1)))

        # row-wise counts of non-zero entries (from CSR matrix)
        non_zero_counts = np.diff(tfidf_vectors.indptr)

        return sums / non_zero_counts


class EntityTypeDetection():
    # implement type detection (cluster based?)
    pass


def main(args):
    df_dict = utils.load_train_data(args.data_directory)
    # noun_phrase_patterns = [
    #     '(<DT|PRP\$|RBR>?<JJ.*>*<NN.*>+)', '(<CD><NN><TO><CD><NN>)',
    #     '(<JJR>?<IN>?<CD>?<NN>+)', '(<POS><NN>)'
    # ]
    # grammar = f'NP:  {{{"|".join(noun_phrase_patterns)}}}'
    grammar = r'NP: {<[CDJNP].*>+}'
    chunk_parser = NounPhraseDetection(grammar)

    sample = 'Here is some sample text. And some more!'
    noun_phrases = chunk_parser.extract(sample)
    print(noun_phrases)

    corpus = [
        'This is the first document.', 'This document is the second document.',
        'And this is the third one.', 'Is this the first document?'
    ]
    entity_extractor = EntityDetection(chunk_parser)
    # entity_candidates = entity_extractor.candidates(sample)
    entity_extractor.fit_tfidf(corpus + [sample])
    temp = entity_extractor.vectorizer.transform(
        ['another document', 'second document'])

    # print(entity_candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)