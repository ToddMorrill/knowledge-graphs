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
from numpy.lib.ufunclike import fix
nltk.download('conll2000')  # noun phrase evaluation
nltk.download('stopwords')  # stopwords
from nltk.corpus import conll2000
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import kg.ner.utils as utils


class NounPhraseDetection(nltk.RegexpParser):
    def __init__(self, grammar=r'NP: {<[CDJNP].*>+}'):
        super().__init__(grammar)

    def extract(self, text, preprocess=True, single_word_proper_nouns=True):
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
        all_phrases = []
        for sentence in preprocessed_sentences:
            tree = self.parse(sentence)
            # tree.subtrees(filter=lambda t: t.label() == 'NP')
            i = 0
            while i < len(tree):
                if isinstance(tree[i], nltk.Tree) and tree[i].label() == 'NP':
                    temp_phrase = tree[i].leaves()
                    # phrase is only one word and not a proper noun, exclude it
                    not_proper_noun = single_word_proper_nouns and (
                        len(temp_phrase) == 1) and (temp_phrase[0][1] != 'NNP')
                    if not_proper_noun:
                        all_phrases.append((temp_phrase[0][0], False))
                        i += 1
                        continue
                    noun_phrase_text = ' '.join(
                        [token for token, pos in temp_phrase])
                    noun_phrases.append(noun_phrase_text)
                    all_phrases.append((noun_phrase_text, True))
                    i += 1
                else:
                    temp_phrase = []
                    while not isinstance(tree[i], nltk.Tree):
                        temp_phrase.append(tree[i])
                        i += 1
                        if i == len(tree):
                            break
                    phrase_text = ' '.join(
                        [token for token, pos in temp_phrase])
                    all_phrases.append((phrase_text, False))

        # unflag stop words
        stop_words_removed = []
        for phrase in all_phrases:
            # if noun phrase == True and is a stop word
            if phrase[1] and (phrase[0] in stop_words):
                stop_words_removed.append((phrase[0], False))
            else:
                stop_words_removed.append(phrase)

        # unflag punction
        final_phrases = []
        for phrase in stop_words_removed:
            # if noun phrase == True and all punctuation
            if phrase[1] and (phrase[0] is all(char in punct
                                               for char in phrase)):
                final_phrases.append((phrase[0], False))
            else:
                final_phrases.append(phrase)

        return final_phrases


class EntityDetection():
    # implement TF-IDF/TextRank filters
    def __init__(self, parser) -> None:
        self.parser = parser

    def candidates(self, text, preprocess=True, single_word_proper_nouns=True):
        tagged_phrases = self.parser.extract(text, preprocess,
                                             single_word_proper_nouns)
        return tagged_phrases

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

    def score_phrases(self, phrases, noun_phrase_flags=True):
        if noun_phrase_flags:
            phrases, flags = zip(*phrases)

        tfidf_vectors = self.vectorizer.transform(phrases)

        # average non-zero entries
        # row sums
        sums = np.squeeze(np.asarray(tfidf_vectors.sum(axis=1)))

        # row-wise counts of non-zero entries (from CSR matrix)
        # non_zero_counts = np.diff(tfidf_vectors.indptr)

        # TODO: find a better way to compute this, accounting for the vocab that vectorizer uses
        token_counts = self.vectorizer._count_vocab(
            raw_documents=phrases, fixed_vocab=False)[1].toarray()
        token_counts = np.squeeze(np.asarray(token_counts.sum(axis=1)))

        scores = sums / token_counts

        if noun_phrase_flags:
            return list(zip(phrases, flags, scores))
        return list(zip(phrases, scores))


class EntityTypeDetection():
    # implement type detection (cluster based?)
    pass


def main(args):
    df_dict = utils.load_train_data(args.data_directory)
    train_df = df_dict['train.csv']
    # gather up articles
    articles = train_df.groupby(['Article_ID'], )['Token'].apply(
        lambda x: ' '.join([str(y) for y in list(x)])).values.tolist()

    chunk_parser = NounPhraseDetection()

    sample = 'Here is some sample text. And some more!'
    noun_phrases = chunk_parser.extract(articles[0])
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
    sample_phrases = [
        'document, document, document', 'This document, document in',
        'Washington'
    ]
    results = entity_extractor.score_phrases(noun_phrases)
    breakpoint()

    # print(entity_candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)