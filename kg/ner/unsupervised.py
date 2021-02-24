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
from collections import Counter
import itertools

import string
from typing import final

import networkx
import nltk
from nltk import chunk
from numpy.core.fromnumeric import var
nltk.download('conll2000')  # noun phrase evaluation
nltk.download('stopwords')  # stopwords
from nltk.corpus import conll2000
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import kg.ner.utils as utils


class NounPhraseDetection(nltk.RegexpParser):
    def __init__(self, grammar=r'NP: {<[CDJNP].*>+}'):
        super().__init__(grammar)

    def extract(self,
                document,
                preprocess=True,
                single_word_proper_nouns=True):
        # exclude candidates that are stop words or entirely punctuation
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))

        # optionally preprocess (tokenize sentences/words, tag POS)
        if preprocess:
            preprocessed_sentences = utils.preprocess(document)
        else:
            preprocessed_sentences = document

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


class EntityScore(object):
    # implement TF-IDF/TextRank filters
    def __init__(self, parser) -> None:
        self.parser = parser

    def candidates(self,
                   document,
                   preprocess=True,
                   single_word_proper_nouns=True):
        tagged_phrases = self.parser.extract(document, preprocess,
                                             single_word_proper_nouns)

        return tagged_phrases

    def candidates_documents(self,
                             documents,
                             preprocess=True,
                             single_word_proper_nouns=True):
        tagged_phrases = []
        for document in documents:
            phrases = self.candidates(document, preprocess,
                                      single_word_proper_nouns)
            for phrase in phrases:
                tagged_phrases.append(phrase)

        return tagged_phrases

    def fit(self, documents, preprocess=True):
        raise NotImplementedError

    def score_phrases(self, phrases, noun_phrase_flags=True):
        raise NotImplementedError


class TFIDFScore(EntityScore):
    # implement TF-IDF/TextRank filters
    def __init__(self, parser) -> None:
        super().__init__(parser)

    def fit(self, documents, preprocess=True):
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


class TextRankScore(EntityScore):
    # adjectives, nouns, verbs
    def __init__(self,
                 document,
                 preprocess,
                 parser,
                 included_pos=['JJ', 'NN', 'VB'],
                 window_size=2) -> None:
        super().__init__(parser)
        if preprocess:
            self.document = utils.preprocess(document)
        else:
            self.document = document
        self.included_pos = included_pos
        self.window_size = window_size

    def _keep_token(self, token_pos):
        token, pos = token_pos
        for included in self.included_pos:
            if pos.startswith(included):
                return True
        return False

    def _get_nodes(self):
        # TODO: lemmatize tokens
        nodes = []
        for sentence in self.document:
            for token, pos in sentence:
                if self._keep_token((token, pos)):
                    nodes.append((token.lower(), pos))
        return nodes

    def _get_edges(self):
        edges = []
        for sentence in self.document:
            filtered_sentence = []
            for token, pos in sentence:
                if self._keep_token((token, pos)):
                    filtered_sentence.append((token.lower(), pos))

            for hop in range(self.window_size):
                for idx, node in enumerate(filtered_sentence[:-1 - hop]):
                    neighbor = filtered_sentence[hop + idx + 1]
                    edges.append((node, neighbor))

        # include weight on the edge: (2, 3, {'weight': 3.1415})
        weighted_edges = [(*edge, {
            "weight": weight
        }) for edge, weight in Counter(edges).items()]

        return weighted_edges

    def fit(self):
        # initalize graph
        self.graph = networkx.DiGraph()

        # get nodes
        self.graph.add_nodes_from(self._get_nodes())

        # get edges
        self.graph.add_edges_from(self._get_edges())

        # get scores for words
        self.token_scores = networkx.pagerank(self.graph)

    def score_phrases(self,
                      phrases,
                      noun_phrase_flags=True,
                      preprocess=True):
        if noun_phrase_flags:
            phrases, flags = zip(*phrases)
        
        # NB: should really be returning pos tags from self.candidates
        # possible to introduce pos errors when working on these fragments
        # get pos tags
        phrases = utils.tokenize_sentences(phrases)
        phrases = utils.tag_pos(phrases)

        scores = []
        for phrase in phrases:
            phrase_score = 0
            tokens_found = 0
            for token, pos in phrase:
                if (token.lower(), pos) in self.token_scores:
                    phrase_score += self.token_scores[(token.lower(), pos)]
                    tokens_found += 1
            # score normalized by length
            score = phrase_score / tokens_found if tokens_found > 0 else 0
            scores.append(score)

        # define a percentile rank for noun phrases (0 for non-noun phrases)
        noun_phrase_scores = []
        for idx, phrase in enumerate(phrases):
            if flags[idx]:
                noun_phrase_scores.append(scores[idx])
        
        final_ranks = []
        for idx, phrase in enumerate(phrases):
            if flags[idx]:
                percentile_rank = stats.percentileofscore(noun_phrase_scores, scores[idx])
                final_ranks.append((phrase, flags[idx], percentile_rank/100))
            else:
                # assign a score of 0.0 to non-noun phrases
                final_ranks.append((phrase, flags[idx], 0.0))                

        return final_ranks


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

    # sample = 'Here is some sample text. And some more!'
    # noun_phrases = chunk_parser.extract(articles[0])
    # print(noun_phrases)

    # corpus = [
    #     'This is the first document.', 'This document is the second document.',
    #     'And this is the third one.', 'Is this the first document?'
    # ]
    # entity_extractor = TFIDFScore(chunk_parser)
    # # entity_candidates = entity_extractor.candidates(sample)
    # entity_extractor.fit_tfidf(corpus + [sample])
    # temp = entity_extractor.vectorizer.transform(
    #     ['another document', 'second document'])
    # sample_phrases = [
    #     'document, document, document', 'This document, document in',
    #     'Washington'
    # ]
    # results = entity_extractor.score_phrases(noun_phrases)
    scorer = TextRankScore(articles[0], preprocess=True, parser=chunk_parser)
    scorer.fit()
    candidates = scorer.candidates(articles[0])
    scores = scorer.score_phrases(candidates)
    breakpoint()

    # candidates = scorer.candidates_documents(corpus)
    # candidates = scorer.candidates(corpus[0])
    # breakpoint()
    # candidate_noun_phrases = [tokens for tokens, flag in candidates if flag]
    # scored_words = scorer.fit_document(candidate_noun_phrases)
    # scored_candidates = scorer.score_phrases(candidates, scored_words)
    # breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)