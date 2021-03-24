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
3) TODO: Assign types to entities.
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
import math
import string

import networkx
import nltk
nltk.download('conll2000')  # noun phrase evaluation
nltk.download('stopwords')  # stopwords
from nltk.corpus import conll2000
import numpy as np
import pandas as pd
from scipy import stats
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import kg.ner.utils as utils
from kg.ner.cluster import Cluster


class NounPhraseDetector(nltk.RegexpParser):
    """A class that detects noun phrase chunks in text documents.
    
    This class inherits from nltk.RegexpParser and in particular, makes use of
    the parse() and evaluate() methods.
    """
    def __init__(self, grammar: str = r'NP: {<[CDJNP].*>+}') -> None:
        """Initialize the parser by specifying an NLTK-style grammar.

        Args:
            grammar (str, optional): NLTK-style grammar specification. Defaults to r'NP: {<[CDJNP].*>+}'.
        """
        super().__init__(grammar)

    def extract(self,
                document: str,
                preprocess: bool = True,
                single_word_proper_nouns: bool = True) -> list:
        """Extract noun phrases from the passed text document. This method 
        returns the entire document, with noun phrases marked True and all other 
        parts of speech marked False. Stopwords and punctuation are marked 
        False.

        Args:
            document (str): Text document from noun phrases are extracted.
            preprocess (bool, optional): Tokenize and assign POS tags. Defaults 
                to True.
            single_word_proper_nouns (bool, optional): If True, assign False to 
                single noun phrase tokens that are not proper nouns. Defaults to 
                True.

        Returns:
            list: Tuples of ('text phrase', True/False) from the document.
        """
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


class ProperNounDetector(object):
    def __init__(self) -> None:
        pass

    def _flag_nnp(self, document):
        groupings = []
        for sentence in document:
            for key, group in itertools.groupby(sentence, key=lambda x: x[1]):
                grouping = ' '.join([token for token, pos in list(group)])
                if key == 'NNP':
                    groupings.append((grouping, True))
                else:
                    groupings.append((grouping, False))
        return groupings

    def extract(self, document: str, preprocess: bool = True) -> list:
        # optionally preprocess (tokenize sentences/words, tag POS)
        if preprocess:
            document = utils.preprocess(document)

        # get proper noun phrases
        return self._flag_nnp(document)


class EntityScorer(object):
    """An abstract base class that scores the likelihood of a phrase being an
    entity.
    """
    def __init__(self, parser: NounPhraseDetector) -> None:
        """Initialize the class with a NounPhraseDetector that can extract 
        phrases.

        Args:
            parser (NounPhraseDetector): A noun phrase detector that implements
                the extract() method.
        """
        self.parser = parser

    def extract(self,
                document: str,
                preprocess: bool = True,
                single_word_proper_nouns: bool = True) -> list:
        """Extract a list of entity candidates from the pass document.

        Args:
            document (str): Text document from noun phrases are extracted.
            preprocess (bool, optional): Tokenize and assign POS tags. Defaults 
                to True.
            single_word_proper_nouns (bool, optional): If True, assign False to 
                single noun phrase tokens that are not proper nouns. Defaults to 
                True.

        Returns:
            list: Tuples of ('text phrase', True/False) from the document.
        """
        tagged_phrases = self.parser.extract(document, preprocess,
                                             single_word_proper_nouns)

        return tagged_phrases

    def fit(self):
        raise NotImplementedError

    def score_phrases(self):
        raise NotImplementedError


class TFIDFScorer(EntityScorer):
    """A class that computes the TF-IDF matrix for a text corpus and scores 
    assigns a TF-IDF score to phrases.

    This class inherits from EntityScorer, which implements the candidates() 
    method.
    """
    def __init__(self, parser: NounPhraseDetector) -> None:
        """Initialize the class with a NounPhraseDetector that can extract 
        phrases.

        Args:
            parser (NounPhraseDetector): A noun phrase detector that implements
                the extract() method.
        """
        super().__init__(parser)

    def fit(self, documents: list) -> None:
        """Fit a TF-IDF model using the documents provided.

        Args:
            documents (list): List of text documents.
        """
        # TODO: determine the best tokenizer/casing to use
        # fit global idf scores
        # not using any normalization here because single tokens will be
        # assigned a score of 1 (highest score possible under l2 norm)
        self.vectorizer = TfidfVectorizer(norm=None)
        self.vectorizer.fit(documents)

    def score_phrases(self,
                      phrases: list,
                      noun_phrase_flags: bool = True) -> list:
        """Assign a TF-IDF score to all the phrases passed.

        Args:
            phrases (list): List of phrases to assign a score to.
            noun_phrase_flags (bool, optional): If True, phrases should be a 
                list of tuples of (phrase, True/False) indicating if the phrase 
                is a noun phrase or not. Defaults to True.

        Returns:
            list: Tuples of (phrase, flags, scores).
        """
        if noun_phrase_flags:
            phrases, flags = zip(*phrases)

        tfidf_vectors = self.vectorizer.transform(phrases)

        # average non-zero entries
        # row sums
        sums = np.squeeze(np.asarray(tfidf_vectors.sum(axis=1)))

        # row-wise counts of non-zero entries (from CSR matrix)
        # non_zero_counts = np.diff(tfidf_vectors.indptr)

        # TODO: find a better way to compute this, accounting for the vocab that
        #  vectorizer uses
        token_counts = self.vectorizer._count_vocab(
            raw_documents=phrases, fixed_vocab=False)[1].toarray()
        token_counts = np.squeeze(np.asarray(token_counts.sum(axis=1)))

        scores = sums / token_counts

        if noun_phrase_flags:
            return list(zip(phrases, flags, scores))
        return list(zip(phrases, scores))


class TextRankScorer(EntityScorer):
    """A class that computes the TextRank score for tokens found in a passed
    document and assigns a TextRank score to phrases.

    This class inherits from EntityScorer, which implements the candidates() 
    method.

    This class is still a work-in-progress. Many thanks to pytextrank for their
    reference implementation. https://github.com/DerwenAI/pytextrank/blob/master/pytextrank/base.py
    """
    def __init__(self,
                 document,
                 preprocess,
                 parser,
                 included_pos=['JJ', 'NN', 'VB'],
                 window_size=2) -> None:
        """Initialize the class on a text document."""
        super().__init__(parser)
        if preprocess:
            self.document = utils.preprocess(document)
        else:
            self.document = document
        self.included_pos = included_pos
        self.window_size = window_size

    def _keep_token(self, token_pos):
        """Determine if the token will be kept """
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

    def _normalized_score(self, phrase, score):
        not_included = len(
            [token for token, pos in phrase if self._keep_token((token, pos))])
        not_included_discount = len(phrase) / (len(phrase) +
                                               (2.0 * not_included) + 1.0)

        # use a root mean square (RMS) to normalize the contributions
        # of all the tokens
        interim_score = math.sqrt(score / (len(phrase) + not_included))

        return interim_score * not_included_discount

    def score_phrases(self,
                      phrases,
                      noun_phrase_flags=True,
                      preprocess=True,
                      normalize_score=True):
        if noun_phrase_flags:
            phrases, flags = zip(*phrases)

        # TODO: address the tokenization isusue
        phrases = [phrase.split() for phrase in phrases]
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
                if normalize_score:
                    score = self._normalized_score(phrase, scores[idx])
                else:
                    score = scores[idx]
                noun_phrase_scores.append(score)

        final_ranks = []
        for idx, phrase in enumerate(phrases):
            # prepare complete phrase
            phrase = ' '.join([token for token, pos in phrase])
            if flags[idx]:
                percentile_rank = stats.percentileofscore(
                    noun_phrase_scores, scores[idx])
                final_ranks.append((phrase, flags[idx], percentile_rank / 100))
            else:
                # assign a score of 0.0 to non-noun phrases
                final_ranks.append((phrase, flags[idx], 0.0))

        return final_ranks


class ClusterEntityTypeDetector(Cluster):
    """Experimental cluster based entity type detection."""
    def __init__(self, documents):
        self.vectorizer = SentenceTransformer(
            'paraphrase-distilroberta-base-v1')
        self.documents = documents
        phrase_embeddings = self.encode(documents)
        super().__init__(phrase_embeddings)

    def encode(self, documents):
        return self.vectorizer.encode(documents)

    def sample_clusters(self):
        # can only be called after fit()
        df = pd.DataFrame(zip(self.documents, self.model.labels_),
                          columns=['Phrase', 'Label'])
        for clus in range(self.k):
            print("Cluster: {}".format(clus))
            sample_size = min(len(df[df['Label'] == clus]), 10)
            print(df[df['Label'] == clus]['Phrase'].sample(
                sample_size).values.tolist())


class CosineEntityTypeDetector(object):
    def __init__(self, entity_phrases) -> None:
        self.vectorizer = SentenceTransformer(
            'paraphrase-distilroberta-base-v1')
        self.entity_phrases = self.encode(entity_phrases)

    def encode(self, documents):
        return self.vectorizer.encode(documents, convert_to_tensor=True)

    def predict(self, documents, invert=True):
        phrase_embeddings = self.encode(documents)
        # cosine distance between phrases and entity indicator
        scores = sentence_transformers.util.pytorch_cos_sim(
            phrase_embeddings, self.entity_phrases)
        if invert:
            scores = 1 - scores
        predictions = scores.argmax(dim=1).numpy()
        return predictions


def main(args):
    df_dict = utils.load_train_data(args.data_directory)
    train_df = df_dict['train.csv']
    train_df['NER_Tag_Flag'] = train_df['NER_Tag'] != 'O'

    # gather up articles
    articles = train_df.groupby(['Article_ID'])['Token'].apply(
        lambda x: ' '.join([str(y) for y in list(x)])).values.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)