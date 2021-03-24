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
from nltk.corpus.reader import tagged
nltk.download('conll2000')  # noun phrase evaluation
nltk.download('stopwords')  # stopwords
from nltk.corpus import conll2000
import numpy as np
import pandas as pd
from scipy import stats
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import spacy
from spacy.tokens import Doc

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

    sentences = train_df.groupby(
        ['Article_ID',
         'Sentence_ID'])['Token'].apply(lambda x: [str(y) for y in list(x)])

    tokenized_articles = sentences.reset_index().groupby(
        ['Article_ID'])['Token'].apply(list).values.tolist()

    nlp = spacy.load('en_core_web_lg')

    tagged_tokens = []
    for article in tokenized_articles:
        # manually create spaCy Doc
        words = []
        sent_starts = []
        for sentence in article:
            for idx, word in enumerate(sentence):
                if idx == 0:
                    sent_starts.append(True)
                else:
                    sent_starts.append(False)
                words.append(word)

        doc = Doc(nlp.vocab, words=words, sent_starts=sent_starts)
        temp_tagged_tokens = []
        for token in nlp.get_pipe('ner')(doc):
            ent_type = token.ent_type_ if token.ent_type_ else 'O'
            temp_tagged_tokens.append((token.text, ent_type))

        tagged_tokens.extend(temp_tagged_tokens)

    prediction_df = pd.DataFrame(
        tagged_tokens, columns=['Predicted_Phrase', 'Predicted_NER_Tag'])

    from kg.ner.evaluate import merge_dfs

    eval_df = merge_dfs(train_df, prediction_df)
    eval_df['Predicted_NER_Tag'].value_counts()
    tag_type_mapping = {
        'O': 'O',
        'PERSON': 'PER',
        'ORG': 'ORG',
        'GPE': 'ORG',
        'DATE': 'MISC',  # ?
        'CARDINAL': 'MISC',  # ?
        'NORP': 'MISC',
        'MONEY': 'MISC',
        'QUANTITY': 'MISC',
        'TIME': 'MISC',
        'ORDINAL': 'MISC',
        'PERCENT': 'MISC',
        'EVENT': 'MISC',
        'LOC': 'MISC',
        'WORK_OF_ART': 'MISC',
        'PRODUCT': 'MISC',
        'FAC': 'MISC',
        'LAW': 'MISC',
        'LANGUAGE': 'MISC'
    }
    eval_df['Entity_Type'] = eval_df['Predicted_NER_Tag'].apply(lambda x: tag_type_mapping[x])
    print(
        classification_report(eval_df['NER_Tag_Normalized'],
                                eval_df['Entity_Type']))
    breakpoint()

    chunk_parser = ProperNounDetector()

    # keep a global phrase index, add placeholder cluster id
    # map proper noun phrase index to global phrase index
    # make all phrases title cased
    # cluster noun phrases
    # assign cluster id to global phrase sequence
    # split on spaces and compare to ground truth training set
    # for each cluster id, find the most common NER type and assign that to that cluster

    # get candidate phrases
    candidates = []
    for article in articles:
        # manually tokenize because nltk tokenizer is converting 'C$' -> ['C', '$'] and throwing off comparison
        sentences = nltk.sent_tokenize(article)
        article = [sentence.split() for sentence in sentences]
        article = utils.tag_pos(article)
        candidates.extend(chunk_parser.extract(article, preprocess=False))

    # keep a global phrase index, add placeholder cluster id
    phrase, flag = zip(*candidates)
    candidates = list(zip(phrase, flag, ['O'] * len(candidates)))

    # map proper noun phrase index to global phrase index
    # pull out proper nouns to be encoded (make title case)
    global_idx = []
    proper_noun_phrases = []
    for idx, phrase in enumerate(candidates):
        phrase, flag, cluster_id = phrase
        if flag:
            global_idx.append(idx)
            proper_noun_phrases.append(phrase.title())

    entity_phrases = [
        'not a person', 'not a location', 'not a organization',
        'not a miscellaneous entity'
    ]
    type_detector = CosineEntityTypeDetector(entity_phrases)
    type_predictions = type_detector.predict(proper_noun_phrases)

    label_entity_type_mapping = {
        0: 'PER',
        1: 'LOC',
        2: 'ORG',
        3: 'MISC',
        4: 'O'
    }
    # convert prediction class
    for idx, label in enumerate(type_predictions):
        phrase, flag, default_entity_type = candidates[global_idx[idx]]
        # set entity type
        entity_type = label_entity_type_mapping[label]
        candidates[global_idx[idx]] = (phrase, flag, entity_type)

    from kg.ner.evaluate import prepare_scored_phrases, merge_dfs
    prediction_df = prepare_scored_phrases(
        candidates,
        columns=['Predicted_Phrase', 'Predicted_Entity_Flag', 'Entity_Type'])
    train_eval_df = merge_dfs(train_df, prediction_df)

    print(
        classification_report(train_eval_df['NER_Tag_Normalized'],
                              train_eval_df['Entity_Type']))
    breakpoint()

    # type_detector = ClusterEntityTypeDetector(proper_noun_phrases)
    # # number of classes in CoNLL-2003 data
    # type_detector.fit(k=4)
    # type_detector.k
    # for idx, label in enumerate(type_detector.model.labels_):
    #     phrase, flag, cluster_id = candidates[global_idx[idx]]
    #     # set cluster_id
    #     candidates[global_idx[idx]] = (phrase, flag, label)

    # print(type_detector.sample_clusters())

    # # split on spaces and compare to ground truth training set
    # from kg.ner.evaluate import prepare_scored_phrases, merge_dfs
    # prediction_df = prepare_scored_phrases(
    #     candidates,
    #     columns=['Predicted_Phrase', 'Predicted_Entity_Flag', 'Cluster_ID'])
    # train_eval_df = merge_dfs(train_df, prediction_df)

    # # for each cluster id, find the most common NER type and assign that to that cluster
    # cluster_id_ner_tag_map = {}
    # for val in train_eval_df['Cluster_ID'].unique():
    #     subset_df = train_eval_df[train_eval_df['Cluster_ID'] == val]
    #     # get NER tag with most votes
    #     # NB: this might be skewed by the fact that some NER tags have more tokens on average
    #     cluster_id_ner_tag_map[val] = subset_df[
    #         'NER_Tag_Normalized'].value_counts().index[0]

    # # assign NER tags and evaluate
    # train_eval_df['Predicted_NER_Tag'] = train_eval_df['Cluster_ID'].apply(
    #     lambda x: cluster_id_ner_tag_map[x])

    # from sklearn.metrics import classification_report
    # print(
    #     classification_report(train_eval_df['NER_Tag_Normalized'],
    #                           train_eval_df['Predicted_NER_Tag']))

    # type_detector = EntityTypeDetector(proper_noun_phrases)
    # type_detector.find_k_fit(k_max=4, k_step=1, var_explained=20)
    # type_detector.k
    # type_detector.model.labels_
    # print(type_detector.sample_clusters())
    breakpoint()

    cluster_phrases = [phrase for phrase, flag in noun_phrases if flag]
    vectorizer = SentenceTransformer('paraphrase-distilroberta-base-v1')
    phrase_embeddings = vectorizer.encode(cluster_phrases)
    cluster = Cluster(phrase_embeddings)
    # cluster.fit(10)
    cluster.find_k_fit(k_max=80)
    cluster.plot_elbows()

    df = pd.DataFrame(zip(cluster_phrases, cluster.model.labels_),
                      columns=['Phrase', 'Label'])
    for clus in range(cluster.k):
        print("Cluster: {}".format(clus))
        sample_size = min(len(df[df['Label'] == clus]), 10)
        print(df[df['Label'] == clus]['Phrase'].sample(
            sample_size).values.tolist())
        #print("\n")
    breakpoint()
    # sample = 'Here is some sample text. And some more!'
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
    # scorer = TextRankScorer(articles[0], preprocess=True, parser=chunk_parser)
    # scorer.fit()
    # candidates = scorer.candidates(articles[0])
    # scores = scorer.score_phrases(candidates)
    # breakpoint()

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