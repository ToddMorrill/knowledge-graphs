"""
This module implements supervised methods for extracting noun phrases, entities,
and typed entities.

Implementation notes:
1) Extract noun phrases:
    - Train and evaluate noun phrase detection on CoNLL-2000 data.
    Bigram Chunker:
        IOB Accuracy:  93.3%%
        Precision:     82.3%%
        Recall:        86.8%%
        F-Measure:     84.5%%
    Maxent Chunker:
        IOB Accuracy:  96.4%%
        Precision:     89.7%%
        Recall:        92.4%%
        F-Measure:     91.0%%
2) Supervised approaches to entity extraction
    - Train deep learning model directly
    - Pretrained model
3) Supervised approaches to assigning types to entities
    - Train deep learning model directly

Examples:
    $ python -m kg.ner.supervised \
        --data-directory /Users/tmorrill002/Documents/datasets/conll/transformed
"""
import argparse
from itertools import groupby
import os
import pickle
from types import SimpleNamespace
from typing import Union

import nltk
# nltk.download('maxent_ne_chunker')  # pretrained NER model
# nltk.download('words')  # used in pretrained NER model
# nltk.download('conll2000')
from nltk.corpus import conll2000
from numpy.random import sample
import pandas as pd
from sklearn.metrics import classification_report
import spacy
from spacy.tokens import Doc
import torch
import yaml

from kg.ner.model import LSTM, get_predictions, translate_predictions
from kg.ner.preprocess import Preprocessor
import kg.ner.utils as utils


class BigramChunker(nltk.ChunkParserI):
    """A class that uses bigrams to predict noun phrases.
    """
    def __init__(self, train_sentences: list) -> None:
        """Initialize the chunker with training sentences mapped to chunk tags.

        Args:
            train_sentences (list): List of sentences in nltk.Tree format.
        """
        train_data = [[
            (pos_tag, iob_tag)
            for word, pos_tag, iob_tag in nltk.chunk.tree2conlltags(sent)
        ] for sent in train_sentences]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence: list) -> nltk.Tree:
        """Assigns a chunk tag to the tokens in the pass sentence.

        Args:
            sentence (list): Tuples of (word, part-of-speech).

        Returns:
            nltk.Tree: Tree of tokens, part-of-speech tags, and chunk tags.
        """
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag)
                     for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


class MaxEntNPChunkTagger(nltk.TaggerI):
    """Maximum Entropy classifier that identifies noun phrase chunks.
    """
    def __init__(self, train_sentences: list) -> None:
        """Train the Maximum Entropy classifier on the training sentences.

        Args:
            train_sentences (list): Tuples of ((word, pos), iob_tag).
        """
        train_set = []
        for tagged_sent in train_sentences:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = self._npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, trace=0)

    def tag(self, sentence: list) -> list:
        """Tag the sentence's tokens with IOB chunk tags.

        Args:
            sentence (list): Tuples of (word, pos).

        Returns:
            list: Tuples of ((word, pos), iob_tag).
        """
        history = []
        for i, word in enumerate(sentence):
            featureset = self._npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

    def _tags_since_dt(self, sentence: list, i: int) -> str:
        """Utility function to compute the 'tags since last determiner'.

        Args:
            sentence (list): Tuples of (word, pos).
            i (int): Sentence index.

        Returns:
            str: '+' delimited string of POS tags.
        """
        tags = set()
        for word, pos in sentence[:i]:
            if pos == 'DT':
                tags = set()
            else:
                tags.add(pos)
        return '+'.join(sorted(tags))

    def _npchunk_features(self, sentence: list, i: int, history: list) -> dict:
        """Create features for the classifier.

        Args:
            sentence (list): Tuples of (word, pos).
            i (int): Sentence index.
            history (list): History of IOB tags.

        Returns:
            dict: Set of features.
        """
        word, pos = sentence[i]
        if i == 0:
            prevword, prevpos = '<START>', '<START>'
        else:
            prevword, prevpos = sentence[i - 1]
        if i == len(sentence) - 1:
            nextword, nextpos = '<END>', '<END>'
        else:
            nextword, nextpos = sentence[i + 1]
        return {
            'pos': pos,
            'word': word,
            'prevpos': prevpos,
            'nextpos': nextpos,
            'prevword': prevword,
            'nextword': nextword,
            'prevpos+pos': f'{prevpos}+{pos}',
            'pos+nextpos': f'{pos}+{nextpos}',
            'tags-since-dt': self._tags_since_dt(sentence, i)
        }


class MaxEntChunker(nltk.ChunkParserI):
    """Wrapper around the MaxEntNPChunkTagger that conforms to the 
    nltk.ChunkParserI interface.
    """
    def __init__(self, train_sentences: list) -> None:
        """Train the Maximum Entropy Classifier on the train sentences.

        Args:
            train_sentences (list): nltk.Tree objects.
        """
        tagged_sentences = [[
            ((word, pos), iob_tag)
            for (word, pos, iob_tag) in nltk.chunk.tree2conlltags(sentence)
        ] for sentence in train_sentences]
        self.tagger = MaxEntNPChunkTagger(tagged_sentences)

    def parse(self, sentence: list) -> nltk.Tree:
        """Tag sentence's tokens with IOB chunk tags.

        Args:
            sentence (list): Tuples of (word, pos).

        Returns:
            nltk.Tree: Tree tagged with POS and IOB tags.
        """
        tagged_sentences = self.tagger.tag(sentence)
        conlltags = [(word, pos, iob_tag)
                     for ((word, pos), iob_tag) in tagged_sentences]
        return nltk.chunk.conlltags2tree(conlltags)


class PretrainedEntityDetector(object):
    """Pretrained NLTK named entity recognition model"""
    def __init__(self, binary: bool = True) -> None:
        """Initialize the detector.

        Args:
            binary (bool, optional): If True, make binary entity predictions, 
            otherwise make multiclass predictions. Defaults to True.
        """
        self.binary = binary
        self.categories = [
            "LOCATION", "ORGANIZATION", "PERSON", "DURATION", "DATE",
            "CARDINAL", "PERCENT", "MONEY", "MEASURE", "FACILITY", "GPE"
        ]
        self.categories += ["LOC", "PER", "ORG"]

    def extract(self, document: str, preprocess: bool = True) -> list:
        """Extract entities from the passed document.

        Args:
            document (str): String document.
            preprocess (bool, optional): Optionally split sentences, tokenize, and tag parts-of-speech. Defaults to True.

        Returns:
            list: Tuples of (phrase, is_entity).
        """
        # optionally preprocess (tokenize sentences/words, tag POS)
        if preprocess:
            document = utils.preprocess(document)

        candidates = []
        for sentence in document:
            tree = nltk.ne_chunk(sentence, binary=self.binary)
            extractions = self._extract(tree)
            candidates += extractions
        return candidates

    def _extract(self, tree: nltk.Tree) -> list:
        """Utility function to extract entities from a nltk.Tree.

        Args:
            tree (nltk.Tree): Tree containing tokens and entity tags.

        Returns:
            list: Tuples of (phrase, is_entity).
        """
        extractions = []
        i = 0
        while i < len(tree):
            if isinstance(tree[i], nltk.Tree) and (
                (tree[i].label() == 'NE') or
                (tree[i].label() in self.categories)):
                phrase = tree[i].leaves()
                phrase_text = ' '.join([token for token, pos in phrase])
                if self.binary:
                    extractions.append((phrase_text, True))
                else:
                    extractions.append((phrase_text, tree[i].label()))
                i += 1
            else:
                phrase = []
                while not isinstance(tree[i], nltk.Tree):
                    phrase.append(tree[i])
                    i += 1
                    if i == len(tree):
                        break
                phrase_text = ' '.join([token for token, pos in phrase])
                extractions.append((phrase_text, False))
        return extractions


class SpacyEntityTypeDetector(object):
    def __init__(self,
                 entity_type_mapping=None,
                 model='en_core_web_lg') -> None:
        self.entity_type_mapping = entity_type_mapping
        self.nlp = spacy.load(model)

    @staticmethod
    def _get_ent_spans(document):
        # TODO: access spaCy information to perfectly reconstruct the sentence
        spans = []
        subspan = []
        for tok in document:
            if tok.ent_iob_ == 'B':
                # dump existing span
                if subspan:
                    spans.append(subspan)
                    subspan = []
                subspan.append((tok, tok.ent_type_))
            elif tok.ent_iob_ == 'I':
                subspan.append((tok, tok.ent_type_))
            else:
                if subspan:
                    spans.append(subspan)
                spans.append([(tok, False)])
                subspan = []
        return spans

    @staticmethod
    def _cleanup_spans(spans):
        final_spans = []
        for span in spans:
            tokens, entity_types = zip(*span)
            joined_tokens = ' '.join([x.text for x in tokens])
            final_spans.append((joined_tokens, entity_types[0]))
        return final_spans

    def extract(self, document: str):
        output = self.nlp(document)
        output = self._cleanup_spans(self._get_ent_spans(output))
        return output

    @staticmethod
    def prepare_for_spacy(df):
        sentences = df.groupby([
            'Article_ID', 'Sentence_ID'
        ])['Token'].apply(lambda x: [str(y) for y in list(x)])
        tokenized_articles = sentences.reset_index().groupby(
            ['Article_ID'])['Token'].apply(list).values.tolist()
        return tokenized_articles

    def tag_docs(self, tokenized_documents):
        # accepts list of documents, where each document is a list of tokenized sentences
        tagged_tokens = []
        for article in tokenized_documents:
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

            doc = Doc(self.nlp.vocab, words=words, sent_starts=sent_starts)
            temp_tagged_tokens = []
            # TODO: is the the best way to do this?
            for token in self.nlp.get_pipe('ner')(doc):
                ent_type = token.ent_type_ if token.ent_type_ else 'O'
                temp_tagged_tokens.append((token.text, ent_type))

            tagged_tokens.extend(temp_tagged_tokens)
        prediction_df = pd.DataFrame(
            tagged_tokens, columns=['Predicted_Phrase', 'Predicted_NER_Tag'])

        return prediction_df

    def _normalize_type(self, df):
        df['Predicted_NER_Tag_Original'] = df['Predicted_NER_Tag']
        # overwrite with mapping
        df['Predicted_NER_Tag'] = df['Predicted_NER_Tag'].apply(
            lambda x: self.entity_type_mapping[x])
        return df

    def fit(self, train_df):
        tokenized_docs = self.prepare_for_spacy(train_df)
        prediction_df = self.tag_docs(tokenized_docs)
        eval_df = utils.merge_dfs(train_df, prediction_df)
        eval_df['Predicted_NER_Tag'].value_counts()

        # https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf (page 21)
        # let the data drive the mapping from predicted NER to CoNLL-2003 tags
        grouped_data = eval_df.groupby(
            ['Predicted_NER_Tag',
             'NER_Tag_Normalized']).agg(Count=('NER_Tag_Normalized', 'count'))
        mapping_df = grouped_data.loc[grouped_data.groupby(
            ['Predicted_NER_Tag'])['Count'].idxmax()].reset_index()
        self.entity_type_mapping = dict(
            zip(mapping_df['Predicted_NER_Tag'],
                mapping_df['NER_Tag_Normalized']))

        eval_df = self._normalize_type(eval_df)

        return eval_df

    def predict(self, df, ground_truth_df=None):
        tokenized_docs = self.prepare_for_spacy(df)
        prediction_df = self.tag_docs(tokenized_docs)
        if ground_truth_df is not None:
            prediction_df = utils.merge_dfs(df, prediction_df)
        prediction_df = self._normalize_type(prediction_df)
        return prediction_df

    def evaluate(self, eval_df):
        print(
            classification_report(eval_df['NER_Tag_Normalized'],
                                  eval_df['Entity_Type']))


class PyTorchTypeDetector(object):
    def __init__(self, config_file_path) -> None:
        # load config file
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = SimpleNamespace(**config)

        # load preprocessor
        preprocessor_file_path = os.path.join(self.config.run_dir,
                                              'preprocessor.pickle')
        with open(preprocessor_file_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        # load model
        model_file_path = os.path.join(self.config.run_dir, 'model.pt')
        state = torch.load(model_file_path)
        self.model = LSTM(self.config)
        self.model.load_state_dict(state['model'])

    @staticmethod
    def _consolidate_output(sentence):
        spans = []
        for key, group in groupby(sentence, lambda x: x[1]):
            # join strings together
            tokens = [x[0] for x in list(group)]
            joined_string = ' '.join(tokens)
            # TODO: clean this up.
            # this was implemented so that this output will work nicely with kg.ner.unsupervised.prepare_entity_html
            if key == 'O':
                spans.append((joined_string, False))
            else:
                spans.append((joined_string, key))
        return spans

    def extract(self, document: Union[str, list]):
        if isinstance(document, str):
            document = [document]

        prepared_sentences = self.preprocessor.preprocess(document)
        output = self.model(prepared_sentences)
        sample_predictions = get_predictions(output,
                                             lengths=prepared_sentences[1],
                                             concatenate=False)
        preds = translate_predictions(sample_predictions,
                                      self.preprocessor.idx_to_label)

        tokenized_sentences = []
        for doc in document:
            tokenized_sentences.append(self.preprocessor._tokenize(doc))

        final_output = []
        for idx, pred in enumerate(preds):
            tokens_tags = list(zip(tokenized_sentences[idx], pred))
            # TODO: the model should really indicate IOB tags
            # consolidate sequences of the output
            grouped_tokens_tags = self._consolidate_output(tokens_tags)
            final_output.extend(grouped_tokens_tags)

        return final_output


def main(args):
    train_sentences = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    test_sentences = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

    # print('Bigram Chunker:')
    # bigram_chunker = BigramChunker(train_sentences)
    # print(bigram_chunker.evaluate(test_sentences))

    # print('Maxent Chunker:')
    # chunker = ConsecutiveNPChunker(train_sentences)
    # print(chunker.evaluate(test_sentences))

    df_dict = utils.load_train_data(args.data_directory)
    train_df = df_dict['train.csv']
    # gather up articles
    articles = train_df.groupby(['Article_ID'], )['Token'].apply(
        lambda x: ' '.join([str(y) for y in list(x)])).values.tolist()

    pretrained_ne_detector = PretrainedEntityDetector(binary=True)
    candidates = []
    for article in articles:
        candidates += pretrained_ne_detector.extract(article)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)