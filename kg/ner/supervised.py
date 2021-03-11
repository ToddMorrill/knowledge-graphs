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
    $ python supervised.py \
        --data-directory /Users/tmorrill002/Documents/datasets/conll/transformed
"""
import argparse

import nltk
nltk.download('maxent_ne_chunker')  # pretrained NER model
nltk.download('words')  # used in pretrained NER model
nltk.download('conll2000')
from nltk.corpus import conll2000

import kg.ner.utils as utils


class BigramChunker(nltk.ChunkParserI):
    """A class that uses bigrams to predict noun phrases.
    """
    def __init__(self, train_sentences: list) -> None:
        """Initialize the chunker with training sentences mapped to chunk tags.

        Args:
            train_sentences (list): List of sentences in nltk.Tree format.
        """
        train_data = [[(pos, iob_tag) for word, pos, iob_tag in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sentences]
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


class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = self._npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = self._npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

    def _tags_since_dt(self, sentence, i):
        tags = set()
        for word, pos in sentence[:i]:
            if pos == 'DT':
                tags = set()
            else:
                tags.add(pos)
        return '+'.join(sorted(tags))

    def _npchunk_features(self, sentence, i, history):
        word, pos = sentence[i]
        if i == 0:
            prevword, prevpos = "<START>", "<START>"
        else:
            prevword, prevpos = sentence[i - 1]
        if i == len(sentence) - 1:
            nextword, nextpos = "<END>", "<END>"
        else:
            nextword, nextpos = sentence[i + 1]
        return {
            "pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevword": prevword,
            "nextword": nextword,
            "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": self._tags_since_dt(sentence, i)
        }


class MaxEntChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c)
                         for (w, t, c) in nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


class PretrainedNEDetector(object):
    def __init__(self, binary=True) -> None:
        self.binary = binary

    def extract(self, document: str, preprocess: bool = True):
        # optionally preprocess (tokenize sentences/words, tag POS)
        if preprocess:
            document = utils.preprocess(document)

        candidates = []
        for sentence in document:
            tree = nltk.ne_chunk(sentence, binary=self.binary)
            extractions = self._extract(tree)
            candidates += extractions
        return candidates

    def _extract(self, tree):
        extractions = []
        i = 0
        while i < len(tree):
            if isinstance(tree[i], nltk.Tree) and tree[i].label() == 'NE':
                phrase = tree[i].leaves()
                phrase_text = ' '.join([token for token, pos in phrase])
                extractions.append((phrase_text, True))
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

    pretrained_ne_detector = PretrainedNEDetector(binary=True)
    candidates = []
    for article in articles:
        candidates += pretrained_ne_detector.extract(article)
    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)