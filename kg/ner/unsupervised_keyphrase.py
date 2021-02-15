"""
This module will implement some unsupervised keyphrase extraction experiments.

Implementation notes:
1) Extract noun phrases.
2) Score the phrases using:
    - TFIDF
    - TextRank
3) Empirically determine an appropriate cutoff threshold using the validation set.
4) Score overlap with entities.

TODO: in another module
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
from nltk.corpus import conll2000
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('conll2000')
# nltk.config_megam('/Users/tmorrill002/Documents/MEGAM/megam-64')

import kg.ner.utils as utils


def tokenize_text(document):
    return nltk.sent_tokenize(document)


def tokenize_sentences(sentences):
    return [nltk.word_tokenize(sentence) for sentence in sentences]


def tag_pos(sentences):
    return [nltk.pos_tag(sentence) for sentence in sentences]


def preprocess(document):
    sentences = tokenize_text(document)
    sentences = tokenize_sentences(sentences)
    sentences = tag_pos(sentences)
    return sentences


def extract_noun_chunks(document, parser, print_tree=False):
    preprocessed_sentences = preprocess(document)
    results = []
    for sentence in preprocessed_sentences:
        result = parser.parse(sentence)
        results.append(result)
        print(result)
        if print_tree:
            result.draw()
    return results


def tests(chunk_parser):
    sample_document = 'the little yellow dog barked at the cat.'
    sample_results = extract_noun_chunks(sample_document,
                                         chunk_parser,
                                         print_tree=True)

    easy_tests = [
        'another sharp dive', 'trade figures', 'any new policy measures',
        'earlier stages', 'Panamanian dictator Manuel Noriega'
    ]
    for test in easy_tests:
        results = extract_noun_chunks(test, chunk_parser, print_tree=True)

    harder_tests = [
        'his Mansion House speech', 'the price cutting', '3% to 4%',
        'more than 10%', 'the fastest developing trends', '\'s skill'
    ]
    for test in harder_tests:
        results = extract_noun_chunks(test, chunk_parser, print_tree=True)


class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
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
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set,
                                                      # algorithm='megam',
                                                      trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c)
                         for (w, t, c) in nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevword": prevword,
            "nextword": nextword,
            "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)}

def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


def main(args):
    # df_dict = utils.load_train_data(args.data_directory)
    # NP: {<DT|PP\$>?<JJ.*>*<RBR>*<NN.*>+}
    # NP: {<DT|PRP\$>?<JJ>*<NN.*>+}
    noun_phrase_patterns = [
        '(<DT|PRP\$|RBR>?<JJ.*>*<NN.*>+)', '(<CD><NN><TO><CD><NN>)',
        '(<JJR>?<IN>?<CD>?<NN>+)', '(<POS><NN>)'
    ]
    grammar = f'NP:  {{{"|".join(noun_phrase_patterns)}}}'
    chunk_parser = nltk.RegexpParser(grammar=grammar)

    test = False
    if test:
        tests(chunk_parser)

    print('Custom RegExpParser:')
    test_sentences = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    print(chunk_parser.evaluate(test_sentences))

    print('Bigram Chunker:')
    train_sentences = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    bigram_chunker = BigramChunker(train_sentences)
    print(bigram_chunker.evaluate(test_sentences))

    print('Maxent Chunker:')
    chunker = ConsecutiveNPChunker(train_sentences)
    print(chunker.evaluate(test_sentences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)