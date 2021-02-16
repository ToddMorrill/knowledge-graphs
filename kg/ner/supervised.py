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
    - Train Bigram/MaxEnt classifier directly?
3) Supervised approaches to assigning types to entities
    - Predict types of existing entities?
    - Predict entities and types in a single model?
    - Score predictions against ground truth CoNLL-2003 typed entities.

Examples:
    $ python unsupervised.py \
        --data-directory /Users/tmorrill002/Documents/datasets/conll/transformed
"""
import argparse

import nltk
nltk.download('maxent_ne_chunker') # pretrained NER model
nltk.download('words') # used in pretrained NER model

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
        self.classifier = nltk.MaxentClassifier.train(
            train_set,
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
        "tags-since-dt": tags_since_dt(sentence, i)
    }


def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


class PretrainedNEDetection(nltk.ne_chunk):
    pass

def main(args):
    # df_dict = utils.load_train_data(args.data_directory)
    # NP: {<DT|PP\$>?<JJ.*>*<RBR>*<NN.*>+}
    # NP: {<DT|PRP\$>?<JJ>*<NN.*>+}
    # noun_phrase_patterns = [
    #     '(<DT|PRP\$|RBR>?<JJ.*>*<NN.*>+)', '(<CD><NN><TO><CD><NN>)',
    #     '(<JJR>?<IN>?<CD>?<NN>+)', '(<POS><NN>)'
    # ]
    # grammar = f'NP:  {{{"|".join(noun_phrase_patterns)}}}'
    grammar = r"NP: {<[CDJNP].*>+}"
    # chunk_parser = nltk.RegexpParser(grammar=grammar)
    chunk_parser = UnsupervisedKeyphrase(grammar)

    test = True
    if test:
        tests(chunk_parser)

    breakpoint()

    print('Custom RegExpParser:')
    test_sentences = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    breakpoint()
    print(chunk_parser.evaluate(test_sentences))

    print('Bigram Chunker:')
    train_sentences = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    bigram_chunker = BigramChunker(train_sentences)
    print(bigram_chunker.evaluate(test_sentences))

    # print('Maxent Chunker:')
    # chunker = ConsecutiveNPChunker(train_sentences)
    # print(chunker.evaluate(test_sentences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)