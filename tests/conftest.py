import nltk


def test_documents():
    return [
        'the little yellow dog barked at the cat.', 'another sharp dive',
        'trade figures', 'any new policy measures', 'earlier stages',
        'Panamanian dictator Manuel Noriega', 'his Mansion House speech',
        'the price cutting', '3% to 4%', 'more than 10%',
        'the fastest developing trends', '\'s skill'
    ]


def test_parse_documents():
    parsed_documents = [
        nltk.Tree('S', [
            nltk.Tree('NP', [('the', 'DT'), ('little', 'JJ'), ('yellow', 'JJ'),
                             ('dog', 'NN')]), ('barked', 'VBD'), ('at', 'IN'),
            nltk.Tree('NP', [('the', 'DT'), ('cat', 'NN')]), ('.', '.')
        ]),
        nltk.Tree('S', [
            nltk.Tree('NP', [('another', 'DT'), ('sharp', 'JJ'),
                             ('dive', 'NN')])
        ]),
        nltk.Tree('S',
                  [nltk.Tree('NP', [('trade', 'NN'), ('figures', 'NNS')])]),
        nltk.Tree('S', [
            nltk.Tree('NP', [('any', 'DT'), ('new', 'JJ'), ('policy', 'NN'),
                             ('measures', 'NNS')])
        ]),
        nltk.Tree('S',
                  [nltk.Tree('NP', [('earlier', 'RBR'), ('stages', 'NNS')])]),
        nltk.Tree('S', [
            nltk.Tree('NP', [('Panamanian', 'JJ'), ('dictator', 'NN'),
                             ('Manuel', 'NNP'), ('Noriega', 'NNP')])
        ]),
        nltk.Tree('S', [
            nltk.Tree('NP', [('his', 'PRP$'), ('Mansion', 'NNP'),
                             ('House', 'NNP'), ('speech', 'NN')])
        ]),
        nltk.Tree('S', [
            nltk.Tree('NP', [('the', 'DT'), ('price', 'NN'),
                             ('cutting', 'NN')])
        ]),
        nltk.Tree('S', [
            nltk.Tree('NP', [('3', 'CD'), ('%', 'NN'), ('to', 'TO'),
                             ('4', 'CD'), ('%', 'NN')])
        ]),
        nltk.Tree('S', [
            nltk.Tree('NP', [('more', 'JJR'), ('than', 'IN'), ('10', 'CD'),
                             ('%', 'NN')])
        ]),
        nltk.Tree('S', [
            nltk.Tree('NP', [('the', 'DT'), ('fastest', 'JJS'),
                             ('developing', 'NN'), ('trends', 'NNS')])
        ]),
        nltk.Tree('S', [nltk.Tree('NP', [("'s", 'POS'), ('skill', 'NN')])])
    ]
    # kg.ner.utils.parse_document yields a list around each tree
    return [[document] for document in parsed_documents]


def test_extracts():
    noun_phrases = [[('the little yellow dog', True), ('barked at', False),
                     ('the cat', True), ('.', False)],
                    [('another sharp dive', True)], [('trade figures', True)],
                    [('any new policy measures', True)],
                    [('earlier stages', True)],
                    [('Panamanian dictator Manuel Noriega', True)],
                    [('his Mansion House speech', True)],
                    [('the price cutting', True)], [('3 % to 4 %', True)],
                    [('more than 10 %', True)],
                    [('the fastest developing trends', True)],
                    [("'s skill", True)]]
    return noun_phrases