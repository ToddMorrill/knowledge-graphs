import pytest
import nltk

from kg.ner.utils import extract_noun_chunks
from kg.ner.unsupervised import NounPhraseDetection

from conftest import test_sentences, test_sentence_solutions


@pytest.fixture(scope='module')
def chunk_parser():
    return NounPhraseDetection()


@pytest.mark.parametrize("test_input,expected",
                         zip(test_sentences(), test_sentence_solutions()))
def test_sentence(test_input, expected, chunk_parser):
    results = extract_noun_chunks(test_input, chunk_parser, print_tree=False)
    assert results == expected