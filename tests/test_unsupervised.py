import pytest
import nltk

from kg.ner.utils import parse_document
from kg.ner.unsupervised import NounPhraseDetector

from conftest import test_documents, test_parse_documents, test_extracts


@pytest.fixture(scope='module')
def chunk_parser():
    return NounPhraseDetector()


@pytest.mark.parametrize("test_input,expected",
                         zip(test_documents(), test_parse_documents()))
def test_parse_document(test_input, expected, chunk_parser):
    results = parse_document(test_input, chunk_parser, print_tree=False)
    assert results == expected

@pytest.mark.parametrize("test_input,expected",
                         zip(test_documents(), test_extracts()))
def test_extract(test_input, expected, chunk_parser):
    result = chunk_parser.extract(test_input)
    assert result == expected


# docs = test_documents()
# chunk_parser = NounPhraseDetection()
# result = chunk_parser.extract(docs[0])
# breakpoint()