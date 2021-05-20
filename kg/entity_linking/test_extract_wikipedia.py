import os

import pytest

from kg.entity_linking.extract_wikipedia import WikiFileExtractor


@pytest.fixture()
def sample_file_path():
    wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401/'
    sample_file = 'enwiki-20210401-pages-articles-multistream1.xml-p1p41242.bz2'
    sample = os.path.join(wiki_dir, sample_file)
    return sample


def test_wikifileextractor_get_context(sample_file_path):
    extractor = WikiFileExtractor(sample_file_path)
    context, root = extractor._get_context()
    expected = {
        '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation':
        'http://www.mediawiki.org/xml/export-0.10/ http://www.mediawiki.org/xml/export-0.10.xsd',
        'version': '0.10',
        '{http://www.w3.org/XML/1998/namespace}lang': 'en'
    }
    assert root.attrib == expected


def test_wikifileextractor_get_page(sample_file_path):
    extractor = WikiFileExtractor(sample_file_path)
    page = extractor.get_page()
    expected = {
        'title':
        'AccessibleComputing',
        'ns':
        0,
        'id':
        20842734,
        'redirect':
        'Computer accessibility',
        'text':
        '#REDIRECT [[Computer accessibility]]\n\n{{rcat shell|\n{{R from move}}\n{{R from CamelCase}}\n{{R unprintworthy}}\n}}'
    }
    assert page == expected
