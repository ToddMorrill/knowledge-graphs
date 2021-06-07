from collections import Counter
import multiprocessing as mp
import os

import pytest

from kg.entity_linking.extract_wikipedia import WikiFileExtractor, LinkExtractor


@pytest.fixture()
def sample_file_path():
    wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401/'
    sample_file = 'enwiki-20210401-pages-articles-multistream1.xml-p1p41242.bz2'
    sample = os.path.join(wiki_dir, sample_file)
    return sample


def test_get_page(sample_file_path):
    wiki_file_extractor = WikiFileExtractor(sample_file_path)
    page = wiki_file_extractor.get_page()
    expected_page = {
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
    assert page == expected_page


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


@pytest.fixture()
def first_page(sample_file_path):
    wiki_file_extractor = WikiFileExtractor(sample_file_path)
    page = wiki_file_extractor.get_page()
    return page


def test_get_links(first_page):
    link_extractor = LinkExtractor(first_page)
    link_dict = link_extractor.extract_links()
    expected_links = {
        'Accessible Computing': Counter({'Computer accessibility': 1})
    }
    assert link_dict == expected_links


def test_save_load_json(first_page):
    link_extractor = LinkExtractor(first_page)
    link_dict = link_extractor.extract_links()
    wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/'
    save_dir = 'links_20210401'
    save_dir = os.path.join(wiki_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    filename = link_extractor.get_json_filename()
    save_file_path = os.path.join(save_dir, filename)
    link_extractor.save_json(link_dict, save_file_path)
    loaded_dict = link_extractor.load_json(save_file_path)
    expected_load = {'Accessible Computing': {'Computer accessibility': 1}}
    assert loaded_dict == expected_load