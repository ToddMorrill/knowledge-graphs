"""This module implements a demo of the NER and entity linking systems."""
import argparse

from kg.ner.supervised import SpacyEntityTypeDetector
from kg.entity_linking.entity_db import query_db


def prepare_wiki_links(original_entities, entity_types, wiki_entities):
    links = []
    for i in range(len(original_entities)):
        if wiki_entities[i] is None:
            continue
        prepared_entity = wiki_entities[i].replace(' ', '_')
        link = f'https://en.wikipedia.org/wiki/{prepared_entity}'
        links.append((original_entities[i], entity_types[i], link))
    return links


def prepare_html(original_text, links, include_entity_types=True):
    revised_text = original_text
    for entity_text, entity_type, link in links:
        html = f'<a href="{link}">{entity_text}</a>'
        # TODO: need a better solution. The entity tags may be inconsistent
        # e.g. New York may have multiple entity types in the same sentence
        if include_entity_types:
            entity_type_text = f'{entity_text} ({entity_type})'
            revised_text = revised_text.replace(entity_text, entity_type_text)
        revised_text = revised_text.replace(entity_text, html)
    return revised_text


def main(args):
    type_detector = SpacyEntityTypeDetector()
    sample_sentence = "A nor’easter grounded flights departing from New York City this weekend. LaGuardia, JFK, and Newark were all impacted though transportation officials expect normal operations to resume by early Monday morning."
    results = type_detector.nlp(sample_sentence)
    db_file_path = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401_sqlite/db.sqlite'
    entity_text = [x.text for x in results.ents]
    entity_types = [x.label_ for x in results.ents]
    # db = SqliteDict(db_file_path)
    wiki_entities = query_db(db_file_path, entity_text)
    # convert wiki_entities to html links
    links = prepare_wiki_links(entity_text, entity_types, wiki_entities)
    revised_text = prepare_html(sample_sentence, links)
    with open('demo.html', 'w') as f:
        f.write(revised_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
