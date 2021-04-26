"""This module processes files from the Wikipedia dump in parallel and counts of anchor texts to entities from Wikipedia links.

Examples:
    $ python extract_wiki_file.py
"""
import bz2
from collections import defaultdict, Counter
import json
import multiprocessing as mp
import os
import time
import xml.etree.ElementTree as etree

import mwparserfromhell

from kg.entity_linking import utils


class WikiFileExtractor(object):
    """Overview of the parsing process:
        1) etree.iterparse incrementally builds up the XML tree one line at a time
        2) the root node stores children nodes underneath it, which correspond to wikipedia pages
        3) the 'start' event corresponds to an opening tag (e.g. <page>) and the 'end' event corresponds to a closing tag (e.g. </page>)
        4) the elem.text method gathers all text between the start and end of the element, which is built up incrementally as more lines are parsed
        5) the root is cleared once a page is parsed so that there is only one child node under the root at a time (keeps memory footprint low)
    """
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.context, self.root = self._get_context(file_path)

    def _get_context(self, file_path):
        # open the compressed file (close in get_page)
        self.fp = bz2.BZ2File(file_path, 'r')

        # handles first element in the iteration, which is later cleared to
        # avoid persisting everything in memory
        # get an iterable
        context = etree.iterparse(self.fp, events=("start", "end"))

        # turn it into an iterator
        context = iter(context)

        # get the root element
        event, root = next(context)
        return context, root

    def get_page(self):
        """Extract the page details from the XML tree."""
        extracted_content = {}
        for event, elem in self.context:
            # clean tags: '{http://www.mediawiki.org/xml/export-0.10/}mediawiki' -> 'mediawiki'
            tag = utils.strip_tag_name(elem.tag)

            # capture information when the close tag is read (e.g. <\page>)
            if event == 'end':
                # get text content that has been accumulated
                if tag == 'title':
                    extracted_content['title'] = elem.text
                elif tag == 'text':
                    extracted_content['text'] = elem.text
                elif tag == 'id':
                    extracted_content['id'] = int(elem.text)
                elif tag == 'redirect':
                    extracted_content['redirect'] = elem.attrib['title']
                elif tag == 'ns':
                    extracted_content['ns'] = int(elem.text)
                elif tag == 'page':
                    # keep memory footprint low, clear all the children nodes
                    self.root.clear()
                    # read one complete page
                    return extracted_content

        # if nothing left to iterate on, close file, return None
        self.fp.close()
        return None


class LinkExtractor(object):
    def __init__(self, page: dict) -> None:
        self.page = page

    def _get_entity_anchor_pairs(self):
        """Get pairs of entities and anchors from a page.
        Issues to solve for:
            1. Extracted text doesn't start and end with 2 open/close brackets
            2. More than 1 pipe
        """
        # create the wiki article
        # grateful for this library!!
        wiki_page = mwparserfromhell.parse(self.page['text'])

        # find the wikilinks
        wikilinks = [x for x in wiki_page.filter_wikilinks()]

        entity_anchor_pairs = []
        issues = []
        for link in wikilinks:
            # links look like this: '[[Political movement|movement]]'
            # the first part is the entity (link to another wiki page)
            # the second is the anchor text (i.e. surface form)

            if link[:2] != '[[' or link[-2:] != ']]':
                # issue if doens't start with '[[' and end with ']]'
                issues.append(link)
                continue

            trimmed_link = link[2:-2]
            link_parts = trimmed_link.split('|')
            if len(link_parts) > 2:
                # issue if more than 1 pipe
                issues.append(link)
                continue

            if len(link_parts) == 1:
                # possible that the link is related to a category e.g. 'Category:Anti-capitalism'
                if link_parts[0].startswith('Category:'):
                    # TODO: determine if we want to handle categories differently
                    # strip category tag from the surface form
                    clean_category = link_parts[0].split('Category:')[-1]
                    # e.g. entity, anchor = 'Category:Anti-capitalism', 'Anti-capitalism'
                    entity, anchor = link_parts[0], clean_category
                else:
                    # i.e. anchor text is the same as the entity name
                    entity, anchor = link_parts[0], link_parts[0]
            elif len(link_parts) == 2:
                # expected format
                entity, anchor = link_parts[0], link_parts[1]

            entity_anchor_pairs.append((entity, anchor))

        return entity_anchor_pairs

    def extract_links(self):
        # anchor_text -> {entity: count, another_entity: count}
        anchor_to_entities = defaultdict(Counter)

        # add redirects to dictionary (probably needs some string cleanup, mostly camel case)
        if 'redirect' in self.page:
            # e.g. page['title'] = 'AfghanistanHistory'
            # page['redirect'] = 'History of Afghanistan'
            anchor_text = self.page['title']
            entity = self.page['redirect']
            anchor_to_entities[anchor_text][entity] += 1

            # no other links on page, simply return
            return anchor_to_entities

        # add the title to the dictionary
        # useful if no other page links to this page
        title = self.page['title']
        anchor_to_entities[title][title] += 1

        # get links and update anchor, entity occurrence counts
        # result will look like:
        # 'Carl Levy': Counter({'Carl Levy (political scientist)': 1})
        # 'capitalism': Counter({'Anarchism and capitalism': 2, 'capitalism': 1})
        entity_anchor_pairs = self._get_entity_anchor_pairs()
        for entity, anchor in entity_anchor_pairs:
            anchor_to_entities[anchor][entity] += 1

        return anchor_to_entities

    @staticmethod
    def save_json(dictionary, file_path):
        with open(file_path, 'w') as f:
            json.dump(dictionary, f)

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as f:
            dictionary = json.load(f)
        return dictionary

    def get_json_filename(self):
        # use title for file name
        # replace spaces with underscores
        title_underscores = self.page['title'].replace(' ', '_')
        # replace slashes with double_underscores
        title_underscores = title_underscores.replace('/', '__')
        return f'{title_underscores}.json'

    @staticmethod
    def combine_dicts(dict_one, dict_two):
        for key in dict_two:
            if key in dict_one:
                dict_one[key].update(dict_two[key])
            else:
                dict_one[key] = dict_two[key]
        return dict_one


def page_extractor(infile_queue, page_queue):
    while True:
        file = infile_queue.get()
        if file is None:
            # notify the consumers that one of the producers exited
            page_queue.put(None)
            print('Page extractor exiting.')
            return None

        # process the file, extract all pages
        extractor = WikiFileExtractor(file)
        while True:
            page = extractor.get_page()
            # if file exhausted, no pages left to process, move on to next file
            if page is None:
                break
            # o/w put the page onto the queue for further processing
            page_queue.put(page)


def link_extractor(page_queue, output_queue, page_extractors_exited,
                   page_extractor_count):
    while True:
        with page_extractors_exited.get_lock():
            # if no more input expected and the queue is empty, exit process
            if page_extractors_exited.value == page_extractor_count:
                if page_queue.empty():
                    # notify downstream consumers that one of the consumers exited
                    output_queue.put(None)
                    print('Link extractor exiting')
                    return None
            else:
                page = page_queue.get()
                if page is None:
                    # a producer process exited
                    page_extractors_exited.value += 1
                    continue

        # do work, NB: must be outside with context to get true parallelism
        extractor = LinkExtractor(page)
        links_dict = extractor.extract_links()
        filename = extractor.get_json_filename()
        output_queue.put((filename, links_dict))


def saver(output_queue, outfile_queue, link_extractors_exited,
          link_extractor_count, save_dir):
    while True:
        with link_extractors_exited.get_lock():
            # if no more input expected and the queue is empty, exit process
            if link_extractors_exited.value == link_extractor_count:
                if output_queue.empty():
                    # notify downstream consumers that one of the workers exited
                    outfile_queue.put(None)
                    print('Saver exiting.')
                    return None
            else:
                item = output_queue.get()
                if item is None:
                    # a producer process exited
                    link_extractors_exited.value += 1
                    continue

        # do work, NB: must be outside with context to get true parallelism
        filename, links_dict = item
        save_file_path = os.path.join(save_dir, filename)

        # TODO: consider reducing all these dictionaries to a single dict before writing to disk
        # TODO: handle these exceptions better (related to file names not being UTF-8)
        # these errors were taking down the saver processes and deadlocking the system
        try:
            LinkExtractor.save_json(links_dict, save_file_path)
        except Exception as e:
            print(f'Error saving {save_file_path}.')
        outfile_queue.put(save_file_path)


def list_files(wiki_dir):
    files = os.listdir(wiki_dir)
    files = [os.path.join(wiki_dir, file) for file in files]

    # filter out index files
    files = [file for file in files if 'index' not in file]

    return files


def get_queue(initial_items=[], sentinel_items=[], maxsize=0):
    """maxsize=0 means there is no limit on the size."""
    queue = mp.Queue(maxsize=maxsize)
    for item in initial_items:
        queue.put(item)
    for item in sentinel_items:
        queue.put(item)
    return queue


def process_one_file(file_path):
    """This function takes ~0:28:28 to complete and ~5Gb of memory, whereas the
    parallel version takes ~0:07:32."""
    start = time.time()
    wiki_file_extractor = WikiFileExtractor(file_path)
    pages = []
    while True:
        page = wiki_file_extractor.get_page()
        # if file exhausted, no pages left to process, move on to next file
        if page is None:
            break
        # o/w put the page onto the queue for further processing
        pages.append(page)

    link_dicts = []
    for page in pages:
        link_extractor = LinkExtractor(page)
        link_dict = link_extractor.extract_links()
        link_dicts.append(link_dict)

    for dict_ in link_dicts:
        item = dict_

    end = time.time()
    duration = end - start
    print(f'Time to process one file: {utils.hms_string(duration)}')


if __name__ == '__main__':
    start = time.time()

    PAGE_EXTRACTOR_COUNT = mp.cpu_count()
    LINK_EXTRACTOR_COUNT = mp.cpu_count()
    SAVER_COUNT = mp.cpu_count()
    PAGE_QUEUE_MAXSIZE = 0  # capped at 32768 ~5Gb worth of pages
    OUTPUT_QUEUE_MAXSIZE = 0  # capped at 32768

    wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401/'

    # create output folder for all the links
    save_dir = os.path.join(wiki_dir, 'links_20210401')
    os.makedirs(save_dir, exist_ok=True)

    # get wikidump files
    files = list_files(wiki_dir)

    # add sentinel values to the queue
    # when the producer sees the sentinel value, it should exit
    page_extractor_sentinels = [None] * PAGE_EXTRACTOR_COUNT
    # test pipeline with first file
    infile_queue = get_queue(initial_items=files[:1],
                             sentinel_items=page_extractor_sentinels)

    # keep memory footprint low by limiting queue size
    page_queue = get_queue(maxsize=PAGE_QUEUE_MAXSIZE)

    # get output_queue
    output_queue = get_queue(maxsize=OUTPUT_QUEUE_MAXSIZE)

    # get outfile_queue
    outfile_queue = get_queue()

    # keep track of how many workers are running
    page_extractors_exited = mp.Value('i', 0)
    link_extractors_exited = mp.Value('i', 0)

    # start page_extractor processes
    # TODO: abstract to function
    page_extractor_processes = []
    for i in range(PAGE_EXTRACTOR_COUNT):
        p = mp.Process(target=page_extractor, args=(infile_queue, page_queue))
        p.start()
        p.name = f'page-extractor-{i}'
        page_extractor_processes.append(p)

    # start link_extractor processes
    link_extractor_processes = []
    for i in range(LINK_EXTRACTOR_COUNT):
        p = mp.Process(target=link_extractor,
                       args=(
                           page_queue,
                           output_queue,
                           page_extractors_exited,
                           PAGE_EXTRACTOR_COUNT,
                       ))
        p.start()
        p.name = f'link-extractor-{i}'
        link_extractor_processes.append(p)

    # # combine everything into one dictionary
    # master_dict = defaultdict(Counter)
    # while True:
    #     if link_extractors_exited.value == LINK_EXTRACTOR_COUNT:
    #         if output_queue.empty():
    #             break
    #     item = output_queue.get()
    #     if item is None:
    #         link_extractors_exited.value += 1
    #     else:
    #         # combine dictionaries
    #         filename, links_dict = item
    #         master_dict = LinkExtractor.combine_dicts(master_dict, links_dict)
    
    # out_file_path = os.path.join(save_dir, 'dictionary_file_1.json')
    # LinkExtractor.save_json(master_dict, out_file_path)
    
    # start saver processes
    saver_processes = []
    for i in range(SAVER_COUNT):
        p = mp.Process(target=saver,
                       args=(output_queue, outfile_queue,
                             link_extractors_exited, LINK_EXTRACTOR_COUNT,
                             save_dir))
        p.start()
        p.name = f'saver-{i}'
        saver_processes.append(p)

    savers_exited = []
    while True:
        if len(savers_exited) == SAVER_COUNT:
            if outfile_queue.empty():
                break
        item = outfile_queue.get()
        if item is None:
            savers_exited.append(item)

    assert outfile_queue.empty()
    assert len(savers_exited) == SAVER_COUNT

    for p in page_extractor_processes:
        p.join()

    for p in link_extractor_processes:
        p.join()
    
    for p in saver_processes:
        p.join()

    infile_queue.close()
    page_queue.close()
    output_queue.close()
    outfile_queue.close()

    end = time.time()
    duration = end - start
    print(f'Total run time: {utils.hms_string(duration)}')