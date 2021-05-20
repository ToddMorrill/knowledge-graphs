"""This module processes files from the Wikipedia dump in parallel and counts
  mappings of anchor texts to entities derived from Wikipedia links.

Examples:
    $ python extract_wikipedia.py

TODO:
    - argparse
    - overhaul parallel processing pipeline
    - review function to process a single file (or a sample of pages) for testing
    - add more tests
"""
import argparse
import bz2
from collections import defaultdict, Counter
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from typing import Union
import uuid
import xml.etree.ElementTree as etree

import mwparserfromhell

from kg.entity_linking import utils


class WikiFileExtractor(object):
    """This class extracts the most commonly desired content from Wikipedia
      dump pages. The get_page method can be customized to suit any use case.
    
    Overview of the page extraction process:
        1) etree.iterparse incrementally builds up the XML tree one line at a
          time, acting as a simple state machine, marking 'start' and 'end'
          states for tags encountered.
        2) The root node stores child nodes underneath it, which correspond to
          Wikipedia pages.
        3) The 'start' event corresponds to an opening tag (e.g. <page>) and
          the 'end' event corresponds to a closing tag (e.g. </page>).
        4) The elem.text method gathers all text between the start and end of
          the element, which is built up incrementally as more lines are 
          parsed.
        5) The child (i.e. a page) under the root is cleared (i.e. deleted)
          once a page is parsed so that there is only one child node under the
          root at a time, which keeps the memory footprint low. This greatly
          simplifies the analysis in parallel settings.
    """
    def __init__(self, file_path: str) -> None:
        """Initializes the class with a local file path to the file to be
          extracted.

        Args:
            file_path (str): A local file path to the file to be extracted.
        """
        self.file_path = file_path
        self.context, self.root = self._get_context()

    def _get_context(self) -> tuple[iter, Union[etree.Element, None]]:
        # open the compressed file (close in get_page)
        self.fp = bz2.BZ2File(self.file_path, 'r')

        # handles first element in the iteration, which is later cleared to
        # avoid persisting everything in memory
        # get an iterable
        context = etree.iterparse(self.fp, events=("start", "end"))

        # turn it into an iterator
        context = iter(context)

        # get the root element
        try:
            event, root = next(context)
        except Exception as e:
            # was getting errors around OSError: Invalid data stream
            # probably due to .DS_Store files
            print(f'Error opening {self.file_path}')
            context, root = iter([]), None
        return context, root

    def get_page(self) -> Union[dict, None]:
        """Extracts the page's contents from the XML tree.

        Returns:
            Union[dict, None]: A dictionary containing various page attributes
              such as the title, text, etc.
        """
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
                    # keep memory footprint low, clear the child node
                    self.root.clear()
                    # read one complete page
                    return extracted_content

        # if nothing left to iterate on, close file, return None
        self.fp.close()
        return None


class LinkExtractor(object):
    """This class extracts mappings from anchor texts to Wikipedia pages
      derived from links found in the raw text of a page. This class operates
      on 1 page at a time."""
    def __init__(self, page: dict) -> None:
        """Initializes the class with a dictionary containing various page
          attributes. This dictionary must contain the following keys: title,
          text, ns, redirect (optional).

        Args:
            page (dict): A dictionary containing various page attributes
              such as the title, text, etc.
        """
        self.page = page

    def _get_entity_anchor_pairs(self) -> list[tuple[str, str]]:
        """Get pairs of entities and anchors derived from links on the page.
        
        TODO: Issues to solve for:
            1. Extracted text doesn't start and end with 2 open/close brackets
            2. More than 1 pipe
        """
        # create the wiki article
        # so grateful for this library!!
        wiki_page = mwparserfromhell.parse(self.page['text'])

        # find the wikilinks
        wikilinks = [x for x in wiki_page.filter_wikilinks()]

        entity_anchor_pairs = []
        issues = []
        for link in wikilinks:
            # links generally look like this: '[[Political movement|movement]]'
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

            # case when link looked like this: [[Computer accessibility]]
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
            # expected format e.g. [[Political movement|movement]]
            elif len(link_parts) == 2:
                entity, anchor = link_parts[0], link_parts[1]

            entity_anchor_pairs.append((entity, anchor))

        return entity_anchor_pairs

    def _handle_colons(self, title):
        # TODO: Is this the best way handle colons?
        # since we're working with the pages/articles multistream, we receive:
        # articles, templates, media/file descriptions, and primary meta-pages
        # legit use of colon for an article: Dune: The Butlerian Jihad

        # not interested in colons for for templates, categories, etc.
        # Template:Template link with parameters -> Template link with parameters
        # Category:People -> People
        # anything namespace other than 0 (i.e. main/article namespace) should
        # have colon info stripped
        # learn more about namespaces:
        # https://en.wikipedia.org/wiki/Wikipedia:What_is_an_article%3F

        if self.page['ns'] != 0:
            title = title.split(':')[-1]
        return title

    @staticmethod
    def _handle_commas(title):
        # TODO: Is this the best way to process commas?
        # Windsor, Berkshire -> Windsor
        # Diana, Princess of Wales -> Princess Diana of Wales or Diana
        # Strip text after comma for starters
        return title.split(',')[0]

    @staticmethod
    def _handle_parentheses(title):
        # TODO: Is this the best way to handle parenthetical information?
        # e.g.  Mercury (element), Mercury (mythology), Mercury (planet)
        # Mercury (element) -> Mercury
        return title.split('(')[0]

    @staticmethod
    def _handle_camel_case(title):
        # clean up camel case text to get a more meaningful surface form
        # e.g. AfghanistanHistory -> Afghanistan History
        pattern = r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))'
        return re.sub(pattern, r' \1', title)

    def extract_links(self) -> defaultdict[str, Counter[str, int]]:
        # return dict will look like this
        # {anchor_text_1: {entity: count, another_entity: count},
        #  anchor_text_2: {entity: count, another_entity: count}}
        anchor_to_entities = defaultdict(Counter)

        # add redirects to dictionary
        if 'redirect' in self.page:
            # e.g. page['title'] = 'AfghanistanHistory'
            # page['redirect'] = 'History of Afghanistan'
            anchor_text = self.page['title']
            anchor_text = self._handle_colons(anchor_text)
            anchor_text = self._handle_commas(anchor_text)
            anchor_text = self._handle_parentheses(anchor_text)
            if ' ' not in anchor_text:
                anchor_text = self._handle_camel_case(anchor_text)

            entity = self.page['redirect']
            anchor_to_entities[anchor_text][entity] += 1

            # no other links on page, simply return
            return anchor_to_entities

        # add the title to the dictionary
        # useful if no other page links to this page

        # understand titles better:
        # https://en.wikipedia.org/wiki/Wikipedia:Article_titles#Disambiguation
        title = self.page['title']
        title = self._handle_colons(title)
        title = self._handle_commas(title)
        title = self._handle_parentheses(title)
        anchor_to_entities[title][self.page['title']] += 1

        # get links from page text and update anchor, entity occurrence counts
        # result will look like:
        # {'Carl Levy': Counter({'Carl Levy (political scientist)': 1}),
        # 'capitalism': Counter({'Anarchism and capitalism': 2, 'Capitalism': 1})}
        entity_anchor_pairs = self._get_entity_anchor_pairs()
        for entity, anchor in entity_anchor_pairs:
            # TODO: do we need to do any processing of surface forms?
            anchor_to_entities[anchor][entity] += 1

        return anchor_to_entities

    @staticmethod
    def save_json(dictionary: dict, file_path: str) -> None:
        """Saves an arbitrary dictionary to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(dictionary, f)

    @staticmethod
    def load_json(file_path: str) -> dict:
        """Loads an arbitrary JSON file into a dictionary."""
        with open(file_path, 'r') as f:
            dictionary = json.load(f)
        return dictionary

    def get_json_filename(self):
        """Prepares a JSON filename for a single page."""
        # use title for file name
        # replace spaces with underscores
        title_underscores = self.page['title'].replace(' ', '_')
        # replace slashes with double_underscores
        title_underscores = title_underscores.replace('/', '__')
        return f'{title_underscores}.json'

    @staticmethod
    def combine_dicts(dict_one: dict, dict_two: dict) -> dict:
        """Merges two dictionaries together."""
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
        # TODO: extract chunks of pages to reduce process overhead
        extractor = WikiFileExtractor(file)
        counter = 0
        pages_batch = []
        while True:
            # put 1000 pages into the queue at a time to reduce inter-process communication overhead
            if counter == 1000:
                # put the pages onto the queue for further processing
                page_queue.put(pages_batch)
                pages_batch = []
                counter = 0
            page = extractor.get_page()
            # if file exhausted, no pages left to process, move on to next file
            if page is None:
                # first put the partial batch onto the queue
                if len(pages_batch) > 0:
                    page_queue.put(pages_batch)
                break
            pages_batch.append(page)
            counter += 1


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
                pages_batch = page_queue.get()
                if pages_batch is None:
                    # a producer process exited
                    page_extractors_exited.value += 1
                    continue

        # do work, NB: must be outside with context to get true parallelism
        # unpack all the pages in the batch
        links_dict_batch = []
        for page in pages_batch:
            extractor = LinkExtractor(page)
            links_dict = extractor.extract_links()
            filename = extractor.get_json_filename()
            links_dict_batch.append((filename, links_dict))

        output_queue.put(links_dict_batch)


def save_partial_dict(save_dir, temp_dict):
    file_id = uuid.uuid4().hex[:8]
    file_name = f'partition_{file_id}.json'
    save_file_path = os.path.join(save_dir, file_name)
    # TODO: handle these exceptions better (related to file names not being UTF-8)
    # these errors were taking down the saver processes and deadlocking the system
    try:
        LinkExtractor.save_json(temp_dict, save_file_path)
    except Exception as e:
        print(f'Error saving {save_file_path}.')
    return save_file_path


def saver(output_queue, outfile_queue, link_extractors_exited,
          link_extractor_count, save_dir):
    counter = 0
    temp_dict = defaultdict(Counter)
    while True:
        # save new dictionary every 100_000 pages to reduce number of small files
        # ~50Mb files
        if counter >= 100_000:
            # save existing dictionary and create a new one
            save_file_path = save_partial_dict(save_dir, temp_dict)
            outfile_queue.put(save_file_path)

            # new dict, reset counter
            temp_dict = defaultdict(Counter)
            counter = 0

        # get item from queue
        with link_extractors_exited.get_lock():
            # if no more input expected and the queue is empty, exit process
            if link_extractors_exited.value == link_extractor_count:
                if output_queue.empty():
                    # save any final dictionaries currently being processed
                    if len(temp_dict) > 0:
                        save_file_path = save_partial_dict(save_dir, temp_dict)
                        outfile_queue.put(save_file_path)

                    # notify downstream consumers that one of the workers exited
                    outfile_queue.put(None)
                    print('Saver exiting.')
                    return None
            else:
                links_dict_batch = output_queue.get()
                if links_dict_batch is None:
                    # a producer process exited
                    link_extractors_exited.value += 1
                    continue

        # do work, NB: must be outside with context to get true parallelism
        for pagename, links_dict in links_dict_batch:
            temp_dict = LinkExtractor.combine_dicts(temp_dict, links_dict)
            counter += 1


def list_files(wiki_dir):
    files = os.listdir(wiki_dir)
    # TODO: combine into one loop

    files = [os.path.join(wiki_dir, file) for file in files]

    # filter out directories
    files = [file for file in files if os.path.isfile(file)]

    # filter out index files
    files = [file for file in files if 'index' not in file]

    # remove .DS_Store files
    files = [file for file in files if '.DS_Store' not in file]

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


from itertools import chain
from collections import deque


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(
        0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


if __name__ == '__main__':

    # # test time/memory to extract 1000 pages

    # wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/'
    # file = 'enwiki-20210401-pages-articles-multistream1.xml-p1p41242.bz2'
    # file_path = os.path.join(wiki_dir, file)
    # import psutil
    # process = psutil.Process(os.getpid())
    # start_size = process.memory_info().rss  # in bytes
    # print(f'Starting process memory size: {start_size / float(2**20):,.2f}')

    # start = time.time()
    # wiki_file_extractor = WikiFileExtractor(file_path)
    # pages = []
    # for _ in range(1000):
    #     page = wiki_file_extractor.get_page()
    #     pages.append(page)
    # end = time.time()
    # extract_pages_duration = end - start
    # print(f'Time to extract 1,000 pages: {utils.hms_string(extract_pages_duration)}')

    # # pages_size = sys.getsizeof(pages)
    # # pages_size = total_size(pages)
    # plus_pages_size = process.memory_info().rss
    # pages_size = plus_pages_size - start_size
    # print(f'Size of 1,000 pages: {pages_size / float(2**20):,.2f} Mb')

    # # test time/memory to extract links from 1000 pages
    # start = time.time()
    # link_dicts = []
    # for page in pages:
    #     link_extractor = LinkExtractor(page)
    #     link_dict = link_extractor.extract_links()
    #     link_dicts.append(link_dict)
    # end = time.time()
    # extract_links_duration = end - start
    # print(f'Time to extract links from 1,000 pages: {utils.hms_string(extract_links_duration)}')

    # # link_dicts_size = sys.getsizeof(link_dicts)
    # # link_dicts_size = total_size(link_dicts)
    # plus_link_dicts_size = process.memory_info().rss
    # link_dicts_size = plus_link_dicts_size - plus_pages_size
    # print(f'Size of 1,000 pages worth of links: {link_dicts_size / float(2**20):,.2f} Mb')

    # # test time/memory to update a dictionary built from 1000 pages
    # start = time.time()
    # temp_dict = defaultdict(Counter)
    # for link_dict in link_dicts:
    #     temp_dict = LinkExtractor.combine_dicts(temp_dict, link_dict)
    # end = time.time()
    # combine_dicts_duration = end - start
    # print(f'Time to combine 1,000 dictionaries: {utils.hms_string(combine_dicts_duration)}')

    # plus_combine_dicts_size = process.memory_info().rss
    # combined_dict_size = plus_combine_dicts_size - plus_link_dicts_size
    # print(f'Size of dict for 1,000 pages worth of links: {combined_dict_size / float(2**20):,.2f} Mb')

    # # I want to keep the maximize throughput while keeping the whole application under 16Gb of memory
    # # want to find the right pages_queue_size
    # memory_budget_mb = 16000
    # one_k_pages_mb = 75
    # one_k_links_dict_mb = 69
    # one_hundred_k_links_dict_mb = 10*100
    # processes = 16
    # page_queue_maxsize = 1
    # pages_mb = one_k_pages_mb * processes
    # links_dict_mb = one_k_links_dict_mb * processes
    # combined_dict_mb = one_hundred_k_links_dict_mb * 2

    # budget_remaining_for_queue = memory_budget_mb - pages_mb - links_dict_mb - combined_dict_mb
    # page_queue_size = budget_remaining_for_queue / one_k_pages_mb
    # print(f'Approx. page queue size: {page_queue_size}')
    # # how many batches of 1,000 among 6m documents
    # batches = 6_000_000 / 1_000
    # compute_time_secs = (extract_links_duration * batches) / processes
    # print(f'Approx. time to compute the job: {utils.hms_string(compute_time_secs)}')
    # # ~155.9 for the page_queue_size
    # exit(0)

    start = time.time()

    PAGE_EXTRACTOR_COUNT = mp.cpu_count()
    LINK_EXTRACTOR_COUNT = mp.cpu_count()
    SAVER_COUNT = 2
    PAGE_QUEUE_MAXSIZE = 100
    OUTPUT_QUEUE_MAXSIZE = 0  # capped at 32768

    wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401/'

    # create output folder for all the links
    save_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/links_20210401'
    # if dir already there, need to remove (filenames are unique and files will be additive)
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # get wikidump files (test with 1 file)
    files = list_files(wiki_dir)
    print(f'Processing {len(files)} Wikipedia dump files.')

    # add sentinel values to the queue
    # when the producer sees the sentinel value, it should exit
    page_extractor_sentinels = [None] * PAGE_EXTRACTOR_COUNT
    # test pipeline with first file
    infile_queue = get_queue(initial_items=files,
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