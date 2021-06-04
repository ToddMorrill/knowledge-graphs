"""This module processes files from the Wikipedia dump in parallel and counts
  mappings of anchor texts to entities derived from Wikipedia links.

Examples:
    $ python extract_wikipedia.py \
        --wiki-dir /Users/tmorrill002/Documents/datasets/wikipedia/20210401/ \
        --save-dir /Users/tmorrill002/Documents/datasets/wikipedia/links_20210401 \
        --config extract_wikipedia.yaml
"""
import argparse
import bz2
from collections import defaultdict, Counter
import json
import multiprocessing as mp
import os
import re
import time
from typing import Union
import uuid
import xml.etree.ElementTree as etree

import mwparserfromhell
import yaml

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

    @staticmethod
    def save_partial_dict(save_dir: str, partial_dict: dict) -> str:
        """This function saves a sharded JSON file."""
        # create a unique filename for the partition
        file_id = uuid.uuid4().hex[:8]
        file_name = f'partition_{file_id}.json'
        save_file_path = os.path.join(save_dir, file_name)
        # TODO: handle these exceptions better (related to file names not being UTF-8)
        # these errors were taking down the saver processes and deadlocking the system
        try:
            LinkExtractor.save_json(partial_dict, save_file_path)
        except Exception as e:
            print(f'Error saving {save_file_path}.')
        return save_file_path


def page_extractor(infile_queue: mp.Queue, page_queue: mp.Queue,
                   config: dict[str, int]) -> None:
    """Worker function that extracts 1000 pages from a saved Wikipedia file at
      a time and moves the results to the page_queue.

    Args:
        infile_queue (mp.Queue): Queue of Wikipedia files to be processed.
        page_queue (mp.Queue): Queue of individual Wikipedia pages to be 
          further processed.
        config (dict[str, int]): Dictionary of parameters configuring the 
          extraction job.
    """
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
            # put multiple pages into the queue at a time to reduce
            # inter-process communication overhead
            if counter == config['num_pages_batch_size']:
                # put the pages onto the queue for further processing
                page_queue.put(pages_batch)
                pages_batch = []
                counter = 0
            page = extractor.get_page()
            # if file exhausted, no pages left to process, move on to next file
            if page is None:
                # put the partial batch onto the queue
                if len(pages_batch) > 0:
                    page_queue.put(pages_batch)
                break
            pages_batch.append(page)
            counter += 1


def link_extractor(page_queue: mp.Queue, output_queue: mp.Queue,
                   page_extractors_exited: mp.Value,
                   config: dict[str, int]) -> None:
    """Worker function that extracts links from the Wikipedia pages and moves
      the results to the output_queue.
    
      This worker exits when all expected paged_extractor workers have exited 
      (i.e. no more items will be put into the queue) and the page_queue is
      empty.

    Args:
        page_queue (mp.Queue): Queue of individual Wikipedia pages to be 
          further processed.
        output_queue (mp.Queue): Queue containing dictionaries that will be
          batched together and saved to sharded JSON files.
        page_extractors_exited (mp.Value): Number of page extractor workers
          exited.
        config (dict[str, int]): Dictionary of parameters configuring the 
          extraction job.
    """
    while True:
        with page_extractors_exited.get_lock():
            # if no more input expected and the queue is empty, exit process
            if page_extractors_exited.value == config[
                    'page_extractor_workers']:
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
            # TODO: remove filename from being passed
            links_dict_batch.append((filename, links_dict))

        output_queue.put(links_dict_batch)


def saver(output_queue: mp.Queue, outfile_queue: mp.Queue,
          link_extractors_exited: mp.Value, config: dict[str, int]) -> None:
    """Worker function that batches dictionaries of surface forms into
      a JSON file with 100,000 pages worth of links.

    Args:
        output_queue (mp.Queue): Queue containing dictionaries that will be
          batched together and saved to sharded JSON files.
        outfile_queue (mp.Queue): Queue containing file paths for all the saved
          JSON files.
        link_extractors_exited (mp.Value): Number of link extractor workers
          exited.
        config (dict[str, int]): Dictionary of parameters configuring the 
          extraction job.
    """
    counter = 0
    temp_dict = defaultdict(Counter)
    while True:
        # save new dictionary every 100_000 pages to reduce number of small files
        # ~50Mb files
        if counter >= config['num_pages_in_json']:
            # save existing dictionary and create a new one
            save_file_path = LinkExtractor.save_partial_dict(
                config['save_dir'], temp_dict)
            outfile_queue.put(save_file_path)

            # new dict, reset counter
            temp_dict = defaultdict(Counter)
            counter = 0

        # get item from queue
        with link_extractors_exited.get_lock():
            # if no more input expected and the queue is empty, exit process
            if link_extractors_exited.value == config[
                    'link_extractor_workers']:
                if output_queue.empty():
                    # save any final dictionaries currently being processed
                    if len(temp_dict) > 0:
                        save_file_path = LinkExtractor.save_partial_dict(
                            config['save_dir'], temp_dict)
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


def process_one_file(file_path: str) -> None:
    """Function to test the time and memory usage required to process 1 
      Wikipedia dump file. This function takes ~0:28:28 to complete and ~5Gb of
      memory, whereas the parallel version takes ~0:07:32.

    Args:
        file_path (str): File path for 1 Wikipedia dump file.
    """
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

    # TODO: to truly simulate the complete job, should combine dictionaries
    # and save to disk

    end = time.time()
    duration = end - start
    print(f'Time to process one file: {utils.hms_string(duration)}')


def main(args):
    start = time.time()

    # configure the application parameters
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if config['page_extractor_workers'] == -1:
        config['page_extractor_workers'] = mp.cpu_count()
    if config['link_extractor_workers'] == -1:
        config['link_extractor_workers'] = mp.cpu_count()

    # get wikidump files (test with 1 file)
    files = utils.list_wiki_files(args.wiki_dir)
    print(f'Processing {len(files)} Wikipedia dump files.')

    # create the save directory
    utils.create_folder(args.save_dir, remove_existing=True)
    # add save_dir to config dict for later use
    config['save_dir'] = args.save_dir

    # add sentinel values to the queue
    # when the producer sees the sentinel value, it should exit
    page_extractor_sentinels = [None] * config['page_extractor_workers']
    infile_queue = utils.get_queue(initial_items=files,
                                   sentinel_items=page_extractor_sentinels)

    # keep memory footprint low by limiting queue size
    page_queue = utils.get_queue(maxsize=config['page_queue_maxsize'])

    # get output_queue
    output_queue = utils.get_queue(maxsize=config['output_queue_maxsize'])

    # get outfile_queue
    outfile_queue = utils.get_queue()

    # keep track of how many workers are running
    page_extractors_exited = mp.Value('i', 0)
    link_extractors_exited = mp.Value('i', 0)

    # start page_extractor processes
    page_extractor_processes = utils.start_workers(
        config['page_extractor_workers'],
        page_extractor,
        args=(infile_queue, page_queue, config),
        name='page-extractor')

    # start link_extractor processes
    link_extractor_processes = utils.start_workers(
        config['link_extractor_workers'],
        link_extractor,
        args=(
            page_queue,
            output_queue,
            page_extractors_exited,
            config,
        ),
        name='link-extractor')

    # start saver processes
    saver_processes = utils.start_workers(config['saver_workers'],
                                          saver,
                                          args=(output_queue, outfile_queue,
                                                link_extractors_exited,
                                                args.save_dir, config),
                                          name='saver')

    # wait for the savers to exit, which signifies the job is complete
    savers_exited = []
    while True:
        if len(savers_exited) == config['saver_workers']:
            if outfile_queue.empty():
                break
        item = outfile_queue.get()
        if item is None:
            savers_exited.append(item)

    assert outfile_queue.empty()
    assert len(savers_exited) == config['saver_workers']

    # join all processes and close queues
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wiki-dir',
        help='Wikidump directory including date (e.g. 20210401) to extract.')
    parser.add_argument(
        '--save-dir',
        help=('Optional directory where the downloaded Wikipedia dump will be',
              'saved. Defaults to current directory.'),
        default='.')
    parser.add_argument(
        '--config',
        help='YAML file used to configure the parameters of the extraction job.'
    )
    args = parser.parse_args()

    main(args)