import bz2
from collections import defaultdict, Counter
import multiprocessing as mp
import os
import xml.etree.ElementTree as etree

import mwparserfromhell

from kg.entity_linking import utils


class WikiFileExtractor(object):
    """**Overview of the parsing process:**
    1) etree.iterparse incrementally builds up the XML tree one line at a time
    2) the root node stores children nodes underneath it, which correspond to wikipedia pages
    3) the 'start' event corresponds to an opening tag (e.g. \<page\>) and the 'end' event corresponds to a closing tag (e.g. \</page\>)
    4) the elem.text method gathers all text between the start and end of the element, which is built up incrementally as more lines are parsed
    5) the root is cleared once a page is parsed so that there is only one child node under the root at a time (keeps memory footprint low)
    """
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.context, self.root = self._get_context(file_path)

    def _get_context(self, file_path):
        # open the compressed file
        fp = bz2.BZ2File(file_path, 'r')

        # handles first element in the iteration, which is later cleared to
        # avoid persisting everything in memory
        # get an iterable
        context = etree.iterparse(fp, events=("start", "end"))

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

        # if nothing left to iterate on, return None
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
        entity_anchor_pairs = self._get_entity_anchor_pairs(page)
        for entity, anchor in entity_anchor_pairs:
            anchor_to_entities[anchor][entity] += 1

        return anchor_to_entities


def process_page(extractor_queue, page_queue):
    # TODO: what happens when you call get on an empty queue?
    extractor = extractor_queue.get()
    page = extractor.get_page()
    # if there are still pages to be processed, add the extractor back to the queue
    if page:
        extractor_queue.put(extractor)
        page_queue.put(page)



if __name__ == '__main__':
    wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401/'
    files = os.listdir(wiki_dir)
    files = [os.path.join(wiki_dir, file) for file in files]
    
    # filter out index files
    files = [file for file in files if 'index' not in file]

    # add extractor to queue for each file
    # should be low memory
    extractor_queue = mp.Queue()
    for file in files[:2]:
        extractor = WikiFileExtractor(file)
        extractor_queue.put(extractor)
    
    # TODO: create a pool of workers, but test this using one worker for starters
    # keep memory footprint low by limiting queue size
    page_queue = mp.Queue(maxsize=50)
    process_page(extractor_queue, page_queue)
    breakpoint()


    
    
    # Sample code to test 1 file

    # wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/'
    # file = 'enwiki-20210401-pages-articles-multistream1.xml-p1p41242.bz2'
    # file_path = os.path.join(wiki_dir, file)

    # wiki_file_extractor = WikiFileExtractor(file_path)

    # page_count = 0
    # page = True
    # while page:
    #     page = wiki_file_extractor.get_page()
    #     page_count += 1
    #     # TODO: this is where the page should be put into a queue for more processing
    #     # queue.put(page)

    # print(page_count)

    # # this is where we should pop from the queue to process a page
    # # queue.get(page)
    # link_extractor = LinkExtractor(page)
    # link_dict = link_extractor.extract_links()

