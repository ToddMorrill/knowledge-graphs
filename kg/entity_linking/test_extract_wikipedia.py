from collections import Counter
import multiprocessing as mp
import os

import pytest

from kg.entity_linking.extract_wikipedia import WikiFileExtractor, LinkExtractor
from kg.entity_linking.extract_wikipedia import page_extractor, link_extractor, saver, list_files, get_queue


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


# def test_page_extractor():
#     PAGE_EXTRACTOR_COUNT = mp.cpu_count()
#     PAGE_QUEUE_MAXSIZE = 50

#     wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401/'

#     # create output folder for all the links
#     save_dir = os.path.join(wiki_dir, 'links_20210401')
#     os.makedirs(save_dir, exist_ok=True)

#     # get wikidump files
#     files = list_files(wiki_dir)

#     # add sentinel values to the queue
#     # when the producer sees the sentinel value, it should exit
#     page_extractor_sentinels = [None] * PAGE_EXTRACTOR_COUNT
#     # test pipeline with first file
#     infile_queue = get_queue(initial_items=files[:1],
#                              sentinel_items=page_extractor_sentinels)

#     # keep memory footprint low by limiting queue size
#     page_queue = get_queue(maxsize=PAGE_QUEUE_MAXSIZE)

#     # start page_extractor processes
#     # TODO: abstract to function
#     page_extractor_processes = []
#     for i in range(PAGE_EXTRACTOR_COUNT):
#         p = mp.Process(target=page_extractor, args=(infile_queue, page_queue))
#         p.start()
#         p.name = f'page-extractor-{i}'
#         page_extractor_processes.append(p)

#     page_extractors_exited = []
#     while True:
#         if len(page_extractors_exited) == PAGE_EXTRACTOR_COUNT:
#             if page_queue.empty():
#                 break
#         item = page_queue.get()
#         if item is None:
#             page_extractors_exited.append(item)

#     assert page_queue.empty()
#     assert len(page_extractors_exited) == PAGE_EXTRACTOR_COUNT

#     for p in page_extractor_processes:
#         p.join()

#     infile_queue.close()
#     page_queue.close()

# def test_page_extractor_link_extractor():
#     PAGE_EXTRACTOR_COUNT = mp.cpu_count()
#     LINK_EXTRACTOR_COUNT = mp.cpu_count()
#     PAGE_QUEUE_MAXSIZE = 10_000 # 10k pages in memory at once
#     OUTPUT_QUEUE_MAXSIZE = 0 # capped at 32768

#     wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401/'

#     # create output folder for all the links
#     save_dir = os.path.join(wiki_dir, 'links_20210401')
#     os.makedirs(save_dir, exist_ok=True)

#     # get wikidump files
#     files = list_files(wiki_dir)

#     # add sentinel values to the queue
#     # when the producer sees the sentinel value, it should exit
#     page_extractor_sentinels = [None] * PAGE_EXTRACTOR_COUNT
#     # test pipeline with first file
#     infile_queue = get_queue(initial_items=files[:1],
#                              sentinel_items=page_extractor_sentinels)

#     # keep memory footprint low by limiting queue size
#     page_queue = get_queue(maxsize=PAGE_QUEUE_MAXSIZE)

#     # get output_queue
#     output_queue = get_queue(maxsize=OUTPUT_QUEUE_MAXSIZE)

#     # keep track of how many workers are running
#     page_extractors_exited = mp.Value('i', 0)
#     link_extractors_exited = mp.Value('i', 0)

#     # start page_extractor processes
#     # TODO: abstract to function
#     page_extractor_processes = []
#     for i in range(PAGE_EXTRACTOR_COUNT):
#         p = mp.Process(target=page_extractor, args=(infile_queue, page_queue))
#         p.start()
#         p.name = f'page-extractor-{i}'
#         page_extractor_processes.append(p)

#     # start link_extractor processes
#     link_extractor_processes = []
#     for i in range(LINK_EXTRACTOR_COUNT):
#         p = mp.Process(target=link_extractor,
#                        args=(
#                            page_queue,
#                            output_queue,
#                            page_extractors_exited,
#                            PAGE_EXTRACTOR_COUNT,
#                        ))
#         p.start()
#         p.name = f'link-extractor-{i}'
#         link_extractor_processes.append(p)

#     link_extractors_exited = []
#     while True:
#         if len(link_extractors_exited) == LINK_EXTRACTOR_COUNT:
#             if output_queue.empty():
#                 break
#         item = output_queue.get()
#         if item is None:
#             link_extractors_exited.append(item)

#     assert output_queue.empty()
#     assert len(link_extractors_exited) == LINK_EXTRACTOR_COUNT

#     for p in page_extractor_processes:
#         p.join()

#     for p in link_extractor_processes:
#         p.join()

#     infile_queue.close()
#     page_queue.close()
#     output_queue.close()

# def test_page_extractor_link_extractor_saver():
#     PAGE_EXTRACTOR_COUNT = mp.cpu_count()
#     LINK_EXTRACTOR_COUNT = mp.cpu_count()
#     SAVER_COUNT = mp.cpu_count()
#     PAGE_QUEUE_MAXSIZE = 10_000  # 10k pages in memory at once
#     OUTPUT_QUEUE_MAXSIZE = 0  # capped at 32768

#     wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/20210401/'

#     # create output folder for all the links
#     save_dir = os.path.join(wiki_dir, 'links_20210401')
#     os.makedirs(save_dir, exist_ok=True)

#     # get wikidump files
#     files = list_files(wiki_dir)

#     # add sentinel values to the queue
#     # when the producer sees the sentinel value, it should exit
#     page_extractor_sentinels = [None] * PAGE_EXTRACTOR_COUNT
#     # test pipeline with first file
#     infile_queue = get_queue(initial_items=files[:1],
#                              sentinel_items=page_extractor_sentinels)

#     # keep memory footprint low by limiting queue size
#     page_queue = get_queue(maxsize=PAGE_QUEUE_MAXSIZE)

#     # get output_queue
#     output_queue = get_queue(maxsize=OUTPUT_QUEUE_MAXSIZE)

#     # get outfile_queue
#     outfile_queue = get_queue()

#     # keep track of how many workers are running
#     page_extractors_exited = mp.Value('i', 0)
#     link_extractors_exited = mp.Value('i', 0)

#     # start page_extractor processes
#     # TODO: abstract to function
#     page_extractor_processes = []
#     for i in range(PAGE_EXTRACTOR_COUNT):
#         p = mp.Process(target=page_extractor, args=(infile_queue, page_queue))
#         p.start()
#         p.name = f'page-extractor-{i}'
#         page_extractor_processes.append(p)

#     # start link_extractor processes
#     link_extractor_processes = []
#     for i in range(LINK_EXTRACTOR_COUNT):
#         p = mp.Process(target=link_extractor,
#                        args=(
#                            page_queue,
#                            output_queue,
#                            page_extractors_exited,
#                            PAGE_EXTRACTOR_COUNT,
#                        ))
#         p.start()
#         p.name = f'link-extractor-{i}'
#         link_extractor_processes.append(p)

#     # start saver processes
#     saver_processes = []
#     for i in range(SAVER_COUNT):
#         p = mp.Process(target=saver,
#                        args=(output_queue, outfile_queue,
#                              link_extractors_exited, LINK_EXTRACTOR_COUNT,
#                              save_dir))
#         p.start()
#         p.name = f'saver-{i}'
#         saver_processes.append(p)

#     savers_exited = []
#     while True:
#         if len(savers_exited) == SAVER_COUNT:
#             if outfile_queue.empty():
#                 break
#         item = outfile_queue.get()
#         if item is None:
#             savers_exited.append(item)

#     assert outfile_queue.empty()
#     assert len(savers_exited) == SAVER_COUNT

#     for p in page_extractor_processes:
#         p.join()

#     for p in link_extractor_processes:
#         p.join()

#     for p in saver_processes:
#         p.join()

#     infile_queue.close()
#     page_queue.close()
#     output_queue.close()
#     outfile_queue.close()

# def test_first_50_pages():
#     """This test takes about 40 seconds to complete."""
#     wiki_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/'
#     file = 'enwiki-20210401-pages-articles-multistream1.xml-p1p41242.bz2'
#     file_path = os.path.join(wiki_dir, file)

#     # add files to queue
#     extractor_queue = mp.Queue()
#     extractor_queue.put(file_path)

#     # add sentinel value to the queue for all the producers
#     # when the producer sees the sentinel value, it should exit
#     n_producers = 1
#     for _ in range(n_producers):
#         extractor_queue.put(None)

#     # keep memory footprint low by limiting queue size
#     page_queue = mp.Queue(maxsize=50)

#     # start producer process (1 process for starters)
#     producer_processes = []
#     for i in range(n_producers):
#         p = mp.Process(target=page_extractor,
#                        args=(extractor_queue, page_queue))
#         p.start()
#         p.name = f'producer-process-{i}'
#         producer_processes.append(p)

#     # manually compare pages for first file
#     queue_pages = []
#     for _ in range(50):
#         queue_pages.append(page_queue.get())

#     # burn through the remainder of the queue (bit of a waste of time but oh well)
#     # alternative would be to add an argument to extract_pages to limit on the
#     producer_exited = False
#     while True:
#         if producer_exited:
#             if page_queue.empty():
#                 break
#         item = page_queue.get()
#         if item is None:
#             producer_exited = True

#     # manually get the first 50 pages
#     wiki_file_extractor = WikiFileExtractor(file_path)

#     expected_pages = []
#     for _ in range(50):
#         page = wiki_file_extractor.get_page()
#         if page is None:
#             break
#         expected_pages.append(page)

#     for p in producer_processes:
#         p.join()

#     extractor_queue.close()
#     page_queue.close()

#     assert queue_pages == expected_pages