"""This module calibrates the parameters used in the parallel Wikipedia
  extraction pipeline. The objective is to minimize processing time by taking
  advantage of all available CPUs and keeping them 100% utilized on productive
  work. See mp_example.md for a motivating example of the architecture.

Examples:
    $ python calibrate.py \
        --file-path /Users/tmorrill002/Documents/datasets/wikipedia/enwiki-20210401-pages-articles-multistream1.xml-p1p41242.bz2 \
        --mem-limit-gb 16 
"""
import argparse
from collections import defaultdict, Counter
from functools import partial
import multiprocessing
import os
import psutil
import time

from kg.entity_linking.extract_wikipedia import WikiFileExtractor, LinkExtractor, save_partial_dict
from kg.entity_linking import utils


def extract_pages(file_path, num_pages=1000):
    """Simulates the work done to read and partially process a file."""
    wiki_file_extractor = WikiFileExtractor(file_path)
    pages = []
    for _ in range(num_pages):
        page = wiki_file_extractor.get_page()
        pages.append(page)
    return pages


def extract_links(pages):
    """Simulates the work done to read and partially process a file."""
    link_dicts = []
    for page in pages:
        link_extractor = LinkExtractor(page)
        link_dict = link_extractor.extract_links()
        link_dicts.append(link_dict)
    return link_dicts


def combine_save(link_dicts):
    """Simulates the work done to merge dictionaries and save to disk."""
    temp_dict = defaultdict(Counter)
    for link_dict in link_dicts:
        temp_dict = LinkExtractor.combine_dicts(temp_dict, link_dict)
    # be sure to delete this file later
    save_file_path = save_partial_dict(save_dir='.', temp_dict=temp_dict)
    return save_file_path


def time_memory(function, description):
    process = psutil.Process(os.getpid())
    start_size = process.memory_info().rss  # in bytes

    start = time.time()
    # this is where the work is done
    result = function()
    end = time.time()
    duration_secs = end - start
    print(f'Time to compute {description}: {utils.hms_string(duration_secs)}')

    end_size = process.memory_info().rss
    memory_used = end_size - start_size
    memory_mb = memory_used / float(2**20)
    print(f'Memory to compute {description}: {memory_mb:,.2f} Mb')
    return result, duration_secs, memory_mb


def main(args):
    num_pages = 1000

    # reading from disk and page extraction simulation
    extract_pages_partial = partial(extract_pages, args.file_path)
    pages_result = time_memory(
        extract_pages_partial,
        description=f'page extraction for {num_pages:,} pages')
    pages, pages_duration, pages_memory = pages_result

    # extracting and processing links
    link_extractor_partial = partial(extract_links, pages)
    links_result = time_memory(
        link_extractor_partial,
        description=f'link extraction for {num_pages:,} pages')
    link_dicts, links_duration, links_memory = links_result

    # merge dictionaries save
    combine_save_partial = partial(combine_save, link_dicts)
    combine_save_result = time_memory(
        combine_save_partial,
        description=f'dictionary combine and save for {num_pages:,} pages')
    save_file_path, combine_save_duration, combine_save_memory = combine_save_result

    # clean up file
    os.remove(save_file_path)

    # maximize throughput while keeping the whole application within memory/CPU limits
    # want to find the right pages_queue_size
    if args.mem_limit_gb is not None:
        memory_budget_mb = args.mem_limit_gb * 1000
        total_available = psutil.virtual_memory().total / 1e+6
        if total_available < memory_budget_mb:
            # WARNING: this may cause the machine to use swap memory
            memory_budget_mb = total_available
    else:
        memory_budget_mb = (psutil.virtual_memory().total / 1e+6) * .75
    
    if args.cpu_limit is not None:
        cpu_budget = args.cpu_limit
        if multiprocessing.cpu_count() < cpu_budget:
            cpu_budget = multiprocessing.cpu_count()
    else:
        cpu_budget = multiprocessing.cpu_count()
    
    # based on the above statistics, we want to saturate memory and processors
    # with pages and link dictionaries 
    # set limit on the number of pages in memory at any point in time
    # (i.e. queue size x number of pages per block)
    # need to solve for 1 free parameter, the page_queue_maxsize

    # page processes will store cpu_budget * pages_memory
    processes_page_memory = cpu_budget * pages_memory

    # link processes will store cpu_budget * links_memory
    processes_links_memory = cpu_budget * links_memory

    # given that processing links is the bottleneck and combining/saving takes
    # very little time, the links queue will almost always be empty. 
    # ignoring in this analysis

    # given how little time is required to combine/save dicts, just use 1 worker
    # there are ~10m articles + info pages on wikipedia, if we save a 
    # dictionary every 100,000 pages we'll get 10,000,000 / 100,000 = 100 files
    num_pages_in_json = 100_000

    # get memory used to store 100_000 pages worth of links
    link_dict_batches = num_pages_in_json / num_pages
    link_dict_memory = link_dict_batches * links_memory

    memory_used = processes_page_memory + processes_links_memory + link_dict_memory

    # solve for free parameter, page_queue_memory
    remaining_memory = memory_budget_mb - memory_used
    page_queue_maxsize = remaining_memory / pages_memory

    print(f'Given paremeters specified, the page queue maxsize should be set to: {page_queue_maxsize}')
    
    total_memory_used = memory_used + (page_queue_maxsize*pages_memory)
    print(f'Total memory used: {total_memory_used} MB')


    # given link extraction is the bottleneck, use that for an estimate of the 
    # time to finish the compute job
    # how many batches of num_pages among 10m documents
    batches = 10_000_000 / num_pages
    compute_time_secs = (links_duration * batches) / cpu_budget
    print(
        f'Approx. time to compute the job: {utils.hms_string(compute_time_secs)}'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file-path',
        help='File path of the sample Wikipedia dump file to be processed.',
        required=True)
    parser.add_argument(
        '--mem-limit-gb',
        help='Maximum number of Gb the application should use. If not specified, this defaults to 75% of available memory.',
        type=float)
    parser.add_argument(
        '--cpu-limit',
        help='Maximum number of CPUs the application should use. If not specified, this defaults to 100% of available CPUs.')
    args = parser.parse_args()

    main(args)