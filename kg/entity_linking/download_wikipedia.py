# download_wikipedia
# Copyright 2021 by Jeff Heaton, released under the MIT License
# https://github.com/jeffheaton
"""This module downloads a complete dump of Wikipedia data.

Examples:
    $ python download_wikipedia.py
"""

import json
from multiprocessing import Pool
import os
import time
import urllib
import urllib.request
from xml.etree.ElementTree import dump  # nonsense that you have to do this

from bs4 import BeautifulSoup

from kg.entity_linking import utils

WIKIPEDIA_URL = 'https://dumps.wikimedia.org/'
WIKIPEDIA_LANG = 'enwiki'


class WikiDump(object):
    """This class identifies which Wikipedia dump to work with."""
    def __init__(self, wiki_lang=WIKIPEDIA_LANG, wiki_base_url=WIKIPEDIA_URL):
        self.wiki_lang = wiki_lang
        self.wiki_base_url = wiki_base_url
        self.wiki_url = os.path.join(self.wiki_base_url, self.wiki_lang)

    def get_wikidumps_available(self):
        # gathers all links found on this page: https://dumps.wikimedia.org/enwiki/
        index = urllib.request.urlopen(self.wiki_url).read()
        soup_index = BeautifulSoup(index, 'html.parser')
        # Find the links on the page
        return [
            a['href'] for a in soup_index.find_all('a') if a.has_attr('href')
        ]

    def get_dumpstatus(self, wiki_date):
        # e.g. https://dumps.wikimedia.org/enwiki/20210420/dumpstatus.json
        dump_url = os.path.join(self.wiki_url, str(wiki_date))
        status_file = os.path.join(dump_url, 'dumpstatus.json')
        dump_json = urllib.request.urlopen(status_file).read()
        status = json.loads(dump_json)
        return status

    def ready(self, wiki_date):
        # check if this dump is ready or in progress
        status = self.get_dumpstatus(wiki_date)
        # True if ready, False o/w
        return status['jobs']['metacurrentdump']['status'] == 'done'

    def get_latest_dump(self):
        dumps = self.get_wikidumps_available()

        # get most recently dated dump from: https://dumps.wikimedia.org/enwiki/
        lst = []
        for dump in dumps:
            # ignore '../' and 'latest/
            if dump in ['../', 'latest/']:
                continue

            # strip '/' from the end of the url
            dump = dump[:-1]
            lst.append(int(dump))  # e.g. 20210420

        # most recent first
        lst.sort(reverse=True)

        # if dump not ready, fall back to previous one
        if not self.ready(lst[0]):
            return lst[1]
        return lst[0]


class DownloadWikifile(object):
    def __init__(self,
                 wiki_lang=WIKIPEDIA_LANG,
                 wiki_base_url=WIKIPEDIA_URL,
                 wiki_date=None):
        self.wiki_lang = wiki_lang
        self.wiki_base_url = wiki_base_url

        # if no dump date specified, get the latest date
        self.dump_site = WikiDump(self.wiki_lang, wiki_base_url)
        if not wiki_date:
            self.wiki_date = self.dump_site.get_latest_dump()
        else:
            # check that this wiki dump data is ready
            if not self.dump_site.ready(wiki_date):
                latest_dump = self.dump_site.get_latest_dump()
                raise ValueError(
                    f'Wikipedia dump for {wiki_date} is not complete yet. Try {latest_dump}.'
                )
            self.wiki_date = wiki_date
        self.dump_url = os.path.join(self.wiki_base_url, self.wiki_lang,
                                     str(self.wiki_date))

    def download_file(self, file, meta, target_path):
        source_url = urllib.parse.urljoin(self.dump_url, meta['url'])
        target_file = os.path.join(target_path, file)
        if os.path.exists(target_file):
            sha1_local = utils.sha1_file(target_file)
            if sha1_local != meta['sha1']:
                # will need to download
                print(f'Corrupt: {file}')
                os.remove('file')
            else:
                print(f'Exists: {file}')
                return 'already downloaded'
        else:
            # probably don't need to print this
            print(target_path)
            print(f'Missing: {file}')
            
        try:
            save_path, http_msg = urllib.request.urlretrieve(source_url, target_file)
            print(f'Downloaded: {file}')
            return save_path, http_msg
        except urllib.error.URLError as e:
            # remove file if it got corrupted
            if os.path.exists(target_file):
                os.remove(target_file)
            print(f'Download Error: {file}')
            print(e)
            return repr(e)  
                          

    def download(self, path):
        """Download articles multistream dump."""
        start = time.time()
        # create save directory
        target_path = os.path.join(path, str(self.wiki_date))
        os.makedirs(target_path, exist_ok=True)

        # get list of files associated with the dump
        dump_status = self.dump_site.get_dumpstatus(self.wiki_date)
        # prefer to download the multistream files for better parallelization
        # e.g. enwiki-20210401-pages-articles-multistream1.xml-p1p41242.bz2
        # e.g. enwiki-20210401-pages-articles-multistream-index1.txt-p1p41242.bz2
        files = dump_status['jobs']['articlesmultistreamdump']['files']

        # generate arg list
        args_list = []
        for file in list(files.keys()):
            meta = files[file]
            args_list.append((file, meta, target_path))
        
        # try to download in parallel
        # with 5 processes you may get throttled with
        # HTTP Error 503: Service Temporarily Unavailable 
        with Pool(5) as p:
            while args_list:
                results = p.starmap(self.download_file, args_list)
                
                # check results to determine any files were throttled
                remaining_files = []
                for idx, result in enumerate(results):
                    if result == 'already downloaded':
                       continue
                    # if the save path is returned, all good
                    elif result[0] == os.path.join(target_path, args_list[idx][0]):
                        continue
                    # continue processing
                    else:
                        remaining_files.append(args_list[idx])
                
                # try to reprocess
                args_list = remaining_files
        
        end = time.time()
        duration = end - start
        print(f'Total time taken to download Wikipedia: {utils.hms_string(duration)}.')

        results
        results    


if __name__ == '__main__':
    save_dir = '/Users/tmorrill002/Documents/datasets/wikipedia'
    dl = DownloadWikifile(wiki_lang=WIKIPEDIA_LANG,
                          wiki_base_url=WIKIPEDIA_URL,
                          wiki_date=None)
    dl.download(save_dir)