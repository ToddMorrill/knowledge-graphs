# Copyright 2021 by Jeff Heaton, Todd Morrill released under the MIT License
"""This module downloads a complete dump of Wikipedia data.

Examples:
    $ python download_wikipedia.py \
        --save-dir /Users/tmorrill002/Documents/datasets/wikipedia
"""
import argparse
from http.client import HTTPMessage
import json
from multiprocessing import Pool
import os
import time
from typing import Union
import urllib
import urllib.request
from xml.etree.ElementTree import dump  # nonsense that you have to do this

from bs4 import BeautifulSoup
import requests

from kg.entity_linking import utils


class WikiDump(object):
    """This class identifies which Wikipedia dump to work with."""
    def __init__(self, wiki_lang: str = 'enwiki') -> None:
        """Initialize the class, optionally passing a language specifier. 

        Args:
            wiki_lang (str, optional): Language string (e.g. 'enwiki') 
              specifying which Wikidump to download. Defaults to 'enwiki'.
        """
        self.wiki_lang = wiki_lang
        self.wiki_base_url = 'https://dumps.wikimedia.org/'
        self.wiki_url = os.path.join(self.wiki_base_url, self.wiki_lang)

    def _get_wikidumps_available(self) -> list[str]:
        """Gathers all the date links available on 
          https://dumps.wikimedia.org/enwiki/ or the comparable page for the 
          language specified. 

        Returns:
            list[str]: A list of date strings (e.g. 20210420).
        """
        # gathers all links found on this page:
        # https://dumps.wikimedia.org/enwiki/
        index_page = requests.get(self.wiki_url).text
        soup_index = BeautifulSoup(index_page, 'html.parser')
        # Find the links to dumps on the page
        return [
            a['href'] for a in soup_index.find_all('a') if a.has_attr('href')
        ]

    def get_dumpstatus(self, wiki_date: str) -> dict:
        """Retrieves the dumpstatus.json file from a URL such as 
          https://dumps.wikimedia.org/enwiki/20210420/dumpstatus.json using the
          date passed.

        Args:
            wiki_date (str): A date string (e.g. 20210420) for the dump.

        Returns:
            dict: A dictionary containing information about the dump status. 
        """
        dump_url = os.path.join(self.wiki_url, str(wiki_date))
        status_page = os.path.join(dump_url, 'dumpstatus.json')
        dump_json = requests.get(status_page).text
        status = json.loads(dump_json)
        return status

    def ready(self, wiki_date: str) -> bool:
        """Checks if the Wiki dump is ready or in progress."""
        status = self.get_dumpstatus(wiki_date)
        # True if ready, o/w False
        return status['jobs']['metacurrentdump']['status'] == 'done'

    def get_latest_dump(self) -> str:
        """Retrieves the date (e.g. 20210420) for the latest dump that is ready
          for download.

        Returns:
            str: A date string (e.g. 20210420) for the dump.
        """
        dumps = self._get_wikidumps_available()

        # get most recently dated dump from:
        # https://dumps.wikimedia.org/enwiki/
        dates = []
        for dump in dumps:
            # ignore '../' and 'latest/
            if dump in ['../', 'latest/']:
                continue

            # strip '/' from the end of the url
            dump = dump[:-1]
            dates.append(int(dump))  # e.g. 20210420

        # most recent first
        dates.sort(reverse=True)

        # if dump not ready, fall back to previous one
        if not self.ready(dates[0]):
            return dates[1]
        return dates[0]


class WikiDumpDownloader(object):
    """This class downloads an entire Wikipedia dump."""
    def __init__(self,
                 wiki_lang: str = 'enwiki',
                 wiki_date: str = None) -> None:
        """Initializes the class with an optional language and date.

        Args:
            wiki_lang (str, optional): A Wikipedia dump language (e.g. enwiki)
              to download. Defaults to 'enwiki'.
            wiki_date (str, optional): A date string (e.g. 20210420) for the
              dump. Defaults to None.

        Raises:
            ValueError: This occurs if the wiki_date passed is invalid.
        """
        self.wiki_lang = wiki_lang
        self.wiki_base_url = 'https://dumps.wikimedia.org/'

        # if no dump date specified, get the latest date
        self.dump_site = WikiDump(self.wiki_lang)
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

    def download_file(self, save_file_path: str,
                      meta: dict) -> Union[str, tuple[str, HTTPMessage]]:
        """Downloads a single file from Wikipedia, first checking if the file is already downloaded. Any error messages are relayed to the calling function.

        Args:
            save_file_path (str): The local file path where the file will be saved.
            meta (dict): A dictionary that contains the URL and SHA1 hash digest information about the file.

        Returns:
            Union[str, tuple[str, HTTPMessage]]: [description]
        """
        # urljoin does the following:
        # self.dump_url = 'https://dumps.wikimedia.org/enwiki/20210501'
        # meta['url'] = '/enwiki/20210501/enwiki-20210501-pages...bz2'
        # source_url = 'https://dumps.wikimedia.org/enwiki/20210501/enwiki-20210501-pages...bz2'
        source_url = urllib.parse.urljoin(self.dump_url, meta['url'])

        # check sha1 hash of the existing file to determine if it's valid
        if os.path.exists(save_file_path):
            sha1_local = utils.sha1_file(save_file_path)
            if sha1_local != meta['sha1']:
                # will need to download
                # TODO: use logging here
                # print(f'Corrupt: {file}')
                os.remove(save_file_path)
            else:
                # TODO: use logging here
                # print(f'Exists: {file}')
                return 'already downloaded'

        # if you attempt to download in parallel, you'll likely get errors
        try:
            # TODO: is there a corresponding function in requests?
            save_path, http_msg = urllib.request.urlretrieve(
                source_url, save_file_path)
            # TODO: use logging here
            # print(f'Downloaded: {file}')
            return save_path, http_msg
        except urllib.error.URLError as e:
            # remove file if it got corrupted
            if os.path.exists(save_file_path):
                os.remove(save_file_path)
            # TODO: logging
            # print(f'Download Error: {file}')
            # print(e)
            return repr(e)

    def download(self, save_dir: str) -> None:
        """Downloads the articles multistream dump from Wikipedia.

        Args:
            save_dir (str): Local directory where the downloads will be saved.
        """
        start = time.time()
        # create save directory
        save_dir = os.path.join(save_dir, str(self.wiki_date))
        os.makedirs(save_dir, exist_ok=True)

        # get list of files associated with the dump
        dump_status = self.dump_site.get_dumpstatus(self.wiki_date)
        # prefer to download the multistream files for better parallelization
        # e.g. enwiki-20210401-pages-articles-multistream1.xml-p1p41242.bz2
        # e.g. enwiki-20210401-pages-articles-multistream-index1.txt-p1p41242.bz2
        files = dump_status['jobs']['articlesmultistreamdump']['files']

        # generate arg list
        args_list = []
        for file in list(files.keys()):
            # fully specified path where the file will be saved
            save_file_path = os.path.join(save_dir, file)
            meta = files[file]
            args_list.append((save_file_path, meta))

        # try to download in parallel
        # with 5 processes you will get throttled
        # HTTP Error 503: Service Temporarily Unavailable
        # TODO: just remove the parallel code here, not working well + complexity
        # TODO: clock before and after parallelism just to know
        with Pool(2) as p:
            # keep trying to download until it succeeds
            while args_list:
                results = []

                # treat args_list like a fifo queue
                # stagger calls to avoid API throttling
                for args in args_list:
                    # not ideal when all files are already downloaded
                    time.sleep(1)
                    # save the future result
                    results.append(p.apply_async(self.download_file,
                                                 args=args))

                # check results to determine if any files were throttled
                remaining_files = []
                for idx, future in enumerate(results):
                    result = future.get()
                    if result == 'already downloaded':
                        continue
                    # if the save path is returned, it was processed correctly
                    elif result[0] == args_list[idx][0]:
                        continue
                    # reprocess if there was an error
                    else:
                        remaining_files.append(args_list[idx])

                # reprocess
                args_list = remaining_files

        end = time.time()
        duration = end - start
        print(
            f'Total time taken to download Wikipedia: {utils.hms_string(duration)}.'
        )


def main(args):
    downloader = WikiDumpDownloader(wiki_date=args.wiki_date)
    downloader.download(args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-dir',
        help=('Optional directory where the downloaded Wikipedia dump will be',
              'saved. Defaults to current directory.'),
        default='.')
    parser.add_argument(
        '--wiki-date',
        help=
        ('Optional Wikidump date (e.g. 20210401) to download. If not passed,',
         ' the module will download the most recent Wikidump.'))
    args = parser.parse_args()

    main(args)