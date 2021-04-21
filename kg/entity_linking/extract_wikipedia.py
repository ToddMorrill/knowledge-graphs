# process_wikipedia
# Copyright 2021 by Jeff Heaton, released under the MIT License
# https://github.com/jeffheaton
"""This module processes the downloaded Wikipedia files in parallel.

Examples:
    $ python extract_wikipedia.py
"""

import bz2
import glob
import multiprocessing as mp
import os
import queue
import time
import traceback
import xml.etree.ElementTree as etree

from kg.entity_linking import utils
from kg.entity_linking.process_wikipedia import ProcessPages

WORKER_REPORT = 1000
ENTIRE_TASK_REPORT = 100000
QUEUE_SIZE = 50
ENCODING = 'utf-8'
GET_TIMEOUT = 1 * 60
GET_RETRY = 5


class ExtractWikipediaFile(object):
    """This class is spun up once per worker process."""
    def __init__(self, worker):
        self.total_count = 0
        self.article_count = 0
        self.redirect_count = 0
        self.template_count = 0
        self.worker = worker

    def extract_file(self, path):
        start_time = time.time()

        title = None
        redirect = ''
        count = 0
        with bz2.BZ2File(path, 'r') as fp:
            is_first = True
            for event, elem in etree.iterparse(fp, events=('start', 'end')):
                tname = utils.strip_tag_name(elem.tag)
                if is_first:
                    root = elem
                    is_first = False

                if event == 'start':
                    if tname == 'page':
                        title = ''
                        id = -1
                        redirect = ''
                        inrevision = False
                        ns = 0
                    elif tname == 'revision':
                        # Do not pick up on revision id's
                        inrevision = True
                else:
                    if tname == 'title':
                        title = elem.text
                    elif tname == 'text':
                        text = elem.text
                    elif tname == 'id' and not inrevision:
                        id = int(elem.text)
                    elif tname == 'redirect':
                        redirect = elem.attrib['title']
                    elif tname == 'ns':
                        ns = int(elem.text)
                    elif tname == 'page':
                        self.total_count += 1

                        try:
                            if ns == 10:
                                self.template_count += 1
                                self.worker.process_template(id, title)
                            elif ns == 0:
                                if len(redirect) > 0:
                                    self.article_count += 1
                                    self.worker.process_redirect(
                                        id, title, redirect)
                                else:
                                    self.redirect_count += 1
                                    #print(f"Article: {title}")
                                    self.worker.process_article(
                                        id, title, text)
                        except Exception as e:
                            print(f"Error processing: {id}:{title}")
                            print(e)

                        title = ""
                        redirect = ""
                        text = ""
                        ns = -100
                        if self.total_count > 1 and (self.total_count %
                                                     WORKER_REPORT) == 0:
                            self.worker.report_progress(self.total_count)
                            self.total_count = 0

                        root.clear()
        self.worker.report_progress(self.total_count)


class ExtractWikipedia(object):
    """This is the main controller class, it runs in the main process and
     aggregates results from the individual processes used to distribute the
     workload."""
    def __init__(self, path, payload, wiki_date=None, files=None):
        # if no date specified, assume most recent dump
        if wiki_date is None:
            wiki_date = ExtractWikipedia.find_latest_dump(path)
            if wiki_date is None:
                raise FileNotFoundError(
                    'No Wikipedia dumps have been downloaded.')

        self.wiki_path = os.path.join(path, str(wiki_date))
        self.total_count = 0
        self.file_count = 0
        self.last_update = 0
        self.payload = payload
        self.files = files  # optional list of files to process
        self.workers_running = 0
        self.workers = 0

    @staticmethod
    def find_latest_dump(path):
        """Find the most recent Wikipedia dump locally."""
        # e.g. ['20210401_bak', '.DS_Store', '20210401', 'enwik9']
        dirs = [
            name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ]
        date_dirs = []
        for dir_ in dirs:
            try:
                i = int(dir_)
                date_dirs.append(i)
            except:
                continue
        date_dirs.sort(reverse=True)

        # if empty, no downloads available
        if not date_dirs:
            return None
        # return most recent date
        return date_dirs[0]

    def process_single_file(self, filename=None):
        start_time = time.time()
        if filename == None:
            files = glob.glob(os.path.join(self.wiki_path, "*.bz2"))
            files.sort()
            filename = files[0]
        self.process([filename])

    # def get_event(self, event_queue):
    #     get_done = False
    #     get_retry = 0
    #     while not get_done:
    #         try:
    #             return event_queue.get(timeout=GET_TIMEOUT)
    #         except queue.Empty:
    #             get_retry += 1
    #             if get_retry <= GET_RETRY:
    #                 print(f"Queue get timeout, retry {get_retry}/{GET_RETRY}")
    #             else:
    #                 print(
    #                     f"Queue timeout failed, retry {GET_RETRY} failed, exiting."
    #                 )
    #                 get_done = True
    #                 return None

    def shutdown(self, processes, event_queue):
        done = False
        print("waiting for workers to write remaining results")
        while not done:
            done = True
            for p in processes:
                p.join(10)
                if p.exitcode == None:
                    done = False
                try:
                    evt = event_queue.get(timeout=10)
                    self.handle_event(evt)
                except queue.Empty:
                    pass

    @staticmethod
    def get_event(workload_queue):
        """Retrieves an event from the queue and waits if one isn't available.
        
        This function may need to wait for the first set of outputs to appear.
        """
        get_done = False
        get_retry = 0
        while not get_done:
            try:
                return workload_queue.get(timeout=GET_TIMEOUT)
            except queue.Empty:
                get_retry += 1
                if get_retry <= GET_RETRY:
                    print(
                        f'Workload get timeout, retry {get_retry}/{GET_RETRY}')
                else:
                    print(
                        f'Workload timeout failed, retry {GET_RETRY} failed, exiting.'
                    )
                    get_done = True
                    return None

    def handle_event(self, evt):
        if 'completed' in evt:
            self.total_count += evt['completed']
            self.current_update = int(self.total_count / ENTIRE_TASK_REPORT)
            if self.current_update != self.last_update:
                print(
                    f'{self.current_update*ENTIRE_TASK_REPORT:,}; files: {self.file_count}/{len(self.files)}, workers: {self.workers_running}/{self.workers}'
                )
                self.last_update = self.current_update
        elif 'file_complete' in evt:
            self.file_count += 1
        elif "**worker done**" in evt:
            self.workers_running -= 1
            print(f"Worker done: {evt['**worker done**']}")

    @staticmethod
    def worker(input_queue, output_queue, config):
        """This function runs on a separate process and processes a particular
         file."""
        try:
            payload_worker = config['payload'].get_worker_class(
                output_queue, config)
            done = False

            while not done:
                path = input_queue.get()

                if path != "**exit**":
                    try:
                        e = ExtractWikipediaFile(payload_worker)
                        e.extract_file(path)
                    except Exception as e:
                        print(f'Error: {repr(e)}')
                        traceback.print_exc()
                    finally:
                        output_queue.put({'file_complete': True})
                else:
                    done = True

        finally:
            output_queue.put({"**worker done**": config['num']})

    def process(self):
        # get files to be processed
        if self.files is None:
            self.files = glob.glob(os.path.join(self.wiki_path, "*.bz2"))
            if len(self.files) == 0:
                raise FileNotFoundError(
                    f'No wiki files located at: {self.wiki_path}')

        start_time = time.time()

        print(f'Processing {len(self.files)} files.')
        cpus = 1  # mp.cpu_count()
        print(f'Detected {cpus} cores.')
        self.workers = cpus * 1
        print(f'Using {self.workers} processes.')

        input_queue = mp.Queue()
        # TODO: why set a max size?
        # this is what keeps memory to a manageable size
        # might get better throughput with a larger queue size
        output_queue = mp.Queue(QUEUE_SIZE)

        # start worker processes
        processes = []
        for i in range(self.workers):
            config = {'payload': self.payload, 'num': i}

            p = mp.Process(target=ExtractWikipedia.worker,
                           args=(input_queue, output_queue, config))
            p.start()
            p.name = f'process-{i}'
            processes.append(p)
        self.workers_running = self.workers

        # add files to be processed to the queue
        for file in self.files:
            input_queue.put(file)

        # why 2*the number of workers?
        for i in range(self.workers * 2):
            input_queue.put('**exit**')

        self.payload.open()
        error_exit = False
        while (self.file_count < len(self.files)) and not error_exit:
            evt = self.get_event(output_queue)
            # handle worker and file count events
            self.handle_event(evt)
            # handle data events
            self.payload.handle_event(evt)

        print(
            f'{self.total_count:,}; files: {self.file_count}/{len(self.files)}'
        )

        self.shutdown(processes, output_queue)
        self.payload.close()
        input_queue.close()
        output_queue.close()

        elapsed_time = time.time() - start_time
        print("Elapsed time: {}".format(utils.hms_string(elapsed_time)))
        print("done")


if __name__ == '__main__':
    save_dir = '/Users/tmorrill002/Documents/datasets/wikipedia'
    wiki = ExtractWikipedia(
        path=save_dir,  # Location you downloaded Wikipedia to
        payload=ProcessPages(
            save_dir)  # where you want the extracted Wikipedia files to go
    )
    wiki.process()