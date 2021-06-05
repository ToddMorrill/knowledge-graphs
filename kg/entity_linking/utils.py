import hashlib
import multiprocessing as mp
import os
import shutil
from typing import Callable

BUF_SIZE = 65536  # 2^16 bytes, read file in chunks for hexdigest


def hms_string(sec_elapsed: int) -> str:
    """Nicely formatted time string.

    Args:
        sec_elapsed (int): Integer number of seconds.

    Returns:
        str: h:m:s formatted string.
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def strip_tag_name(t: str) -> str:
    """Processes and XML string."""
    idx = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t


def sha1_file(path: str) -> str:
    """Creates a SHA1 digest of the specified file."""
    sha1 = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def create_folder(dir: str, remove_existing: bool = False) -> None:
    """Creates a folder, optionally removing an existing folder."""
    # create output folder for all the links
    # if dir already there, need to remove (filenames are unique and files will be additive)
    if remove_existing:
        if os.path.isdir(dir):
            shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def list_wiki_files(wiki_dir: str) -> list:
    """Lists Wikipedia dump files."""
    files = os.listdir(wiki_dir)
    final_files = []
    for file in files:
        temp_file = os.path.join(wiki_dir, file)
        # filter out directories
        if not os.path.isfile(temp_file):
            continue
        # filter out index files
        if 'index' in file or '.DS_Store' in file:
            continue
        final_files.append(temp_file)
    return final_files


def get_queue(initial_items: list = [],
              sentinel_items: list = [],
              maxsize: int = 0) -> mp.Queue:
    """Create a multiprocessing safe queue and optionally add an initial batch
      of items and limit the queue size.

      maxsize=0 means there is no limit on the size.

    Args:
        initial_items (list, optional): Initial queue items. Defaults to [].
        sentinel_items (list, optional): Items used to determine if anything
          else will be added to the queue. Defaults to [].
        maxsize (int, optional): Limit on the number of items in the queue.
          Defaults to 0.

    Returns:
        mp.Queue: The initialized queue.
    """
    queue = mp.Queue(maxsize=maxsize)
    for item in initial_items:
        queue.put(item)
    for item in sentinel_items:
        queue.put(item)
    return queue


def start_workers(num_processes: int, target: Callable, args: tuple,
                  name: str) -> list:
    """Start worker processes and target the specified function.

    Args:
        num_processes (int): Number of worker processes.
        target (Callable): Function the worker should run.
        args (tuple): Arguments to be passed to the function.
        name (str): Name of the processes.

    Returns:
        list: List of process handlers, which can be used to later join the
          processes.
    """
    # start processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=target, args=args)
        p.start()
        p.name = f'{name}-{i}'
        processes.append(p)
    return processes