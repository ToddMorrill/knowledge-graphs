"""
This module downloads CoNLL 2003 data and saves it to the specified repository.

Examples:
    $ python download_unzip.py \
        --url https://www.clips.uantwerpen.be/conll2003/ner.tgz \
        --save-directory /Users/tmorrill002/Documents/datasets/conll/raw
"""
import argparse
import os
import shutil
import subprocess
import tarfile

import requests


def download_save_file(url: str, save_directory: str) -> str:
    """Download a file from the specified URL and save it to specified directory.
    
    Args:
        url (str): URL to download.
        save_directory (str): Destination directory where the download will be saved.
    """
    print(f'Downloading: {url}')
    downloaded_file = requests.get(url)
    head, tail = os.path.split(url)
    save_file_path = os.path.join(save_directory, tail)

    # ensure path exists
    os.makedirs(save_directory, exist_ok=True)

    with open(save_file_path, 'wb') as fd:
        fd.write(downloaded_file.content)
    print(f'Saved: {url}')
    return save_file_path


def handle_tar(file_path, extension, extracted_path, destination_directory):
    """Helper function to extract tar files."""
    tar = tarfile.open(file_path, extension)
    # remove files if they already exist
    if os.path.exists(extracted_path):
        shutil.rmtree(extracted_path)
    tar.extractall(path=destination_directory)
    tar.close()


def unzip(file_path: str) -> str:
    """Unzip file based on its extension.

    Args:
        file_path (str): File to be unzipped.

    Returns:
        str: Path to unzipped file/folder.
    """
    destination_directory, zip_file = os.path.split(file_path)
    extracted_path, _ = os.path.splitext(file_path)

    if file_path.endswith('tar.gz') or file_path.endswith('tgz'):
        handle_tar(file_path, 'r:gz', extracted_path, destination_directory)
    elif file_path.endswith('tar'):
        handle_tar(file_path, 'r:', extracted_path, destination_directory)
    return extracted_path


def main(args):
    # download and unzip files
    save_file_path = download_save_file(args.url, args.save_directory)
    extracted_path = unzip(save_file_path)
    print(f'Unzipped: {extracted_path}')

    # copy the Reuters rcv1.tar.xz dataset to the appropriate directory
    try:
        shutil.copy(args.reuters_file_path, extracted_path)
    except:
        print('No --reuters-file-path passed - will cause issues.')

    # Build the train/test datasets
    print('Attempting to build the CoNLL-2003 English dataset...')
    cmd = ['/bin/bash', f'{extracted_path}/bin/make.eng.2016']
    p = subprocess.Popen(cmd)
    exit_status = p.wait()
    if exit_status != 0:
        print(
            'It appears that you do not have the Reuters Corpus file: rcv1.tar.xz - '
            'either place the file in the root of the extracted ner folder or '
            'visit https://trec.nist.gov/data/reuters/reuters.html to request a copy.'
        )
        print('Downloading data from another source...')
        train_data = 'https://raw.githubusercontent.com/patverga/torch-ner-nlp-from-scratch/master/data/conll2003/eng.train'
        val_data = 'https://raw.githubusercontent.com/patverga/torch-ner-nlp-from-scratch/master/data/conll2003/eng.testa'
        test_data = 'https://raw.githubusercontent.com/patverga/torch-ner-nlp-from-scratch/master/data/conll2003/eng.testb'
        train_file_path = download_save_file(train_data, extracted_path)
        val_file_path = download_save_file(val_data, extracted_path)
        test_file_path = download_save_file(test_data, extracted_path)
    else:
        print('CoNLL-2003 English dataset successfully built.')

    # remove the extra copy of Reuters rcv1.tar.xz to save disk space
    try:
        _, tail = os.path.split(args.reuters_file_path)
        os.remove(os.path.join(extracted_path, tail))
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='URL to download.')
    parser.add_argument(
        '--save-directory',
        type=str,
        help='Destination directory where the download will be saved.')
    parser.add_argument(
        '--reuters-file-path',
        type=str,
        help='File path where the Reuters rcv1.tar.xz dataset is saved.')
    args = parser.parse_args()

    main(args)