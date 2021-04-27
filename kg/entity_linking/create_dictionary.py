"""This module loads and combines all the partial dictionaries mapping anchor texts to entities.

NB: this script requires 20Gb+ of RAM to combine all files.
There are 47,224,683 surface forms in the dictionary.
There are 39,466,809 entities in the dictionary.

Examples:
    $ python create_dictionary.py
"""
from collections import Counter, defaultdict
import os
import time

from kg.entity_linking.extract_wiki_file import LinkExtractor
from kg.entity_linking import utils

if __name__ == '__main__':
    save_dir = '/Users/tmorrill002/Documents/datasets/wikipedia/links_20210401'
    files = [os.path.join(save_dir, file) for file in os.listdir(save_dir)]
    files = [file for file in files if file != '.DS_Store']
    master_dict = defaultdict(Counter)
    start = time.time()
    for file in files:
        temp_dict = LinkExtractor.load_json(file)
        master_dict = LinkExtractor.combine_dicts(master_dict, temp_dict)
    end = time.time()
    duration = end - start
    print(f'Time taken to combine all dictionaries: {utils.hms_string(duration)}')

    print(f'There are {len(master_dict):,} surface forms in the dictionary.')
    entities = set()
    for key in master_dict:
        entities.update(master_dict[key].keys())
    print(f'There are {len(entities):,} entities in the dictionary.')

    # remove infrequently cited terms (e.g. <5 citations)
    for key in list(master_dict.keys()):
        for entity, count in list(master_dict[key].items()):
            if count < 5:
                master_dict[key].pop(entity)
        # i.e. no entities left
        if len(master_dict[key]) == 0:
            master_dict.pop(key)
    
    print('After cleanup:')
    print(f'There are {len(master_dict):,} surface forms in the dictionary.')
    entities = set()
    for key in master_dict:
        entities.update(master_dict[key].keys())
    print(f'There are {len(entities):,} entities in the dictionary.')
    
    outfile_path = '/Users/tmorrill002/Documents/datasets/wikipedia/anchor_text_entity_mappings.json'
    LinkExtractor.save_json(master_dict, outfile_path)

