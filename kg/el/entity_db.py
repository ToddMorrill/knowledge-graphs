"""This module adds all JSON shards to a SQLite database.

It takes about 21 minutes to process 211 shards.

Examples:
    $ python entity_db.py \
        --save-dir /Users/tmorrill002/Documents/datasets/wikipedia/20210401_sqlite \
        populate \
        --link-dir /Users/tmorrill002/Documents/datasets/wikipedia/links_20210401
    
    $ python entity_db.py \
        --save-dir /Users/tmorrill002/Documents/datasets/wikipedia/20210401_sqlite \
        query \
        --queries 'New York' 'Mumbai' 'Shanghai'
        
"""
import argparse
from collections import Counter, defaultdict
import os
import time

from sqlitedict import SqliteDict

from kg.el import utils
from kg.el.extract_wikipedia import LinkExtractor


def merge_dicts(json_files):
    master_dict = defaultdict(Counter)
    for file in json_files:
        temp_dict = LinkExtractor.load_json(file)
        master_dict = LinkExtractor.combine_dicts(master_dict, temp_dict)
    return master_dict


def add_to_db(file_path, dictionary):
    with SqliteDict(file_path) as db:
        for key in dictionary:
            counter_dict = dictionary[key]
            try:
                # adds entity counts together
                db[key] = db[key].update(counter_dict)
            except:
                db[key] = counter_dict
        db.commit()


def batch_add_to_db(json_files, db_file_path, batch_size=10):
    for i in range(0, len(json_files), batch_size):
        batch = json_files[i:i + batch_size]
        # merge a batch of JSON files
        temp_dict = merge_dicts(batch)
        # add batch to DB
        add_to_db(db_file_path, temp_dict)
        return None

def populate_db(link_dir, save_dir, batch_size=10):
    # create save_dir if not '.'
    if save_dir != '.':
        utils.create_folder(save_dir, remove_existing=True)

    save_file_path = os.path.join(save_dir, 'db.sqlite')

    json_shard_files = utils.list_wiki_files(link_dir)
    # much faster to merge dictionaries in memory than in the DB
    # each shard requires about 300Mb in memory
    start = time.time()
    batch_add_to_db(json_shard_files, save_file_path, batch_size=batch_size)
    end = time.time()
    duration = end - start
    print(
        f'Time taken to add {len(json_shard_files)} JSON shards to db: {utils.hms_string(duration)}'
    )

def query_db(file_path, queries, return_top_result=True):
    if isinstance(queries, str):
        queries = [queries]
    results = []
    with SqliteDict(file_path) as db:  # re-open the same DB
        for query in queries:
            results.append(db.get(query))
    
    # select most frequently cited entity (as opposed to complete dictionary)
    if return_top_result:
        top_results = []
        for idx, result in enumerate(results):
            if result is None:
                top_results.append(result)
            else:
                counter = Counter(result)
                top_results.append(counter.most_common()[0][0])
        return top_results
        
    return results


def main(args):
    if args.subparser == 'populate':
        populate_db(args.link_dir, args.save_dir, batch_size=10)
    
    elif args.subparser == 'query':
        file_path = os.path.join(args.save_dir, 'db.sqlite')
        results = query_db(file_path, args.queries)
        for idx, result in enumerate(results):
            print(f'Query: \'{args.queries[idx]}\', Top Hit:\'{result}\'')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-dir',
        help=
        ('Optional directory where the SQLite DB will be saved. Defaults to the current directory.'
         ),
        default='.')
    subparsers = parser.add_subparsers()

    # subparser for the populate command
    populate_parser = subparsers.add_parser('populate', help='Populate the DB from the JSON shards.')
    populate_parser.set_defaults(subparser='populate')
    populate_parser.add_argument(
        '--link-dir',
        help='Directory where JSON link dictionary shards are stored.',
        required=True)

    # subparser for the query command
    query_parser = subparsers.add_parser('query', help='Query the database.')
    query_parser.set_defaults(subparser='query')
    query_parser.add_argument(
        '--queries',
        help='A sequence of queries (e.g. \'New York\', \'Mumbai\', \'Shanghai\').',
        required=True,
        nargs='*')
    args = parser.parse_args()

    main(args)