"""This module adds all JSON shards to a SQLite database.

It takes about 21 minutes to process 211 shards.

Examples:
    $ python -m kg.el.entity_db \
        --save-dir /Users/tmorrill002/Documents/datasets/wikipedia/20210401_sqlite \
        populate \
        --link-dir /Users/tmorrill002/Documents/datasets/wikipedia/links_20210401
    
    $ python -m kg.el.entity_db \
        --save-dir /Users/tmorrill002/Documents/datasets/wikipedia/20210401_sqlite \
        query \
        --queries 'New York' 'Mumbai' 'Shanghai'
        
"""
import argparse
from collections import Counter, defaultdict
import os
import time
from typing import Union

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


class EntityDB(object):
    def __init__(self, db_file_path) -> None:
        # open DB as read-only
        self.db = SqliteDict(db_file_path, flag='r')

    def query(self, queries: Union[str, list], k: int = -1) -> list[str]:
        """Queries the sqlite database.

        Args:
            queries (Union[str, list]): String or list of string queries.
            k (int, optional): -1 refers to all candidates while positive
            integers correspond to the top k results. Defaults to -1.

        Returns:
            list[str]: Query results.
        """
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for query in queries:
            results.append(self.db.get(query))

        # select most frequently cited entity (as opposed to complete dictionary)
        top_results = []
        for idx, result in enumerate(results):
            if result is None:
                top_results.append(result)
            else:
                counter = Counter(result)
                # entities, counts = zip(*counter.most_common())
                if k == -1:
                    top_results.append(counter.most_common())
                elif k >= 0:
                    top_results.append(counter.most_common()[:k])
                else:
                    raise ValueError(f'Argument k must be nonnegative or -1.')
        return top_results

    def __del__(self):
        self.db.close()

    @staticmethod
    def get_wikipedia_link(entity):
        if entity is None:
            return None
        prepared_entity = entity.replace(' ', '_')
        link = f'https://en.wikipedia.org/wiki/{prepared_entity}'
        return link

    def query_result_html(self, query, results):
        html = []
        header = f'<h4>Results for query: <i>{query}</i></h4>'
        html.append(header)
        html.append('<ul>')
        for wiki_entry, count in results:
            link = self.get_wikipedia_link(wiki_entry)
            html.append(f'<li><a href="{link}">{wiki_entry}</a> ({count})</li>')
        html.append('</ul>')
        return '\n'.join(html)


def main(args):
    if args.subparser == 'populate':
        populate_db(args.link_dir, args.save_dir, batch_size=10)

    elif args.subparser == 'query':
        file_path = os.path.join(args.save_dir, 'db.sqlite')
        entity_db = EntityDB(file_path)
        results = entity_db.query(args.queries, k=2)
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
    populate_parser = subparsers.add_parser(
        'populate', help='Populate the DB from the JSON shards.')
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
        help=
        'A sequence of queries (e.g. \'New York\', \'Mumbai\', \'Shanghai\').',
        required=True,
        nargs='*')
    args = parser.parse_args()

    main(args)