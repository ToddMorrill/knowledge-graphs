"""This script evaluates Wikipedia entity linking procedures.

Examples:
    $ python evaluate.py \
        --db-file-path /Users/tmorrill002/Documents/datasets/wikipedia/20210401_sqlite/db.sqlite \
        --aida-file-path /Users/tmorrill002/Documents/datasets/aida-yago2-dataset/transformed/aida_parsed.csv
"""
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score

from kg.el.entity_db import EntityDB

def extract_wikipedia_entity(x):
    if isinstance(x, str):
        return x.split('http://en.wikipedia.org/wiki/')[-1]
    return x

def main(args):
    aida_df = pd.read_csv(args.aida_file_path)
    aida_df['Wikipedia_Entity'] = aida_df['Wikipedia_URL'].apply(lambda x: extract_wikipedia_entity(x))

    # only query actual entities (ignore all other tokens)
    # retain positional info from original DF
    queries_df = aida_df[['Complete_Entity']].dropna()
    entity_db = EntityDB(args.db_file_path)
    query_results = entity_db.query(queries_df['Complete_Entity'].values, k=1)
    # query_results = query_db(args.db_file_path, queries_df['Complete_Entity'].values)
    queries_df['Query_Result'] = query_results
    # fill Nones in Query_Result column with --NME--
    queries_df['Query_Result'] = queries_df['Query_Result'].fillna('--NME--')
    # replace spaces with underscores to compare to Wikipedia URLs
    queries_df['Query_Result'] = queries_df['Query_Result'].apply(lambda x: x.replace(' ', '_'))

    # join query results back to main DF
    aida_df = aida_df.join(queries_df.drop(columns = ['Complete_Entity']))

    # filter down to only entities
    entity_df = aida_df[aida_df['BI_Tag'].notnull()]

    acc = accuracy_score(entity_df['Wikipedia_Entity'], entity_df['Query_Result'])
    print(f'Accuracy between AIDA and the surface form dictionary: {acc*100:.2f}%')

    # examine how often AIDA predicts --NME--, while dictionary yields an answer
    NME_df = entity_df[entity_df['Wikipedia_Entity'] == '--NME--']
    extra_pred_rate = NME_df[NME_df['Query_Result'] != '--NME--'].shape[0] / len(NME_df)
    print(f'The surface form dictionary yields predictions for {extra_pred_rate*100:.2f}% of the entities where AIDA does not have a linked Wikipedia Entity.')
    
    # examine accuracy when an entity exists
    mention_entity_df = entity_df[entity_df['Wikipedia_Entity'] != '--NME--']
    mention_entity_acc = accuracy_score(mention_entity_df['Wikipedia_Entity'], mention_entity_df['Query_Result'])
    print(f'Accuracy among population where AIDA has a Wikipedia entity link: {mention_entity_acc*100:.2f}%')

    # some differences are due to changes to Wikipedia (e.g. Mushtaq_Ahmed  Mushtaq_Ahmed_(cricketer))
    # some are due to capitalization (e.g. KANSAS) but title casing may still not solve the problem if the correct answer is
    # Kansas_City_Chiefs rather than Kansas (state)
    # some of the annotations are simply wrong (e.g. when token is Moslems, the Wiki entry should be Moslems, not Islam)
    print(f'True accuracy is probably 10-20% higher than {acc:.2f}%.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db-file-path',
        help='Fully specified file path where the SQLite DB is located.')
    parser.add_argument(
        '--aida-file-path',
        required=True,
        help='Fully specified file path where the parsed AIDA CSV file is located.')
    args = parser.parse_args()
    main(args)
