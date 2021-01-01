"""
This module implements a baseline NER tagger and employs the following principles:
1) Only select complete named entities which appear in the training data.
2) Longer phrases are preferred over shorter ones.
3) Phrases with more than one entity tag are discarded.

Current baseline metrics (token level evaluation):
Macro average precision: 0.88
Macro average recall: 0.38
Macro F1: 0.46

Examples:
    $ python baseline.py \
        --data-directory /Users/tmorrill002/Documents/datasets/conll/transformed


Implementation notes:
1) Create dictionary of entities from train set. Only unambiguous entities and where entities are subphrases of other entities, use the longer entity.
    - keys will be tokenized entities, values will be the entity type
2) Take max length of the keys to limit the max subsequence of a sentence.
3) Run sliding window over test sentences. Iterate over each token and then expand the subsequence one token at a time until you reach the maximum obtained in step 2 above. Check if these subsequences are present in the training dictionary, else tag with 'O'.
4) Convert back to pandas column vector (may need to normalize the test labels for comparison).
5) Use standard sklearn evaluation on columns of labels to evaluate.

Experiments:
1) add validation set to train_dict - small increase for precision/recall/f1
2) try lowercasing everything - decrease in precision. recall/f1 steady.
3) check implementation for bugs - found bug in the way I was dropping duplicates.
4) Evaluate at the entity level (as opposed to the token level).

TODO: refactor and docstrings.
"""

import argparse
from functools import partial
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def create_train_dict(df):
    # try lowercasing to improve hit rate
    # df['Token'] = df['Token'].str.lower()

    temp_df = df.groupby(['NER_Tag_ID', 'NER_Tag_Normalized'],
                         as_index=False).agg(Entity=('Token', tuple))

    # drop ambiguous entities
    temp_df = temp_df.groupby(['NER_Tag_Normalized', 'Entity'],
                              as_index=False).count()
    temp_df = temp_df[~temp_df.duplicated(subset=['Entity'])]

    temp_df = temp_df[['Entity', 'NER_Tag_Normalized']]
    temp_dict = pd.Series(temp_df['NER_Tag_Normalized'].values,
                          index=temp_df['Entity']).to_dict()

    # Longer phrases are preferred over shorter ones.
    initial_keys = list(temp_dict.keys())
    longest_strings = []
    for substring in initial_keys:
        temp_indicators = []
        for key in initial_keys:
            temp_substring = ' '.join(substring)
            temp_key = ' '.join(key)
            if temp_substring in temp_key:
                temp_indicators.append(1)
            else:
                temp_indicators.append(0)
        # if the only substring is itself, add to longest_strings
        if sum(temp_indicators) == 1:
            longest_strings.append(substring)
    train_dict = {}
    for key in longest_strings:
        train_dict[key] = temp_dict[key]
    return train_dict


def get_tags(train_dict, sentence, max_key_len):
    labeled_sentence = [('placeholder', 'O')] * len(sentence)
    i = 0
    while i != len(sentence):
        broke = False
        for j in range(1, max_key_len):
            end_index = min(i + j, len(sentence))
            # lowercase_sub_sentence = [x.lower() for x in sentence[i:end_index]]
            # lowercase_sub_sentence = tuple(lowercase_sub_sentence)
            sub_sentence = tuple(sentence[i:end_index])
            if sub_sentence in train_dict:
                entity_type = train_dict[sub_sentence]
                tags = [(tok, entity_type) for tok in sentence[i:end_index]]
                labeled_sentence[i:end_index] = tags
                i += j
                broke = True
                break
            else:
                tags = [(tok, 'O') for tok in sentence[i:end_index]]
                labeled_sentence[i:end_index] = tags
        # if nothing found
        if not broke:
            i += 1
    return labeled_sentence


def increment_id(x, increment):
    if x == np.nan:
        return np.nan
    return x + increment


def add_tag_ids(df):
    tag_id = df['NER_Tag_ID'].max() + 1
    for idx, row in df.iterrows():
        if math.isnan(row['NER_Tag_ID']):
            df.loc[idx, 'NER_Tag_ID'] = tag_id
            tag_id += 1
    return df


def main(args):
    file_names = ['train.csv', 'validation.csv', 'test.csv']

    # load data
    df_dict = {}
    for file_name in file_names:
        file_path = os.path.join(args.data_directory, file_name)
        df_dict[file_name] = pd.read_csv(file_path)

    # # increment id tags
    # max_article_id = df_dict['train.csv']['Article_ID'].max() + 1
    # max_ner_tag_id = df_dict['train.csv']['NER_Tag_ID'].max() + 1
    # increment_article_id = partial(increment_id, increment=max_article_id)
    # increment_tag_id = partial(increment_id, increment=max_ner_tag_id)

    # df_dict['validation.csv']['Article_ID'] = df_dict['validation.csv']['Article_ID'].apply(increment_article_id)
    # df_dict['validation.csv']['NER_Tag_ID'] = df_dict['validation.csv']['NER_Tag_ID'].apply(increment_tag_id)

    # # combine train/validation set
    # train_val_df = pd.concat((df_dict['train.csv'], df_dict['validation.csv']), axis=0)

    # create training dictionary
    train_dict = create_train_dict(df_dict['train.csv'])

    # get max key length
    max_key_len = max([len(x) for x in list(train_dict.keys())])

    # transform test sentences
    test_sentences_df = df_dict['test.csv'].groupby(
        ['Article_ID', 'Sentence_ID'],
        as_index=False).agg(Sentence=('Token', list))
    # test_sentence = test_sentences_df.iloc[0]['Sentence']
    # pred_sentence = get_tags(train_dict, test_sentence, max_key_len)
    get_tags_partial = partial(get_tags,
                               train_dict=train_dict,
                               max_key_len=max_key_len)
    test_sentences_df['Tagged_Sentence'] = test_sentences_df['Sentence'].apply(
        lambda x: get_tags_partial(sentence=x))
    test_sentences_df = test_sentences_df[[
        'Article_ID', 'Sentence_ID', 'Tagged_Sentence'
    ]]
    test_sentences_df = test_sentences_df.explode('Tagged_Sentence')
    test_sentences_df[[
        'Token', 'NER_Tag_Prediction'
    ]] = test_sentences_df['Tagged_Sentence'].apply(pd.Series)
    test_sentences_df = test_sentences_df.drop(columns=['Tagged_Sentence'])
    # test_sentences_df.groupby('NER_Tag_Prediction').count()
    test_preds_df = pd.concat(
        (df_dict['test.csv'], test_sentences_df[['NER_Tag_Prediction'
                                                 ]].reset_index(drop=True)),
        axis=1)
    print('Token level performance:')
    print(
        classification_report(test_preds_df['NER_Tag_Normalized'],
                              test_preds_df['NER_Tag_Prediction']))

    # add in unique ids so we don't drop the 'O' category in the groupby
    test_preds_df = add_tag_ids(test_preds_df)

    # should really be grouping by the predicted entities
    test_preds_entity_df = test_preds_df.groupby(
        ['NER_Tag_ID', 'NER_Tag_Normalized'],
        as_index=False)[['Token', 'NER_Tag_Prediction'
                         ]].agg(Tokens=('Token', list),
                                NER_Tag_Predictions=('NER_Tag_Prediction',
                                                     list))

    # majority vote
    test_preds_entity_df['NER_Tag_Prediction'] = test_preds_entity_df[
        'NER_Tag_Predictions'].apply(lambda x: max(set(x), key=x.count))
    print('Entity level performance:')
    print(
        classification_report(test_preds_entity_df['NER_Tag_Normalized'],
                              test_preds_entity_df['NER_Tag_Prediction']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')

    args = parser.parse_args()

    main(args)
