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
3) check implementation for bugs - found bug in the way I was dropping duplicates, major increase in performance.
4) Evaluate at the entity level (as opposed to the token level) - ~1% decrease across the board.
"""

import argparse
from functools import partial
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import kg.ner.utils as utils


def create_train_dict(df: pd.DataFrame, lowercase: bool = False) -> dict:
    """Create dictionary containing entities and their tags from the training set.

    Args:
        df (pd.DataFrame): Training df.
        lowercase (bool, optional): Optionally lowercase tokens. Defaults to False.

    Returns:
        dict: Dictionary with entities as keys and tags as values, e.g. {('Washington', 'DC'): 'LOC', ...}.
    """
    if lowercase:
        # try lowercasing to improve hit rate
        df['Token'] = df['Token'].str.lower()

    # consolidate entities into tuples
    temp_df = df.groupby(['NER_Tag_ID', 'NER_Tag_Normalized'],
                         as_index=False).agg(Entity=('Token', tuple))

    # drop ambiguous entities
    temp_df = temp_df.groupby(['NER_Tag_Normalized', 'Entity'],
                              as_index=False).count()
    temp_df = temp_df[~temp_df.duplicated(subset=['Entity'])]

    # create a dictionary, where keys are entities and values are NER tags
    # e.g. {('Washington', 'DC'): 'LOC', ...}
    temp_df = temp_df[['Entity', 'NER_Tag_Normalized']]
    temp_dict = pd.Series(temp_df['NER_Tag_Normalized'].values,
                          index=temp_df['Entity']).to_dict()

    # Longer phrases are preferred over shorter ones.
    # This is an O(n^2) solution - may be faster approaches.
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

    # filter out shorter substrings from the original dictionary
    train_dict = {}
    for key in longest_strings:
        train_dict[key] = temp_dict[key]
    return train_dict


def get_tags(train_dict: dict,
             sentence: list,
             lowercase: bool = False) -> list:
    """Greedy approach that looks for tags starting at each token position in the sentence
     and extends the window size one at a time up to a maximum size of max_key_len. If an 
     entity is found in the train_dict, the token read position advances past the found entity
     and continues looking for entities, else the read position advances by one.

    Args:
        train_dict (dict): Dictionary with entities as keys and tags as values, e.g. {('Washington', 'DC'): 'LOC', ...}.
        sentence (list): Tokenized sentence to tag.
        lowercase (bool, optional): Optionally lowercase tokens before looking up entities in train_dict. Defaults to False.

    Returns:
        list: Tagged sentence, which is a list of tuples, e.g. [('Washington', 'LOC'), ('DC', 'LOC'), ('is', 'O), ...].
    """
    # get max key length to limit the search window
    max_key_len = max([len(x) for x in list(train_dict.keys())])

    labeled_sentence = [('<placeholder>', 'O')] * len(sentence)
    i = 0
    while i != len(sentence):
        broke = False
        for j in range(1, max_key_len):
            end_index = min(i + j, len(sentence))
            # optionally lower case when looking up in train_dict
            if lowercase:
                lowercase_sub_sentence = [
                    x.lower() for x in sentence[i:end_index]
                ]
                sub_sentence = tuple(lowercase_sub_sentence)
            else:
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
    """Utility function to increase an ID by increment."""
    if x == np.nan:
        return np.nan
    return x + increment


def add_tag_ids(df):
    """Utility function to backfill NaN IDs."""
    tag_id = df['NER_Tag_ID'].max() + 1
    for idx, row in df.iterrows():
        if math.isnan(row['NER_Tag_ID']):
            df.loc[idx, 'NER_Tag_ID'] = tag_id
            tag_id += 1
    return df


def concat_train_val(train_df, val_df):
    """Utility function to concatenate the train and validation set, while maintaining globally unique ids."""
    # increment id tags
    max_article_id = train_df['Article_ID'].max() + 1
    max_ner_tag_id = train_df['NER_Tag_ID'].max() + 1
    increment_article_id = partial(increment_id, increment=max_article_id)
    increment_tag_id = partial(increment_id, increment=max_ner_tag_id)
    val_df['Article_ID'] = val_df['Article_ID'].apply(increment_article_id)
    val_df['NER_Tag_ID'] = val_df['NER_Tag_ID'].apply(increment_tag_id)

    # combine train/validation set
    train_val_df = pd.concat((train_df, val_df), axis=0)
    return train_val_df


def get_predictions(df: pd.DataFrame,
                    train_dict: dict,
                    lowercase: bool = False) -> pd.DataFrame:
    """Make predictions for each token in the given dataset.

    Args:
        df (pd.DataFrame): DF to make predictions on.
        train_dict (dict): Dictionary with entities as keys and tags as values, e.g. {('Washington', 'DC'): 'LOC', ...}.
        lowercase (bool, optional): Optionally lowercase tokens before looking up entities in train_dict. Defaults to False.

    Returns:
        pd.DataFrame: New DF with token level predictions.
    """
    # transform test sentences to lists
    sentences_df = df.groupby(['Article_ID', 'Sentence_ID'],
                              as_index=False).agg(Sentence=('Token', list))

    # get token tags per sentence
    get_tags_partial = partial(get_tags,
                               train_dict=train_dict,
                               lowercase=lowercase)
    sentences_df['Tagged_Sentence'] = sentences_df['Sentence'].apply(
        lambda x: get_tags_partial(sentence=x))
    sentences_df = sentences_df[[
        'Article_ID', 'Sentence_ID', 'Tagged_Sentence'
    ]]

    # NB: this line takes a long time
    # pivot the sentences into rows
    sentences_df = sentences_df.explode('Tagged_Sentence')

    # unpack the tuples of tokens and tag predictions into separate columns
    sentences_df[['Token', 'NER_Tag_Prediction'
                  ]] = sentences_df['Tagged_Sentence'].apply(pd.Series)
    sentences_df = sentences_df.drop(columns=['Tagged_Sentence'])

    # join back to original df to compare predictions to ground truth
    preds_df = pd.concat(
        (df, sentences_df[['NER_Tag_Prediction']].reset_index(drop=True)),
        axis=1)
    return preds_df


def evaluate(preds_df: pd.DataFrame) -> None:
    """Evaluate entity predictions at the token level and at the entity level.

    Args:
        preds_df (pd.DataFrame): DF to evaluate.
    """
    # evaluate predictions at the token level
    print('Token level performance:')
    print(
        classification_report(preds_df['NER_Tag_Normalized'],
                              preds_df['NER_Tag_Prediction']))

    # add in unique ids so we don't lose the 'O' category in the groupby
    preds_df = add_tag_ids(preds_df)

    # evaluate predictions at the entity level
    # should really be grouping by the predicted entities but this is close enough
    preds_entity_df = preds_df.groupby(
        ['NER_Tag_ID', 'NER_Tag_Normalized'],
        as_index=False)[['Token', 'NER_Tag_Prediction'
                         ]].agg(Tokens=('Token', list),
                                NER_Tag_Predictions=('NER_Tag_Prediction',
                                                     list))

    # majority vote
    preds_entity_df['NER_Tag_Prediction'] = preds_entity_df[
        'NER_Tag_Predictions'].apply(lambda x: max(set(x), key=x.count))
    print('Entity level performance:')
    print(
        classification_report(preds_entity_df['NER_Tag_Normalized'],
                              preds_entity_df['NER_Tag_Prediction']))


def main(args):
    file_names = ['train.csv', 'validation.csv', 'test.csv']

    # load data
    df_dict = utils.load_train_data(args.data_directory)

    # optionally include validation set into the train_dict
    if args.use_validation:
        train_df = concat_train_val(df_dict['train.csv'],
                                    df_dict['validation.csv'])
    else:
        train_df = df_dict['train.csv']

    # create training dictionary
    train_dict = create_train_dict(train_df, args.lowercase)

    # get predictions and evaluate on all sets
    for f in file_names:
        print(f'Evaluating: {f}')
        preds_df = get_predictions(df_dict[f], train_dict, args.lowercase)

        # evaluate predictions
        # NB: evaluating the training set is a good sanity check. Naturally, we
        # expect the performance on the training set to be quite high.
        evaluate(preds_df)
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    parser.add_argument(
        '--use-validation',
        action='store_true',
        default=False,
        help=
        'Optionally include the validation set (in addition to the train set) in the dictionary of entities and tags.'
    )
    parser.add_argument(
        '--lowercase',
        action='store_true',
        default=False,
        help=
        'Optionally lowercase tokens when creating the training dictionary and looking up entities.'
    )

    args = parser.parse_args()

    main(args)
