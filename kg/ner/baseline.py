"""
This module will implement a baseline NER tagger.

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
"""

import argparse
import os

import pandas as pd


def create_train_dict(df):
    temp_df = df.groupby(['NER_Tag_ID', 'NER_Tag_Normalized'],
                         as_index=False).agg(Entity=('Token', tuple))
    temp_df = temp_df[['Entity', 'NER_Tag_Normalized']]
    # drop all duplicates
    temp_df = temp_df.drop_duplicates(subset='Entity', keep=False)
    temp_dict = pd.Series(temp_df['NER_Tag_Normalized'].values,
                          index=temp_df['Entity']).to_dict()
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
    labeled_sentence = [('placeholder', 'placeholder')] * len(sentence)
    i = 0
    while i != len(sentence):
        for j in range(1, max_key_len):
            end_index = min(i+j, len(sentence))
            sub_sentence = tuple(sentence[i:end_index])
            if sub_sentence in train_dict:
                entity_type = train_dict[sub_sentence]
                tags = [(tok, entity_type) for tok in sub_sentence]
                labeled_sentence[i:end_index] = tags
                i += j
                break
            else:
                tags = [(tok, 'O') for tok in sub_sentence]
                labeled_sentence[i:end_index] = tags
        i += 1
    return labeled_sentence

def main(args):
    file_names = ['train.csv', 'validation.csv', 'test.csv']

    # load data
    df_dict = {}
    for file_name in file_names:
        file_path = os.path.join(args.data_directory, file_name)
        df_dict[file_name] = pd.read_csv(file_path)

    # create training dictionary
    train_dict = create_train_dict(df_dict['train.csv'])

    # get max key length
    max_key_len = max([len(x) for x in list(train_dict.keys())])
    
    # transform test sentences
    test_sentences_df = df_dict['test.csv'].groupby(['Sentence_ID'], as_index=False).agg(Sentence=('Token', list))
    test_sentence = test_sentences_df.iloc[0]['Sentence']
    pred_sentence = get_tags(train_dict, test_sentence, max_key_len)
    breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')

    args = parser.parse_args()

    main(args)
