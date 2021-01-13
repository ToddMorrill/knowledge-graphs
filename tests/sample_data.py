"""Create a small sample from the train.csv file that can be stored on Github.  
"""
import random

import pandas as pd

random.seed(42)

train_file_path = '/Users/tmorrill002/Documents/datasets/conll/transformed/train.csv'
val_file_path = '/Users/tmorrill002/Documents/datasets/conll/transformed/validation.csv'
test_file_path = '/Users/tmorrill002/Documents/datasets/conll/transformed/test.csv'

train_df = pd.read_csv(train_file_path)
val_df = pd.read_csv(val_file_path)
test_df = pd.read_csv(test_file_path)


def save_sample(df, split):
    max_id = df['Article_ID'].max()

    articles = []
    for _ in range(min(len(df), 10)):
        articles.append(random.randint(0, max_id))

    df_sample = df[df['Article_ID'].isin(articles)]
    df_sample = df_sample.drop_duplicates()
    df_sample.to_csv(f'data_samples/{split}.csv', index=False)


save_sample(train_df, 'train')
save_sample(val_df, 'validation')
save_sample(test_df, 'test')