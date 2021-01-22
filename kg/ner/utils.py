import os

import pandas as pd


def load_train_data(data_directory):
    file_names = ['train.csv', 'validation.csv', 'test.csv']

    # load data
    df_dict = {}
    for file_name in file_names:
        file_path = os.path.join(data_directory, file_name)
        df_dict[file_name] = pd.read_csv(file_path)

    return df_dict