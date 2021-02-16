import os

import nltk
import pandas as pd


def load_train_data(data_directory: str) -> dict:
    """Loads training data from train.csv, validation.csv, and test.csv, returning a dictionary of DFs.

    Args:
        data_directory (str): Directory containing the training data CSV files.

    Returns:
        dict: Dictionary of DFs.
    """
    file_names = ['train.csv', 'validation.csv', 'test.csv']

    # load data
    df_dict = {}
    for file_name in file_names:
        file_path = os.path.join(data_directory, file_name)
        df_dict[file_name] = pd.read_csv(file_path)

    return df_dict


def tokenize_text(document):
    return nltk.sent_tokenize(document)


def tokenize_sentences(sentences):
    return [nltk.word_tokenize(sentence) for sentence in sentences]


def tag_pos(sentences):
    return [nltk.pos_tag(sentence) for sentence in sentences]


def preprocess(document):
    sentences = tokenize_text(document)
    sentences = tokenize_sentences(sentences)
    sentences = tag_pos(sentences)
    return sentences


def extract_noun_chunks(document, parser, print_tree=False):
    preprocessed_sentences = preprocess(document)
    results = []
    for sentence in preprocessed_sentences:
        result = parser.parse(sentence)
        results.append(result)
        print(result)
        if print_tree:
            result.draw()
    return results