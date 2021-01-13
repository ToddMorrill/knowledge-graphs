"""
preprocess.py tests.
"""
from random import sample
import pandas as pd
import pytest

from kg.ner.preprocess import CoNLL2003Dataset, Preprocessor


@pytest.fixture(scope='module')
def sample_train_data():
    train_df = pd.read_csv('data_samples/train.csv')
    return train_df

@pytest.fixture(scope='module')
def train_dataset_no_transform(sample_train_data):
    train_dataset = CoNLL2003Dataset(sample_train_data)
    return train_dataset

def test_dataset_len(sample_train_data, train_dataset_no_transform):
    # check that the length of the train_dataset is as expected
    assert len(train_dataset_no_transform) == len(sample_train_data.groupby(['Article_ID', 'Sentence_ID']))

def test_dataset_getitem_no_transform(sample_train_data, train_dataset_no_transform):
    first_sentence_idx = sample_train_data.iloc[0]['Article_ID']
    mask = (sample_train_data['Article_ID'] == first_sentence_idx) & (sample_train_data['Sentence_ID'] == 0)
    first_sentence_df = sample_train_data[mask]
    tokens = first_sentence_df['Token'].values.tolist()
    labels = first_sentence_df['NER_Tag_Normalized'].values.tolist()
    assert train_dataset_no_transform[0] == (tokens, labels)

@pytest.fixture(scope='module')
def preprocessor():
    preprocessor_instance = Preprocessor('configs/baseline.yaml')
    return preprocessor_instance

def test_get_train_datasets(preprocessor):
    datasets = preprocessor.get_train_datasets()
    assert (len(datasets[0]) > 0) and (len(datasets[1]) > 0) and (len(datasets[2]) > 0)

def test_get_train_dataloaders(preprocessor):
    dataloaders = preprocessor.get_train_dataloaders()
    assert (len(next(iter(dataloaders[0]))) > 0) and (len(next(iter(dataloaders[1]))) > 0) and (len(next(iter(dataloaders[2]))) > 0)

def test_preprocess(preprocessor):
    sample_sentence = 'Todd Morrill lives in New York City.'
    word_index_tensor, lengths = preprocessor.preprocess(sample_sentence)
    sample_sentence_len = len(sample_sentence.split(' '))
    assert (len(word_index_tensor[0]) == sample_sentence_len) and (sample_sentence_len == lengths[0])