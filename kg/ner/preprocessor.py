"""
This module contains preprocessing code to prepare data for training and inference.

Examples:
    $ python preprocessor.py \
        --config configs/baseline.yaml
"""

import argparse
from collections import Counter
import os
from types import SimpleNamespace

import pandas as pd
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import yaml


class CoNLL2003Dataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, label_dict, transform=None):
        self.df = df
        self.vocab = vocab
        self.label_dict = label_dict
        self.transform = transform
        self.sentences, self.labels = self._prepare_data()

    def _prepare_data(self):
        # groups data into sequences of tokens and labels
        temp_df = self.df.groupby(['Article_ID', 'Sentence_ID'],
                                  as_index=False).agg(
                                      Sentence=('Token', list),
                                      Labels=('NER_Tag_Normalized', list))
        sentences = temp_df['Sentence'].values.tolist()
        labels = temp_df['Labels'].values.tolist()
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        if self.transform:
            raise NotImplementedError

        indices = []
        for token in self.sentences[idx]:
            indices.append(self.vocab[token])
        labels = []
        for label in self.labels[idx]:
            labels.append(self.label_dict[label])

        return torch.tensor(indices), torch.tensor(labels)


class Preprocessor(object):
    def __init__(self, config):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        self.config = SimpleNamespace(**config)

        self.vocab, self.label_dict = self._create_vocabs()

    def _create_vocabs(self):
        # load train data to build the dictionaries
        train_df = pd.read_csv(os.path.join(self.config.data_dir, 'train.csv'))

        # create vocabulary
        vocab = torchtext.vocab.Vocab(
            Counter(train_df['Token'].value_counts().to_dict()))

        # create label dictionary
        label_dict = {}
        i = 0
        for k in train_df['NER_Tag_Normalized'].unique():
            label_dict[k] = i
            i += 1
        return vocab, label_dict

    def _load_csv(self, file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def _collate_fn(batch):
        sentence_indices, sentence_labels = zip(*batch)
        sentence_lens = [len(x) for x in sentence_indices]

        # vocab['<pad>'] = 1
        sentences_padded = pad_sequence(sentence_indices,
                                        batch_first=True,
                                        padding_value=1)
        labels_padded = pad_sequence(sentence_labels,
                                     batch_first=True,
                                     padding_value=-1)

        return (sentences_padded, sentence_lens), labels_padded

    def get_train_datasets(self):
        train_file_path = os.path.join(self.config.data_dir, 'train.csv')
        val_file_path = os.path.join(self.config.data_dir, 'validation.csv')
        test_file_path = os.path.join(self.config.data_dir, 'test.csv')

        train_dataset = CoNLL2003Dataset(
            self._load_csv(train_file_path), self.vocab, self.label_dict)
        val_dataset = CoNLL2003Dataset(
            self._load_csv(val_file_path), self.vocab, self.label_dict)
        test_dataset = CoNLL2003Dataset(
            self._load_csv(test_file_path), self.vocab, self.label_dict)
        return train_dataset, val_dataset, test_dataset

    def get_train_dataloaders(self):
        train_dataset, val_dataset, test_dataset = self.get_train_datasets()
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=16,
            collate_fn=Preprocessor._collate_fn,
            shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=16, collate_fn=Preprocessor._collate_fn)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, collate_fn=Preprocessor._collate_fn)
        return train_dataloader, val_dataloader, test_dataloader
