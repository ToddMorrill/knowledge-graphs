"""
This module contains preprocessing code to prepare data for training and inference.

Examples:
    $ python preprocess.py \
        --config configs/baseline.yaml
"""

import argparse
from collections import Counter
import os
from types import SimpleNamespace

import pandas as pd
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
import yaml
from yaml import parser


class CoNLL2003Dataset(torch.utils.data.Dataset):
    """Custom dataset to contain the CoNLL2003 dataset.
    """
    def __init__(self, df: pd.DataFrame, transform: list = None) -> None:
        """Initializes the dataset and prepares sequences of tokens and labels.

        Args:
            df (pd.DataFrame): DF containing training examples.
            transform (list, optional): List of transforms (e.g. index lookups, etc.). Defaults to None.
        """
        self.df = df
        self.transform = transform
        self.sentences, self.labels = self._prepare_data()

    def _prepare_data(self) -> tuple:
        """Groups data into sequences of tokens and labels.

        Returns:
            tuple: sentences, labels
        """
        temp_df = self.df.groupby(['Article_ID', 'Sentence_ID'],
                                  as_index=False).agg(
                                      Sentence=('Token', list),
                                      Labels=('NER_Tag_Normalized', list))
        sentences = temp_df['Sentence'].values.tolist()
        labels = temp_df['Labels'].values.tolist()
        return sentences, labels

    def __len__(self) -> int:
        """Retrieve the length of the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.sentences)

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves the idx item from the dataset, potentially transformed.

        Args:
            idx (int): idx item from the dataset.

        Returns:
            tuple: sentences, labels
        """
        if self.transform is None:
            return self.sentences[idx], self.labels[idx]

        # TODO: probably should wrap this in a for-loop
        indices = self.transform[0](self.sentences[idx])
        labels = self.transform[1](self.labels[idx])

        return indices, labels


class Preprocessor(object):
    """Preproccessor class to handle data preparation at train and inference time.
    """
    def __init__(self, config: str) -> None:
        """Initialize the preprocessor and generate vocabulary and label dictionary based on the training set.

        Args:
            config (str): File path to the configuration yaml file.
        """
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        self.config = SimpleNamespace(**config)

        self.vocab, self.label_dict = self._create_vocabs()
        self.idx_to_label = {v: k for k, v in self.label_dict.items()}

    def _create_vocabs(self) -> tuple:
        """Generate vocabulary object and label dictionary.

        Returns:
            tuple: vocab, label_dict
        """
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

    @staticmethod
    def _collate_fn(batch: tuple, train: bool = True) -> tuple:
        """Custom collate function that combines variable length sequences into padded batches.

        Args:
            batch (tuple): sentence_indices, sentences_labels OR just sentences_indices (a list).
            train (bool, optional): If train=True, expects tuple of 
            sentence_indices, sentences_labels, else just a list of sentence_indices. Defaults to True.

        Returns:
            tuple: (sentences_padded, sentence_lens), labels_padded if train=True, else (sentences_padded, sentence_lens).
        """
        if train:
            sentence_indices, sentence_labels = zip(*batch)
        else:
            sentence_indices = batch
        sentence_lens = [len(x) for x in sentence_indices]

        # vocab['<pad>'] = 1
        sentences_padded = pad_sequence(sentence_indices,
                                        batch_first=True,
                                        padding_value=1)
        if train:
            labels_padded = pad_sequence(sentence_labels,
                                         batch_first=True,
                                         padding_value=-1)

            return (sentences_padded, sentence_lens), labels_padded
        else:
            return (sentences_padded, sentence_lens)

    def get_train_datasets(self) -> tuple:
        """Generates all the datasets needed for model training.

        Returns:
            tuple: train_dataset, val_dataset, test_dataset
        """
        train_file_path = os.path.join(self.config.data_dir, 'train.csv')
        val_file_path = os.path.join(self.config.data_dir, 'validation.csv')
        test_file_path = os.path.join(self.config.data_dir, 'test.csv')

        transform = [self._transform_sentence, self._transform_labels]

        train_dataset = CoNLL2003Dataset(pd.read_csv(train_file_path),
                                         transform)
        val_dataset = CoNLL2003Dataset(pd.read_csv(val_file_path), transform)
        test_dataset = CoNLL2003Dataset(pd.read_csv(test_file_path), transform)
        return train_dataset, val_dataset, test_dataset

    def get_train_dataloaders(self) -> tuple:
        """Generates all the dataloaders needed for model training.

        Returns:
            tuple: train_dataloader, val_dataloader, test_dataloader
        """
        train_dataset, val_dataset, test_dataset = self.get_train_datasets()
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn,
            shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn)
        return train_dataloader, val_dataloader, test_dataloader

    @staticmethod
    def _tokenize(sentence: str) -> list:
        """Utility function to tokenize sentences.

        Args:
            sentence (str): Sentence string.

        Returns:
            list: Tokenized sentence.
        """
        return sentence.split(' ')

    def _transform_sentence(self, sentence: list) -> torch.tensor:
        """Transform function that accepts a sentence as a string or tokenized list and returns vocabulary indices.

        Args:
            sentence (list): Tokenized list or sentence string.

        Returns:
            torch.tensor: Vocabulary indices.
        """
        if isinstance(sentence, str):
            sentence = self._tokenize(sentence)
        indices = []
        for token in sentence:
            indices.append(self.vocab[token])
        return torch.tensor(indices)

    def _transform_labels(self, label_sequence: list) -> torch.tensor:
        """Transform function that accepts a sequence of labels and returns label indices.

        Args:
            label_sequence (list): Sequence of string labels.

        Returns:
            torch.tensor: Label indices.
        """
        labels = []
        for label in label_sequence:
            labels.append(self.label_dict[label])
        return torch.tensor(labels)

    def preprocess(self, sentences: list) -> tuple:
        """Preprocess any arbitrary list of string sentences and return indices that can be fed into the model.

        Args:
            sentences (list): List of sentences to tokenize and retrieve indices for.

        Returns:
            tuple: (sentences_padded, sentence_lens)
        """
        # TODO: see if there is a way to reuse the CoNLL2003Dataset class + dataloaders
        # for guaranteed consistency with the way that we're preparing training data
        preprocessed = []
        if isinstance(sentences, str):
            preprocessed.append(self._transform_sentence(sentences))
        else:
            for sentence in sentences:
                preprocessed.append(self._transform_sentence(sentence))
        return self._collate_fn(preprocessed, train=False)


def main(args):
    # contains vocab and label_dict embedded in the transform function
    preprocessor = Preprocessor(args.config)
    sample_sentence = 'Todd Morrill lives in New York City.'
    prepared_sentence = preprocessor.preprocess(sample_sentence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='File path where the model configuration file is located.',
        required=True)
    args = parser.parse_args()
    main(args)