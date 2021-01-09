"""
This module contains a class to parse CoNLL-2003 data into a user-friendly format.


Examples:
    $ python parse.py \
        --data-directory /Users/tmorrill002/Documents/datasets/conll/raw/ner \
        --save-directory /Users/tmorrill002/Documents/datasets/conll/transformed
"""
import argparse
import os

import pandas as pd
import numpy as np


class CoNLLParser(object):
    """This class parses CoNLL-2003 English data into a user-friendly format and 
    provides facilities to check the validity of the parsed data.
    """
    def __init__(self, data_file_path: str) -> None:
        """Initializes an instance of the CoNLLParser and parses the data.

        Args:
            data_file_path (str): Fully specified file path to a portion (e.g. train, test, etc.) 
            of the CoNLL-2003 data.
        """
        self.data_file_path = data_file_path
        self._read_data()
        self._split_documents()
        self._split_sentences()
        self._split_tokens()
        self._split_tags()

    def _read_data(self) -> None:
        """Read raw data.
        """
        with open(self.data_file_path, 'r') as f:
            self.data = f.read()

    def _split_documents(self) -> None:
        """Split document by the delimiters.
        """
        temp = self.data.split('-DOCSTART- -X- O O\n\n')
        if len(temp) > 1:
            # train/val set
            self.data = temp[1:]
        else:
            # test set
            self.data = self.data.split('-DOCSTART- -X- -X- O\n\n')[1:]

    def _split_sentences(self) -> None:
        """Split sentences by the delimiters.
        """
        split_data = []
        for doc in self.data:
            split_data.append(doc.split('\n\n'))
        self.data = split_data

    def _split_tokens(self) -> None:
        """Split tokens by the delimiters.
        """
        split_data = []
        for doc in self.data:
            split_doc = []
            for sentence in doc:
                tokens = sentence.split('\n')
                # remove blank lines
                tokens = [tok for tok in tokens if tok != '']
                split_doc.append(tokens)
            split_data.append(split_doc)
        self.data = split_data

    def _split_tags(self) -> None:
        """Split tags by the delimiters.
        """
        split_data = []
        for doc in self.data:
            split_doc = []
            for sentence in doc:
                split_sentence = []
                for example in sentence:
                    tags = example.split(' ')
                    split_sentence.append(tags)
                split_doc.append(split_sentence)
            split_data.append(split_doc)
        self.data = split_data

    def count_articles(self) -> int:
        """Count the number of unique articles.

        Returns:
            int: Count of unique articles.
        """
        return len(self.data)

    def count_sentences(self) -> int:
        """Count the number of unique sentences.

        Returns:
            int: Count of unique sentences.
        """
        return sum([len(doc) for doc in self.data])

    def count_tokens(self) -> int:
        """Count the number of tokens.

        Returns:
            int: Count of tokens.
        """
        token_count = 0
        for doc in self.data:
            for sentence in doc:
                token_count += len(sentence)
        return token_count

    def _increment_tag(self, tag_type: str, i: int, sentence: list) -> int:
        """Utility function to increment the count of a particular entity type.

        Args:
            tag_type (str): Type of tag (e.g. 'LOC', 'MISC', 'ORG', 'PER').
            i (int): Index of the token position in the sentence.
            sentence (list): List of lists of the form: [['token1', pos_tag, chunk_tag, ner_tag], ['token2', ...]].

        Returns:
            int: Updated index of the token position in the sentence.
        """
        self.tag_counts[tag_type] += 1
        ner_tag = f'I-{tag_type}'
        while ner_tag == f'I-{tag_type}':
            # advance the index by one and check the ner_tag
            i += 1
            if i == len(sentence):
                break
            word, pos_tag, chunk_tag, ner_tag = sentence[i][:4]
        return i

    def count_tags(self) -> dict:
        """Count the number of unique entities by type.

        Returns:
            dict: Counts of entities by type.
        """
        self.tag_counts = {'LOC': 0, 'MISC': 0, 'ORG': 0, 'PER': 0}
        for doc in self.data:
            for sentence in doc:
                i = 0
                while i != len(sentence):
                    word, pos_tag, chunk_tag, ner_tag = sentence[i][:4]

                    # check if it's a LOC tag
                    if ner_tag == 'I-LOC' or ner_tag == 'B-LOC':
                        i = self._increment_tag('LOC', i, sentence)

                    # check if it's a MISC tag
                    elif ner_tag == 'I-MISC' or ner_tag == 'B-MISC':
                        i = self._increment_tag('MISC', i, sentence)

                    # check if it's an ORG tag
                    elif ner_tag == 'I-ORG' or ner_tag == 'B-ORG':
                        i = self._increment_tag('ORG', i, sentence)

                    # check if it's an PER tag
                    elif ner_tag == 'I-PER' or ner_tag == 'B-PER':
                        i = self._increment_tag('PER', i, sentence)

                    # O tag
                    else:
                        i += 1
        return self.tag_counts

    def _add_tag_id(self, tag_type: str, j: int, k: int, i: int,
                    sentence: list, global_tag_id: int) -> tuple:
        """Utility function to increment the count of a particular entity type.

        Args:
            tag_type (str): Type of tag (e.g. 'LOC', 'MISC', 'ORG', 'PER').
            j (int): Index of the article.
            k (int): Index of the sentence in the article.
            i (int): Index of the token position in the sentence.
            sentence (list): List of lists of the form: [['token1', pos_tag, chunk_tag, ner_tag], ['token2', ...]].
            global_tag_id (int): Global tag ID among all the entities.

        Returns:
            tuple: Updated index of the token position in the sentence, and the global tag ID.
        """
        temp_tag = f'I-{tag_type}'
        m = 0
        while temp_tag == f'I-{tag_type}':
            if m != 0:
                self.data[j][k][i] = [
                    word, pos_tag, chunk_tag, temp_tag, global_tag_id
                ]
            # advance the index by one and check the ner_tag
            i += 1
            if i == len(sentence):
                break
            word, pos_tag, chunk_tag, temp_tag = sentence[i][:4]
            m += 1
        global_tag_id += 1
        return i, global_tag_id

    def add_tag_ids(self) -> None:
        """Modifies the data and adds entity tag IDs.
        """
        print('NB: this will modify raw the data.')
        global_tag_id = 0
        for j, doc in enumerate(self.data):
            for k, sentence in enumerate(doc):
                i = 0
                while i != len(sentence):
                    word, pos_tag, chunk_tag, ner_tag = sentence[i][:4]

                    # check if it's a LOC tag
                    if ner_tag == 'I-LOC' or ner_tag == 'B-LOC':
                        self.data[j][k][i] = [
                            word, pos_tag, chunk_tag, ner_tag, global_tag_id
                        ]
                        i, global_tag_id = self._add_tag_id(
                            'LOC', j, k, i, sentence, global_tag_id)

                    # check if it's a MISC tag
                    elif ner_tag == 'I-MISC' or ner_tag == 'B-MISC':
                        self.data[j][k][i] = [
                            word, pos_tag, chunk_tag, ner_tag, global_tag_id
                        ]
                        i, global_tag_id = self._add_tag_id(
                            'MISC', j, k, i, sentence, global_tag_id)

                    # check if it's an ORG tag
                    elif ner_tag == 'I-ORG' or ner_tag == 'B-ORG':
                        self.data[j][k][i] = [
                            word, pos_tag, chunk_tag, ner_tag, global_tag_id
                        ]
                        i, global_tag_id = self._add_tag_id(
                            'ORG', j, k, i, sentence, global_tag_id)

                    # check if it's an PER tag
                    elif ner_tag == 'I-PER' or ner_tag == 'B-PER':
                        self.data[j][k][i] = [
                            word, pos_tag, chunk_tag, ner_tag, global_tag_id
                        ]
                        i, global_tag_id = self._add_tag_id(
                            'PER', j, k, i, sentence, global_tag_id)

                    # O tag
                    else:
                        if i == len(sentence):
                            break
                        word, pos_tag, chunk_tag, ner_tag = sentence[i][:4]
                        self.data[j][k][i] = [
                            word, pos_tag, chunk_tag, ner_tag, np.nan
                        ]
                        i += 1

    def _normalize_data(self) -> list:
        """Flatten list of lists and add IDs so that this function is invertible.
        Each element of the normalized list is a token with it's corresponding IDs and tags.

        Returns:
            list: Flattened list of tokens and tags.
        """
        normalized_data = []
        for i, doc in enumerate(self.data):
            for j, sentence in enumerate(doc):
                for k, example in enumerate(sentence):
                    temp = [i, j, k] + example
                    normalized_data.append(temp)
        return normalized_data

    def _normalize_tag(self, x: str) -> str:
        """Convert tags to a uniform format by stripping out 'I-' and 'B-'.

        Args:
            x (str): IOB style tag (e.g. 'I-LOC' or 'B-LOC').

        Returns:
            str: Normalized entity tag.
        """
        if x == 'O':
            return 'O'
        else:
            # e.g. 'I-LOC' -> 'LOC' or 'B-LOC' -> 'LOC'
            return x[2:]

    def to_df(self) -> pd.DataFrame:
        """Converts parsed CoNLL-2003 data into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DF of parsed data.
        """
        cols = [
            'Article_ID', 'Sentence_ID', 'Token_ID', 'Token', 'POS_Tag',
            'Chunk_Tag', 'NER_Tag', 'NER_Tag_ID'
        ]
        normalized_data = self._normalize_data()

        if len(normalized_data[0]) == len(cols):
            self.df = pd.DataFrame(normalized_data, columns=cols)
        # add_tag_ids has not been called yet
        else:
            self.df = pd.DataFrame(normalized_data, columns=cols[:-1])

        self.df['NER_Tag_Normalized'] = self.df['NER_Tag'].apply(
            lambda x: self._normalize_tag(x))
        return self.df

    def save(self, save_directory: str, data_split: str) -> None:
        """Save CSV file of the DF of parsed data.

        Args:
            save_directory (str): Directory where the data will be saved.
            data_split (str): Split of the data (e.g. 'train', 'validation', 'test').
        """
        if not hasattr(self, 'df'):
            _ = self.to_df()
        os.makedirs(save_directory, exist_ok=True)
        save_file_path = os.path.join(save_directory, f'{data_split}.csv')
        self.df.to_csv(save_file_path, index=False)


def main(args):
    train_file_path = os.path.join(args.data_directory, 'eng.train')
    val_file_path = os.path.join(args.data_directory, 'eng.testa')
    test_file_path = os.path.join(args.data_directory, 'eng.testb')

    train_data = CoNLLParser(train_file_path)
    val_data = CoNLLParser(val_file_path)
    test_data = CoNLLParser(test_file_path)

    # TBD: do I need to save a pickle of the parsed data in list format?
    # or is the .csv format sufficient?
    train_data.add_tag_ids()
    val_data.add_tag_ids()
    test_data.add_tag_ids()

    train_df = train_data.to_df()
    val_df = val_data.to_df()
    test_df = test_data.to_df()

    train_data.save(args.save_directory, 'train')
    val_data.save(args.save_directory, 'validation')
    test_data.save(args.save_directory, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    parser.add_argument(
        '--save-directory',
        type=str,
        help='Destination directory where the parsed data will be saved.')
    args = parser.parse_args()

    main(args)