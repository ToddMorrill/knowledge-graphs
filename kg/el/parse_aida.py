"""This module parses the AIDA CoNLL dataset, which can be found at the link
below:
https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads

Examples:
    $ python parse_aida.py \
        --aida-data-dir /Users/tmorrill002/Documents/datasets/aida-yago2-dataset \
        --save-dir /Users/tmorrill002/Documents/datasets/aida-yago2-dataset/transformed
"""
import argparse
import os
import re

import numpy as np
import pandas as pd


class AIDAParser(object):
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, 'AIDA-YAGO2-dataset.tsv')
        self.doc_id = None  # (e.g. the 1 in (1 EU))
        self.doc_name = None  # (e.g. the EU in (1 EU))

        self.data = []

    @staticmethod
    def _detect_line_type(line):
        if line.startswith('-DOCSTART-'):
            return 'DOCSTART'
        elif line == '':
            return 'NEWSENTENCE'
        else:
            return 'TOKEN'

    def _handle_doc_start(self, line: str) -> None:
        # takes '(1 EU)\n' and sets
        # self.doc_id = 1, self.doc_name = EU
        match_groups = re.findall('(\d+(?:test[ab])?) (.+)', line)[0]
        self.doc_id, self.doc_name = match_groups[0], match_groups[1][:-1]
        # reset self.sentence_number for new document
        # increment within document
        self.sentence_number = 0

    def _handle_new_sentence(self, line) -> None:
        self.sentence_number += 1

    def _probe_value(self, parts, index):
        if index < len(parts):
            return parts[index]
        return '--NME--'

    def _handle_new_token(self, line) -> None:
        parsed_line = []
        # prepend with doc_id, doc_name, and sentence number
        parsed_line.append(self.doc_id)
        parsed_line.append(self.doc_name)
        parsed_line.append(self.sentence_number)

        # parse line
        parts = line.split('\t')
        # guaranteed to at least have a token
        parsed_line.append(parts[0])
        if len(parts) == 1:
            parsed_line.extend([np.nan] * 6)
        # if len(parts) > 1, then will have B/I tag and complete entity
        else:
            # B/I tag
            parsed_line.append(parts[1])
            # complete entity
            parsed_line.append(parts[2])
            # YAGO entity or --NME-- (i.e. no entity found)
            parsed_line.append(parts[3])

            # probe remaining values
            # not sure if YAGO can be --NME-- while other fields are populated
            parsed_line.append(self._probe_value(parts, 4))  # Wikipedia URL
            parsed_line.append(self._probe_value(parts, 5))  # Wikipedia ID
            parsed_line.append(self._probe_value(parts, 6))  # Freebase ID

        # retain document information
        self.data.append(parsed_line)

    def parse(self, return_df=True) -> list:
        # open the file for reading
        self.f = open(self.file_path, 'r')

        process_line_funcs = {
            'DOCSTART': self._handle_doc_start,
            'NEWSENTENCE': self._handle_new_sentence,
            'TOKEN': self._handle_new_token
        }
        for line in self.f.readlines():
            line = line[:-1]  # strip \n
            line_type = self._detect_line_type(line)
            process_line_funcs[line_type](line)

        # close
        self.f.close()
        if return_df:
            columns = [
                'Document_ID', 'Document_Name', 'Sentence_Number', 'Token',
                'BI_Tag', 'Complete_Entity', 'YAGO2_Entity', 'Wikipedia_URL',
                'Wikipedia_ID', 'Freebase_MID'
            ]
            return pd.DataFrame(self.data, columns=columns)

        return self.data


def main(args):
    parser = AIDAParser(args.aida_data_dir)
    aida_df = parser.parse()

    os.makedirs(args.save_dir, exist_ok=True)
    save_file_path = os.path.join(args.save_dir, 'aida_parsed.csv')
    aida_df.to_csv(save_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aida-data-dir',
        required=True,
        help='Directory where the AIDA CoNLL dataset is stored.')
    parser.add_argument(
        '--save-dir',
        help='Directory where a CSV of the parsed file will be stored.')
    args = parser.parse_args()
    main(args)
