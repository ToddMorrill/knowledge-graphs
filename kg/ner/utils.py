import os

import nltk
nltk.download('punkt')  # word tokenizer
nltk.download('averaged_perceptron_tagger')  # pos tagger
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


def parse_document(document, parser, print_tree=False):
    preprocessed_sentences = preprocess(document)
    results = []
    for sentence in preprocessed_sentences:
        result = parser.parse(sentence)
        results.append(result)
        print(result)
        if print_tree:
            result.draw()
    return results


def prepare_report_df(report):
    accuracy = report.pop('accuracy')
    df = pd.DataFrame(report).T
    df.index.name = 'Class'
    df = df.reset_index()
    df.columns = [x.title() for x in df.columns]
    df['Class'] = df['Class'].apply(lambda x: x.title())
    df['Support'] = df['Support'].astype(int)
    return df


def generate_table(df,
                   index=False,
                   column_format=None,
                   caption=None,
                   float_format='%.2f'):
    # column_format='c | c | c',
    # caption=('test caption', 'test'),
    # label='tab:test'
    table_string = df.to_latex(index=index,
                               column_format=column_format,
                               caption=caption,
                               float_format=float_format,
                               bold_rows=True)

    if caption:
        # if you add a caption, it will enclose everything in table environment
        table_split = table_string.split('\n')
        table_split[0] = table_split[0] + '[ht]'  # inline with text
        table_string = '\n'.join(table_split)

    # TODO: remove \toprule, \midrule, \bottomrule, add preferred borders
    # TODO: bold column headers and row labels
    return table_string


def save_table(table_string, file_path):
    with open(file_path, 'w') as f:
        f.write(table_string)


def latex_table(report, file_path):
    df = prepare_report_df(report)
    table_string = generate_table(df)
    save_table(table_string, file_path)