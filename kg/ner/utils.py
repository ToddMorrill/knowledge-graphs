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


def tokenize_sentences(sentences: list) -> list:
    """Word tokenize each sentence in a list of sentences.

    Args:
        sentences (list): List of sentences.

    Returns:
        list: List of tokenized sentences.
    """
    return [nltk.word_tokenize(sentence) for sentence in sentences]


def tag_pos(sentences: list) -> list:
    """Tag parts of speech of tokens in each sentence.

    Args:
        sentences (list): List of word tokenized sentences.

    Returns:
        list: List of part-of-speech tagged sentences.
    """
    return [nltk.pos_tag(sentence) for sentence in sentences]


def preprocess(document: str) -> list:
    """Preprocess document by splitting sentences, tokenizing sentences, and 
    tagging parts-of-speech.

    Args:
        document (str): String document.

    Returns:
        list: Preprocessed document.
    """
    sentences = nltk.sent_tokenize(document)
    sentences = tokenize_sentences(sentences)
    sentences = tag_pos(sentences)
    return sentences


def parse_document(document: str, parser, print_tree: bool = False) -> list:
    """Parse a document and generate an nltk.Tree for each sentence.

    Args:
        document (str): String document.
        parser (TBD): nltk style parser that returns a parse tree.
        print_tree (bool, optional): Optionally display the parse tree for each sentence. Defaults to False.

    Returns:
        list: List of nltk.Trees.
    """
    preprocessed_sentences = preprocess(document)
    results = []
    for sentence in preprocessed_sentences:
        result = parser.parse(sentence)
        results.append(result)
        print(result)
        if print_tree:
            result.draw()
    return results


def prepare_report_df(report: dict) -> pd.DataFrame:
    """Convert sklearn classification report to a dataframe. 

    Args:
        report (dict): sklearn classification report.

    Returns:
        pd.DataFrame: Formatted dataframe.
    """
    accuracy = report.pop('accuracy')
    df = pd.DataFrame(report).T
    df.index.name = 'Class'
    df = df.reset_index()
    df.columns = [x.title() for x in df.columns]
    df['Class'] = df['Class'].apply(lambda x: x.title())
    # remove column name
    df = df.rename(columns={'Class': ''})
    df['Support'] = df['Support'].astype(int)
    return df


def generate_table(df: pd.DataFrame,
                   index: bool = False,
                   column_format: str = None,
                   caption: str = None,
                   float_format: str = '%.2f') -> str:
    """Generate a LaTeX table based on the passed dataframe.

    Args:
        df (pd.DataFrame): Dataframe.
        index (bool, optional): If True, include the dataframe index in the LaTeX table. Defaults to False.
        column_format (str, optional): LaTeX column format (e.g. 'c | c | c'). Defaults to None.
        caption (str, optional): Table caption. Defaults to None.
        float_format (str, optional): Formatting for floats. Defaults to '%.2f'.

    Returns:
        str: [description]
    """
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


def save_table(table_string: str, file_path: str) -> None:
    """Save the passed LaTeX table.

    Args:
        table_string (str): LaTeX table.
        file_path (str): File destination.
    """
    with open(file_path, 'w') as f:
        f.write(table_string)


def latex_table(report: dict, file_path: str) -> None:
    """Convert an sklearn style classification report into a LaTeX table and 
    save the result.

    Args:
        report (dict): sklearn style classification report.
        file_path (str): File destination.
    """
    df = prepare_report_df(report)
    table_string = generate_table(df)
    save_table(table_string, file_path)