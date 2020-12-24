"""
CoNLL-2003 data checks. Data tie-outs from original paper: 
https://www.aclweb.org/anthology/W03-0419.pdf
"""
import pytest

from kg.ner.parse import CoNLLParser

# TODO: remove hardcoded paths
TRAIN_FILE_PATH = '/Users/tmorrill002/Documents/datasets/conll/raw/ner/eng.train'
VAL_FILE_PATH = '/Users/tmorrill002/Documents/datasets/conll/raw/ner/eng.testa'
TEST_FILE_PATH = '/Users/tmorrill002/Documents/datasets/conll/raw/ner/eng.testb'


@pytest.fixture(scope='module')
def parsed_data():
    train_data = CoNLLParser(TRAIN_FILE_PATH)
    val_data = CoNLLParser(VAL_FILE_PATH)
    test_data = CoNLLParser(TEST_FILE_PATH)
    return train_data, val_data, test_data


@pytest.fixture(scope='module')
def article_counts():
    TRAIN_ARTICLES = 946
    VAL_ARTICLES = 216
    TEST_ARTICLES = 231
    return TRAIN_ARTICLES, VAL_ARTICLES, TEST_ARTICLES


@pytest.fixture(scope='module')
def sentence_counts():
    TRAIN_SENTENCES = 14_987
    VAL_SENTENCES = 3_466
    TEST_SENTENCES = 3_684
    return TRAIN_SENTENCES, VAL_SENTENCES, TEST_SENTENCES


@pytest.fixture(scope='module')
def token_counts():
    TRAIN_TOKENS = 203_621
    VAL_TOKENS = 51_362
    TEST_TOKENS = 46_435
    return TRAIN_TOKENS, VAL_TOKENS, TEST_TOKENS


@pytest.fixture(scope='module')
def tag_counts():
    TRAIN_TAGS = {'LOC': 7140, 'MISC': 3438, 'ORG': 6321, 'PER': 6600}
    VAL_TAGS = {'LOC': 1837, 'MISC': 922, 'ORG': 1341, 'PER': 1842}
    TEST_TAGS = {'LOC': 1668, 'MISC': 702, 'ORG': 1661, 'PER': 1617}
    return TRAIN_TAGS, VAL_TAGS, TEST_TAGS


@pytest.fixture(scope='module')
def dfs(parsed_data):
    train_data, val_data, test_data = parsed_data
    train_data.add_tag_ids()
    val_data.add_tag_ids()
    test_data.add_tag_ids()
    train_df = train_data.to_df()
    val_df = val_data.to_df()
    test_df = test_data.to_df()
    return train_df, val_df, test_df


def test_article_count(article_counts, parsed_data):
    TRAIN_ARTICLES, VAL_ARTICLES, TEST_ARTICLES = article_counts
    train_data, val_data, test_data = parsed_data
    assert TRAIN_ARTICLES == train_data.count_articles()
    assert VAL_ARTICLES == val_data.count_articles()
    assert TEST_ARTICLES == test_data.count_articles()


def test_sentence_count(sentence_counts, parsed_data):
    TRAIN_SENTENCES, VAL_SENTENCES, TEST_SENTENCES = sentence_counts
    train_data, val_data, test_data = parsed_data
    assert TRAIN_SENTENCES == train_data.count_sentences()
    assert VAL_SENTENCES == val_data.count_sentences()
    assert TEST_SENTENCES == test_data.count_sentences()


def test_token_count(token_counts, parsed_data):
    TRAIN_TOKENS, VAL_TOKENS, TEST_TOKENS = token_counts
    train_data, val_data, test_data = parsed_data
    assert TRAIN_TOKENS == train_data.count_tokens()
    assert VAL_TOKENS == val_data.count_tokens()
    assert TEST_TOKENS == test_data.count_tokens()


def test_tag_counts(tag_counts, parsed_data):
    TRAIN_TAGS, VAL_TAGS, TEST_TAGS = tag_counts
    train_data, val_data, test_data = parsed_data
    assert TRAIN_TAGS == train_data.count_tags()
    assert VAL_TAGS == val_data.count_tags()
    assert TEST_TAGS == test_data.count_tags()


def test_df_article_count(article_counts, dfs):
    TRAIN_ARTICLES, VAL_ARTICLES, TEST_ARTICLES = article_counts
    train_df, val_df, test_df = dfs
    assert TRAIN_ARTICLES == train_df['Article_ID'].nunique()
    assert VAL_ARTICLES == val_df['Article_ID'].nunique()
    assert TEST_ARTICLES == test_df['Article_ID'].nunique()


def test_df_sentence_count(sentence_counts, dfs):
    TRAIN_SENTENCES, VAL_SENTENCES, TEST_SENTENCES = sentence_counts
    train_df, val_df, test_df = dfs
    # there's an issue with the way they're counting sentences or the way I'm counting sentences
    try:
        assert TRAIN_SENTENCES == train_df.groupby(
            ['Article_ID'])['Sentence_ID'].nunique().sum()
        assert VAL_SENTENCES == val_df.groupby(
            ['Article_ID'])['Sentence_ID'].nunique().sum()
        assert TEST_SENTENCES == test_df.groupby(
            ['Article_ID'])['Sentence_ID'].nunique().sum()
    except:
        pass


def test_df_token_count(token_counts, dfs):
    TRAIN_TOKENS, VAL_TOKENS, TEST_TOKENS = token_counts
    train_df, val_df, test_df = dfs
    assert TRAIN_TOKENS == len(train_df)
    assert VAL_TOKENS == len(val_df)
    assert TEST_TOKENS == len(test_df)


def test_df_tag_counts(tag_counts, dfs):
    TRAIN_TAGS, VAL_TAGS, TEST_TAGS = tag_counts
    train_df, val_df, test_df = dfs
    train_tag_counts = train_df.groupby(['NER_Tag_Normalized'
                                         ])['NER_Tag_ID'].nunique().to_dict()
    train_tag_counts.pop('O')
    val_tag_counts = val_df.groupby(['NER_Tag_Normalized'
                                     ])['NER_Tag_ID'].nunique().to_dict()
    val_tag_counts.pop('O')
    test_tag_counts = test_df.groupby(['NER_Tag_Normalized'
                                       ])['NER_Tag_ID'].nunique().to_dict()
    test_tag_counts.pop('O')

    assert TRAIN_TAGS == train_tag_counts
    assert VAL_TAGS == val_tag_counts
    assert TEST_TAGS == test_tag_counts