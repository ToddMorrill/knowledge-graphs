"""Evaluate the named entity recognition procedures.

Examples:
    $ python evaluate.py \
        --data-directory /Users/tmorrill002/Documents/datasets/conll/transformed
"""
import argparse
import os
import itertools

import matplotlib.pyplot as plt
import nltk
from nltk import text
from nltk.corpus import conll2000
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, classification_report

import kg.ner.utils as utils
from kg.ner.unsupervised import NounPhraseDetector, TFIDFScorer, TextRankScorer
from kg.ner.supervised import BigramChunker, MaxEntChunker, PretrainedNEDetector


def nouns_over_NER(df, noun_col='Chunk_Tag', ner_col='NER_Tag_Normalized'):
    """What fraction of NER tags are Noun Phrases?"""
    print('What fraction of NER tags are Noun Phrases?')
    ner_df = df[df[ner_col] != 'O']
    noun_phrase_token_count = len(ner_df[(ner_df[noun_col] == 'I-NP') |
                                         (ner_df[noun_col] == 'B-NP')])
    print(
        f'Count of noun phrase tokens among NER tokens: {noun_phrase_token_count}'
    )
    print(f'Count of NER tokens: {len(ner_df)}')
    print(
        f'Percent of NER tokens that are part of noun phrases: {round(noun_phrase_token_count / len(ner_df),4) * 100}%'
    )
    print()


def NER_over_nouns(df, noun_col='Chunk_Tag', ner_col='NER_Tag_Normalized'):
    """What fraction of Noun Phrase tokens are NER tagged?"""
    print('What fraction of Noun Phrase tokens are NER tagged?')
    noun_phrase_df = df[(df[noun_col] == 'I-NP') | (df[noun_col] == 'B-NP')]
    ner_tag_token_count = len(noun_phrase_df[noun_phrase_df[ner_col] != 'O'])
    print(
        f'Count of NER tokens among noun phrase tokens: {ner_tag_token_count}')
    print(f'Count of noun phrase tokens: {len(noun_phrase_df)}')
    print(
        f'Percent of noun phrase tokens that are part of NER tags: {round(ner_tag_token_count / len(noun_phrase_df), 4) * 100}%'
    )
    print()


def NER_tags_noun_phrases(df):
    conclusions = """
    Conclusions:
    1. NER Tags are almost exclusively noun phrases (97%), which means that noun 
        phrase candidates will yield high recall downstream tasks.
    2. Noun phrases encompass a lot more than NER tags, which means noun phrase 
        candidates will yield low precision and other techniques should be used 
        to reduce the number of false positives in downstream tasks.
    """
    nouns_over_NER(df)
    NER_over_nouns(df)
    print(conclusions)


def proper_nouns_over_NER(df,
                          noun_col='POS_Tag',
                          ner_col='NER_Tag_Normalized'):
    """What fraction of NER tags are proper nouns?"""
    print('What fraction of NER tags are proper nouns?')
    ner_df = df[df[ner_col] != 'O']
    nnp_token_count = len(ner_df[ner_df[noun_col] == 'NNP'])
    print(f'Count of proper noun tokens among NER tokens: {nnp_token_count}')
    print(f'Count of NER tokens: {len(ner_df)}')
    print(
        f'Percent of NER tokens that are proper noun tokens: {round(nnp_token_count / len(ner_df),4) * 100}%'
    )
    print()


def NER_over_proper_nouns(df,
                          noun_col='POS_Tag',
                          ner_col='NER_Tag_Normalized'):
    """What fraction of proper noun tokens are NER tagged?"""
    print('What fraction of proper noun tokens are NER tagged?')
    proper_noun_df = df[df[noun_col] == 'NNP']
    ner_tag_token_count = len(proper_noun_df[proper_noun_df[ner_col] != 'O'])
    print(
        f'Count of NER tokens among proper noun tokens: {ner_tag_token_count}')
    print(f'Count of proper noun tokens: {len(proper_noun_df)}')
    print(
        f'Percent of noun phrase tokens that are part of NER tags: {round(ner_tag_token_count / len(proper_noun_df), 4) * 100}%'
    )
    print()


def NER_tags_proper_nouns(df):
    conclusions = """
    Conclusions:
    1. NER Tags are almost exclusively proper nouns (~85%), which means that 
        proper noun candidates will high recall for entity prediction tasks.
    2. Proper nouns don't encompass much more than NER tags (~84%), which means proper
        nouns will have high precision for entity prediction tasks.
    """
    proper_nouns_over_NER(df)
    NER_over_proper_nouns(df)
    print(conclusions)


def evaluate_noun_phrase_detection(detector,
                                   train_sentences,
                                   test_sentences,
                                   method='unsupervised',
                                   table_directory=None):
    print(
        f'Evaluate {method} Noun Phrase Detection Against CoNLL-2000 chunking task.'
    )
    if train_sentences:
        detector = detector(train_sentences)

    results = detector.evaluate(test_sentences)
    print(results)
    if table_directory:
        columns = ['IOB Accuracy', 'Precision', 'Recall', 'F1-Score']
        scores = [
            results.accuracy(),
            results.precision(),
            results.recall(),
            results.f_measure()
        ]
        df = pd.DataFrame([scores], columns=columns)
        table_string = utils.generate_table(df)
        utils.save_table(
            table_string,
            os.path.join(table_directory,
                         f'{method}_noun_phrase_classification_report.tex'))


def get_candidate_phrases(articles, entity_scorer):
    # get candidate phrases
    candidates = []
    for article in articles:
        # manually tokenize because nltk tokenizer is converting 'C$' -> ['C', '$'] and throwing off comparison
        sentences = utils.tokenize_text(article)
        article = [sentence.split() for sentence in sentences]
        article = utils.tag_pos(article)
        for candidate in entity_scorer.extract(article, preprocess=False):
            candidates.append(candidate)
    return candidates


def flag_nnp(article):
    groupings = []
    for sentence in article:
        for key, group in itertools.groupby(sentence, key=lambda x: x[1]):
            grouping = ' '.join([token for token, pos in list(group)])
            if key == 'NNP':
                groupings.append((grouping, True))
            else:
                groupings.append((grouping, False))
    return groupings


def get_nnp_candidate_phrases(articles):
    # get candidate phrases
    candidates = []
    for article in articles:
        # manually tokenize because nltk tokenizer is converting 'C$' -> ['C', '$'] and throwing off comparison
        sentences = utils.tokenize_text(article)
        article = [sentence.split() for sentence in sentences]
        article = utils.tag_pos(article)
        candidates += flag_nnp(article)
    return candidates


def prepare_scored_phrases(scored_candidates):
    df = pd.DataFrame(
        scored_candidates,
        columns=['Predicted_Phrase', 'Predicted_Entity_Flag', 'Score'])
    df['Phrase_ID'] = df.index
    df['Predicted_Phrase'] = df['Predicted_Phrase'].apply(lambda x: x.split())
    df = df.explode('Predicted_Phrase')
    # punctuation isn't getting assigned a score, fill with zero for now
    df['Score'] = df['Score'].fillna(0.0)
    # TODO: Investigate why this is happening
    df['Predicted_Phrase'] = df['Predicted_Phrase'].replace('``', '"')
    return df


def prepare_nnp_phrases(candidates):
    df = pd.DataFrame(candidates,
                      columns=['Predicted_Phrase', 'Predicted_Entity_Flag'])
    df['Phrase_ID'] = df.index
    df['Predicted_Phrase'] = df['Predicted_Phrase'].apply(lambda x: x.split())
    df = df.explode('Predicted_Phrase')
    df['Predicted_Phrase'] = df['Predicted_Phrase'].replace('``', '"')
    return df


def merge_dfs(df, prediction_df):
    assert len(df) == len(prediction_df)
    eval_df = pd.concat((df, prediction_df.reset_index(drop=True)), axis=1)
    # TODO: why are some CoNLL-2003 tokens NaN?
    eval_df = eval_df.dropna(subset=['Token'])
    assert (
        eval_df['Token'] == eval_df['Predicted_Phrase']).sum() == len(eval_df)
    return eval_df


def optimize_threshold(eval_df,
                       scoring_method='TFIDF',
                       optimization_steps=32,
                       table_directory=None):
    # optimize threshold to maximize macro f1
    start = eval_df['Score'].describe()['25%']
    stop = eval_df['Score'].describe()['75%']
    step = (stop - start) / optimization_steps

    predictions = []
    for thresh in np.arange(start, stop, step=step):
        predictions.append((thresh, (eval_df['Predicted_Entity_Flag'] &
                                     (eval_df['Score'] >= thresh)).values))

    scores = []
    for prediction in predictions:
        report = classification_report(eval_df['NER_Tag_Flag'],
                                       prediction[1],
                                       output_dict=True)
        scores.append((prediction[0], report['True']['f1-score'],
                       report['True']['precision'], report['True']['recall']))

    optimized_threshold = max(scores, key=lambda x: x[1])[0]

    print(
        f'Optimize {scoring_method} threshold ({optimized_threshold}) to maximize positive class (Entity = True) F1 score.'
    )
    print(f'Range searched - start: {start}, stop: {stop}, step: {step}')
    print(
        classification_report(eval_df['NER_Tag_Flag'],
                              (eval_df['Predicted_Entity_Flag'] &
                               (eval_df['Score'] >= optimized_threshold))))
    if table_directory:
        report = classification_report(
            eval_df['NER_Tag_Flag'],
            (eval_df['Predicted_Entity_Flag'] &
             (eval_df['Score'] >= optimized_threshold)),
            output_dict=True)
        table_file_path = os.path.join(
            table_directory,
            f'{scoring_method}_entity_classification_report.tex')
        # NB: this may overwrite an existing report
        utils.latex_table(report, table_file_path)

    return optimized_threshold, scores


def evaluate(eval_df,
             scoring_method='TFIDF',
             optimization_steps=64,
             table_directory=None):
    print()
    # baseline of just using noun phrases to identify entities (high recall, low precision)
    print(
        f'{scoring_method} baseline using noun phrases to identify entities:')
    print(
        classification_report(eval_df['NER_Tag_Flag'],
                              eval_df['Predicted_Entity_Flag']))
    print()
    if table_directory:
        report = classification_report(eval_df['NER_Tag_Flag'],
                                       eval_df['Predicted_Entity_Flag'],
                                       output_dict=True)
        table_file_path = os.path.join(
            table_directory,
            f'{scoring_method}_entity_classification_report.tex')
        # if optimization_steps=None, this will be the final table, else optimize_threshold will overwrite
        utils.latex_table(report, table_file_path)

    if optimization_steps:
        # use score and a threshold
        median_threshold = eval_df['Score'].describe()['50%']
        eval_df[f'Predicted_Entity_Flag_{scoring_method}_Median'] = (
            eval_df['Predicted_Entity_Flag'] &
            (eval_df['Score'] > median_threshold))
        print(
            f'Use the median {scoring_method} score ({round(median_threshold, 4)}) as a threshold to identify entities.'
        )
        print(
            classification_report(
                eval_df['NER_Tag_Flag'],
                eval_df[f'Predicted_Entity_Flag_{scoring_method}_Median']))

        return optimize_threshold(eval_df, scoring_method, optimization_steps,
                                  table_directory)


def evaluate_tfidf_entity_detection(train_documents, train_df, test_documents,
                                    test_df, table_directory):
    chunk_parser = NounPhraseDetector()
    entity_scorer = TFIDFScorer(chunk_parser)
    # fit TFIDF model and tune threshold
    entity_scorer.fit(train_documents)
    candidates = get_candidate_phrases(train_documents, entity_scorer)
    scores = entity_scorer.score_phrases(candidates)
    prediction_df = prepare_scored_phrases(scores)
    eval_df = merge_dfs(train_df, prediction_df)
    optimized_threshold, perf_scores = evaluate(eval_df,
                                                scoring_method='TFIDF_Train',
                                                optimization_steps=64,
                                                table_directory=None)

    # get predictions for test set and evaluate
    candidates = get_candidate_phrases(test_documents, entity_scorer)
    scores = entity_scorer.score_phrases(candidates)
    prediction_df = prepare_scored_phrases(scores)
    eval_df = merge_dfs(test_df, prediction_df)
    eval_df['Predicted_Entity_Flag'] = eval_df['Score'] >= optimized_threshold
    evaluate(eval_df,
             scoring_method='TFIDF_Test',
             optimization_steps=None,
             table_directory=table_directory)

    return perf_scores


def evaluate_textrank_entity_detection(train_documents, train_df,
                                       test_documents, test_df,
                                       table_directory):
    chunk_parser = NounPhraseDetector()
    # get scored phrases
    scored_candidates = []
    for idx, article in enumerate(train_documents):
        # fit TextRank on article
        scorer = TextRankScorer(article, preprocess=True, parser=chunk_parser)
        scorer.fit()

        # manually tokenize because nltk tokenizer is converting 'C$' -> ['C', '$'] and throwing off comparison
        sentences = utils.tokenize_text(article)
        article = [sentence.split() for sentence in sentences]
        article = utils.tag_pos(article)
        candidates = scorer.extract(article, preprocess=False)

        # score candidates
        for candidate in scorer.score_phrases(candidates):
            scored_candidates.append(candidate)
    prediction_df = prepare_scored_phrases(scored_candidates)
    eval_df = merge_dfs(train_df, prediction_df)
    optimized_threshold, perf_scores = evaluate(
        eval_df,
        scoring_method='TextRank_Train',
        optimization_steps=64,
        table_directory=None)

    # get scored phrases
    scored_candidates = []
    for idx, article in enumerate(test_documents):
        # fit TextRank on article
        scorer = TextRankScorer(article, preprocess=True, parser=chunk_parser)
        scorer.fit()

        # manually tokenize because nltk tokenizer is converting 'C$' -> ['C', '$'] and throwing off comparison
        sentences = utils.tokenize_text(article)
        article = [sentence.split() for sentence in sentences]
        article = utils.tag_pos(article)
        candidates = scorer.extract(article, preprocess=False)

        # score candidates
        for candidate in scorer.score_phrases(candidates):
            scored_candidates.append(candidate)
    prediction_df = prepare_scored_phrases(scored_candidates)
    eval_df = merge_dfs(test_df, prediction_df)
    eval_df['Predicted_Entity_Flag'] = eval_df['Score'] >= optimized_threshold
    evaluate(eval_df,
             scoring_method='TextRank_Test',
             optimization_steps=None,
             table_directory=table_directory)

    return perf_scores


def evaluate_cosine_entity_detection(train_documents, train_df, test_documents,
                                     test_df, table_directory):
    chunk_parser = NounPhraseDetector()
    candidates = get_candidate_phrases(train_documents, chunk_parser)
    phrases, flags = zip(*candidates)

    # consine distance between phrases and entity indicator
    vectorizer = SentenceTransformer(
        'paraphrase-distilroberta-base-v1')  # ('stsb-distilbert-base')
    phrase_embeddings = vectorizer.encode(phrases, convert_to_tensor=True)
    entity_indicator = vectorizer.encode(['this is not a named entity'],
                                         convert_to_tensor=True)
    scores = 1 - util.pytorch_cos_sim(phrase_embeddings, entity_indicator)
    scores = np.squeeze(scores).numpy()

    prediction_df = prepare_scored_phrases(zip(phrases, flags, scores))
    eval_df = merge_dfs(train_df, prediction_df)
    optimized_threshold, perf_scores = evaluate(eval_df,
                                                scoring_method='Cosine_Train',
                                                optimization_steps=64,
                                                table_directory=None)

    candidates = get_candidate_phrases(test_documents, chunk_parser)
    phrases, flags = zip(*candidates)

    # consine distance between phrases and entity indicator
    phrase_embeddings = vectorizer.encode(phrases, convert_to_tensor=True)
    entity_indicator = vectorizer.encode(['this is not a named entity'],
                                         convert_to_tensor=True)
    scores = 1 - util.pytorch_cos_sim(phrase_embeddings, entity_indicator)
    scores = np.squeeze(scores).numpy()

    prediction_df = prepare_scored_phrases(zip(phrases, flags, scores))
    eval_df = merge_dfs(test_df, prediction_df)
    eval_df['Predicted_Entity_Flag'] = eval_df['Score'] >= optimized_threshold
    evaluate(eval_df,
             scoring_method='Cosine_Test',
             optimization_steps=None,
             table_directory=table_directory)
    return perf_scores


def evaluate_nnp_entity_detection(documents, df, table_directory):
    candidates = get_nnp_candidate_phrases(documents)
    prediction_df = prepare_nnp_phrases(candidates)
    eval_df = merge_dfs(df, prediction_df)
    results = evaluate(eval_df,
                       scoring_method='NNP',
                       optimization_steps=None,
                       table_directory=table_directory)
    return results


def evaluate_pretrained_entity_detector(documents, df, table_directory):
    pretrained_ne_detector = PretrainedNEDetector(binary=True)
    candidates = get_candidate_phrases(documents, pretrained_ne_detector)
    prediction_df = prepare_nnp_phrases(candidates)
    eval_df = merge_dfs(df, prediction_df)
    results = evaluate(eval_df,
                       scoring_method='Pretrained',
                       optimization_steps=None,
                       table_directory=table_directory)
    return results


def plot_precision_recall(results,
                          scoring_method='TFIDF',
                          image_directory=None):
    results_df = pd.DataFrame(
        results, columns=['Threshold', 'F1', 'Precision', 'Recall'])
    results_df.index = results_df['Threshold']
    results_df.drop(columns=['Threshold']).plot()
    results_directory = 'results'
    os.makedirs(results_directory, exist_ok=True)
    filename = f'{scoring_method}_precision_recall_f1.png'
    plt.savefig(os.path.join(results_directory, filename), bbox_inches='tight')
    if image_directory:
        plt.savefig(os.path.join(image_directory, filename))


def main(args):
    # reporting directory
    reporting_directory = '/Users/tmorrill002/Documents/knowledge-graphs/reports/entity_detection'
    os.makedirs(reporting_directory, exist_ok=True)
    table_directory = os.path.join(reporting_directory, 'tables')
    os.makedirs(table_directory, exist_ok=True)
    image_directory = os.path.join(reporting_directory, 'images')
    os.makedirs(image_directory, exist_ok=True)

    # CoNLL-2000 chunking task data
    train_sentences = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    test_sentences = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

    # CoNLL-2003 NER task data
    df_dict = utils.load_train_data(args.data_directory)
    train_df, val_df, test_df = df_dict['train.csv'], df_dict[
        'validation.csv'], df_dict['test.csv']
    train_df['NER_Tag_Flag'] = train_df['NER_Tag'] != 'O'
    val_df['NER_Tag_Flag'] = val_df['NER_Tag'] != 'O'
    test_df['NER_Tag_Flag'] = test_df['NER_Tag'] != 'O'
    # confirm hypothesis that noun phrases are a superset of entities
    NER_tags_noun_phrases(train_df)

    # confirm hypothesis that proper nouns and entities are essentially equal
    NER_tags_proper_nouns(train_df)

    # evaluate unsupervised noun phrase detector
    evaluate_noun_phrase_detection(NounPhraseDetector(),
                                   train_sentences=None,
                                   test_sentences=test_sentences,
                                   method='unsupervised',
                                   table_directory=table_directory)

    # evaluate bigram noun phrase detector
    # evaluate_noun_phrase_detection(BigramChunker,
    #                                train_sentences,
    #                                test_sentences,
    #                                method='bigram',
    #                                table_directory=table_directory)

    # # evaluate maxent noun phrase detector
    # evaluate_noun_phrase_detection(MaxEntChunker,
    #                                train_sentences,
    #                                test_sentences,
    #                                method='MaximumEntropy',
    #                                table_directory=table_directory)

    # gather up articles
    train_articles = train_df.groupby(['Article_ID'], )['Token'].apply(
        lambda x: ' '.join([str(y) for y in list(x)])).values.tolist()
    val_articles = val_df.groupby(['Article_ID'], )['Token'].apply(
        lambda x: ' '.join([str(y) for y in list(x)])).values.tolist()
    test_articles = test_df.groupby(['Article_ID'], )['Token'].apply(
        lambda x: ' '.join([str(y) for y in list(x)])).values.tolist()

    # evaluate pretrained NTLK entity detector
    evaluate_pretrained_entity_detector(test_articles, test_df,
                                        table_directory)

    # evaluate NNP prediction method
    evaluate_nnp_entity_detection(test_articles, test_df, table_directory)

    # evaluate TFIDF scoring method
    train_val_articles = train_articles + val_articles
    train_val_df = pd.concat((train_df, val_df)).reset_index(drop=True)
    tfidf_results = evaluate_tfidf_entity_detection(train_val_articles,
                                                    train_val_df,
                                                    test_articles, test_df,
                                                    table_directory)
    plot_precision_recall(tfidf_results,
                          scoring_method='TFIDF',
                          image_directory=image_directory)

    # evaluate TextRank scoring method
    # TODO: debug
    textrank_results = evaluate_textrank_entity_detection(
        train_val_articles, train_val_df, test_articles, test_df,
        table_directory)
    plot_precision_recall(textrank_results,
                          scoring_method='TextRank',
                          image_directory=image_directory)

    # evaluate cosine scoring method
    cosine_results = evaluate_cosine_entity_detection(train_val_articles,
                                                      train_val_df,
                                                      test_articles, test_df,
                                                      table_directory)
    plot_precision_recall(cosine_results,
                          scoring_method='Cosine',
                          image_directory=image_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-directory',
        type=str,
        help='Directory where train, validation, and test data are stored.')
    args = parser.parse_args()
    main(args)