"""
This module implements the training routine for the specified model.

Examples:
    $ python train.py \
        --config configs/baseline.yaml

Current supervised results (token level evaluation):
Macro average precision: 0.80
Macro average recall: 0.67
Macro F1: 0.72
"""
import argparse
import os
import time
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.optim as optim
import yaml

from kg.ner.model import LSTM
from kg.ner.preprocess import Preprocessor


def loss_fn(outputs, labels):
    labels = labels.reshape(-1)
    mask = (labels >= 0).float()
    labels = labels % outputs.shape[1]
    num_tokens = mask.sum()
    return -torch.sum(
        outputs[range(outputs.shape[0]), labels] * mask) / num_tokens


def get_predictions(output, lengths, concatenate=True):
    # extract predictions
    max_len = max(lengths)
    preds = output.argmax(dim=1)
    i = 0
    preds_list = []
    for length in lengths:
        start = i * max_len
        stop = start + length
        preds_list.append(preds[start:stop])
        i += 1
    if concatenate:
        return torch.cat(preds_list)
    return preds_list


def recover_labels(padded_labels, lengths):
    # extract labels
    max_len = max(lengths)
    labels_vector = padded_labels.reshape(-1)
    i = 0
    labels_list = []
    for length in lengths:
        start = i * max_len
        stop = start + length
        labels_list.append(labels_vector[start:stop])
        i += 1
    return torch.cat(labels_list)


def accuracy(output, sentences, labels):
    batch_preds = get_predictions(output, sentences[1])
    batch_labels = recover_labels(labels, sentences[1])
    raw_acc = accuracy_score(batch_labels, batch_preds)
    # acc = round(raw_acc * 100, 2)
    return raw_acc


def evaluate(model, dataloader, split='Validation', print_report=False):
    # monitor validation loss
    model.eval()
    eval_loss_scores = []
    eval_preds = []
    eval_labels = []
    for sentences, labels in dataloader:
        output = model(sentences)
        loss = loss_fn(output, labels)
        eval_loss_scores.append(loss.item())
        eval_preds += get_predictions(output, sentences[1]).tolist()
        eval_labels += recover_labels(labels, sentences[1]).tolist()
    eval_loss = round(np.mean(eval_loss_scores), 4)
    eval_acc = round(accuracy_score(eval_labels, eval_preds) * 100, 2)
    print(f'{split} loss (average): {eval_loss}')
    print(f'{split} accuracy score: {eval_acc}%')
    if print_report:
        label_set = list(set(eval_preds).union(set(eval_labels)))
        print(
            classification_report(eval_labels,
                                  eval_preds,
                                  labels=label_set,
                                  target_names=list(label_dict.keys())))
    return eval_loss


def train_epoch(model, i, dataloader, optimizer):
    model.train()
    print(f'Epoch number: {i}')
    j = 0
    for sentences, labels in dataloader:
        output = model(sentences)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j % 100 == 0:
            batch_preds = get_predictions(output, sentences[1])
            batch_labels = recover_labels(labels, sentences[1])
            raw_acc = accuracy_score(batch_labels, batch_preds)
            print(
                f'Sample train batch - loss value: {round(loss.item(), 4)} \t accuracy score: {round(raw_acc*100, 2)}%'
            )
        j += 1
    return model, optimizer


def save_model(model, optimizer, i, config):
    # save the model
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': i
    }
    save_file_path = os.path.join(config.run_dir, 'model.pt')
    torch.save(state, save_file_path)
    print('Saved best model.')
    return save_file_path


def train(model, dataloaders, optimizer, config):
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    running_patience = config.patience
    best_val_loss = float('inf')
    for i in range(1, config.epochs + 1):
        model, optimizer = train_epoch(model, i, train_dataloader, optimizer)
        val_loss = evaluate(model, val_dataloader, 'Validation')

        # stopping and saving criterion
        if val_loss < best_val_loss:
            running_patience = config.patience
            best_val_loss = val_loss

            # save the model
            save_file_path = save_model(model, optimizer, i, config)

        else:
            running_patience -= 1
            print(f'Patience at {running_patience}.')
            if running_patience == 0:
                print(
                    f'Model has not improved for {config.patience} epochs. Stopping training.'
                )
                break

        print()

    print('Loading best model from saved checkpoints.')
    state = torch.load(save_file_path)
    model.load_state_dict(state['model'])

    # run final validation eval
    print('Running final evaluations.')
    val_loss = evaluate(model, val_dataloader, 'Validation')

    # run test set eval
    test_loss = evaluate(model, test_dataloader, 'Test', print_report=True)
    return model


def translate_predictions(batch, label_dict):
    preds = []
    for pred_sequence in batch:
        pred_list = []
        for pred in pred_sequence.tolist():
            pred_list.append(label_dict[pred])
        preds.append(pred_list)
    return preds


def main(args):
    preprocessor = Preprocessor(args.config)

    # TODO: clean this up
    global label_dict
    label_dict = preprocessor.label_dict

    dataloaders = preprocessor.get_train_dataloaders()

    config = preprocessor.config
    config.vocab_size = len(preprocessor.vocab)
    config.num_classes = len(preprocessor.label_dict)

    # create run directory
    runtime = time.strftime('%Y%m%d-%H%M%S')
    config.run_dir = os.path.join(config.model_dir, 'runs', runtime)
    os.makedirs(config.run_dir, exist_ok=True)

    model = LSTM(config)
    optimizer = optim.Adam(model.parameters())

    train(model, dataloaders, optimizer, config)

    # pickle/save the preprocessor
    preprocessor_file_path = os.path.join(config.run_dir,
                                          'preprocessor.pickle')
    with open(preprocessor_file_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    # save the config file
    config_file_path = os.path.join(config.run_dir, 'config.yaml')
    with open(config_file_path, 'w') as f:
        yaml.dump(config, f)

    # sample prediction with reloaded model
    model_file_path = os.path.join(config.run_dir, 'model.pt')
    state = torch.load(model_file_path)
    model = LSTM(config)
    model.load_state_dict(state['model'])

    sample_sentences = [
        'Todd Morrill lives in New York City.',
        'Emily is a math student at Columbia University.',
        'Bob works at the United Nations.'
    ]
    prepared_sentences = preprocessor.preprocess(sample_sentences)

    sample_output = model(prepared_sentences)
    sample_predictions = get_predictions(sample_output,
                                         lengths=prepared_sentences[1],
                                         concatenate=False)
    preds = translate_predictions(sample_predictions,
                                  preprocessor.idx_to_label)
    for idx, sent in enumerate(sample_sentences):
        print(f'Sample sentence:    {sent}')
        print(f'Sample predictions: {preds[idx]}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='File path where the model configuration file is located.',
        required=True)

    args = parser.parse_args()
    main(args)