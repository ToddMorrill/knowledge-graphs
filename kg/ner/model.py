"""
This module implements some PyTorch models for the NER task.

Examples:
    $ python model.py
"""
import argparse
from types import SimpleNamespace

import torch
from torch._C import T, dtype
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.sparse import Embedding
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    """Vanilla LSTM."""
    def __init__(self, config: SimpleNamespace) -> None:
        """Define components of the network.

        Args:
            config (SimpleNamespace): Configuration mappings.
        """
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, batch: tuple) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            batch (tuple): sentences, lengths

        Returns:
            torch.Tensor: Log of the softmax for each token.
        """
        sentences, lengths = batch
        embeds = self.embedding(sentences)
        embeds = pack_padded_sequence(embeds,
                                      lengths,
                                      batch_first=True,
                                      enforce_sorted=False)
        packed_activations, _ = self.lstm(embeds)
        activations, lengths = pad_packed_sequence(packed_activations,
                                                   batch_first=True,
                                                   padding_value=1)

        activations = activations.reshape(-1, activations.shape[2])
        logits = self.fc(activations)
        return F.log_softmax(logits, dim=1)


def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Custom loss function that ignores PAD tokens when computing the loss.

    Args:
        outputs (torch.Tensor): log_softmax from the net.
        labels (torch.Tensor): Ground truth labels, where -1 corresponds to a padded token.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    labels = labels.reshape(-1)  # convert to vector
    mask = (labels >= 0).float()  # 1 if greater than 0, 0 if -1
    # bring the -1's into a valid index range
    labels = labels % outputs.shape[1]
    num_tokens = mask.sum()  # sum the actual number of tokens
    # extract the value at the ground truth index
    log_probs = outputs[range(outputs.shape[0]), labels]
    # mask padded tokens from the sum
    total_loss = -torch.sum(log_probs * mask)
    loss = total_loss / num_tokens  # average the loss
    return loss


def get_predictions(output: torch.Tensor,
                    lengths: list,
                    concatenate: bool = True) -> list:
    """Utility function to remove the padded tokens and retrieve the predicted index.

    Args:
        output (torch.Tensor): log_softmax from the net.
        lengths (list): Lengths of original sequences in terms of tokens.
        concatenate (bool, optional): If True, concatenate into one tensor, else return list of lists. Defaults to True.

    Returns:
        list: List of lists of predicted indices.
    """
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


def recover_labels(padded_labels: torch.Tensor, lengths: list) -> torch.Tensor:
    """Utility function to remove the padded labels and retrieve the ground truth labels.

    Args:
        padded_labels (torch.Tensor): Vector of labels including -1's.
        lengths (list): Lengths of original sequences in terms of tokens.

    Returns:
        torch.Tensor: Unpadded sequence of labels.
    """
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


def translate_predictions(batch: list, label_dict: dict) -> list:
    """Iterate over list of lists of predicted indices and convert them to string labels.

    Args:
        batch (list): List of lists of predicted indices.
        label_dict (dict): Index to string dictionary.

    Returns:
        list: List of lists of string labels.
    """
    preds = []
    for pred_sequence in batch:
        pred_list = []
        for pred in pred_sequence.tolist():
            pred_list.append(label_dict[pred])
        preds.append(pred_list)
    return preds


def main(args):
    config = {
        'vocab_size': 10,
        'embedding_dim': 5,
        'hidden_size': 7,
        'num_classes': 9,
        'batch_size': 11
    }
    config = SimpleNamespace(**config)

    model = LSTM(config)
    # max_sequence_length = 13
    small_batch = [
        torch.tensor([5, 6, 3, 0]),
        torch.tensor([2, 8, 4, 9, 9, 3])
    ]
    lengths = torch.tensor([len(x) for x in small_batch])
    padded_batch = pad_sequence(small_batch, batch_first=True, padding_value=1)
    batch = (padded_batch, lengths)
    expected_shape = (len(small_batch) * lengths.max(), config.num_classes)
    output = model(batch)

    # check output shape
    assert output.shape == expected_shape

    # check that everything sums to 1 row-wise
    # assert output.sum(dim=1).bool().all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
