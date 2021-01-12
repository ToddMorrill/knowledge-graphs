"""
This module implements some PyTorch models for the NER task.

Examples:
    $ python model.py
"""
import argparse
from types import SimpleNamespace

import torch
from torch._C import dtype
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
