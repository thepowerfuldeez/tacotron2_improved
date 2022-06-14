from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from model.layers import ConvNorm


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, n_symbols, encoder_n_convolutions,
                 encoder_embedding_dim, encoder_lstm_hidden_dim, encoder_kernel_size):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(n_symbols, encoder_embedding_dim)
        std = sqrt(2.0 / (n_symbols + encoder_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(encoder_embedding_dim,
                            encoder_lstm_hidden_dim, 1,
                            batch_first=True, bidirectional=True)

    @torch.jit.ignore
    def forward(self, x, input_lengths):
        x = self.embedding(x).transpose(1, 2)

        for conv in self.convolutions:
            # leaky_relu instead of relu in original impl
            x = F.dropout(F.leaky_relu(conv(x), 0.01), 0.5, self.training)

        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    @torch.jit.export
    def infer(self, x, input_lengths):
        device = x.device
        x = self.embedding(x.to(device)).transpose(1, 2)
        # [batch, symbols_len, encoder_dim]

        for conv in self.convolutions:
            # leaky_relu instead of relu in original impl
            x = F.dropout(F.leaky_relu(conv(x), 0.01), 0.5, self.training)

        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs
