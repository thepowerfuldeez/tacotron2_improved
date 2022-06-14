from typing import Tuple

import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from utils.audio_processing import dynamic_range_compression, dynamic_range_decompression


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


class ZoneOutCell(torch.nn.Module):
    # Adapted from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2/decoder.py
    """ZoneOut Cell module.
    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.
    Examples:
        >> lstm = torch.nn.LSTMCell(16, 32)
        >> lstm = ZoneOutCell(lstm, 0.5)
    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305
    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch
    """

    def __init__(self, cell, zoneout_rate=0.1):
        """Initialize zone out cell module.
        Args:
            cell (torch.nn.Module): Pytorch recurrent cell module
                e.g. `torch.nn.Module.LSTMCell`.
            zoneout_rate (float, optional): Probability of zoneout from 0.0 to 1.0.
        """
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_rate = zoneout_rate
        if zoneout_rate > 1.0 or zoneout_rate < 0.0:
            raise ValueError(
                "zoneout probability must be in the range from 0.0 to 1.0."
            )

    def forward(self, inputs: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        """Calculate forward propagation.
        Args:
            inputs (Tensor): Batch of input tensor (B, input_size).
            hidden (tuple):
                - Tensor: Batch of initial hidden states (B, hidden_size).
                - Tensor: Batch of initial cell states (B, hidden_size).
        Returns:
            tuple:
                - Tensor: Batch of next hidden states (B, hidden_size).
                - Tensor: Batch of next cell states (B, hidden_size).
        """
        next_hidden = self.cell(inputs, hidden)

        # self.zoneout_rate should be tuple
        prob = torch.ones(2) * self.zoneout_rate
        res: Tuple[torch.Tensor, torch.Tensor] = (
            self._zoneout(hidden[0], next_hidden[0], prob[0]),
            self._zoneout(hidden[1], next_hidden[1], prob[1])
        )
        return res

    def _zoneout(self, h: torch.Tensor, next_h: torch.Tensor, prob: float) -> torch.Tensor:
        if self.training:
            # dropout
            mask = torch.zeros_like(h).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0, center=False):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        # self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', torch.hann_window(win_length))

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        # similar to librosa, reflect-pad the input
        input_data = F.pad(
            y.unsqueeze(1),
            (int((self.filter_length - self.hop_length) / 2), int((self.filter_length - self.hop_length) / 2)),
            mode='reflect')
        input_data = input_data.squeeze(1)

        magnitudes = torch.stft(input_data, self.filter_length, hop_length=self.hop_length, win_length=self.win_length,
                                window=self.hann_window, return_complex=False,
                                center=self.center, pad_mode='reflect', normalized=False, onesided=True)

        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1) + (1e-9))

        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)

        energy = magnitudes.norm(dim=1)

        return mel_output, energy
