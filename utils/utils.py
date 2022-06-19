import torch
import numpy as np
from soundfile import read
from scipy.stats import betabinom, pearsonr


def load_wav(full_path) -> (np.ndarray, int):
    data, sampling_rate = read(full_path)
    return data.astype(np.float32), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f if len(line.strip().split(split)) == 2]
    return filepaths_and_text


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


def to_gpu(x):
    """move tensor to gpu"""
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def parse_batch(batch):
    """
    Parse batch from dataloader
    """
    # TODO: fix with n_frames=2
    # max_input_length = max(input_lengths)
    # max_output_length = max(output_lengths)
    # if max_input_length != text_padded.shape[1]:
    #     text_padded = text_padded[:, :max_input_length]
    # if max_output_length != mel_padded.shape[2]:
    #     mel_padded = mel_padded[..., :max_output_length]
    #     gate_padded = gate_padded[..., :max_output_length]

    batch['text_padded'] = to_gpu(batch['text_padded']).long()
    batch['input_lengths'] = to_gpu(batch['input_lengths']).long()
    batch['max_len'] = torch.max(batch['input_lengths'].data).item()
    batch['mel_padded'] = to_gpu(batch['mel_padded']).float()
    batch['gate_padded'] = to_gpu(batch['gate_padded']).float()
    batch['output_lengths'] = to_gpu(batch['output_lengths']).long()

    return batch


def parse_outputs(outputs, mask_padding, n_mel_channels, n_frames_per_step):
    """
    Parse model output

    type: (List[torch.Tensor], torch.Tensor) -> List[torch.Tensor]
    """

    if mask_padding and outputs['mel_lengths'] is not None:
        mask: torch.Tensor = get_mask_from_lengths(outputs['mel_lengths'])
        mask = mask.expand(n_mel_channels, mask.size(0), mask.size(1))

        if mask.size(2) % n_frames_per_step != 0:
            # pad mask with True value
            to_append = torch.ones(
                mask.size(0), mask.size(1),
                (n_frames_per_step - mask.size(2) % n_frames_per_step)).bool().to(mask.device)
            mask = torch.cat([mask, to_append], dim=-1)
        if mask.size(2) < outputs['mel_outputs'].size(2):
            # pad with false
            to_append = torch.zeros(
                mask.size(0), mask.size(1),
                (outputs['mel_outputs'].size(2) - mask.size(2))).bool().to(mask.device)
            mask = torch.cat([mask, to_append], dim=-1)
        mask = mask.permute(1, 0, 2)

        outputs['mel_outputs'].masked_fill_(mask, 0.0)
        outputs['mel_outputs_postnet'].masked_fill_(mask, 0.0)
        outputs['gate_outputs'].masked_fill_(mask[:, 0, :], 1e3)  # gate energies

    return outputs


def beta_binomial_prior_distribution(phoneme_count, mel_count):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = i, M + 1 - i
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def alignment_diagonal_score(alignments, lengths):
    betabinomial = beta_binomial_prior_distribution(alignments.size(2), alignments.size(1))
    res = torch.tensor(0., dtype=alignments.dtype, device=alignments.device)
    for i, l in enumerate(lengths):
        res += torch.tensor(pearsonr(betabinomial[:, :l].reshape(-1), alignments[i, :, :l].view(-1).cpu())[0],
                            dtype=alignments.dtype, device=alignments.device)
    return res / alignments.size(0)


def alignment_confidence_score(alignments, lengths, binary=False):
    """
    Atention confidence
    Compute how diagonal alignment predictions are. It is useful
    to measure the alignment consistency of a model
    Args:
        alignments (torch.Tensor): batch of alignments.  (batch, out_seq_len, encoder_steps)
        lengths: (batch, )
        binary (bool): if True, ignore scores and consider attention
        as a binary mask.
    Shape:
        alignments : batch x decoder_steps x encoder_steps
    """
    maxs = alignments.max(dim=1)[0]
    if binary:
        maxs[maxs > 0] = 1

    res = torch.tensor(0., dtype=alignments.dtype, device=alignments.device)
    for i, l in enumerate(lengths):
        res += maxs[i, :l].mean()

    return res / alignments.size(0)


@torch.jit.script
class StandardScaler:
    """
    Mean-std scaler. Compute stats using compute_statistics.py first!
    """

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, x: torch.Tensor):
        x -= self.mean_.to(x.device)
        x /= self.scale_.to(x.device)
        return x

    def inverse_transform(self, x: torch.Tensor):
        x *= self.scale_.to(x.device)
        x += self.mean_.to(x.device)
        return x


def vocoder_infer(generator, mel):
    """
    Mel to Audio using supported traced vocoder
    """
    with torch.no_grad():
        audio = generator(mel.float()).cpu().numpy()[0]
    audio = audio / np.abs(audio).max()
    return audio
