import random
import re
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from tqdm.auto import tqdm
from scipy.stats import betabinom

from model import layers
from text import text_to_sequence, cmudict, _clean_text, get_arpabet
from utils.utils import load_wav, load_filepaths_and_text, StandardScaler


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.data.text_cleaners
        self.max_wav_value = hparams.data.max_wav_value
        self.load_mel_from_disk = hparams.data.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.data.filter_length, hparams.data.hop_length, hparams.data.win_length,
            hparams.data.n_mel_channels, hparams.data.sampling_rate, hparams.data.mel_fmin,
            hparams.data.mel_fmax)
        random.seed(hparams.experiment.seed)
        random.shuffle(self.audiopaths_and_text)

        self.cmudict = cmudict.CMUDict(hparams.data.cmudict_path, keep_ambiguous=False)
        self.skip_heteronyms = hparams.data.skip_heteronyms
        self.p_arpabet = hparams.data.p_arpabet

        self.setup_scaler(hparams.data.stats_path)
        # should be about the batch size
        self.batch_group_size = int(hparams.data.batch_group_size * len(self.audiopaths_and_text))
        self.max_seq_len = getattr(hparams, "max_seq_len", None) or 300
        self.max_mel_len = getattr(hparams, "max_mel_len", None) or 860  # 12s.
        self.sort_items(drop=True)

    def sort_items(self, drop=False):
        r"""Sort instances based on text length in ascending order"""
        # drop=True means that you drop too long mels (longer than 1033 = 12s. by default)
        # or too long texts (longer than 300 by default)
        # it takes more time so it's better to set drop=True only at initialization
        lengths = np.array([len(ins[1]) for ins in self.audiopaths_and_text])

        # perform semi-sorted batching according to the paper
        # https://www.isca-speech.org/archive/pdfs/interspeech_2021/ge21_interspeech.pdf
        r = 0.1  # local randomization factor
        a = (np.max(lengths) - np.min(lengths)) * r
        lengths_for_sort = lengths + np.random.uniform(-a // 2, a // 2)

        idxs = np.argsort(lengths_for_sort)
        new_items = []
        ignored = []
        if drop:
            for i, idx in tqdm(list(enumerate(idxs))):
                length = lengths[idx]
                audiopath = Path(self.audiopaths_and_text[idx][0])
                mel_path = Path(str(audiopath).replace("wavs", "mels")).with_suffix(".pt")

                if not mel_path.exists() or length > self.max_seq_len or \
                        torch.load(str(mel_path), map_location='cpu').size(-1) > self.max_mel_len:
                    ignored.append(idx)
                else:
                    new_items.append(self.audiopaths_and_text[idx])
        else:
            for i, idx in enumerate(idxs):
                new_items.append(self.audiopaths_and_text[idx])
        # shuffle batch groups
        if self.batch_group_size > 0:
            for i in range(len(new_items) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_items = new_items[offset:end_offset]
                random.shuffle(temp_items)
                new_items[offset:end_offset] = temp_items
        self.audiopaths_and_text = new_items

        if len(ignored) != 0:
            print(" | > Max length sequence: {}".format(np.max(lengths)))
            print(" | > Min length sequence: {}".format(np.min(lengths)))
            print(" | > Avg length sequence: {}".format(np.mean(lengths)))
            print(f" | > Num. instances discarded by max-min (max={self.max_seq_len}) seq limits: {len(ignored)}")
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    def setup_scaler(self, stats_path):
        stats = np.load(stats_path, allow_pickle=True).item()
        mel_mean, mel_std = torch.tensor(stats['mel_mean']), torch.tensor(stats['mel_std'])
        self.mel_scaler = StandardScaler(mel_mean, mel_std)

    def get_scaler(self, scaler, speaker=0):
        if scaler is None:
            scaler = self.mel_scaler
        return scaler

    def normalize(self, S, scaler=None, speaker=0):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        # pylint: disable=no-else-return
        # S = S.clone()
        # mean-var scaling
        assert S.shape[0] == self.stft.n_mel_channels
        scaler = self.get_scaler(scaler, speaker)
        mel = scaler.transform(S.T).T
        return mel

    def denormalize(self, S, scaler=None, speaker=0):
        """denormalize values"""
        # pylint: disable=no-else-return
        S_denorm = S.clone()
        scaler = self.get_scaler(scaler, speaker)
        # mean-var scaling
        if S_denorm.shape[0] == self.stft.n_mel_channels:
            return scaler.inverse_transform(S_denorm.T).T

    def get_data(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        speaker = 0
        text, cleaned_text = self.get_text(text, return_clean_text=True)
        mel = self.get_mel(audiopath)
        mel = self.normalize(mel)

        item = {
            "text": text,
            "mel": mel,
            "speaker": torch.tensor(speaker),
            "audiopath": audiopath,
            "cleaned_text": cleaned_text,
        }
        return item

    def load_mel_from_path(self, audiopath):
        audiopath = Path(audiopath)
        mel_path = Path(str(audiopath).replace("wavs", "mels")).with_suffix(".pt")
        return torch.load(mel_path, map_location='cpu').detach()

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if audio.max() > 2:
                audio_norm = audio / self.max_wav_value
            else:
                audio_norm = audio
            audio_norm = normalize(audio_norm) * 0.95
            audio_norm = torch.from_numpy(audio_norm).unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec, _ = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = self.load_mel_from_path(filename)
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text, preparing_alignments=False, return_clean_text=False):
        cleaned_text = _clean_text(text, self.text_cleaners)
        words = re.findall(r'\S*\{.*?\}\S*|\S+', cleaned_text)
        result_text = []
        for word in words:
            if random.random() < self.p_arpabet:
                if "<" not in word:
                    result_text.append(get_arpabet(word, self.cmudict, self.skip_heteronyms))
                else:
                    p_ind = word.index("<")
                    result_text.append(get_arpabet(word[:p_ind], self.cmudict, self.skip_heteronyms) + word[p_ind:])
            else:
                result_text.append(word)
        result_text = ' '.join(result_text)
        text_norm = torch.IntTensor(text_to_sequence(result_text))
        if preparing_alignments:
            return text_norm, result_text
        if return_clean_text:
            return text_norm, cleaned_text
        return text_norm

    def __getitem__(self, index):
        return self.get_data(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per setep
    :param: inference – include audiopath (for inference, teacher forcing purposes)
    """

    def __init__(self, n_frames_per_step, inference=False):
        self.n_frames_per_step = n_frames_per_step
        self.inference = inference

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, bert_feats]
        """
        # Right zero-pad all one-hot text sequences to max input length

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['text']) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]['text']
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0]['mel'].size(0)  # 80
        max_target_len = max([x['mel'].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (self.n_frames_per_step - (max_target_len % self.n_frames_per_step))
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['mel']
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        cleaned_texts = [batch[ids_sorted_decreasing[i]]['cleaned_text'] for i in range(len(ids_sorted_decreasing))]

        output = {
            "text_padded": text_padded,
            "input_lengths": input_lengths,
            "mel_padded": mel_padded,
            "gate_padded": gate_padded,
            "output_lengths": output_lengths,
            "speaker": torch.stack(
                [batch[ids_sorted_decreasing[i]]['speaker'] for i in range(len(ids_sorted_decreasing))]),
            "cleaned_texts": cleaned_texts
        }
        if self.inference:
            output.update({
                "audiopath": [batch[ids_sorted_decreasing[i]]['audiopath'] for i in range(len(ids_sorted_decreasing))]
            })

        return output
