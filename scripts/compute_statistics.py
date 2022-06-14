"""Save pitch, mels, energy and normalization coefficients. Normalize audio and remove outliers for pitch and energy."""

import argparse
from pathlib import Path
from multiprocessing.pool import Pool

import torch
import pyworld as pw
import numpy as np
from tqdm import tqdm
from librosa.util import normalize

from model import layers
from utils.utils import load_filepaths_and_text, load_wav
from utils.hparams import create_hparams


def process_audio(audiopath, max_wav_value=32768):
    audio, sampling_rate = load_wav(audiopath)

    if audio.max() > 2:
        audio_norm = audio / max_wav_value
    else:
        audio_norm = audio
    audio_norm = normalize(audio_norm) * 0.97
    return audio_norm


def save_mel(audiopath, mel):
    audiopath = Path(audiopath)
    new_path = Path(str(audiopath).replace("wavs", "mels")).with_suffix(".pt")
    if not new_path.parent.exists():
        (new_path.parent).mkdir(exist_ok=True, parents=True)
    torch.save(mel, new_path)


def save_pitch(audiopath, pitch):
    audiopath = Path(audiopath)
    new_path = Path(str(audiopath).replace("wavs", "pitch")).with_suffix(".pt")
    if not new_path.parent.exists():
        (new_path.parent).mkdir(exist_ok=True, parents=True)
    torch.save(pitch, new_path)


def save_energy(audiopath, energy):
    audiopath = Path(audiopath)
    new_path = Path(str(audiopath).replace("wavs", "energy")).with_suffix(".pt")
    if not new_path.parent.exists():
        (new_path.parent).mkdir(exist_ok=True, parents=True)
    torch.save(energy, new_path)


def get_pitch(wav, hparams, repeat=False):
    """
    get f0 from wav
    if repeat=true, resulting length will be as the length of audio
    """
    pitch, t = pw.dio(
        wav.astype(np.float64),
        hparams.data.sampling_rate,
        frame_period=hparams.data.hop_length / hparams.data.sampling_rate * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, hparams.data.sampling_rate)
    pitch = torch.from_numpy(pitch)
    if repeat:
        pitch = pitch.repeat_interleave(len(wav) // len(pitch))
    return pitch


def remove_outliers(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]


def compute_and_save_feats(audiopath_and_text):
    audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
    wav = process_audio(audiopath)
    audio_norm = torch.from_numpy(wav).unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    mel, energy = _stft.mel_spectrogram(audio_norm)
    mel, energy = mel.squeeze(0), energy.squeeze(0)

    # TODO: revisit, sometimes it helps with kinda corrupted audios
    # if mel.min() < -11:
    #     print("clip mel")
    #     mel = torch.clip(mel, -11)

    pitch = get_pitch(wav, hparams)

    save_mel(audiopath, mel)
    save_pitch(audiopath, pitch)
    save_energy(audiopath, energy)
    return mel, pitch, energy


if __name__ == "__main__":
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Compute mean and variance of spectrogtram features.")
    parser.add_argument('--config', type=str,
                        required=False, help='comma separated name=value pairs')
    args = parser.parse_args()

    hparams = create_hparams(args.config)

    # load the meta data of target dataset
    audiopaths_and_text = load_filepaths_and_text(hparams.data.training_files)
    val_audiopaths_and_text = load_filepaths_and_text(hparams.data.validation_files)
    print(f" > There are {len(audiopaths_and_text)} files.")

    # they are in global state now
    mel_sum, mel_square_sum = 0, 0
    energy_sum, energy_square_sum = 0, 0
    pitch_sum, pitch_square_sum = 0, 0
    energy_min, energy_max = int(1e6), -int(1e6)
    pitch_min, pitch_max = int(1e6), -int(1e6)
    N = 0
    max_wav_value = hparams.data.max_wav_value
    _stft = layers.TacotronSTFT(
        hparams.data.filter_length, hparams.data.hop_length, hparams.data.win_length,
        hparams.data.n_mel_channels, hparams.data.sampling_rate, hparams.data.mel_fmin,
        hparams.data.mel_fmax)

    # with Pool(32) as p:
    #     for (mel, pitch, energy) in \
    #             tqdm(p.imap_unordered(compute_and_save_feats, audiopaths_and_text), total=len(audiopaths_and_text)):
    if 1:
        for audiopath_and_text in tqdm(audiopaths_and_text):
            mel, pitch, energy = compute_and_save_feats(audiopath_and_text)

            pitch = remove_outliers(pitch)
            energy = remove_outliers(energy)

            # compute stats
            N += mel.shape[1]
            mel_sum += mel.sum(1)
            mel_square_sum += (mel ** 2).sum(axis=1)

            energy_sum += energy.sum()
            energy_square_sum += (energy ** 2).sum()
            if energy.min() < energy_min:
                energy_min = energy.min()
            if energy.max() > energy_max:
                energy_max = energy.max()

            pitch_sum += pitch.sum()
            pitch_square_sum += (pitch ** 2).sum()
            if len(pitch):
                if pitch.min() < pitch_min:
                    pitch_min = pitch.min()
                if pitch.max() > pitch_max:
                    pitch_max = pitch.max()

    for audiopath_and_text in tqdm(val_audiopaths_and_text):
        compute_and_save_feats(audiopath_and_text)

    mel_mean = mel_sum / N
    mel_scale = np.sqrt(mel_square_sum / N - mel_mean ** 2)

    energy_mean = energy_sum / N
    energy_scale = np.sqrt(energy_square_sum / N - energy_mean ** 2)

    pitch_mean = pitch_sum / N
    pitch_scale = np.sqrt(pitch_square_sum / N - pitch_mean ** 2)

    output_file_path = hparams.data.stats_path
    stats = {
        'mel_mean': mel_mean,
        'mel_std': mel_scale,
        'energy_mean': energy_mean,
        'energy_std': energy_scale,
        'energy_min': energy_min,
        'energy_max': energy_max,
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_scale,
        'pitch_min': pitch_min,
        'pitch_max': pitch_max,
    }

    print(f' > Train Ds length: {N * hparams.data.hop_length / hparams.data.sampling_rate / 3600} h.')
    print(f' > Avg mel spec mean: {mel_mean.mean()}')
    print(f' > Avg mel spec scale: {mel_scale.mean()}')
    print(f' > Avg energy: {energy_mean}')
    print(f' > Std energy: {energy_scale}')
    print(f' > Avg f0: {pitch_mean}')
    print(f' > Std f0: {pitch_scale}')

    np.save(output_file_path, stats, allow_pickle=True)
    print(f' > stats saved to {output_file_path}')
