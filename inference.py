from string import punctuation
from pathlib import Path
from typing import Dict

import torch
import numpy as np
import matplotlib.pylab as plt
from g2p_en import G2p

from scripts.make_bert_vectors import get_vectors_for_words, pt_model, tokenizer
from text import text_to_sequence, _clean_text, get_phones, cmudict, _symbol_to_id
from train import load_model, load_checkpoint
from utils.utils import StandardScaler, alignment_confidence_score, alignment_diagonal_score
from utils.hparams import create_hparams

g2p = G2p()
cmudict = cmudict.CMUDict("data/cmudict_dictionary", True)


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')


def load_model_from_folder(checkpoint_dir: Path, device='cuda', tpse=True):
    """
    Load trained model from checkpoint folder
    """
    checkpoint_path = str(sorted(checkpoint_dir.glob("*checkpoint*"), key=lambda p: p.stat().st_mtime)[-1])
    config_file = checkpoint_dir / "config.yaml"
    hparams = create_hparams(config_file)
    hparams.experiment.distributed_run = False
    denormalizer = Denormalizer(hparams.data.stats_path, hparams.model.multispeaker, hparams.model.n_speakers)
    model = load_model(hparams, device=device)
    load_checkpoint(checkpoint_path, model)

    if tpse:
        tpse_predictor_path = checkpoint_dir / "tpse_predictor_weights.pth"
        model.init_infer(hparams.model.encoder_lstm_hidden_dim * 2,  # as we do bidirectional lstm
                         hparams.model.bert_embedding_dim, hparams.model.gst_embedding_dim,
                         hparams.model.gst_tpse_gru_hidden_size, hparams.model.gst_tpse_num_layers,
                         tpse_predictor_path)
    model.eval()
    if torch.device(device) == torch.device("cuda"):
        _ = model.cuda().half()
    return model, denormalizer


def text_len2max_decoder_steps(length, symbol_len=0.09, sr=22050, hop=256):
    """Simple heuristic to infer max decoder steps from input sequence length to save compute"""
    return max(150, int(length * 1.5 * symbol_len * (sr / hop)))


def get_text_for_inference(text, verbose, input_text_phones=None,
                           use_g2p=False, device='cuda', cleaners=("flowtron_cleaners",)):
    """
    Prepare text for inference. Infer BERT for features, add pauses, add punctuation
    """

    if text[-1] not in punctuation:
        text += "."
    # clean pauses before infering BERT
    text_ = text.replace("<p0>", "").replace("<p1>", "").replace("<p2>", "")
    # infer BERT on input text
    bert_feats, groups = get_vectors_for_words(pt_model, tokenizer,
                                               _clean_text(text_, ['english_cleaners']), device)
    bert_feats = bert_feats.unsqueeze(0).to(device)

    # if no pauses specified, use heuristic and replace commas with short pauses, periods with middle pauses
    # TODO: Check if it causes problems for input text like apples,carrots,beetroots instead of apples, carrots ...
    text = text.replace(". ", ".<p1> ").replace(", ", ",<p0> ")
    # run text cleaners
    cleaned_text = _clean_text(text, cleaners)

    text = cleaned_text
    if use_g2p:
        if input_text_phones is not None:
            text_phones = input_text_phones
        else:
            text_phones = get_phones(g2p, cmudict, text)
            if verbose:
                print(text)
    else:
        text_phones = text.lower()
        if verbose:
            print(text)

    # double stop-token for short sequences
    if len(text) < 20:
        sequence = np.array(text_to_sequence(text_phones) + [_symbol_to_id['<eos>']])[None, :]
    else:
        sequence = np.array(text_to_sequence(text_phones))[None, :]

    sequence = torch.LongTensor(sequence).to(device)
    if torch.device(device) == torch.device("cuda"):
        bert_feats = bert_feats.half()
    max_steps = text_len2max_decoder_steps(len(cleaned_text))
    return sequence, cleaned_text, text_phones, bert_feats, max_steps


def text2mel(model, input_text, denormalizer, input_text_phones=None,
             use_g2p=True, tpse=True, verbose=False, plot=False,
             gst_vector_ind=None, gst_vector_coef=0.3, device='cuda',
             cleaners=("flowtron_cleaners",)):
    text = input_text
    sequence, cleaned_text, text_phones, bert_feats, max_steps = get_text_for_inference(
        text, verbose, input_text_phones, use_g2p, device, cleaners=cleaners)
    input_lengths = torch.tensor([sequence.size(1)]).to(sequence.device)

    with torch.no_grad():
        outputs = model.inference(
            sequence, input_lengths,
            bert_feats=bert_feats if tpse is True else None,
            max_decoder_steps=max_steps,
            gst_vector_ind=gst_vector_ind, gst_vector_coef=gst_vector_coef
        )

        alignments = outputs['alignments']
        mel_outputs = outputs['mel_outputs']
        mel_outputs_postnet = outputs['mel_outputs_postnet']
        out_lengths = outputs['mel_lengths']
        if denormalizer is not None:
            if mel_outputs_postnet.size(0) == 1:
                mel_outputs_postnet = denormalizer.denormalize(
                    mel_outputs_postnet[0, :, :out_lengths[0]]).unsqueeze(0).to(alignments.device)
            else:
                for i in range(len(mel_outputs_postnet)):
                    mel_outputs_postnet[i, :, :out_lengths[i]] = denormalizer.denormalize(
                        mel_outputs_postnet[i, :, :out_lengths[i]].cpu()).to(alignments.device)
    if plot:
        plot_data(
            (mel_outputs.cpu().float().data.numpy()[0],
             mel_outputs_postnet.cpu().float().data.numpy()[0],
             alignments.float().cpu().data.numpy().T)
        )
    return mel_outputs_postnet, alignments, cleaned_text, mel_outputs_postnet.shape[-1] != max_steps


def text2mel_traced(model, input_text, input_phones=None,
                    use_g2p=True, verbose=False, device='cuda',
                    cleaners=("flowtron_cleaners",), transition_agent_bias=0.0, n_tries=3):
    """
    Main method for running model. It should be already traced
    """

    text = input_text
    sequence, cleaned_text, text_phones, bert_feats, max_steps = get_text_for_inference(
        text, verbose, input_phones, use_g2p, device, cleaners=cleaners)

    mel_outputs_postnet_max, diagonality_max, confidence_max = None, 0, 0

    # try n_tries and select best mel-spectrogram
    for _ in range(n_tries):
        with torch.no_grad():
            if transition_agent_bias != 0.0:
                mel_outputs_postnet, mel_lengths, alignments = model(
                    sequence,
                    torch.tensor([sequence.size(1)]).to(sequence.device),
                    bert_feats.to(sequence.device),
                    torch.tensor([len(text.split())]).to(sequence.device),
                    max_decoder_steps=torch.tensor(max_steps).long().to(sequence.device),
                    transition_agent_bias=transition_agent_bias
                )
            else:
                mel_outputs_postnet, mel_lengths, alignments = model(
                    sequence,
                    torch.tensor([sequence.size(1)]).to(sequence.device),
                    bert_feats.to(sequence.device),
                    torch.tensor([len(text.split())]).to(sequence.device),
                    max_decoder_steps=torch.tensor(max_steps).long().to(sequence.device)
                )
            align_reshaped = alignments[:, :, 0].T
            confidence = alignment_confidence_score(align_reshaped, [sequence.size(1)]).item()
            diagonality = alignment_diagonal_score(align_reshaped, [sequence.size(1)]).item()
        if diagonality > diagonality_max or mel_outputs_postnet_max is None:
            mel_outputs_postnet_max = mel_outputs_postnet
            confidence_max = confidence
            diagonality_max = diagonality

    return mel_outputs_postnet_max, cleaned_text, text_phones, mel_outputs_postnet_max.shape[-1] != max_steps, (
        confidence_max, diagonality_max)


class Denormalizer:
    # for traced model
    def __init__(self, stats_path, n_speakers=1):
        self.n_speakers: int = n_speakers
        self.n_mel_channels: int = 80
        self.scalers: Dict[int, StandardScaler] = {}
        self.setup_scaler(stats_path)
        self.inverse_spk_mapping: Dict[int, int] = {i: spk for i, spk in enumerate(sorted(self.scalers))}

    @torch.jit.ignore
    def setup_scaler(self, stats_path):
        stats: Dict = np.load(stats_path, allow_pickle=True).item()
        mel_mean, mel_std = torch.tensor(stats['mel_mean']).cpu(), \
                            torch.tensor(stats['mel_std']).cpu()
        self.scalers = {0: StandardScaler(mel_mean, mel_std)}

    @torch.jit.export
    def denormalize(self, S, speaker=0):
        """denormalize values"""
        # pylint: disable=no-else-return
        S_denorm = S.clone()
        scaler = self.scalers[0]
        return scaler.inverse_transform(S_denorm.T).T
