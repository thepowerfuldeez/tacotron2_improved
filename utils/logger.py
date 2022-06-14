import random
import torch
import wandb

from text import sequence_to_text
from .plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy


class Tacotron2Logger:
    def __init__(self):
        pass

    def log_training(self, reduced_loss, reduced_gate_loss, reduced_attn_loss,
                     grad_norm, learning_rate, mean_mel_len, duration, iteration):
        wandb.log({
            "training.loss": reduced_loss,
            "training.gate_loss": reduced_gate_loss,
            "training.attn_loss": reduced_attn_loss,
            "grad.norm": grad_norm,
            "learning.rate": learning_rate,
            "duration": duration,
            "mean_mel_len": mean_mel_len,
        }, step=iteration)

    def log_validation(self, reduced_loss, reduced_gate_loss, reduced_attn_loss, align_error,
                       model, dataset, batch, outputs, texts, input_lengts, iteration):
        mel_outputs, gate_outputs, alignments = outputs['mel_outputs_postnet'], outputs['gate_outputs'], \
                                                outputs['alignments']
        mel_targets, gate_targets = batch['mel_padded'], batch['gate_padded']

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        text = sequence_to_text(texts[idx][:input_lengts[idx]])

        wandb.log({
            "validation.loss": reduced_loss,
            "validation.gate_loss": reduced_gate_loss,
            "validation.attn_loss": reduced_attn_loss,
            "align.error": align_error,
            "alignment": plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            "mel_target": plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            "mel_predicted": plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            "gate": [wandb.Image(plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()))],
        }, step=iteration)

    def log_inference(self, outputs, avg_attention_confidence, iteration):
        mel_outputs, gate_outputs, alignments = outputs['mel_outputs_postnet'], outputs['gate_outputs'], \
                                                outputs['alignments']

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        wandb.log({
            "inference.attention_confidence": avg_attention_confidence,
            "inference.alignment": plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            "inference.mel_predicted": plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy())
        }, step=iteration)

    def log_evaluation(self, wer, iteration):
        wandb.log({
            "evaluation.wer": wer,
        }, step=iteration)

    def log_confidence(self, avg_attention_confidence, iteration, type='inference'):
        wandb.log({
            f"{type}.attention_confidence": avg_attention_confidence,
        }, step=iteration)
