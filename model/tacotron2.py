import torch
from torch import nn

from model.encoder import Encoder
from model.decoder import Decoder, Postnet
from model.gst import GST, TextEncoder


class Tacotron2(nn.Module):
    def __init__(self, mask_padding, n_mel_channels, n_frames_per_step, max_decoder_steps,

                 n_symbols, encoder_kernel_size,
                 encoder_n_convolutions, encoder_embedding_dim,
                 encoder_lstm_hidden_dim,

                 use_gst, gst_fusion_type, gst_embedding_dim,
                 gst_reference_encoder_dim, gst_num_heads, gst_num_style_tokens,

                 attention_type, attention_rnn_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,

                 decoder_rnn_dim, prenet_dim, prenet_noise,

                 gate_threshold, use_zoneout, p_attention_dropout, p_decoder_dropout,
                 p_teacher_forcing,

                 postnet_embedding_dim, postnet_kernel_size,
                 postnet_n_convolutions, postnet_bn_disable_running_stats,
                 windowing_attention, win_attention_back, win_attention_front):
        super(Tacotron2, self).__init__()
        self.mask_padding = mask_padding
        self.n_frames_per_step = n_frames_per_step
        self.encoder = Encoder(n_symbols, encoder_n_convolutions,
                               encoder_embedding_dim, encoder_lstm_hidden_dim, encoder_kernel_size)
        self.decoder = Decoder(n_mel_channels, n_frames_per_step,
                               max_decoder_steps,
                               encoder_lstm_hidden_dim * 2,  # as we are using bidirectional lstm in encoder
                               use_gst, gst_fusion_type, gst_embedding_dim,
                               attention_type, attention_dim, attention_location_n_filters,
                               attention_location_kernel_size,
                               attention_rnn_dim, decoder_rnn_dim,
                               prenet_dim, prenet_noise, gate_threshold,
                               use_zoneout,
                               p_attention_dropout, p_decoder_dropout, p_teacher_forcing,
                               windowing_attention, win_attention_back, win_attention_front)
        self.postnet = Postnet(n_mel_channels, postnet_embedding_dim,
                               postnet_kernel_size,
                               postnet_n_convolutions, postnet_bn_disable_running_stats)

        self.use_gst = use_gst
        self.gst_fusion_type = gst_fusion_type
        if self.use_gst:
            self.gst = GST(gst_reference_encoder_dim, n_mel_channels,
                           gst_num_heads, gst_num_style_tokens, gst_embedding_dim)

    def forward(self, batch):
        text_lengths, output_lengths = batch['input_lengths'].data, batch['output_lengths'].data
        encoder_outputs = self.encoder(batch['text_padded'], text_lengths)

        if self.use_gst:
            style_emb = self.gst(batch['mel_padded'], output_lengths)
            style_emb = style_emb.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            if self.gst_fusion_type == 'concat':
                encoder_outputs = torch.cat((encoder_outputs, style_emb), -1)
            else:
                encoder_outputs = encoder_outputs + style_emb

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, batch['mel_padded'],
            memory_lengths=text_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        output = {
            "mel_outputs": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "gate_outputs": gate_outputs,
            "alignments": alignments,
            "mel_lengths": batch['output_lengths']
        }
        return output

    def inference_tpse(self, encoder_outputs, input_lengths, bert_feats):
        bert_lens = torch.tensor([bert_feat.size(0) for bert_feat in bert_feats],
                                 device=encoder_outputs.device)
        return self.gst.inference(encoder_outputs, input_lengths, bert_feats, bert_lens)

    def get_gst_vector(self, gst_vector_ind, gst_vector_coef=0.3):
        query = torch.zeros(
            1, 1, self.gst.ref_encoder_dim,
            device=self.gst.style_token_layer.style_tokens.device,
            dtype=self.gst.style_token_layer.style_tokens.dtype
        )
        GST = torch.tanh(self.gst.style_token_layer.style_tokens)
        key = GST[gst_vector_ind].unsqueeze(0).expand(1, -1, -1) * gst_vector_coef
        return self.gst.style_token_layer.attention(query, key).squeeze(1)

    def gst_style_transfer(self, reference_mel, out_lens=None):
        # reference_mel: shape [mel_len, num_mels] or [batch_size, mel_len, num_mels]
        if reference_mel.ndim != 3:
            reference_mel = reference_mel.unsqueeze(0)
        device = self.gst.style_token_layer.style_tokens.device
        dtype = self.gst.style_token_layer.style_tokens.dtype
        if out_lens is None:
            out_lens = torch.tensor([reference_mel.size(1)]).repeat(reference_mel.size(0))
        # batch_size, 1, token embedding size
        return self.gst(reference_mel.to(device).to(dtype), out_lens).squeeze(1)

    def inference(self, inputs, input_lengths,
                  bert_feats=None, max_decoder_steps=None,
                  gst_vector_ind=None, gst_vector_coef=0.3, gst_reference_mel=None):
        """
        Perform inference of Tacotron2 model
        :param inputs: text inputs (already converted to phonemes if needed, and numbers afterwise)
        :param bert_feats: bert features for TPSE-head
        :param verbose: print gate output value at every decoder step
        :param max_decoder_steps: maximum decoder steps for inference, defined by heuristic based on text len
        :param speaker: speaker index for multispeaker setting, defaults to 0 as in speaker adaptation setting
        # for speaker one can also expect ready-to-use speaker embedding

        :param gst_vector_ind: GST trained vector index for inference, instead of style vector
        :param gst_vector_coef: GST trained vector coefficient for inference, defaults to 0.3 as in paper
        :return:
        """
        encoder_outputs = self.encoder.infer(inputs, input_lengths)

        if self.use_gst:
            # first branch: inference using tpse model
            if hasattr(self.gst, "text_encoder") and bert_feats is not None:
                style_emb = self.inference_tpse(encoder_outputs, input_lengths, bert_feats)
            # second branch: inference using specific gst token index
            elif gst_vector_ind is not None:
                style_emb = self.get_gst_vector(gst_vector_ind, gst_vector_coef)
            # third branch: inference using reference mel spectrogram
            elif gst_reference_mel is not None:
                style_emb = self.gst_style_transfer(gst_reference_mel)
            # not using gst at inference
            else:
                style_emb = torch.zeros(encoder_outputs.size(0), self.gst.embedding_dim,
                                        device=encoder_outputs.device, dtype=encoder_outputs.dtype)

            style_emb = style_emb.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            if self.gst_fusion_type == 'concat':
                encoder_outputs = torch.cat((encoder_outputs, style_emb), -1)
            else:
                encoder_outputs = encoder_outputs + style_emb

        if max_decoder_steps is None:
            max_decoder_steps = self.decoder.max_decoder_steps
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
            encoder_outputs, input_lengths, max_decoder_steps)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = {
            "mel_outputs": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "gate_outputs": gate_outputs,
            "alignments": alignments,
            "mel_lengths": mel_lengths
        }
        return outputs

    def init_infer(self, encoder_embedding_dim,
                   bert_embedding_dim, gst_embedding_dim,
                   gst_tpse_gru_hidden_size, gst_tpse_num_layers,
                   tpse_predictor_path):
        "Load tpse predictor model and put it into inference graph"
        assert self.use_gst
        self.gst.text_encoder = TextEncoder(encoder_embedding_dim,
                                            bert_embedding_dim, gst_embedding_dim,
                                            gst_tpse_gru_hidden_size, gst_tpse_num_layers)
        self.gst.text_encoder.load_state_dict(torch.load(tpse_predictor_path, map_location='cpu'))

    def infer_encoder_outputs(self, inputs, input_lengths, bert_feats, bert_lens):
        # for tracing
        encoder_outputs = self.encoder.infer(inputs, input_lengths)

        if self.use_gst:
            style_emb = self.gst_text_encoder(encoder_outputs, input_lengths, bert_feats, bert_lens)
            style_emb = style_emb.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            if self.gst_fusion_type == 'concat':
                encoder_outputs = torch.cat((encoder_outputs, style_emb), -1)
            else:
                encoder_outputs = encoder_outputs + style_emb
        return encoder_outputs

    def infer_decoder_with_postnet(self, encoder_outputs, input_lengths, max_decoder_steps,
                                   transition_agent_bias: float):
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
            encoder_outputs, input_lengths, max_decoder_steps, transition_agent_bias
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        bs = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, bs, bs).transpose(0, 2)

        return mel_outputs_postnet, mel_lengths, alignments

    def infer_singlespeaker(self, inputs, input_lengths, bert_feats, bert_lens, max_decoder_steps,
                            transition_agent_bias: float = 0.0):
        encoder_outputs = self.infer_encoder_outputs(inputs, input_lengths, bert_feats, bert_lens)
        return self.infer_decoder_with_postnet(encoder_outputs, input_lengths, max_decoder_steps, transition_agent_bias)
