import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """NN module – second prediction pathway of TP-GST,
    creating fixed size text embedding from set of embeddings from Tacotron's text encoder AND BERT outputs
    and predicting gst embedding via mlp
    you should use L1 loss with stop-gradient as stated in paper
    """

    def __init__(self, tacotron_encoder_dim, bert_embedding_dim, gst_embedding_dim,
                 gru_hidden_size, mlp_num_layers, use_bert=True):
        """
        :param: embedding_dim: size of tacotron text encoder embedding
        """
        super().__init__()
        text_embedding_dim = gru_hidden_size
        bert_text_embedding_dim = gru_hidden_size

        self.tacotron_encoder_recurrence = nn.GRU(
            input_size=tacotron_encoder_dim,
            hidden_size=text_embedding_dim,
            batch_first=True)

        self.bert_encoder_recurrence = nn.GRU(
            input_size=bert_embedding_dim,
            hidden_size=bert_text_embedding_dim,
            batch_first=True)

        filters = [gru_hidden_size + gru_hidden_size] + [gst_embedding_dim] * mlp_num_layers
        linears = [
            nn.Sequential(nn.Linear(filters[i], filters[i + 1]), nn.BatchNorm1d(filters[i + 1]),
                          nn.ReLU(inplace=True)) if i < mlp_num_layers - 1
            else nn.Sequential(nn.Linear(filters[i], filters[i + 1]), nn.BatchNorm1d(filters[i + 1]))
            for i in range(mlp_num_layers)
        ]
        self.linears = nn.Sequential(*linears)

    @torch.jit.export
    def infer(self, inputs, input_lens, bert_outs, bert_lens):
        device = inputs.device
        input_lens = input_lens.cpu()

        # inputs shape (N, T, E)
        _, out = self.tacotron_encoder_recurrence(
            torch.nn.utils.rnn.pack_padded_sequence(inputs.to(device), input_lens,
                                                    batch_first=True, enforce_sorted=False))
        out = out.transpose(0, 1).contiguous()
        out = out.view(out.size(0), -1)
        # out: tensor [batch_size, encoding_size=128]

        bert_lens = bert_lens.cpu()
        _, out_bert = self.bert_encoder_recurrence(
            torch.nn.utils.rnn.pack_padded_sequence(bert_outs.to(device), bert_lens,
                                                    batch_first=True, enforce_sorted=False))
        out_bert = out_bert.transpose(0, 1).contiguous()
        out_bert = out_bert.view(out_bert.size(0), -1)
        x = torch.cat((out, out_bert), -1)

        x = self.linears(x)
        style_emb = torch.tanh(x)
        return style_emb

    @torch.jit.ignore
    def forward(self, inputs, input_lens, bert_outs, bert_lens):
        device = inputs.device
        self.tacotron_encoder_recurrence.flatten_parameters()

        input_lens = input_lens.cpu()
        # inputs shape (N, T, E)
        _, out = self.tacotron_encoder_recurrence(
            torch.nn.utils.rnn.pack_padded_sequence(inputs.to(device), input_lens,
                                                    batch_first=True, enforce_sorted=False))
        out = out.transpose(0, 1).contiguous()
        out = out.view(out.size(0), -1)
        # out: tensor [batch_size, encoding_size=128]

        self.bert_encoder_recurrence.flatten_parameters()
        bert_lens = bert_lens.cpu()
        _, out_bert = self.bert_encoder_recurrence(
            torch.nn.utils.rnn.pack_padded_sequence(bert_outs.to(device), bert_lens,
                                                    batch_first=True, enforce_sorted=False))
        out_bert = out_bert.transpose(0, 1).contiguous()
        out_bert = out_bert.view(out_bert.size(0), -1)
        x = torch.cat((out, out_bert), -1)

        x = self.linears(x)
        style_emb = torch.tanh(x)
        return style_emb


class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = []
        for i in range(num_layers):
            convs += [
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    # Do not use bias due to the following batch norm
                    bias=False
                ),
                nn.BatchNorm2d(filters[i + 1]),
                nn.ReLU(inplace=True),
            ]

        self.convs = nn.Sequential(*convs)

        post_conv_height = self.calculate_post_conv_height(
            num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height,
            hidden_size=embedding_dim,
            batch_first=True)

    def forward(self, inputs, input_lengths=None):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        x = self.convs(x).transpose(1, 2)

        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]

        self.recurrence.flatten_parameters()
        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            input_lengths = input_lengths.cpu().numpy().astype(int)
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True, enforce_sorted=False)

        _, out = self.recurrence(x)
        # out: 3D tensor [seq_len==1, batch_size, encoding_size=128]

        return out.squeeze(0)

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad,
                                   n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height


class StyleTokenLayer(nn.Module):
    """NN Module attending to style tokens based on prosody encodings."""

    def __init__(self, num_heads, num_style_tokens,
                 reference_encoder_embedding_dim, embedding_dim):
        super().__init__()

        query_dim = reference_encoder_embedding_dim
        key_dim = embedding_dim // num_heads

        gst_embs = torch.randn(num_style_tokens, key_dim)
        self.register_parameter("style_tokens", nn.Parameter(gst_embs))

        self.attention = MultiHeadAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            num_units=embedding_dim,
            num_heads=num_heads)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        prosody_encoding = inputs.unsqueeze(1)
        # prosody_encoding: 3D tensor [batch_size, 1, encoding_size==128]
        tokens = torch.tanh(self.style_tokens).unsqueeze(0).expand(batch_size, -1, -1)
        # tokens: 3D tensor [batch_size, num tokens, token embedding size]
        style_embed = self.attention(prosody_encoding, tokens)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        """
        :param gst_vector_ind: particular index to use
        :param gst_vector_coef: coef for score at gst vector ind, defaults to 0.3
        :return:
        """
        queries = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        queries = torch.stack(torch.split(queries, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(queries, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    """Global Style Token Module for factorizing prosody in speech.

    See https://arxiv.org/pdf/1803.09017
    https://arxiv.org/pdf/1808.01410.pdf
    """

    def __init__(self, gst_reference_encoder_dim, n_mel_channels,
                 num_heads, num_style_tokens, embedding_dim):
        super().__init__()
        self.ref_encoder_dim = gst_reference_encoder_dim
        self.embedding_dim = embedding_dim

        # this encoder should embed mel spectrograms into emotion vector
        self.ref_encoder = ReferenceEncoder(n_mel_channels, gst_reference_encoder_dim)
        self.style_token_layer = StyleTokenLayer(
            num_heads, num_style_tokens, gst_reference_encoder_dim, embedding_dim
        )

    def forward(self, mels, mel_lengths):
        """
        :param: inputs: mel spectrogram
        :param: text_feats: features from tacotron text encoder
        :param: bert_feats: features from bert used on text
        """
        enc_out = self.ref_encoder(mels, input_lengths=mel_lengths)
        style_embed = self.style_token_layer(enc_out).squeeze(1)
        return style_embed

    def inference(self, text_feats, text_lengths, bert_feats, bert_lens):
        return self.text_encoder(text_feats, text_lengths, bert_feats, bert_lens)
