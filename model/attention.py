import torch
from torch import nn
from torch.nn import functional as F

from model.layers import ConvNorm, LinearNorm


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,
                 windowing_attention, win_attention_back, win_attention_front):
        super(Attention, self).__init__()
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")
        self.windowing = windowing_attention
        self.win_back = win_attention_back
        self.win_front = win_attention_front
        self.attention_weights = torch.zeros(0)
        self.attention_weights_cum = torch.zeros(0)

    def preprocess_inputs(self, memory):
        return self.memory_layer(memory)

    def init_attention(self, memory):
        B = memory.size(0)
        T = memory.size(1)
        self.attention_weights = torch.zeros([B, T], device=memory.device, dtype=memory.dtype)
        self.attention_weights_cum = torch.zeros([B, T], device=memory.device, dtype=memory.dtype)

    def update_attention(self, attention_weights):
        self.attention_weights_cum += attention_weights

    def get_alignment_energies(self, query, processed_memory):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(2)
        return energies

    def apply_windowing(self, attention, inputs):
        win_idx = torch.argmax(attention, 1).long()[0].item()

        back_win = win_idx - self.win_back
        front_win = win_idx + self.win_front
        if back_win > 0:
            attention[:, :back_win] = -float("inf")
        if front_win < inputs.shape[1]:
            attention[:, front_win:] = -float("inf")

        return attention

    def forward(self, attention_hidden_state, memory, processed_memory, mask,
                transition_agent_bias: float = 0.0):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cumulative attention weights
        mask: binary mask for padded data

        transition_agent_bias: is not used, added only for tracing
        """
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory)
        alignment = alignment.masked_fill(mask, self.score_mask_value)
        if not self.training and self.windowing:
            alignment = self.apply_windowing(alignment, memory)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        self.attention_weights = attention_weights
        self.update_attention(attention_weights)
        return attention_context


class ForwardAttention(nn.Module):
    """Bahdanau Attention with various optional modifications. Proposed below.
    - Location sensitive attnetion: https://arxiv.org/abs/1712.05884
    - Forward Attention: https://arxiv.org/abs/1807.06736 + state masking at inference
    - Using sigmoid instead of softmax normalization
    - Attention windowing at inference time
    Note:
        Location Sensitive Attention is an attention mechanism that extends the additive attention mechanism
    to use cumulative attention weights from previous decoder time steps as an additional feature.
        Forward attention considers only the alignment paths that satisfy the monotonic condition at each
    decoder timestep. The modified attention probabilities at each timestep are computed recursively
    using a forward algorithm.
        Transition agent for forward attention is further proposed, which helps the attention mechanism
    to make decisions whether to move forward or stay at each decoder timestep.
        Attention windowing applies a sliding windows to time steps of the input tensor centering at the last
    time step with the largest attention weight. It is especially useful at inference to keep the attention
    alignment diagonal.
    Args:
        attention_rnn_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the vakue tensor. In general, the value tensor is the output of the encoder layer.
        attention_dim (int): number of channels of the inner attention layers.
        attention_location_n_filters (int): number of location attention filters.
        attention_location_kernel_size (int): filter size of location attention convolution layer.
        windowing (int): window size for attention windowing. if it is 5, for computing the attention, it only considers the time steps [(t-5), ..., (t+5)] of the input.
        forward_attn_mask (int): enable/disable an explicit masking in forward attention. It is useful to set at especially inference time.
    """

    def __init__(
            self,
            attention_rnn_dim,
            embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            windowing=0,
            forward_attn_mask=False,
    ):
        super().__init__()
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

        # transition agent
        self.ta = nn.Linear(attention_rnn_dim + embedding_dim, 1, bias=True)
        self.windowing = windowing
        # self.win_idx = None
        self.forward_attn_mask = forward_attn_mask

        self.attention_weights = torch.zeros(0)
        self.attention_weights_cum = torch.zeros(0)
        self.alpha = torch.zeros(0)
        self.u = torch.zeros(0)

    # def init_win_idx(self):
    #     self.win_idx = -1
    #     self.win_back = 2
    #     self.win_front = 6

    def init_attention(self, memory):
        B = memory.size(0)
        T = memory.size(1)
        self.attention_weights = torch.zeros([B, T], dtype=memory.dtype, device=memory.device)
        self.attention_weights_cum = torch.zeros([B, T], dtype=memory.dtype, device=memory.device)
        self.alpha = torch.cat([torch.ones([B, 1], dtype=memory.dtype),
                                torch.zeros([B, T], dtype=memory.dtype)[:, :-1] + 1e-7], dim=1).to(memory.device)
        self.u = (0.5 * torch.ones([B, 1], dtype=memory.dtype)).to(memory.device)
        # if self.windowing:
        #     self.init_win_idx()

    def preprocess_inputs(self, inputs):
        return self.memory_layer(inputs)

    def update_attention(self, attention_weights):
        self.attention_weights_cum += attention_weights

    def get_alignment_energies(self, query, processed_memory):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(2)
        return energies

    # def apply_windowing(self, attention_weights, memory):
    #     back_win = self.win_idx - self.win_back
    #     front_win = self.win_idx + self.win_front
    #     if back_win > 0:
    #         attention_weights[:, :back_win] = -float("inf")
    #     if front_win < memory.shape[1]:
    #         attention_weights[:, front_win:] = -float("inf")
    #     # this is a trick to solve a special problem.
    #     # but it does not hurt.
    #     if self.win_idx == -1:
    #         attention_weights[:, 0] = attention_weights.max()
    #     # Update the window
    #     self.win_idx = torch.argmax(attention_weights, 1).long()[0].item()
    #     return attention_weights

    def apply_forward_attention(self, attention_weights):
        # forward attention
        fwd_shifted_alpha = F.pad(self.alpha[:, :-1].clone().to(attention_weights.device), (1, 0, 0, 0))
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha + self.u * fwd_shifted_alpha + 1e-8) * attention_weights
        # force incremental alignment
        if not self.training and self.forward_attn_mask:
            _, n = fwd_shifted_alpha.max(1)
            val, _ = alpha.max(1)
            for b in range(attention_weights.shape[0]):
                alpha[b, n[b] + 3:] = 0
                alpha[b, : (n[b] - 1)] = 0  # ignore all previous states to prevent repetition.
                alpha[b, (n[b] - 2)] = 0.01 * val[b]  # smoothing factor for the prev step
        # renormalize attention weights
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha

    def forward(self, attention_hidden_state, memory, processed_memory, mask, transition_agent_bias: float = 0.0):
        """
        shapes:
            query: [B, C_attn_rnn]
            inputs: [B, T_en, D_en]
            processed_inputs: [B, T_en, D_attn]
            mask: [B, T_en]
        """
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory)
        # apply masking
        alignment = alignment.masked_fill(mask, self.score_mask_value)
        # # apply windowing - only in eval mode
        # if not self.training and self.windowing:
        #     alignment = self.apply_windowing(alignment, memory)

        # attention_weights = torch.sigmoid(alignment) / torch.sigmoid(alignment).sum(dim=1, keepdim=True)
        attention_weights = F.softmax(alignment, dim=1)
        self.update_attention(attention_weights)

        # apply forward attention
        attention_weights = self.apply_forward_attention(attention_weights)
        self.alpha = attention_weights

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        self.attention_weights = attention_weights

        # compute transition agent
        ta_input = torch.cat([attention_context, attention_hidden_state.squeeze(1)], dim=-1)
        self.u = torch.sigmoid(self.ta(ta_input) + transition_agent_bias)
        return attention_context
