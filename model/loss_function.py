from torch import nn
import torch


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
    """
    return ~make_pad_mask(lengths, xs, length_dim)


# https://github.com/gothiswaysir/Transformer_Multi_encoder/blob/952868b01d5e077657a036ced04933ce53dcbf4c/nets/pytorch_backend/e2e_tts_tacotron2.py#L28-L156
class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.
    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.33, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control how close attention to a diagonal.
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)

        B, mel_T, enc_T = self.guided_attn_masks.shape
        losses = self.guided_attn_masks * att_ws[:, :mel_T, :enc_T]
        loss = torch.sum(losses.masked_select(self.masks)) / torch.sum(olens)  # get mean along B and mel_T
        if self.reset_always:
            self._reset_masks()
        return loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = ilens.shape[0]
        max_ilen = int(ilens.max().item())
        max_olen = int(olens.max().item())
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens.view(n_batches), olens.view(n_batches))):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma)
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen, device=olen.device), torch.arange(ilen, device=ilen.device))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
        """
        in_masks = make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


class Tacotron2Loss(nn.Module):
    def __init__(self, use_masking=True, bce_pos_weight=5.0, loss_guided_attn_weight=1.0,
                 loss_guided_attn_weight_decay=1.0, loss_guided_attn_min_value=0.0,
                 n_frames_per_step=1):
        """
        Tacotron2 Loss with improvements

        use_masking: bool – not compute loss on padded values (True by default)
        bce_pos_weight: float – weight of binary cross entropy loss term (values expected to work 1 to 10).
        Improves stop-token prediction

        loss_guided_attn_weight: float – initial guided attention loss weight. see above
        loss_guided_attn_weight_decay: float – decay term for guided attention loss.
        Better to disable guided attn at the end, so determine this value yourself
        loss_guided_attn_min_value: float – if you don't want to disable guided attn loss completely at the end

        n_frames_per_step: int – very important parameter, it should match decoder n_steps output.
        n_steps > 1 significantly improves attention convergence and training speed overall
        """
        super(Tacotron2Loss, self).__init__()
        self.use_masking = use_masking
        self.n_frames_per_step = n_frames_per_step

        self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_pos_weight))
        self.mse_criterion = nn.MSELoss()
        self.use_guided_attn = loss_guided_attn_weight > 0
        if self.use_guided_attn:
            self.guided_attn_criterion = GuidedAttentionLoss()
            self.guided_attn_weight = loss_guided_attn_weight
            self.guided_attn_weight_decay = loss_guided_attn_weight_decay
            self.guided_attn_min_value = loss_guided_attn_min_value

    def forward(self, model_outputs, batch):
        mel_out = model_outputs['mel_outputs']
        mel_out_postnet = model_outputs['mel_outputs_postnet']
        gate_out = model_outputs['gate_outputs']
        attention_weights = model_outputs['alignments']
        mel_target = batch['mel_padded']
        gate_target = batch['gate_padded']
        output_lengths = batch['output_lengths']
        input_lengths = batch['input_lengths']

        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(output_lengths, mel_out_postnet).to(mel_target.device)
            mel_target = mel_target.masked_select(masks)
            mel_out_postnet = mel_out_postnet.masked_select(masks)
            mel_out = mel_out.masked_select(masks)
            gate_target = gate_target.masked_select(masks[:, 0, :])
            gate_out = gate_out.masked_select(masks[:, 0, :])

        mel_loss = self.mse_criterion(mel_out, mel_target) + self.mse_criterion(mel_out_postnet, mel_target)
        gate_loss = self.bce_criterion(gate_out, gate_target)

        if self.use_guided_attn:
            self.guided_attn_weight *= self.guided_attn_weight_decay
            self.guided_attn_weight = max(self.guided_attn_min_value, self.guided_attn_weight)
            attn_loss = self.guided_attn_criterion(
                attention_weights,
                input_lengths,
                output_lengths // self.n_frames_per_step
            ) * self.guided_attn_weight
        else:
            attn_loss = torch.tensor(0., device=mel_target.device)

        loss = mel_loss + gate_loss + attn_loss
        return loss, gate_loss, attn_loss
