import os
import time
import argparse
import math
import shutil
from pathlib import Path

import wandb
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2, Tacotron2Loss
from data_utils import TextMelLoader, TextMelCollate
from utils.logger import Tacotron2Logger
from utils.hparams import create_hparams
from utils.utils import alignment_confidence_score, parse_batch, parse_outputs
from utils.distributed import apply_gradient_allreduce


WANDB_PROJECT_NAME = "tts"


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / num_gpus
    else:
        rt = rt // num_gpus
    return rt


def init_distributed(hparams, world_size, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.experiment.dist_backend, init_method=hparams.experiment.dist_url,
        world_size=world_size, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams, inference=False):
    """
    :param inference: include audiopath in outputs of loader
    """
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.data.training_files, hparams)
    trainset.sort_items()
    valset = TextMelLoader(hparams.data.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.model.n_frames_per_step, inference=inference)

    if hparams.experiment.distributed_run:
        # we already sorted values
        shuffle = False
        train_sampler = DistributedSampler(trainset, shuffle=shuffle)
    else:
        # we already sorted values
        train_sampler = None
        shuffle = False  # True

    train_loader = DataLoader(trainset, num_workers=8, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.training.batch_size, pin_memory=False,
                              # drop_last=True if not inference else False,
                              collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger()
    else:
        logger = None
    return logger


def get_model_params(hparams):
    """
    This function parses config and returns kwargs for model constructor
    """

    model_params = dict(
        # optimization
        mask_padding=hparams.training.mask_padding,
        # audio
        n_mel_channels=hparams.data.n_mel_channels,
        # encoder
        n_symbols=hparams.model.n_symbols,
        encoder_kernel_size=hparams.model.encoder_kernel_size,
        encoder_n_convolutions=hparams.model.encoder_n_convolutions,
        encoder_lstm_hidden_dim=getattr(hparams.model, "encoder_lstm_hidden_dim", hparams.model.encoder_embedding_dim),
        encoder_embedding_dim=hparams.model.encoder_embedding_dim,
        # gst
        use_gst=hparams.model.use_gst,
        gst_fusion_type=hparams.model.gst_fusion_type,
        gst_embedding_dim=hparams.model.gst_embedding_dim,
        gst_reference_encoder_dim=hparams.model.gst_reference_encoder_dim,
        gst_num_heads=hparams.model.gst_num_heads,
        gst_num_style_tokens=hparams.model.gst_num_style_tokens,
        # attention
        attention_type=hparams.model.attention_type,
        attention_rnn_dim=hparams.model.attention_rnn_dim,
        attention_dim=hparams.model.attention_dim,
        # attention location
        attention_location_n_filters=hparams.model.attention_location_n_filters,
        attention_location_kernel_size=hparams.model.attention_location_kernel_size,
        # windowing Attention
        windowing_attention=hparams.model.windowing_attention,
        win_attention_back=hparams.model.win_attention_back,
        win_attention_front=hparams.model.win_attention_front,
        # decoder
        n_frames_per_step=hparams.model.n_frames_per_step,
        decoder_rnn_dim=hparams.model.decoder_rnn_dim,
        prenet_dim=hparams.model.prenet_dim,
        prenet_noise=hparams.model.prenet_noise,
        max_decoder_steps=hparams.model.max_decoder_steps,
        gate_threshold=hparams.model.gate_threshold,
        use_zoneout=hparams.model.use_zoneout,
        p_attention_dropout=hparams.model.p_attention_dropout,
        p_decoder_dropout=hparams.model.p_decoder_dropout,
        p_teacher_forcing=hparams.model.p_teacher_forcing,
        # postnet
        postnet_embedding_dim=hparams.model.postnet_embedding_dim,
        postnet_kernel_size=hparams.model.postnet_kernel_size,
        postnet_n_convolutions=hparams.model.postnet_n_convolutions,
        postnet_bn_disable_running_stats=getattr(hparams.model, "postnet_bn_disable_running_stats", False),
    )
    return model_params


def load_model(hparams, inference=False, device='cuda'):
    model_params = get_model_params(hparams)
    if inference:
        from inference import StandardScaler, Denormalizer
        class Tacotron2__forward_is_infer(Tacotron2):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                denormalizer = Denormalizer(hparams.data.stats_path)
                self.scaler: StandardScaler = denormalizer.scalers[0]

            def forward(self, inputs, input_lengths, bert_feats, bert_lens, max_decoder_steps,
                        transition_agent_bias: float = 0.0):
                out = self.infer_singlespeaker(inputs, input_lengths, bert_feats, bert_lens, max_decoder_steps,
                                               transition_agent_bias)
                mel_outputs_postnet, mel_lengths, alignments = out
                mel_outputs_postnet = self.scaler.inverse_transform(
                    mel_outputs_postnet.cpu().transpose(1, 2)).transpose(1, 2).to(alignments.device)
                return mel_outputs_postnet, mel_lengths, alignments
        model = Tacotron2__forward_is_infer(**model_params).to(device)
    else:
        model = Tacotron2(**model_params).to(device)

    if hparams.experiment.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def save_checkpoint(model, optimizer, scaler, learning_rate, iteration, run_id,
                    output_dir, model_name, local_rank):
    if local_rank == 0:
        checkpoint = {'iteration': iteration,
                      'scaler': scaler.state_dict() if scaler is not None else None,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'learning_rate': learning_rate,
                      'run_id': run_id}

        checkpoint_filename = f"checkpoint_{model_name}_{iteration}.pt"
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")

        symlink_src = checkpoint_filename
        symlink_dst = os.path.join(
            output_dir, "checkpoint_{}_last.pt".format(model_name))
        if os.path.exists(symlink_dst) and os.path.islink(symlink_dst):
            print("Updating symlink", symlink_dst, "to point to", symlink_src)
            os.remove(symlink_dst)

        os.symlink(symlink_src, symlink_dst)


def get_last_checkpoint_filename(output_dir, model_name):
    symlink = os.path.join(output_dir, "checkpoint_{}_last.pt".format(model_name))
    if os.path.exists(symlink):
        print("Loading checkpoint from symlink", symlink)
        return os.path.join(output_dir, os.readlink(symlink))
    else:
        print("No last checkpoint available - starting from epoch 0 ")
        return ""


def load_checkpoint(checkpoint_path, model,
                    scaler=None, optimizer=None, ignore_layers=()):
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    if len(ignore_layers) > 0:
        print(f"Warm starting model from checkpoint '{checkpoint_path}'")
        dummy_dict = model.state_dict()

        ignore_load = set()
        for k, v in checkpoint_dict['state_dict'].items():
            for ignore_k in ignore_layers:
                if ignore_k.endswith("*") and k.startswith(ignore_k[:-1]):
                    ignore_load.add(k)

            if k in ignore_layers or k not in dummy_dict:
                ignore_load.add(k)

        model_dict = {k: v for k, v in checkpoint_dict['state_dict'].items()
                      if k not in ignore_load and k in dummy_dict}
        dummy_dict.update(model_dict)
        checkpoint_dict['state_dict'] = dummy_dict
        print(f"discarded {len(ignore_load)} keys")

    model.load_state_dict(checkpoint_dict['state_dict'])
    if scaler is None and optimizer is None:
        return model
    if len(ignore_layers) == 0:
        if scaler is not None and "scaler" in checkpoint_dict:
            scaler.load_state_dict(checkpoint_dict['scaler'])
        if optimizer is not None and "optimizer" in checkpoint_dict:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])

    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    run_id = checkpoint_dict.get('run_id')
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, scaler, optimizer, learning_rate, iteration, run_id


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=8,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        val_gate_loss = 0.0
        val_attn_loss = 0.0
        val_align_error = 0.0
        for i, batch in enumerate(val_loader):
            batch = parse_batch(batch)
            outputs = model(batch)
            outputs = parse_outputs(outputs, model.mask_padding,
                                    model.decoder.n_mel_channels, model.decoder.n_frames_per_step)

            texts, input_lengths = batch['text_padded'].cpu(), batch['input_lengths'].cpu()
            align_error = 1 - alignment_confidence_score(outputs['alignments'], input_lengths)

            loss, gate_loss, attn_loss = criterion(outputs, batch)

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
                reduced_gate_loss = reduce_tensor(gate_loss.data, n_gpus).item()
                reduced_attn_loss = reduce_tensor(attn_loss.data, n_gpus).item()
                reduced_align_error = reduce_tensor(align_error.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
                reduced_gate_loss = gate_loss.item()
                reduced_attn_loss = attn_loss.item()
                reduced_align_error = align_error.item()
            val_loss += reduced_val_loss
            val_gate_loss += reduced_gate_loss
            val_attn_loss += reduced_attn_loss
            val_align_error += reduced_align_error
        val_loss = val_loss / (i + 1)
        val_gate_loss = val_gate_loss / (i + 1)
        val_align_error = val_align_error / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, val_gate_loss, val_attn_loss, val_align_error,
                              model, valset, batch, outputs, texts, input_lengths, iteration)


def inference(model, valset, batch_size, iteration, n_gpus, collate_fn, logger, distributed_run, rank):
    """
    Inference for training log
    """
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=8,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)
        avg_attention_confidence = 0.0
        c = 0
        for j, batch in enumerate(val_loader):
            batch = parse_batch(batch)
            outputs = model.inference(batch['text_padded'], batch['input_lengths'],
                                      gst_reference_mel=batch['mel_padded'].transpose(-1, -2))
            outputs = parse_outputs(outputs, model.mask_padding,
                                    model.decoder.n_mel_channels, model.decoder.n_frames_per_step)
            att_confidence = alignment_confidence_score(outputs['alignments'], batch['input_lengths'])

            if distributed_run:
                reduced_att_confidence = reduce_tensor(att_confidence.data, n_gpus).item()
            else:
                reduced_att_confidence = att_confidence.item()
            avg_attention_confidence += reduced_att_confidence
        avg_attention_confidence = avg_attention_confidence / (j + 1)

    if rank == 0:
        logger.log_inference(outputs, avg_attention_confidence, iteration)
    model.train()


def cos_decay(from_lr, to_lr, epoch, n_epochs):
    learning_rate = to_lr + 0.5 * (from_lr - to_lr) * \
                    (1 + math.cos(3.14 * epoch / n_epochs))
    return learning_rate


def set_seeds(hparams, n_gpus, rank, group_name):
    if hparams.experiment.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)
    torch.cuda.synchronize()

    torch.manual_seed(hparams.experiment.seed)
    torch.cuda.manual_seed(hparams.experiment.seed)
    np.random.seed(hparams.experiment.seed)


def init_data(hparams, inference=False):
    train_loader, valset, collate_fn = prepare_dataloaders(hparams, inference=inference)
    return train_loader, valset, collate_fn


def init_model(hparams, warm_start):
    model = load_model(hparams)
    if getattr(model.decoder.attention, 'windowing'):
        # disable windowing at training
        model.decoder.attention.windowing = False
    init_learning_rate = hparams.training.learning_rate

    train_only_layers = getattr(hparams, 'train_only_layers', None)
    if train_only_layers:
        s = 0
        assert warm_start, "allow_only_layers flag works only with warm_start"
        params = []
        param_names = []
        for name, param in model.named_parameters():
            is_pref = [name.startswith(pref[:-1]) for pref in train_only_layers if pref[-1] == "*"]
            if name in train_only_layers or (len(is_pref) and all(is_pref)):
                params.append(param)
                param_names.append(name)
                s += torch.prod(torch.tensor(param.size()))
        print(f"trainable params: {s}")
        print("params")
        print(param_names)
    else:
        params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=init_learning_rate,
                                 eps=1e-6, betas=(0.9, 0.98),
                                 weight_decay=hparams.training.weight_decay)

    if hparams.experiment.distributed_run:
        model = apply_gradient_allreduce(model)
    return model, optimizer, init_learning_rate


def init_checkpoint(model, scaler, optimizer, train_loader, init_learning_rate, checkpoint_path, warm_start):
    # Load checkpoint if one exists
    ignore_layers = ()
    if warm_start:
        ignore_layers = hparams.checkpoint.ignore_layers
        if not ignore_layers:
            # sample key to perform warm start with no ignore keys
            ignore_layers = ['abc']
        if any([pref[-1] == "*" for pref in ignore_layers]):
            new_ignore_layers = set()
            for name, param in model.named_parameters():
                for pref in ignore_layers:
                    if pref[-1] != "*" and name in ignore_layers:
                        new_ignore_layers.add(name)
                    elif pref[-1] == "*" and name.startswith(pref[:-1]):
                        new_ignore_layers.add(name)
            new_ignore_layers = list(new_ignore_layers)
            print("ignoring params:")
            print(new_ignore_layers)
            ignore_layers = new_ignore_layers
    model, scaler, optimizer, _learning_rate, iteration, run_id = load_checkpoint(
        checkpoint_path, model, scaler, optimizer, ignore_layers)

    if hparams.training.use_saved_learning_rate:
        init_learning_rate = _learning_rate
    if hparams.checkpoint.start_from_iteration != -1 or warm_start:
        iteration = hparams.checkpoint.start_from_iteration
        run_id = None

    iteration += 1  # next iteration is iteration + 1
    epoch_offset = max(0, int(iteration / len(train_loader)))
    print(f"Starting from {iteration} iteration")
    return model, scaler, optimizer, init_learning_rate, iteration, epoch_offset, run_id


def set_learning_rate(optimizer, hparams, iteration, epoch, base_lr, init_learning_rate):
    if iteration < hparams.training.n_warmup_steps:
        # warmup
        learning_rate = max(base_lr, iteration / hparams.training.n_warmup_steps * init_learning_rate)
    else:
        if not hparams.training.disable_lr_decay:
            # cosine decay with steps
            epoch_div = epoch / hparams.training.epochs
            divider = 1 if epoch_div < 0.6 else 10
            max_epochs = int(hparams.training.epochs * 0.6) if epoch_div < 0.6 else int(hparams.training.epochs * 0.4)
            cur_epoch = epoch if epoch_div < 0.6 else epoch - max_epochs

            from_lr = init_learning_rate / divider
            to_lr = init_learning_rate / (divider * 10) if epoch_div < 0.6 else base_lr
            learning_rate = cos_decay(from_lr, to_lr, cur_epoch, max_epochs)
        else:
            learning_rate = init_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return optimizer, learning_rate


def train(output_directory, log_directory, checkpoint_path, config_path,
          warm_start, n_gpus, rank, group_name, scaler, run_message, custom_run_id, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    run_message: run message shown in wandb
    custom_run_id: your run id instead of auto-generated from wandb, use for e2e training
    hparams (object): comma separated list of "name=value" pairs.
    """
    set_seeds(hparams, n_gpus, rank, group_name)

    # initializations
    train_loader, valset, collate_fn = init_data(hparams)
    model, optimizer, init_learning_rate = init_model(hparams, warm_start)
    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    base_lr = 1e-5
    if checkpoint_path is not None:
        model, scaler, optimizer, init_learning_rate, iteration, epoch_offset, run_id = init_checkpoint(
            model, scaler, optimizer, train_loader, init_learning_rate, checkpoint_path, warm_start)
    else:
        iteration = 0
        epoch_offset = 0
        run_id = None

    if custom_run_id is not None:
        run_id = custom_run_id

    if rank == 0:
        run = wandb.init(
            id=run_id,
            project=WANDB_PROJECT_NAME, name=hparams.experiment.name, resume='allow', config=hparams, notes=run_message)
        run_id = run.id

    criterion = Tacotron2Loss(hparams.training.loss_use_masking, hparams.training.loss_bce_pos_weight,
                              hparams.training.loss_guided_attn_weight * (
                                          hparams.training.loss_guided_attn_weight_decay ** iteration),
                              hparams.training.loss_guided_attn_weight_decay,
                              hparams.training.loss_guided_attn_min_value,
                              hparams.model.n_frames_per_step)
    model.train()
    is_overflow = False

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.training.epochs):
        torch.cuda.synchronize()
        t0 = time.time()

        # when we sort items, we shuffle them in groups
        train_loader.dataset.sort_items()
        avg_attention_confidence = 0.0

        print(f"Epoch: {epoch}")
        for i, batch in enumerate(train_loader):
            torch.cuda.synchronize()
            start = time.perf_counter()

            # set teacher forcing if decay is activated
            model.decoder.p_teacher_forcing = hparams.model.p_teacher_forcing * (
                    hparams.model.p_teacher_forcing_decay_rate ** iteration)
            # set new learning rate for warmup and decay
            optimizer, learning_rate = set_learning_rate(
                optimizer, hparams, iteration, epoch, base_lr, init_learning_rate
            )

            optimizer.zero_grad()
            batch = parse_batch(batch)
            mean_mel_len = batch['output_lengths'].float().mean()

            # MODEL INFERENCE
            with torch.cuda.amp.autocast(enabled=hparams.experiment.fp16_run):
                # forward pass model
                outputs = model(batch)
                outputs = parse_outputs(outputs, model.mask_padding,
                                        model.decoder.n_mel_channels, model.decoder.n_frames_per_step)
                loss, gate_loss, attn_loss = criterion(outputs, batch)
                att_confidence = alignment_confidence_score(outputs['alignments'], batch['input_lengths'])

                if hparams.experiment.distributed_run:
                    reduced_attn_loss = reduce_tensor(attn_loss.data, n_gpus).item()
                    reduced_gate_loss = reduce_tensor(gate_loss.data, n_gpus).item()
                    reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                    reduced_att_confidence = reduce_tensor(att_confidence.data, n_gpus).item()
                else:
                    reduced_attn_loss = attn_loss.item()
                    reduced_gate_loss = gate_loss.item()
                    reduced_loss = loss.item()
                    reduced_att_confidence = att_confidence.item()
                avg_attention_confidence += reduced_att_confidence

            # LOSS COMPUTATION
            if hparams.experiment.fp16_run:
                # model optimizer step in mixed precision mode
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.training.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.training.grad_clip_thresh)
                optimizer.step()

            torch.cuda.synchronize()

            # LOGGING
            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                if iteration % hparams.experiment.log_every == 0:
                    logger.log_training(
                        reduced_loss, reduced_gate_loss, reduced_attn_loss,
                        grad_norm, learning_rate, mean_mel_len, duration, iteration
                    )
            if is_overflow and rank == 0:
                print("overflow")

            # VALIDATE AND SAVE CHECKPOINT
            checkpoint_dir = Path(output_directory) / f"{hparams.experiment.name}_{run_id}"
            checkpoint_dir.mkdir(exist_ok=True, parents=True)

            if not is_overflow and (iteration % hparams.experiment.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.training.batch_size, n_gpus, collate_fn, logger,
                         hparams.experiment.distributed_run, rank)
                inference(model, valset, hparams.training.batch_size, iteration, n_gpus, collate_fn,
                          logger, hparams.experiment.distributed_run, rank)

                if rank == 0 and iteration % hparams.experiment.iters_per_checkpoint == 0:
                    shutil.copy(config_path, checkpoint_dir / "config.yaml")
                    save_checkpoint(model, optimizer, scaler, learning_rate, iteration,
                                    run_id, checkpoint_dir, hparams.experiment.name, rank)

            torch.cuda.synchronize()

            iteration += 1
        avg_attention_confidence = avg_attention_confidence / (i + 1)
        if rank == 0:
            logger.log_confidence(avg_attention_confidence, iteration, "training")
        t1 = time.time()
        print(f"Time spent on epoch: {t1 - t0:.1f} s.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default='outdir/',
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, default='logdir',
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-m', '--message', type=str, default="",
                        required=False, help="message for training run, e.g. what's changed")
    parser.add_argument('--run_id', type=str, default=None,
                        required=False, help="custom run id to set checkpoint dir")
    parser.add_argument('--config', type=str, default="configs/config_singlespeaker.yaml",
                        required=True, help="config for training")
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=2,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.config)

    torch.backends.cudnn.enabled = hparams.experiment.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.experiment.cudnn_benchmark

    print("FP16 Run:", hparams.experiment.fp16_run)
    print("Distributed Run:", hparams.experiment.distributed_run)
    print("cuDNN Enabled:", hparams.experiment.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.experiment.cudnn_benchmark)

    # scalers for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if hparams.experiment.fp16_run else None

    train(args.output_directory, args.log_directory, args.checkpoint_path, args.config,
          args.warm_start, args.n_gpus, args.rank, args.group_name, scaler, args.message, args.run_id, hparams)
