import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from train import init_data, init_distributed, load_checkpoint, load_model
from utils.hparams import create_hparams
from utils.utils import parse_batch


def process_loader(model, loader):
    c = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, batch in tqdm(enumerate(loader)):
            parse_batch(batch)
            outputs = model(batch)
            for path, mel, l in zip(batch['audiopath'],
                                    outputs['mel_outputs_postnet'].cpu().float(),
                                    batch['output_lengths'].cpu()):
                mel = loader.dataset.denormalize(mel[:, :l])
                # if mel.mean().isnan():
                #     mel = mel[:, ~mel.mean(0).isnan()]
                new_path = Path(str(path).replace("wavs", "mels_gen")).with_suffix(".pt")
                new_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(mel, str(new_path))
                c += 1
    print(c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', type=Path, default=None,
                        required=True, help='checkpoint directory (with config)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    config_file = checkpoint_dir / "config.yaml"
    hparams = create_hparams(config_file)
    hparams.experiment.distributed_run = False
    hparams.data.max_mel_len = 3000
    hparams.data.max_seq_len = 900
    if hparams.experiment.distributed_run:
        init_distributed(hparams, args.n_gpus, args.rank, args.group_name)

    checkpoint_path = str(sorted(checkpoint_dir.glob("*checkpoint*"), key=lambda p: p.stat().st_mtime)[-1])
    # note that in order to teacher forcing work correctly, you don't need to put model here in eval() mode
    model = load_model(hparams)
    scaler = torch.cuda.amp.GradScaler() if hparams.experiment.fp16_run else None
    load_checkpoint(checkpoint_path, model, scaler=scaler)
    _ = model.cuda().half()

    train_loader, valset, collate_fn = init_data(hparams, inference=True)
    val_sampler = DistributedSampler(valset, shuffle=False) if hparams.experiment.distributed_run else None
    val_loader = DataLoader(valset, sampler=val_sampler, num_workers=8,
                            shuffle=False, batch_size=hparams.training.batch_size,
                            pin_memory=False, collate_fn=collate_fn)
    process_loader(model, val_loader)
    process_loader(model, train_loader)
