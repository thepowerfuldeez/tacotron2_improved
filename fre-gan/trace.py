import torch
import os
from pathlib import Path
from argparse import ArgumentParser

from generator import FreGAN
# from generator_legacy import HifiGAN
from vocoder_utils import AttrDict
import json


def main():
    p = ArgumentParser()
    p.add_argument("--checkpoint_dir", "-c",
                   help='path to directory containing trained generator', type=Path)
    p.add_argument("--out_path", required=True, help='out path')
    args = p.parse_args()

    checkpoint_file = str(sorted(args.checkpoint_dir.glob("g_*"), key=lambda p: p.stat().st_mtime)[-1])

    config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    device = torch.device('cpu')

    generator = FreGAN(h).to(device)
    # generator = HifiGAN(h).to(device)
    state_dict_g = torch.load(checkpoint_file, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print(f"loaded {checkpoint_file}")

    torch.jit.trace(generator, torch.ones(2, 80, 300).to(device)).save(args.out_path)


if __name__ == "__main__":
    main()
