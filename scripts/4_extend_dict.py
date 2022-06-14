ABOUT = """Update pronunciation dictionary using g2p library (only english)"""

from pathlib import Path
import argparse
from g2p_en import G2p

from tqdm.auto import tqdm

g2p_fn = G2p()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", type=Path, help="directory with downloaded audio files")
    p.add_argument("old_dictionary", type=Path, help="old dictionary prepared for mfa (usually librispeech)")
    p.add_argument("out_file", type=Path, help="new dictionary")

    return p.parse_args()


def extend_dict(input_dir, old_dict_path, out_file):
    # create montreal-forced-aligner corpus dictionary to better alignmets with less unk tokens
    # merge data with librispeech-corpus to get better alignment results
    lines = old_dict_path.read_text().strip('\n').split('\n')
    old_keys = set([x.split(" ", 1)[0] for x in lines])
    c = 0
    for text_file in tqdm(list(input_dir.glob("**/*.lab"))):
        words = [w.strip('.,!?()"').upper().replace("\n", "") for w in text_file.read_text().split()]
        for w in words:
            if w not in old_keys:
                phone = " ".join(filter(lambda s: s not in " '.-?!", g2p_fn(w)))
                if phone.strip():
                    lines.append(f"{w}\t{phone}")
                    c += 1
    out_file.write_text("\n".join(lines))
    print(f"added {c} new keys")


if __name__ == "__main__":
    args = parse_args()
    extend_dict(args.input_dir, args.old_dictionary, args.out_file)
