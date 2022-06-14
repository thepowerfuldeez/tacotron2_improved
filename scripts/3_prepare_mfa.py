ABOUT = """Script checks audio files in training+val data and writes lab files for mfa"""

from pathlib import Path
import argparse
from tqdm.auto import tqdm

from text.cleaners import english_cleaners


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", type=Path, help="directory with dataset labels in txt format")

    return p.parse_args()


def write_labs(input_dir):
    for labels_path in tqdm(list(input_dir.glob("*.txt"))):
        lines = labels_path.read_text().strip("\n").split("\n")
        for line in lines:
            path, text, *_ = line.split("|")
            path = Path(path)
            if path.exists():
                name = path.name
                text = text.replace("<p0>", "").replace("<p1>", "").replace("<p2>", "")
                clean_text = " ".join([w.strip(".,;:?!-'\"()[]").replace(",", "").replace("..", "").replace("...", "")
                                       for w in english_cleaners(text).split()]).strip()
                (path.parent / name).with_suffix(".lab").write_text(clean_text)


if __name__ == "__main__":
    args = parse_args()
    write_labs(args.input_dir)
