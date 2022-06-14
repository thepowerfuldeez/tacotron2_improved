import argparse
import pickle
from pathlib import Path
from itertools import groupby

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

from utils.hparams import create_hparams
from text.cleaners import english_cleaners

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_vectors_for_words(model, tokenizer, text, device='cuda'):
    # text – 5 words
    pt_batch = tokenizer(text.split(), return_tensors='pt', is_split_into_words=True).to(device)
    # pt_batch – 5+ vectors (we know correspondence for which words)
    model = model.to(device)
    out = model(**pt_batch)

    i = 0
    groups = []
    for gr in [list(g) for k, g in groupby(pt_batch.word_ids())]:
        new_gr = []
        for it in gr:
            if it is not None:
                new_gr.append(i)
            i += 1
        if new_gr:
            groups.append(new_gr)
    assert len(out.last_hidden_state.squeeze(0).size()) == 2
    return out.last_hidden_state.squeeze(0), groups


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Pre-compute bert vectors for TPSE training.")
    parser.add_argument("--config", type=str, default="configs/config_ljspeech.yaml")
    parser.add_argument("--out_dir", required=False, default=None,
                        help="directory to save corresponding bert vectors.")
    args = parser.parse_args()

    hparams = create_hparams(args.config)

    if args.out_dir is None:
        lines = Path(hparams.data.training_files).read_text().strip("\n").split("\n")
        p = Path(lines[0].split("|")[0]).parts
        out_dir = Path(*p[:p.index("wavs")]) / "bert_vectors"
        print(f"--out_dir is None, setting as {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for labels_path in [Path(hparams.data.training_files), Path(hparams.data.validation_files)]:
        lines = labels_path.read_text().strip("\n").split("\n")

        for line in tqdm(lines):
            try:
                name, text = line.split("|")
                name = Path(name).stem
                if (out_dir / f"{name}.p").exists():
                    continue
                text = english_cleaners(text.replace("<p0>", "").replace("<p1>", "").replace("<p2>", ""))
                vectors, groups = get_vectors_for_words(pt_model, tokenizer, text)
                assert len(groups) == len(text.split())
                torch.save(vectors, str(out_dir / f"{name}.pt"))
                (out_dir / f"{name}.p").write_bytes(pickle.dumps(groups))
            except:
                print(f"error on line {line}")
                pass


if __name__ == "__main__":
    main()
