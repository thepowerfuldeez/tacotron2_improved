from argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np
from tqdm.auto import tqdm

import flash
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from train import load_model, load_checkpoint, init_data
from utils.hparams import create_hparams
from model.gst import TextEncoder


class StyleDs(Dataset):
    """bert vectors, encoder_outs -> gst vector"""

    def __init__(self, encoder_outs, bert_vectors, gst_vectors):
        super().__init__()
        self.bert_vectors = bert_vectors
        self.encoder_outs = encoder_outs
        self.gst_vectors = gst_vectors

    def __getitem__(self, i):
        return (torch.tensor(self.encoder_outs[i]).float(),
                torch.tensor(self.bert_vectors[i]).float()), torch.tensor(self.gst_vectors[i]).float()

    def __len__(self):
        return len(self.bert_vectors)


class EncBertCollate:
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, bert_feats]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0][0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        encoder_outs_padded = torch.FloatTensor(len(batch), max_input_len, batch[0][0][0].size(-1))
        encoder_outs_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            encoder_outs = batch[ids_sorted_decreasing[i]][0][0]
            encoder_outs_padded[i, :encoder_outs.size(0)] = encoder_outs

        bert_vectors_padded = torch.FloatTensor(len(batch), max([x[0][1].size(0) for x in batch]),
                                                batch[0][0][1].size(-1))
        bert_vectors_padded.zero_()
        bert_vectors_lenghts = torch.LongTensor(len(batch))
        style_vectors = torch.stack([x[1][0] for x in batch])
        for i in range(len(ids_sorted_decreasing)):
            bert_vectors = batch[ids_sorted_decreasing[i]][0][1]
            bert_vectors_padded[i, :bert_vectors.size(0), :] = bert_vectors
            bert_vectors_lenghts[i] = bert_vectors.size(0)

        input_lengths = input_lengths.cpu()
        bert_vectors_lenghts = bert_vectors_lenghts.cpu()

        return (encoder_outs_padded, input_lengths, bert_vectors_padded, bert_vectors_lenghts), style_vectors


class RegressorTask(flash.Task):
    def __init__(
            self,
            model,
            loss_fn,
            # scheduler,
            # scheduler_kwargs,
            optimizer,
            metrics=None,
            learning_rate: float = 1e-3,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            # scheduler=scheduler,
            # scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
        )
        self.save_hyperparameters()
        self.model = model

    def forward(self, x):
        encoder_outs_padded, input_lengths, bert_vectors_padded, bert_vectors_lenghts = x
        return self.model(encoder_outs_padded, input_lengths, bert_vectors_padded, bert_vectors_lenghts)


def main():
    p = ArgumentParser(description="Train tpse module for estimating GST embedding from BERT text embeddings\n"
                                   "Expects dataset has wavs and mels folders already and created bert vectors "
                                   "with script python make_bert_vectors.py")
    p.add_argument("--vectors_dir", required=False, help="if not entered, guess from data path in config")
    p.add_argument('-c', '--checkpoint_dir', type=Path, required=True, help='checkpoint directory (with config)')
    args = p.parse_args()

    checkpoint_dir = args.checkpoint_dir
    config_file = checkpoint_dir / "config.yaml"
    hparams = create_hparams(config_file)
    hparams.experiment.distributed_run = False

    if args.vectors_dir is None:
        lines = Path(hparams.data.training_files).read_text().strip("\n").split("\n")
        p = Path(lines[0].split("|")[0]).parts
        vectors_dir = Path(*p[:p.index("wavs")]) / "bert_vectors"
        print(f"--vectors_dir is None, setting as {vectors_dir}")
        assert vectors_dir.exists(), f"{vectors_dir} must exist"
    else:
        vectors_dir = Path(args.vectors_dir)

    lines = []
    for labels_path in [Path(hparams.data.training_files), Path(hparams.data.validation_files)]:
        lines += labels_path.read_text().strip("\n").split("\n")
    line_id2text = {
        Path(str(line.split("|")[0]).replace("wavs", "mels")).with_suffix(".pt"): line.split("|")[1] for line in lines
        if len(line.split("|")) == 2
    }

    checkpoint_path = str(sorted(checkpoint_dir.glob("*checkpoint*"), key=lambda p: p.stat().st_mtime)[-1])
    model = load_model(hparams).eval()
    model = load_checkpoint(checkpoint_path, model)
    _ = model.cuda().half()
    _, ds, _ = init_data(hparams)

    encoder_outs = []
    bert_inputs = []
    style_outs = []

    for mel_path in tqdm(list(line_id2text)):
        name = mel_path.name
        if not (vectors_dir / name).exists() or not mel_path.exists():
            continue

        with torch.no_grad():
            mel = torch.load(str(mel_path), map_location='cpu')
            mel = ds.normalize(mel).cuda().half()

            text = line_id2text[mel_path]
            text_norm = ds.get_text(text)
            text_ids = text_norm.cuda().unsqueeze(0)

            encoder_outputs = model.encoder(text_ids, torch.tensor([text_ids.size(1)]))[0].cpu().numpy()
            encoder_outs.append(encoder_outputs)

            style_vector = model.gst_style_transfer(mel.unsqueeze(0)).cpu().numpy()
            style_outs.append(style_vector)

            vectors = torch.load(str(vectors_dir / name), map_location='cpu').numpy()
            bert_inputs.append(vectors)
    assert bert_inputs, f"Couldn't find bert embeddings in folder! See python make_bert_vectors.py"
    print(f"read {len(bert_inputs)} items.")

    np.random.seed(42)
    train_idx = np.random.choice(np.arange(len(bert_inputs)), int(len(bert_inputs) * 0.9), replace=False)
    train_idx_set = set(train_idx)
    val_idx = [i for i in np.arange(len(bert_inputs)) if i not in train_idx_set]

    enc_data = [encoder_outs[i] for i in train_idx]
    bert_data = [bert_inputs[i] for i in train_idx]
    gst_data = [style_outs[i] for i in train_idx]
    if len(enc_data) < 1500:
        for _ in range(1500 // len(enc_data)):
            enc_data += enc_data
            bert_data += bert_data
            gst_data += gst_data
    # data
    train, val = StyleDs(enc_data,
                         bert_data,
                         gst_data), \
                 StyleDs([encoder_outs[i] for i in val_idx],
                         [bert_inputs[i] for i in val_idx],
                         [style_outs[i] for i in val_idx])

    model = TextEncoder(hparams.model.encoder_lstm_hidden_dim * 2,  # as we do bidirectional lstm
                        hparams.model.bert_embedding_dim, hparams.model.gst_embedding_dim,
                        hparams.model.gst_tpse_gru_hidden_size, hparams.model.gst_tpse_num_layers)

    # task
    regressor = RegressorTask(model, loss_fn=nn.functional.l1_loss,
                              # scheduler=ExponentialLR, scheduler_kwargs={"gamma": 0.88},
                              optimizer=optim.Adam, learning_rate=1e-4)

    if (checkpoint_dir/'tpse.ckpt').exists():
        (checkpoint_dir/'tpse.ckpt').unlink()
    callback = ModelCheckpoint(checkpoint_dir, filename="tpse", monitor="val_l1_loss", verbose=True,
                               save_weights_only=True, mode='min')
    # train
    flash.Trainer(gpus=[0], auto_select_gpus=True, max_epochs=70, callbacks=[callback]).fit(
        regressor,
        DataLoader(train, num_workers=16, batch_size=16, shuffle=True, collate_fn=EncBertCollate()),
        DataLoader(val, num_workers=16, batch_size=8, collate_fn=EncBertCollate())
    )

    sd = torch.load(str(checkpoint_dir/'tpse.ckpt'))['state_dict']
    from collections import OrderedDict
    sd_ = OrderedDict()
    for k, v in sd.items():
        if k.startswith("model."):
            sd_[k[len("model."):]] = v
        else:
            sd_[k] = v
    torch.save(sd_, str(checkpoint_dir / "tpse_predictor_weights.pth"))


if __name__ == "__main__":
    main()
