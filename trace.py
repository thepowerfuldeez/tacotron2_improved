import torch
from argparse import ArgumentParser
from pathlib import Path

from utils.hparams import create_hparams
from train import load_model, load_checkpoint
from model.gst import TextEncoder


def trace_model(checkpoints_dir, save_path, tpse=True, gst_vector_coef=0.15, gst_vector_ind=5):
    hparams = create_hparams(checkpoints_dir / "config.yaml")
    hparams.experiment.distributed_run = False

    checkpoint_path = str(sorted(checkpoints_dir.glob("*checkpoint*"), key=lambda p: p.stat().st_mtime)[-1])
    model = load_model(hparams, inference=True)
    _ = load_checkpoint(checkpoint_path, model)
    if model.use_gst:
        if not tpse:
            # USE TO SAVE WITH CONSTANT GST VECTOR

            query = torch.zeros(1, 1, model.gst.ref_encoder_dim).cuda().half()
            GST = torch.tanh(model.gst.style_token_layer.style_tokens)
            key = GST[gst_vector_ind].unsqueeze(0).expand(1, -1, -1) * gst_vector_coef
            style_emb = model.gst.style_token_layer.attention(query, key).squeeze(1)

            class ConstantTextEncoderInfer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gst = style_emb

                def forward(self, inputs, input_lens, bert_outs, bert_lens):
                    return self.gst.to(inputs.device)

            tpse_model = ConstantTextEncoderInfer()
        else:
            if not (checkpoints_dir / "tpse_predictor_weights.pth").exists():
                raise RuntimeError(f"Couldn't find tpse weights in checkpoint folder!!! "
                                   f"Run python train_tpse.py -c {checkpoints_dir} --vectors_dir VECTORS_DIR"
                                   f"or set tpse argument to false!")

            class TextEncoderInfer(TextEncoder):
                def forward(self, inputs, input_lens, bert_outs, bert_lens):
                    return self.infer(inputs, input_lens, bert_outs, bert_lens)

            tpse_model = TextEncoderInfer(hparams.model.encoder_lstm_hidden_dim * 2,  # as we do bidirectional lstm
                                          hparams.model.bert_embedding_dim, hparams.model.gst_embedding_dim,
                                          hparams.model.gst_tpse_gru_hidden_size, hparams.model.gst_tpse_num_layers)
            tpse_model.load_state_dict(torch.load(checkpoints_dir / "tpse_predictor_weights.pth"))

        del model.gst

        _ = tpse_model.cuda().eval().half()
        model.gst_text_encoder = tpse_model

    else:
        class Noop(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inputs, input_lens, bert_outs, bert_lens):
                return torch.zeros(1)

        tpse_model = Noop()
        _ = tpse_model.cuda().eval().half()
        model.gst_text_encoder = tpse_model

    _ = model.cuda().eval().half()
    with torch.jit.optimized_execution(True):
        script = torch.jit.script(model)
    script = torch.jit.freeze(script)

    torch.jit.save(script, save_path)
    print(f"successfully traced and saved model to {save_path}")


if __name__ == "__main__":
    p = ArgumentParser(description="trace tacotron2 model using jit")
    p.add_argument("--checkpoint_dir", "-c", help='checkpoint directory with config and optionally tpse weights',
                   type=Path, required=True)
    p.add_argument("--save_path", type=Path, help='path to saved jit model')
    p.add_argument("--disable_tpse", action='store_true', help='whether or not to disable tpse when loading checkpoint')
    p.add_argument("--gst_ind", type=int, help='global style tokens index to query', default=0)
    args = p.parse_args()

    trace_model(args.checkpoint_dir, args.save_path, tpse=not args.disable_tpse, gst_vector_ind=args.gst_ind)
