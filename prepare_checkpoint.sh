PYTHONPATH=. python scripts/make_bert_vectors.py --config "$CHECKPOINT_DIR"/config.yaml && \
python train_tpse.py --checkpoint_dir "$CHECKPOINT_DIR" && \
mkdir -p "$OUT_DIR" && \
python trace.py --checkpoint_dir "$CHECKPOINT_DIR" --save_path "$OUT_DIR"/tacotron2.jit #--disable_tpse
