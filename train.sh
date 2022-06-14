PYTHONPATH=. python scripts/compute_statistics.py --config "$CONFIG"

#export RUN_ID=$(date +%s)
# make new config which points to /storage/george/data/filelists/mike_pauses/
export CHECKPOINT_DIR="/home/frappuccino/outdir/${MODEL_NAME}_${RUN_ID}/"
CUDA_VISIBLE_DEVICES=0,1 python -m multiproc train.py --n_gpus 2 --config "$CONFIG" -o /home/george/outdir/ -m "$MESSAGE" --run_id "$RUN_ID"
#CUDA_VISIBLE_DEVICES=0,1 python -m multiproc train.py --n_gpus 2 --config "$CONFIG" -o /storage/george/outdir/ -m "$MESSAGE" --run_id "$RUN_ID" -c /storage/george/outdir/ljspeech_11k_3232lj43/checkpoint_ljspeech_11k_last.pt --warm_start

mkdir -p "$OUT_PATH"
#cp /storage/george/tts_checkpoints/snoop_lj_11k/v11/generator.pth "$OUT_PATH"

CUDA_VISIBLE_DEVICES=0 CHECKPOINT_DIR=$CHECKPOINT_DIR OUT_DIR="$OUT_PATH" ./prepare_checkpoint.sh
CUDA_VISIBLE_DEVICES=0,1 python teacher_forcing.py -c $CHECKPOINT_DIR
# update demo_dictors.json, demo updates itself automatically
# it will take 6 hours for 12k iterations on one gpu