# CONFIG=path to experiment config; MODEL_NAME=speaker name; RUN_ID=unique experiment id;
# MESSAGE=train message; OUT_PATH=final model folder where traced checkpoint would be
PYTHONPATH=. python scripts/compute_statistics.py --config "$CONFIG"

#export RUN_ID=$(date +%s)
# make new config which points to /storage/frappuccino/data/filelists/mike_pauses/
mkdir -p "$OUT_PATH"
export CHECKPOINT_DIR="/home/frappuccino/outdir/${MODEL_NAME}_${RUN_ID}/"

# it will take 6 hours for 12k iterations on one gpu
CUDA_VISIBLE_DEVICES=0,1 python -m multiproc train.py --n_gpus 2 --config "$CONFIG" -o /home/frappuccino/outdir/ -m "$MESSAGE" --run_id "$RUN_ID"

# if needed to continue training or fine-tuning
#CUDA_VISIBLE_DEVICES=0,1 python -m multiproc train.py --n_gpus 2 --config "$CONFIG" -o /home/frappuccino/outdir/ -m "$MESSAGE" --run_id "$RUN_ID" -c /home/frappuccino/outdir/ljspeech_lj_160622_02l/checkpoint_ljspeech_last.pt --warm_start

# usually you can copy traced vocoder from other model if you do this not first time
#cp /storage/frappuccino/tts_checkpoints/snoop_lj_11k/v11/generator.pth "$OUT_PATH"

CUDA_VISIBLE_DEVICES=0 CHECKPOINT_DIR=$CHECKPOINT_DIR OUT_DIR="$OUT_PATH" ./prepare_checkpoint.sh
# update demo_dictors.json with OUT_PATH, demo updates itself automatically