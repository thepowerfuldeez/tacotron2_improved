# Please prepare train/val split for your dataset, labels should be in LJSpeech format with full paths to wav files
# EXAMPLE:
shuf /storage/george/data/mike/metadata.txt > /storage/george/data/mike/shuf_metadata.txt
mkdir -p /home/george/data/filelists/mike/
cat /storage/george/data/mike/shuf_metadata.txt | head -n 280 > /home/george/data/filelists/mike/train.txt
cat /storage/george/data/mike/shuf_metadata.txt | tail -n 53 > /home/george/data/filelists/mike/val.txt
sed -i -- 's,DUMMY,/storage/george/data/mike/wavs,g' /home/george/data/filelists/mike/*.txt

# Add pauses to dataset
python scripts/3_prepare_mfa.py /storage/george/data/filelists/mike/
python scripts/4_extend_dict.py /storage/george/data/mike/wavs/ /storage/george/data/librispeech-lexicon.txt /storage/george/data/mike/lexicon.txt
conda activate aligner && mfa align /storage/george/data/mike/wavs/ /storage/george/data/mike/lexicon.txt english_us_arpa /storage/george/data/mike/alignments/ -j 16 -v --clean && conda activate tts
PYTHONPATH=. python scripts/make_pauses_from_alignments.py --input_dir /storage/george/data/filelists/mike/ --alignments_path /storage/george/data/mike/alignments/

MODEL_NAME="mike" CONFIG=configs/config_mike.yaml MESSAGE="train mike" OUT_PATH=/storage/george/tts_checkpoints/mike/v1/ RUN_ID="xxxsss1" ./train.sh

# train fre-gan
mkdir l/storage/george/outdir/fregan_checkpoints/mike_ft/
cp /storage/george/outdir/fregan_checkpoints/universal_v1_spk_enc_taco_ft/g_00040000 /storage/george/outdir/fregan_checkpoints/mike_ft/
cp /storage/george/outdir/fregan_checkpoints/universal_v1_spk_enc_taco_ft/do_00040000 /storage/george/outdir/fregan_checkpoints/mike_ft/
# update config.json
cd fre-gan
python train.py --fine_tuning True --config config.json --input_mels_dir /storage/george/data/mike/mels_gen/ --checkpoint_path /storage/george/outdir/fregan_checkpoints/mike_ft/
python trace.py --checkpoint_dir /storage/george/outdir/fregan_checkpoints/mike_ft/ --out_path /storage/george/tts_checkpoints/mike/v1/generator.pth