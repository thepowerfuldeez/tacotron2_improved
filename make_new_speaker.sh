# Please prepare train/val split for your dataset, labels should be in LJSpeech format with full paths to wav files
# EXAMPLE:
shuf /storage/frappuccino/data/mike/metadata.txt > /storage/frappuccino/data/mike/shuf_metadata.txt
mkdir -p /home/frappuccino/data/filelists/mike/
cat /storage/frappuccino/data/mike/shuf_metadata.txt | head -n 280 > /home/frappuccino/data/filelists/mike/train.txt
cat /storage/frappuccino/data/mike/shuf_metadata.txt | tail -n 53 > /home/frappuccino/data/filelists/mike/val.txt
sed -i -- 's,DUMMY,/storage/frappuccino/data/mike/wavs,g' /home/frappuccino/data/filelists/mike/*.txt

# Add pauses to dataset, but first create separate conda env for aligner
python scripts/3_prepare_mfa.py /storage/frappuccino/data/filelists/mike/
python scripts/4_extend_dict.py /storage/frappuccino/data/mike/wavs/ /storage/frappuccino/data/librispeech-lexicon.txt /storage/frappuccino/data/mike/lexicon.txt
conda activate aligner && mfa align /storage/frappuccino/data/mike/wavs/ /storage/frappuccino/data/mike/lexicon.txt english_us_arpa /storage/frappuccino/data/mike/alignments/ -j 16 -v --clean && conda activate tts
PYTHONPATH=. python scripts/make_pauses_from_alignments.py --input_dir /storage/frappuccino/data/filelists/mike/ --alignments_path /storage/frappuccino/data/mike/alignments/

# main script for full training of tacotron2
MODEL_NAME="mike" CONFIG=configs/config_mike.yaml MESSAGE="train mike" OUT_PATH=/storage/frappuccino/tts_checkpoints/mike/v1/ RUN_ID="xxxsss1" ./train.sh

# train fre-gan
mkdir /home/frappuccino/outdir/fregan_checkpoints/mike_ft/
# here I fine-tune different fre-gan version, you can first pre-train fre-gan on LJSpeech to have this folder
cp /storage/frappuccino/outdir/fregan_checkpoints/universal_v1_spk_enc_taco_ft/g_00040000 /storage/frappuccino/outdir/fregan_checkpoints/mike_ft/
cp /storage/frappuccino/outdir/fregan_checkpoints/universal_v1_spk_enc_taco_ft/do_00040000 /storage/frappuccino/outdir/fregan_checkpoints/mike_ft/
# update config.json
cd fre-gan
python train.py --fine_tuning True --config config.json --input_mels_dir /storage/frappuccino/data/mike/mels_gen/ --checkpoint_path /storage/frappuccino/outdir/fregan_checkpoints/mike_ft/
python trace.py --checkpoint_dir /storage/frappuccino/outdir/fregan_checkpoints/mike_ft/ --out_path /storage/frappuccino/tts_checkpoints/mike/v1/generator.pth