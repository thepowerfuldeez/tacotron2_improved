# Tacotron 2 (without wavenet)

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). 

## Pre-requisites
Tested on CUDA 11.3 with CUDNN and pytorch 1.8, python 3.8

## Setup
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
2. Clone this repo and cd into this repo: `cd text-to-speech`
3. Create conda env `conda create -n tts python=3.8 && conda activate tts`
4. Install pytorch `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
5. Install python requirements: `pip install -r requirements.txt`
6. Instal sndfile `sudo apt install libsndfile1`
7. Copy filelists/ folder to your data folder and cd into it
8. Update .wav paths: `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/ljspeech/*.txt`
9. Run `python compute_statistics.py --config CONFIG`


In this version of tacotron2 many new tweaks are added. 
- Batching strategy to minimize padding
- Guided diagonal attention loss
- New forward attention
- Ability for pauses in text
- GST vectors and TPSE from BERT for inference
- Traceability for easy deployment including TPSE module

## Training a new voice
1. Get audio - text pairs, good quality, pay special attention to noises and variety of phrases. Clean very short/long/broken phrases
2. Preprocess audio – clean noise, compress in Audacity if needed, normalize amount of silence before/after phrases. 
In preprocessing scripts audio will be normalized to have norm 1.0 and then mel-spectrograms by mean/std once again.
3. Preprocess texts – in order to add pauses use MFA (montreal forced aligner). You can try without this step at first
4. Make mel-spectrograms and compute mean-var stats for normalization 

## Adding pauses
Additionally and optionally, for adding pauses into training data, you need to get text-to-audio alignments.
It is achieved through montreal-forced-aligner: [example](https://montreal-forced-aligner.readthedocs.io/en/latest/example.html)
1. Follow the instructions [here](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html#installation)
2. Download only the librispeech lexicon: [link](https://drive.google.com/open?id=1dAvxdsHWbtA1ZIh3Ex9DPn9Nemx9M1-L) and put it somewhere
3. Prepare lab files for alignment: `python scripts/3_prepare_mfa.py filelists/`
4. (optional) extend the pronunciation lexicon using `python scripts/4_extend_dict.py WAVS_DIR librispeech-lexicon.txt NEW_LEXICON`
5. download english speech model for mfa `mfa model download acoustic english_us_arpa`
6. run the alignment `mfa align -j 32 WAVS_DIR LEXICON english_us_arpa ALIGNMENTS_PATH -v --clean`
7. after successful alignment (usually 20-30 min) run `PYTHONPATH=. python scripts/make_pauses_from_alignments.py --input_dir filelists/ --alignments_path ALIGNMENTS_PATH`

# TLDR
```bash
python compute_statistics.py --config configs/config_jlspeech.yaml && \
python -m multiproc train.py --config configs/config_ljspeech.yaml -m "[frog] train ljspeech" --n_gpus 2
```
after training, make bert vectors & train tpse and also trace model with embedded scale stats

```bash
CHECKPOINT_DIR=/storage/george/outdir/multispeaker_kk7bspm9/ OUT_DIR=/storage/george/tts_checkpoints/multispeaker/v3_1/ ./prepare_checkpoint.sh
```

## Multi-GPU (distributed) and Automatic Mixed Precision Training
`python -m multiproc train.py --output_directory=outdir --log_directory=logdir --n_gpus 2 -m "first run message"`

## Training
`python train.py --output_directory=outdir --log_directory=logdir -m "first run message"`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Tacotron 2] model
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

logs will be sent to wandb.ai. At the first run it will ask you for your credentials. Register and paste into terminal.
Good results on LJSpeech are expected to come after 6 hours of training on 2 RTX 3090 with fp16 and batch 128 on each gpu (10k iters)

## After model training
1. Run `python make_bert_vectors.py`
2. Train GST-TPSE: `python train_tpse.py --vectors_dir BERT_VECTORS_DIR --mels_dir MELS_DIR -c CHECKPOINT_PATH`
3. (optional) train pause predictor if you have pauses in data: `python train_pause_predictor.py --vectors_dir BERT_VECTORS_DIR`

## Pre-alignment
If using guided pre-alignment you must repeat steps 3 and 5 again:
1. Prepare lab files for new text/wavs: `python scripts/3_prepare_mfa.py filelists_pauses/`
2. run the alignment `mfa align -j 32 WAVS_DIR LEXICON english ALIGNMENTS_PATH -v --clean`
3. Make pre alignments as .npy files `python make_pre_alignment.py --config config_path`

!!! WARNING !!! this script will rewrite some existing wav files (if they have trailing or leadning silence) and also split some wavs if they have long pauses inside.
After running you will have new folder called filelists_pauses/ with new labels. You need to change `training_files` and `validation_files` params in hparams.py


## Inference and demo
1. Download published [HiFi-GAN](https://github.com/jik876/hifi-gan) model
2. `jupyter notebook --ip=0.0.0.0 --no-browser`
3. Load inference.ipynb, follow details inside for tracing of vocoder
4. Additionally, you can run streamlit demo, see `demo.py`

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel vocoder were trained on the same mel-spectrogram representation. 


## Related repos
[HiFi-GAN](https://github.com/jik876/hifi-gan) 


## FAQ
1. Good alignment, starts at the bottom left, end at the upper-right (excluding padding), line is bold, without notable plateau bc of pauses, monotonically increasing 
![](alignment.jpg)
2. You shold catch the moment, when align.error drops, good numbers usually bellow 0.5
3. When training tpse with l1 loss, it should be lower as the average l1 between every vector. 0.04 and less is a good number
4. padding should not be too big in a batch, as it harms gate loss and training speed
5. same applies for trailing or leading silence in audio – it greatly harms alignment training.
6. validation gate loss might go up after training gate loss is still lowering, it is ok
7. You can try setting higher weight_decay, p_attention_dropout until alignment converges
8. Setting n_frames_per_step=2 increases discrepancy between prenet features from previous gt_frame in teacher forcing mode and
also speeds up training in ~2 times as absolute number of frames to predict also decreases. It usually speeds up training and helps
with alignment, but might slightly harm quality of audio. But you can fine-tune with n_frames=1 after alignment converges.
