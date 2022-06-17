# Fre-GAN Vocoder
[Fre-GAN: Adversarial Frequency-consistent Audio Synthesis](https://arxiv.org/abs/2106.02297)

## Training:
```
python train.py --config config.json --input_mels_dir MELS_DIR
python train.py --config config.json --input_mels_dir MELS_GEN_DIR --fine_tuning=True
```

## Citation:
```
@misc{kim2021fregan,
      title={Fre-GAN: Adversarial Frequency-consistent Audio Synthesis}, 
      author={Ji-Hoon Kim and Sang-Hoon Lee and Ji-Hyun Lee and Seong-Whan Lee},
      year={2021},
      eprint={2106.02297},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## References:
* [Hi-Fi-GAN repo](https://github.com/jik876/hifi-gan)
* [WaveSNet repo](https://github.com/LiQiufu/WaveSNet)
