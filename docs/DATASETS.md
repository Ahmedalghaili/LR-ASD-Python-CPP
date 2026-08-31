# Datasets and test media

Datasets and test videos are deliberately excluded from Git because they are
large and have separate licenses.

## AVA ActiveSpeaker

- [Google Research AVA downloads](https://sites.research.google/gr/ava/download/)
- [AVA download repository](https://github.com/cvdfoundation/ava-dataset)

Prepare data with the upstream LR-ASD workflow:

```bash
python train.py --dataPathAVA AVADataPath --downloadAVA
```

Expected structure:

```text
AVADataPath/
├── clips_audios/{train,val,test}/
├── clips_videos/{train,val,test}/
├── csv/
├── orig_audios/
└── orig_videos/
```

Preprocessing can require hundreds of gigabytes and many hours. Evaluate the
complete validation split with:

```bash
python train.py --dataPathAVA AVADataPath --evaluation
```

The upstream project reports 94.45% full-validation mAP. Our deterministic
5,494-frame subset produced 97.20% mAP; it is only a subset result.

## Columbia ASD

```bash
python Columbia_test.py --evalCol --colSavePath colDataPath
```

The upstream project reports 86.1% average F1 with the AVA checkpoint and 96.4%
with the TalkSet-finetuned checkpoint.

## Personal media

Place personal videos under `demo/` or outside the repository. Video, audio,
image, and generated-output formats are ignored by Git.
