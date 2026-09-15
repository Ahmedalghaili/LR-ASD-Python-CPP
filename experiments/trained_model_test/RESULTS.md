# Newly trained LR-ASD evaluation

Best training-log checkpoint: epoch 32. Final checkpoint: epoch 40.

## Full AVA validation re-evaluation

| Model | AP | Accuracy at 0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| original | 94.4850% | 93.60% | 0.8851 | 0.8560 | 0.8703 |
| best | 94.3302% | 93.46% | 0.9052 | 0.8258 | 0.8637 |
| last | 94.2185% | 93.51% | 0.8887 | 0.8473 | 0.8675 |

Best checkpoint versus the original: **-0.1548 percentage points AP**.

All models were evaluated on the same 8,015 tracks / 768,307 annotated frames. Checkpoint loading was strict, predictions were checked for finiteness, and loader label order was checked against the original CSV. AP uses the repository evaluator and class-1 softmax probabilities.

**This is validation, not an untouched test-set result.** The same validation set was used to select epoch 32 during training; this rerun checks reproducibility and comparison under identical preprocessing. SCRFD is not used for this metric: AVA annotation crops are used.

## C++ / SCRFD smoke tests

| Video | Model | Frames | FPS | Tracks |
|---|---|---:|---:|---:|
| wide | original | 534 | 97.12 | 7 |
| wide | trained | 534 | 92.91 | 7 |
| two_person | original | 802 | 76.78 | 2 |
| two_person | trained | 802 | 76.82 | 2 |

One run per model/video; FPS is a smoke-test observation, not a repeated speed benchmark. The detector is SCRFD-2.5G at 480×288. Video-loop timing excludes model loading, MFCC preparation and warmup. The existing C++ raw class-1-logit speaking threshold (zero) is preserved; it differs from the softmax threshold used for validation accuracy.

## Open annotated outputs

- wide: [trained model](results/20260912_003121/wide_trained.mp4), [original model](results/20260912_003121/wide_original.mp4); speaking agreement 92.76% on 566 matched face observations.
- two_person: [trained model](results/20260912_003121/two_person_trained.mp4), [original model](results/20260912_003121/two_person_original.mp4); speaking agreement 94.70% on 1604 matched face observations.

The personal videos have no manual speaking labels. Agreement is not correctness. Detection boxes should match because both runs use the same detector; the ASD checkpoint determines the speaking scores.

The exported checkpoint is [models/lr_asd_epoch32.pt](results/20260912_003121/models/lr_asd_epoch32.pt). Export was checked against eager PyTorch at 5, 10, 25, 100 and 305 frames. Details are in its `.validation.json` file.

Each model subfolder contains full predictions and metrics. `config.json` records checkpoint hashes and GPU details; `pipeline.json` records exact C++ commands and measurements.
