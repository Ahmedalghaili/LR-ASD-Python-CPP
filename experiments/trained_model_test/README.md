# Test the newly trained LR-ASD model

This experiment re-evaluates the best and final checkpoints from
`exps/ava_training_20260910/` and compares them with `weight/pretrain_AVA.model`.
The best checkpoint is selected from the training validation history.

Read [RESULTS.md](RESULTS.md) for the completed results. Raw artifacts are under
`results/<timestamp>/`; `results/latest.txt` identifies the latest run.

The test includes:

1. Full AVA validation evaluation of the best, last and original checkpoints,
   using identical annotation crops and the repository AP evaluator.
2. Strict checkpoint loading, finite-score and label-order checks.
3. TorchScript export of the best checkpoint, checked against eager PyTorch
   for 5, 10, 25, 100 and 305 frames.
4. C++ inference on both existing personal recordings, using SCRFD-2.5G at
   480×288, with original and newly trained LR-ASD weights. Annotated videos,
   predictions, timing and speaking-decision comparisons are retained.

This is a validation recheck, not an independent held-out test result: the
validation set was also used for checkpoint selection during training. The
personal videos do not have manual speaking labels, so model agreement on
those videos is not an accuracy measurement. The C++ tests preserve the
existing raw-logit speaking threshold, whereas validation accuracy uses a
class-1 softmax threshold of 0.5.

From the repository root, run the complete experiment with:

```bash
.venv/bin/python experiments/trained_model_test/scripts/run_test.py
```

Or evaluate an explicit checkpoint:

```bash
.venv/bin/python experiments/trained_model_test/scripts/evaluate.py \
  --checkpoint exps/ava_training_20260910/model/model_0032.model \
  --output experiments/trained_model_test/results/manual_epoch32
```

The output directory for a standalone evaluation must be new. This experiment
uses the existing CUDA environment, prepared AVA validation clips and the
built `experiments/scrfd_cpp/` executable/models. It creates new outputs and
does not replace the original pretrained checkpoint or existing C++ exports.

For the best exported model, use the exact path shown in the results report
with the SCRFD runner's `--asd` option. Example after this run:

```bash
bash experiments/scrfd_cpp/run.sh \
  --detector scrfd \
  --model experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx \
  --asd /path/from/report/models/lr_asd_epoch32.pt \
  --video input.mp4 --audio matching_16k_mono.wav --output output.mp4 \
  --width 480 --height 288 --device cuda
```

Models, full predictions, logs and videos in `results/` are ignored by Git.
The human-readable combined results report remains in the experiment folder.
