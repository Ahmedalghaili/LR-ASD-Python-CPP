# AVA preparation and LR-ASD training

The existing dataset is at `AVADataPath/`. This job prepares the 120 training
and 33 validation videos, validates the required clips, then automatically runs
40 LR-ASD training epochs from scratch. SCRFD remains the inference detector;
training uses the original AVA annotation boxes and LR-ASD architecture.

Run folder: `exps/ava_training_20260910/`.

```bash
cat exps/ava_training_20260910/status.json
tail -f exps/ava_training_20260910/preprocess.log
# Once status changes from "preparing" to "training":
tail -f exps/ava_training_20260910/train.log
```

The launched process is detached from the terminal. Keep the computer awake.
If interrupted or rebooted, resume the pipeline from the repository root:

```bash
nohup .venv/bin/python -u experiments/ava_training/scripts/pipeline.py \
  >> exps/ava_training_20260910/pipeline.log 2>&1 &
```

A file lock prevents duplicate pipeline instances. Preprocessing resumes at
completed video boundaries using `AVADataPath/preparation/{train,val}/*.json`.
Partially prepared videos are processed again; original videos are preserved.
The job stops if free disk space approaches 20 GiB or validation fails.
It only launches training after `AVADataPath/preparation/READY.json` is written
and all loader tracks have the expected image count and a nonempty audio clip.

Training uses 1,500-frame dynamic batches, eight data-loader workers, learning
rate 0.001 and the original 0.95 learning-rate decay. The first real-data
forward/backward test passed at 1,445 frames with about 7.8 GiB peak allocation.
The preflight update is a disposable test, not the full training run or an
initialization checkpoint. Details are saved in `preflight.json`.

Outputs:

- `config.json`: exact preparation and training commands.
- `status.json`: preparing, training, complete, or failed.
- `preprocess.log`, `train.log`, `pipeline.log`: progress and errors.
- `model/model_0001.model`, etc.: per-epoch model weights.
- `training_state.pt`: atomically saved model, optimizer and scheduler state
  for resuming at the next epoch. An interrupted epoch is repeated.
- `score.txt`, `val_res.csv`: validation mAP history and predictions.

Compatibility fixes were applied to the original trainer: Python 3.12-safe
noise selection (including single-item batches), configurable validation loader
workers, evaluation using the active Python interpreter, percentage parsing,
full optimizer/scheduler checkpoint restoration, and removal of obsolete NumPy
aliases and an unused plotting import in the validation evaluator.

The preprocessing script follows the original audio slicing, annotation boxes,
JPEG encoding and frame-index convention. Consecutive frames are read in order
instead of repeatedly seeking for each face. Crops from the first prepared
video were checked against the original timestamp-seek method. Test videos are
already downloaded and are left available; they are not needed by this training
and validation job.
