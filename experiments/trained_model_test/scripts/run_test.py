#!/usr/bin/env python3
"""Recheck trained checkpoints and exercise the selected model in C++/SCRFD."""
from datetime import datetime
import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
TRAIN=PROJECT/'exps/ava_training_20260910'
rows=[]
for line in (TRAIN/'score.txt').read_text().splitlines():
    m=re.match(r'(\d+) epoch,.*?mAP ([\d.]+)%',line)
    if m:rows.append((int(m[1]),float(m[2])))
best=max(rows,key=lambda x:x[1])[0];last=max(x[0] for x in rows)
run=ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
(ROOT/'results/latest.txt').write_text(str(run)+'\n')
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')

def status(phase,**extra):
    info=dict(phase=phase,pid=os.getpid(),updated=datetime.now().isoformat(),**extra)
    (run/'status.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
def call(command,log):
    with log.open('w') as f:subprocess.run(command,cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

checkpoints={'best':TRAIN/f'model/model_{best:04d}.model','last':TRAIN/f'model/model_{last:04d}.model','original':PROJECT/'weight/pretrain_AVA.model'}
config=dict(best_epoch=best,last_epoch=last,selection='highest rounded validation AP in training score.txt',
            checkpoints={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in checkpoints.items()},
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip())
(run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
try:
    for name,checkpoint in checkpoints.items():
        status('validation',checkpoint=name)
        call([sys.executable,str(ROOT/'scripts/evaluate.py'),'--checkpoint',str(checkpoint),'--output',str(run/name)],run/f'{name}.log')
    status('export')
    export=run/'models'/f'lr_asd_epoch{best}.pt'
    call([sys.executable,str(ROOT/'scripts/export.py'),'--checkpoint',str(checkpoints['best']),'--output',str(export)],run/'export.log')
    sys.path.insert(0,str(PROJECT/'experiments/scrfd_cpp/scripts'))
    from benchmark import compare
    videos={'wide':PROJECT/'WhatsApp Video 2026-08-30 at 9.46.47 PM.mp4','two_person':PROJECT/'VID_20260824_145739.mp4'}
    pipeline=[]
    for name,video in videos.items():
        audio=run/f'{name}.wav'
        call(['ffmpeg','-v','error','-y','-i',str(video),'-ac','1','-ar','16000','-vn','-c:a','pcm_s16le',str(audio)],run/f'{name}_audio.log')
        for label,model in [('original',PROJECT/'cpp/models/lr_asd.pt'),('trained',export)]:
            status('cpp_video',video=name,model=label)
            output=run/f'{name}_{label}.mp4'
            command=[str(PROJECT/'experiments/scrfd_cpp/run.sh'),'--detector','scrfd',
                     '--model',str(PROJECT/'experiments/scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx'),
                     '--asd',str(model),'--video',str(video),'--audio',str(audio),'--output',str(output),
                     '--width','480','--height','288','--device','cuda','--warmup','10']
            log=run/f'{name}_{label}.log';call(command,log)
            metrics=json.loads(next(line for line in reversed(log.read_text().splitlines()) if line.startswith('{')))
            metrics.update(video=name,model=label,command=command);pipeline.append(metrics)
    comparisons={name:compare(run/f'{name}_original.mp4.csv',run/f'{name}_trained.mp4.csv') for name in videos}
    (run/'pipeline.json').write_text(json.dumps(dict(runs=pipeline,comparisons=comparisons),indent=2)+'\n')
    metrics={name:json.loads((run/name/'metrics.json').read_text()) for name in checkpoints}
    summary=dict(config=config,validation=metrics,pipeline=pipeline,comparisons=comparisons,run=str(run))
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Newly trained LR-ASD evaluation','',f'Best training-log checkpoint: epoch {best}. Final checkpoint: epoch {last}.',
           '', '## Full AVA validation re-evaluation','',
           '| Model | AP | Accuracy at 0.5 | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|']
    for name in ['original','best','last']:
        m=metrics[name];lines.append(f"| {name} | {m['AP_percent']:.4f}% | {m['accuracy_percent']:.2f}% | {m['precision']:.4f} | {m['recall']:.4f} | {m['F1']:.4f} |")
    delta=metrics['best']['AP_percent']-metrics['original']['AP_percent']
    lines+=['',f'Best checkpoint versus the original: **{delta:+.4f} percentage points AP**.',
            '', 'All models were evaluated on the same 8,015 tracks / 768,307 annotated frames. Checkpoint loading was strict, predictions were checked for finiteness, and loader label order was checked against the original CSV. AP uses the repository evaluator and class-1 softmax probabilities.',
            '', '**This is validation, not an untouched test-set result.** The same validation set was used to select epoch 32 during training; this rerun checks reproducibility and comparison under identical preprocessing. SCRFD is not used for this metric: AVA annotation crops are used.',
            '', '## C++ / SCRFD smoke tests','',
            '| Video | Model | Frames | FPS | Tracks |','|---|---|---:|---:|---:|']
    for m in pipeline:lines.append(f"| {m['video']} | {m['model']} | {m['frames']} | {m['pipeline_fps']:.2f} | {m['tracks_created']} |")
    lines+=['','One run per model/video; FPS is a smoke-test observation, not a repeated speed benchmark. The detector is SCRFD-2.5G at 480×288. Video-loop timing excludes model loading, MFCC preparation and warmup. The existing C++ raw class-1-logit speaking threshold (zero) is preserved; it differs from the softmax threshold used for validation accuracy.',
            '', '## Open annotated outputs','']
    for name,c in comparisons.items():
        lines.append(f"- {name}: [trained model]({name}_trained.mp4), [original model]({name}_original.mp4); speaking agreement {c['speaking_agreement_pct']:.2f}% on {c['matched']} matched face observations.")
    lines+=['','The personal videos have no manual speaking labels. Agreement is not correctness. Detection boxes should match because both runs use the same detector; the ASD checkpoint determines the speaking scores.',
            '', f'The exported checkpoint is [models/{export.name}](models/{export.name}). Export was checked against eager PyTorch at 5, 10, 25, 100 and 305 frames. Details are in its `.validation.json` file.',
            '', 'Each model subfolder contains full predictions and metrics. `config.json` records checkpoint hashes and GPU details; `pipeline.json` records exact C++ commands and measurements.','']
    (run/'REPORT.md').write_text('\n'.join(lines))
    (ROOT/'RESULTS.md').write_text('\n'.join(lines).replace('](models/',f'](results/{run.name}/models/').replace(']('+ 'wide_',']('+f'results/{run.name}/wide_').replace(']('+ 'two_person_',']('+f'results/{run.name}/two_person_'))
    status('complete',report=str(run/'REPORT.md'))
except BaseException as error:
    status('failed',error=str(error));raise
