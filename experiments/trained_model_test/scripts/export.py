#!/usr/bin/env python3
"""Export a trained LR-ASD checkpoint and verify variable-length inference."""
import argparse,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(PROJECT))
import torch
from ASD import ASD
from cpp.export_torchscript import LRASD

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
torch.manual_seed(20260912)
net=ASD();net.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True);net.eval()
wrapper=LRASD(net).cuda().eval()
with torch.inference_mode():
    audio=torch.randn(1,100,13,device='cuda');visual=torch.rand(1,25,112,112,device='cuda')*255
    traced=torch.jit.trace(wrapper,(audio,visual),strict=False)
    traced.save(str(a.output));loaded=torch.jit.load(str(a.output),map_location='cuda').eval()
    checks=[]
    for frames in [5,10,25,100,305]:
        au=torch.randn(1,frames*4,13,device='cuda');vi=torch.rand(1,frames,112,112,device='cuda')*255
        expected=wrapper(au,vi);actual=loaded(au,vi)
        assert tuple(actual.shape)==(frames,2)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        checks.append(dict(frames=frames,max_logit_error=float((actual-expected).abs().max())))
a.output.with_suffix('.validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks),flush=True)
