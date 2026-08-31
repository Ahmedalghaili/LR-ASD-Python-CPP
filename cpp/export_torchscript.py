#!/usr/bin/env python3
"""Export the exact LR-ASD and S3FD checkpoints to TorchScript."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from ASD import ASD
from model.faceDetector.s3fd import S3FD

class LRASD(torch.nn.Module):
    def __init__(self, asd):
        super().__init__(); self.model=asd.model; self.fc=asd.lossAV.FC
    def forward(self,audio,visual):
        a=self.model.forward_audio_frontend(audio)
        v=self.model.forward_visual_frontend(visual)
        x=self.model.forward_audio_visual_backend(a,v)
        return self.fc(x)

class RawS3FD(torch.nn.Module):
    """S3FD convolutional head; decode/NMS intentionally runs in native C++."""
    def __init__(self, net):
        super().__init__(); self.net=net
    def forward(self,x):
        sources=[]
        for k in range(16): x=self.net.vgg[k](x)
        sources.append(self.net.L2Norm3_3(x))
        for k in range(16,23): x=self.net.vgg[k](x)
        sources.append(self.net.L2Norm4_3(x))
        for k in range(23,30): x=self.net.vgg[k](x)
        sources.append(self.net.L2Norm5_3(x))
        for k in range(30,len(self.net.vgg)): x=self.net.vgg[k](x)
        sources.append(x)
        for k,layer in enumerate(self.net.extras):
            x=F.relu(layer(x))
            if k%2==1: sources.append(x)
        loc=[]; conf=[]
        c=self.net.conf[0](sources[0]); c=torch.cat((torch.max(c[:,:3],dim=1,keepdim=True)[0],c[:,3:]),dim=1)
        loc.append(self.net.loc[0](sources[0]).permute(0,2,3,1).contiguous()); conf.append(c.permute(0,2,3,1).contiguous())
        for i in range(1,len(sources)):
            loc.append(self.net.loc[i](sources[i]).permute(0,2,3,1).contiguous())
            conf.append(self.net.conf[i](sources[i]).permute(0,2,3,1).contiguous())
        return torch.cat([z.view(z.size(0),-1,4) for z in loc],1), torch.softmax(torch.cat([z.view(z.size(0),-1,2) for z in conf],1),dim=-1)

def main():
    out=Path('cpp/models'); out.mkdir(parents=True,exist_ok=True)
    asd=ASD(); asd.loadParameters('weight/pretrain_AVA.model'); asd.eval()
    wrapper=LRASD(asd).cuda().eval()
    a=torch.randn(1,100,13,device='cuda'); v=torch.randn(1,25,112,112,device='cuda')
    traced=torch.jit.trace(wrapper,(a,v),strict=False)
    traced.save(str(out/'lr_asd.pt'))
    with torch.inference_mode():
        a2=torch.randn(1,3208,13,device='cuda'); v2=torch.randn(1,802,112,112,device='cuda')
        assert traced(a2,v2).shape == (802,2)
    wrapper=wrapper.cpu()
    traced_cpu=torch.jit.trace(wrapper,(a.cpu(),v.cpu()),strict=False)
    traced_cpu.save(str(out/'lr_asd_cpu.pt'))
    detector=RawS3FD(S3FD(device='cuda').net.eval()).cuda().eval()
    # Fixed detector resolution used for 1920x1080 input at official scale 0.25.
    detector_paths=[]
    for h,w in [(270,480),(206,466)]:
        image=torch.randn(1,3,h,w,device='cuda')
        traced_detector=torch.jit.trace(detector,image,strict=False)
        path=out/f's3fd_{h}x{w}.pt'; traced_detector.save(str(path)); detector_paths.append(path)
    print(out/'lr_asd.pt', out/'lr_asd_cpu.pt', *detector_paths)

if __name__=='__main__': main()
