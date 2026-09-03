#!/usr/bin/env python3
"""Export the official LR-ASD checkpoint and S3FD detector for robot ONNX Runtime."""
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ASD import ASD
from model.faceDetector.s3fd import S3FD


class LRASD(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.model = source.model
        self.fc = source.lossAV.FC

    def forward(self, audio, visual):
        audio = self.model.forward_audio_frontend(audio)
        visual = self.model.forward_visual_frontend(visual)
        fused = self.model.forward_audio_visual_backend(audio, visual)
        return self.fc(fused)


class RawS3FD(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.net = detector.net

    def forward(self, image):
        sources = []
        for index in range(16):
            image = self.net.vgg[index](image)
        sources.append(self.net.L2Norm3_3(image))
        for index in range(16, 23):
            image = self.net.vgg[index](image)
        sources.append(self.net.L2Norm4_3(image))
        for index in range(23, 30):
            image = self.net.vgg[index](image)
        sources.append(self.net.L2Norm5_3(image))
        for index in range(30, len(self.net.vgg)):
            image = self.net.vgg[index](image)
        sources.append(image)
        for index, layer in enumerate(self.net.extras):
            image = F.relu(layer(image))
            if index % 2 == 1:
                sources.append(image)
        locations, confidences = [], []
        confidence = self.net.conf[0](sources[0])
        confidence = torch.cat((torch.max(confidence[:, :3], 1, keepdim=True)[0], confidence[:, 3:]), 1)
        locations.append(self.net.loc[0](sources[0]).permute(0, 2, 3, 1).contiguous())
        confidences.append(confidence.permute(0, 2, 3, 1).contiguous())
        for index in range(1, len(sources)):
            locations.append(self.net.loc[index](sources[index]).permute(0, 2, 3, 1).contiguous())
            confidences.append(self.net.conf[index](sources[index]).permute(0, 2, 3, 1).contiguous())
        locations = torch.cat([item.view(item.size(0), -1, 4) for item in locations], 1)
        confidences = torch.softmax(torch.cat([item.view(item.size(0), -1, 2) for item in confidences], 1), -1)
        return locations, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path.home() / "SoloSeniorWatchRobot_build/lr-asd"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    source = ASD()
    source.loadParameters(str(ROOT / "weight/pretrain_AVA.model"))
    # The original uses MaxPool3d on an unbatched 4-D tensor. PyTorch accepts
    # that shorthand, but ONNX requires rank-consistent pooling. MaxPool2d is
    # mathematically identical for these (1 x 3) audio-frequency pools.
    source.model.audioEncoder.pool1 = torch.nn.MaxPool2d((1, 3), (1, 2), (0, 1))
    source.model.audioEncoder.pool2 = torch.nn.MaxPool2d((1, 3), (1, 2), (0, 1))
    model = LRASD(source).cpu().eval()
    torch.onnx.export(model, (torch.randn(1, 100, 13), torch.randn(1, 25, 112, 112)),
                      output / "lr_asd.onnx", input_names=["audio", "visual"], output_names=["logits"],
                      dynamic_axes={"audio": {1: "audio_frames"}, "visual": {1: "video_frames"},
                                    "logits": {0: "video_frames"}}, opset_version=17, dynamo=False)

    detector = RawS3FD(S3FD(device="cpu")).eval()
    torch.onnx.export(detector, torch.randn(1, 3, 270, 480), output / "s3fd_270x480.onnx",
                      input_names=["image"], output_names=["loc", "conf"], opset_version=17, dynamo=False)
    print(f"Exported robot models to {output}")


if __name__ == "__main__":
    main()
