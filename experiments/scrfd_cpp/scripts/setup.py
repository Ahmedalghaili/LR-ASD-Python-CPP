#!/usr/bin/env python3
"""Download pinned experiment dependencies without changing the project environment."""
from pathlib import Path
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ORT_VERSION = '1.22.0'
MODELS = {
    'scrfd_2.5g_bnkps.onnx': ('https://huggingface.co/MonsterMMORPG/files1/resolve/55b9440/scrfd_2.5g_bnkps.onnx', 'bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce'),
    'scrfd_10g_bnkps.onnx': ('https://huggingface.co/lithiumice/insightface/resolve/1141cd22e2bff0d4036d10ba4151903605a8902d/models/antelopev2/scrfd_10g_bnkps.onnx', '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'),
}

def fetch(url, path, digest=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        subprocess.run(['curl', '-fL', '--retry', '5', '-o', str(temporary), url], check=True)
        temporary.replace(path)
    if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'Checksum mismatch: {path}')

if __name__ == '__main__':
    lib = ROOT/'third_party/python/onnxruntime/capi'
    if not (lib/f'libonnxruntime.so.{ORT_VERSION}').exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(ROOT/'third_party/python'), '--no-deps', f'onnxruntime-gpu=={ORT_VERSION}'], check=True)
    soname = lib/'libonnxruntime.so.1'
    if not soname.exists():
        soname.symlink_to(f'libonnxruntime.so.{ORT_VERSION}')
    fetch('https://raw.githubusercontent.com/deepinsight/insightface/41bf106fa0f9a3c998dd717f2abb5f3c2fada6da/detection/scrfd/tools/scrfd.py', ROOT/'third_party/scrfd_reference.py')
    for name in ['onnxruntime_c_api.h', 'onnxruntime_cxx_api.h', 'onnxruntime_cxx_inline.h', 'onnxruntime_float16.h']:
        fetch(f'https://raw.githubusercontent.com/microsoft/onnxruntime/v{ORT_VERSION}/include/onnxruntime/core/session/{name}', ROOT/'third_party/include'/name)
    for name, (url, digest) in MODELS.items():
        fetch(url, ROOT/'models'/name, digest)
    import onnx
    for name in MODELS:
        model = onnx.load(ROOT/'models'/name)
        # The exports accept dynamic input but carry stale 640x640 output sizes.
        # Change shape metadata only; graph operations and weights stay intact.
        for i, output in enumerate(model.graph.output):
            dim = output.type.tensor_type.shape.dim[0]
            dim.ClearField('dim_value')
            dim.dim_param = f'anchors_stride_{8 << (i % 3)}'
        onnx.checker.check_model(model)
        onnx.save(model, ROOT/'models'/name.replace('.onnx', '.dynamic.onnx'))
    print('Dependencies and model checksums verified; dynamic output metadata prepared.')
