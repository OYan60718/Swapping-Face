#!/usr/bin/env python3
"""
ONNX export script for SimSwap and ArcFace models.
Author: Oliver (Chengyang) Yan
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fs_networks_fix import Generator_Adain_Upsample
import onnxruntime as ort

class ArcFaceNormalized(nn.Module):
    def __init__(self, arcface):
        super().__init__()
        self.arcface = arcface

    def forward(self, x):
        return F.normalize(self.arcface(x), p=2, dim=1)


def load_generator(checkpoint_path, deep = False):
    netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=deep)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    netG.load_state_dict(state)
    netG.eval()
    return netG


def load_arcface(checkpoint_path):
    arcface = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    arcface.eval()
    return ArcFaceNormalized(arcface)


def do_export(model, dummy_inputs, output_path, input_names, output_names, dynamic_axes):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        model, dummy_inputs, output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )


def export_generator(checkpoint_path, output_path, deep = False, verify = False):
    print(f'Loading generator from {checkpoint_path}')
    netG = load_generator(checkpoint_path, deep=deep)

    dummy_img   = torch.randn(1, 3, 256, 256)
    dummy_embed = F.normalize(torch.randn(1, 512), p=2, dim=1)

    print(f'Exporting generator → {output_path}')
    do_export(
        netG, (dummy_img, dummy_embed), output_path,
        input_names=['source_image', 'id_embedding'],
        output_names=['swapped_image'],
        dynamic_axes={
            'source_image':  {0: 'batch'},
            'id_embedding':  {0: 'batch'},
            'swapped_image': {0: 'batch'},
        },
    )
    print(f'  source_image  (B, 3, 256, 256)')
    print(f'  id_embedding  (B, 512)')
    print(f'  → swapped_image (B, 3, 256, 256)')

    if verify:
        _verify_generator(netG, output_path, dummy_img, dummy_embed)


def export_arcface(checkpoint_path, output_path, verify = False):
    print(f'Loading ArcFace from {checkpoint_path}')
    model = load_arcface(checkpoint_path)

    dummy_face = torch.randn(1, 3, 112, 112)

    print(f'Exporting ArcFace → {output_path}')
    do_export(
        model, dummy_face, output_path,
        input_names=['face_112'],
        output_names=['id_embedding'],
        dynamic_axes={
            'face_112':     {0: 'batch'},
            'id_embedding': {0: 'batch'},
        },
    )
    print(f'  face_112      (B, 3, 112, 112)  — aligned face, ImageNet-normalised')
    print(f'  → id_embedding (B, 512)          — L2-normalised')

    if verify:
        _verify_arcface(model, output_path, dummy_face)


def _ort_session(onnx_path):
    return ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])


def _verify_generator(netG, onnx_path, dummy_img, dummy_embed):
    sess = _ort_session(onnx_path)
    print('Verifying generator...')
    with torch.no_grad():
        pt_out = netG(dummy_img, dummy_embed).numpy()
    ort_out = sess.run(
        ['swapped_image'],
        {'source_image': dummy_img.numpy(), 'id_embedding': dummy_embed.numpy()},
    )[0]
    _report_diff(pt_out, ort_out)


def _verify_arcface(model, onnx_path, dummy_face):
    sess = _ort_session(onnx_path)
    print('Verifying ArcFace...')
    with torch.no_grad():
        pt_out = model(dummy_face).numpy()
    ort_out = sess.run(['id_embedding'], {'face_112': dummy_face.numpy()})[0]
    _report_diff(pt_out, ort_out)


def _report_diff(pt_out, ort_out):
    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f'  Max |PyTorch − ONNX| = {max_diff:.2e}')
    print(f'  {"PASSED" if max_diff < 1e-4 else "WARNING: larger diff than expected"}')
