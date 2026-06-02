"""
Evaluation script for FaceSwap model in terms of id cosine similarity, id retrieval accuracy, and posture consistency.
Author: Oliver (Chengyang) Yan

Model and code adapted from:
Chen, R., Chen, X., Ni, B. and Ge, Y., 2020, October. 
Simswap: An efficient framework for high fidelity face swapping.
In Proceedings of the 28th ACM international conference on multimedia (pp. 2003-2011).
"""
import os
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms as T
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
transform     = T.Compose([T.Resize((256, 256)), T.ToTensor()])

model_url  = ('https://storage.googleapis.com/mediapipe-models/'
'face_landmarker/face_landmarker/float16/latest/face_landmarker.task')
model_path = os.path.expanduser('~/.cache/mediapipe/face_landmarker.task')


def _setup_pose_estimator():
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print('[eval] Downloading face_landmarker model...')
        urllib.request.urlretrieve(model_url, model_path)

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return mp, landmarker


def _rotation_to_euler(rmat):
    sy    = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
    pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    roll  = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    return np.array([yaw, pitch, roll])


def _get_pose(landmarker, mp, image_np):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    result = landmarker.detect(mp_img)
    if not result.facial_transformation_matrixes:
        return None
    mat = np.array(result.facial_transformation_matrixes[0])
    return _rotation_to_euler(mat[:3, :3])


def _build_gallery(model, identity_groups):
    embeds = []
    with torch.no_grad():
        for group in tqdm(identity_groups, desc='Building gallery', leave=False):
            img = transform(Image.open(group[0])).unsqueeze(0)
            img_n = img.sub(imagenet_mean).div(imagenet_std).cuda()
            e = F.normalize(
                model.netArc(F.interpolate(img_n, size=(112, 112), mode='bicubic')),
                p=2, dim=1,
            )
            embeds.append(e.squeeze(0).cpu())
    return torch.stack(embeds)


def evaluate(model, test_dataset, opt, step, logger=None, save_dir=None):
    identity_groups = test_dataset.identity_groups
    n_ids = len(identity_groups)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    mp, landmarker = _setup_pose_estimator()

    gallery = _build_gallery(model, identity_groups)

    loader = data.DataLoader(
        test_dataset, batch_size=opt.batchSize,
        shuffle=False, num_workers=0, drop_last=False,
    )

    id_cosines = []
    retrieval_hits = []
    pose_errors = []

    model.netG.eval()
    sample_idx = 0

    def _denorm(t):
        return (t.cpu() * imagenet_std + imagenet_mean).clamp(0, 1)

    with torch.no_grad():
        for img_bg, img_id_src in tqdm(loader, desc='Evaluating', leave=False):
            bs = img_bg.shape[0]
            target_ids = [(sample_idx + j + 1) % n_ids for j in range(bs)]

            img_bg_n = img_bg.sub(imagenet_mean).div(imagenet_std).cuda()
            img_id_n = img_id_src.sub(imagenet_mean).div(imagenet_std).cuda()

            id_embed = F.normalize(
                model.netArc(F.interpolate(img_id_n, size=(112, 112), mode='bicubic')),
                p=2, dim=1,
            )
            img_swapped = model.netG(img_bg_n, id_embed)

            id_embed_sw = F.normalize(
                model.netArc(F.interpolate(img_swapped, size=(112, 112), mode='bicubic')),
                p=2, dim=1,
            )

            id_cosines.extend(model.cosin_metric(id_embed_sw, id_embed).cpu().tolist())

            sims      = id_embed_sw.cpu() @ gallery.T
            predicted = sims.argmax(dim=1)
            for j in range(bs):
                retrieval_hits.append(int(predicted[j].item() == target_ids[j]))

            def to_uint8(t):
                return (
                    (t * imagenet_std + imagenet_mean).clamp(0, 1)
                    .mul(255).byte().permute(0, 2, 3, 1).numpy()
                )
            for bg_img, sw_img in zip(to_uint8(img_bg), to_uint8(img_swapped.cpu())):
                p_bg = _get_pose(landmarker, mp, bg_img)
                p_sw = _get_pose(landmarker, mp, sw_img)
                if p_bg is not None and p_sw is not None:
                    pose_errors.append(float(np.mean(np.abs(p_bg - p_sw))))

            if save_dir is not None:
                bg_vis      = _denorm(img_bg_n)
                id_vis      = _denorm(img_id_n)
                swapped_vis = _denorm(img_swapped)

                grid = torch.cat([bg_vis, id_vis, swapped_vis], dim=0)
                vutils.save_image(
                    vutils.make_grid(grid, nrow=bs),
                    os.path.join(save_dir, f'{sample_idx:05d}_grid.jpg'),
                )
                for j, img in enumerate(swapped_vis):
                    vutils.save_image(img, os.path.join(save_dir, f'{sample_idx + j:05d}_swapped.jpg'))

            sample_idx += bs

    landmarker.close()

    mean_cosine   = float(np.mean(id_cosines))
    retrieval_pct = float(np.mean(retrieval_hits))
    mean_pose     = float(np.mean(pose_errors))

    lines = [
        f'[Eval @ step {step}]',
        f'  ID cosine similarity: {mean_cosine:.4f}',
        f'  ID retrieval top-1: {retrieval_pct:.2f}%',
        f'  Pose error: {mean_pose:.4f} deg',
    ]
    print('\n'.join(lines))

    if logger is not None:
        logger.add_scalar('eval/ID_cosine_similarity', mean_cosine, step)
        logger.add_scalar('eval/ID_retrieval_pct', retrieval_pct, step)
        if mean_pose is not None:
            logger.add_scalar('eval/Pose_error_deg', mean_pose, step)

    model.netG.train()
    return mean_cosine, retrieval_pct, mean_pose
