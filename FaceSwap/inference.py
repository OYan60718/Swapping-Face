#!/usr/bin/env python3
"""
Inference script for FaceSwap model.
This script handles face detection, alignment, and swapping using ONNX models.
Author: Oliver (Chengyang) Yan

Model and code adapted from:
Chen, R., Chen, X., Ni, B. and Ge, Y., 2020, October. 
Simswap: An efficient framework for high fidelity face swapping.
In Proceedings of the 28th ACM international conference on multimedia (pp. 2003-2011).
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from tqdm import tqdm

import onnxruntime as ort

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from insightface_func.utils import face_align_ffhqandnewarc as face_align

# ImageNet normalization parameters
imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Supported file extensions, the code support multiple file extensions
image_ext = {'.jpg', '.jpeg', '.png'}
video_ext = {'.mp4', '.avi', '.mov'}

# load onnx seessions
def load_onnx(onnx_dir):
    available = ort.get_available_providers()
    providers  = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in available]

    arc_path = os.path.join(onnx_dir, 'arcface.onnx')
    gen_path = os.path.join(onnx_dir, 'generator.onnx')

    arc_sess = ort.InferenceSession(arc_path, providers=providers)
    gen_sess = ort.InferenceSession(gen_path, providers=providers)

    active_session = arc_sess.get_providers()[0]
    device = 'GPU (CUDA)' if active_session == 'CUDAExecutionProvider' else 'CPU'

    print(f'ArcFace  device: {device}')
    print(f'Generator device: {device}')
    return arc_sess, gen_sess


# Get ID embedding from the arcface model
def get_id_embedding(arc_sess, face_bgr):
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (112, 112)).astype(np.float32) / 255.0
    face = (face - imagenet_mean) / imagenet_std
    face = face.transpose(2, 0, 1)[np.newaxis]
    return arc_sess.run(['id_embedding'], {'face_112': face})[0]


# Crop the face from the input image
def swap_face_crop(gen_sess, crop_bgr, id_embedding):
    face = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (256, 256)).astype(np.float32) / 255.0
    face = (face - imagenet_mean) / imagenet_std
    face = face.transpose(2, 0, 1)[np.newaxis]

    out = gen_sess.run(['swapped_image'],{'source_image': face, 'id_embedding': id_embedding})[0][0]

    out = out.transpose(1, 2, 0) * imagenet_std + imagenet_mean
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# Paste the swapped face back to the originial frame
def paste_to_bg(frame, swapped_crop, M, crop_size):
    h, w = frame.shape[:2]
    swapped_resized = cv2.resize(swapped_crop, (crop_size, crop_size))

    inv_M = cv2.invertAffineTransform(M)
    swapped_in_frame = cv2.warpAffine(
        swapped_resized, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0,
    )

    inset = max(crop_size // 12, 6)
    mask_crop = np.zeros((crop_size, crop_size), dtype=np.uint8)
    cx = crop_size // 2
    cy = crop_size // 2

    cv2.ellipse(mask_crop, (cx, cy), (cx - inset, cy - inset), 0, 0, 360, 255, -1)
    mask_in_frame = cv2.warpAffine(mask_crop, inv_M, (w, h), borderValue=0)

    corners = np.array([[0, 0], [crop_size, 0], [crop_size, crop_size], [0, crop_size]],dtype=np.float32)
    corners_dst = cv2.transform(corners.reshape(1, -1, 2), inv_M).reshape(-1, 2)
    x_f = int(np.clip(corners_dst[:, 0].mean(), 1, w - 2))
    y_f = int(np.clip(corners_dst[:, 1].mean(), 1, h - 2))

    try:
        return cv2.seamlessClone(swapped_in_frame, frame, mask_in_frame, (x_f, y_f), cv2.NORMAL_CLONE)
    except cv2.error:
        alpha = mask_in_frame.astype(np.float32)[..., np.newaxis] / 255.0
        blended = frame.astype(np.float32) * (1 - alpha) + swapped_in_frame.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)


# A class to detecting faces and facial features, the code is adapted from the InsightFace library, current the method only support single face per image/frame
# The detector locates the face in the image, and estimate facial features for alignment purposes
class FaceDetector:
    def __init__(self, model_path):
        available = ort.get_available_providers()
        providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in available]
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._output_names = [o.name for o in self._sess.get_outputs()]
        self._center_cache = {}
        self.det_threshold = 0.5
        self.det_size = (640, 640)

    # config for the detector
    def prepare_detector(self, ctx_id=0, det_threshold=0.5, det_size=(640, 640)):
        self.det_threshold = det_threshold
        self.det_size = det_size

    # get face and features for alignments with the detector model
    def get(self, img_bgr, crop_size, max_num=0):
        input_w, input_h = self.det_size
        im_h, im_w = img_bgr.shape[:2]

        det_scale = min(input_w / im_w, input_h / im_h)
        new_w, new_h = int(im_w * det_scale), int(im_h * det_scale)
        padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = cv2.resize(img_bgr, (new_w, new_h))

        blob = (padded[:, :, ::-1].astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        net_outs = self._sess.run(self._output_names, {self._input_name: blob})

        all_scores, all_bboxes, all_kpss = [], [], []
        fmc = len([8, 16, 32])

        for idx, stride in enumerate([8, 16, 32]):
            scores    = net_outs[idx].reshape(-1)
            bbox_pred = net_outs[idx + fmc].reshape(-1, 4) * stride
            kps_pred  = net_outs[idx + fmc * 2].reshape(-1, 10) * stride

            feat_h, feat_w = input_h // stride, input_w // stride
            key = (feat_h, feat_w, stride)
            if key not in self._center_cache:
                cx = np.arange(feat_w, dtype=np.float32) * stride
                cy = np.arange(feat_h, dtype=np.float32) * stride
                gx, gy = np.meshgrid(cx, cy)
                centres = np.stack([gx, gy], axis=-1).reshape(-1, 2)
                centres = np.tile(centres[:, np.newaxis, :], (1, 2, 1)).reshape(-1, 2)
                self._center_cache[key] = centres
            ac = self._center_cache[key]

            pos = scores >= self.det_threshold
            if not pos.any():
                continue

            ac_pos = ac[pos]
            bd = bbox_pred[pos]

            bboxes = np.column_stack([
                ac_pos[:, 0] - bd[:, 0],
                ac_pos[:, 1] - bd[:, 1],
                ac_pos[:, 0] + bd[:, 2],
                ac_pos[:, 1] + bd[:, 3],
            ])
            kpss = ac_pos[:, np.newaxis, :] + kps_pred[pos].reshape(-1, 5, 2)

            all_scores.append(scores[pos])
            all_bboxes.append(bboxes)
            all_kpss.append(kpss)

        if not all_scores:
            return None

        scores_cat = np.concatenate(all_scores)
        bboxes_cat = np.concatenate(all_bboxes)
        kpss_cat = np.concatenate(all_kpss)

        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in bboxes_cat]
        keep = cv2.dnn.NMSBoxes(xywh, scores_cat.tolist(), self.det_threshold, 0.4)
        if len(keep) == 0:
            return None
        keep = np.array(keep).flatten()

        best = keep[int(np.argmax(scores_cat[keep]))]
        kps = kpss_cat[best] / det_scale

        M, _ = face_align.estimate_norm(kps, crop_size, mode='None')
        crop = cv2.warpAffine(img_bgr, M, (crop_size, crop_size), borderValue=0.0)
        return [crop], [M]


# Code for processing the image input, including faace detection, alignment, swapping
def run_image(source_path, donor_path, output_path, arc_sess, gen_sess, detector, crop_size):
    source = cv2.imread(source_path)
    donor  = cv2.imread(donor_path)

    # Detect and get the face from the donnor, including the ID embeddings
    donor_result = detector.get(donor, crop_size)
    id_embedding = get_id_embedding(arc_sess, donor_result[0][0])

    # Detect the face from the source image
    result = detector.get(source, crop_size)

    # Swap and paste
    output = source.copy()
    for crop, M in zip(result[0], result[1]):
        swapped = swap_face_crop(gen_sess, crop, id_embedding)
        output  = paste_to_bg(output, swapped, M, crop_size)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, output)
    print(f'Saved: {output_path}')


# Processing for video input
def run_video(source_path, donor_path, output_path, arc_sess, gen_sess, detector, crop_size):
    donor = cv2.imread(donor_path)
    donor_result = detector.get(donor, crop_size)
    id_embedding = get_id_embedding(arc_sess, donor_result[0][0])

    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
         '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', source_path],
        capture_output=True, text=True, timeout=10,
    )
    has_audio = bool(r.stdout.strip())

    tmp_path = (output_path + '.tmp.mp4') if has_audio else output_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))

    read_q = queue.Queue(maxsize=16)
    write_q = queue.Queue(maxsize=16)

    def _reader():
        while True:
            ret, frame = cap.read()
            if not ret:
                read_q.put(None)
                return
            read_q.put(frame)

    def _writer():
        while True:
            frame = write_q.get()
            if frame is None:
                return
            writer.write(frame)

    reader_t = threading.Thread(target=_reader, daemon=True)
    writer_t = threading.Thread(target=_writer, daemon=True)
    reader_t.start()
    writer_t.start()

    processed = 0
    t_start = time.perf_counter()
    for _ in tqdm(range(frame_count), desc='Swapping frames', unit='frame'):
        frame = read_q.get()
        if frame is None:
            break

        result = detector.get(frame, crop_size)

        for crop, M in zip(result[0], result[1]):
            swapped = swap_face_crop(gen_sess, crop, id_embedding)
            frame= paste_to_bg(frame, swapped, M, crop_size)

        write_q.put(frame)
        processed += 1

    write_q.put(None)
    reader_t.join()
    writer_t.join()
    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    average_fps = processed / elapsed

    print(f'Processed {processed} frames in {elapsed:.1f}s, avgerage {average_fps:.1f} FPS')

    if has_audio:
        subprocess.run(
            ['ffmpeg', '-y',
             '-i', tmp_path, '-i', source_path,
             '-c:v', 'copy', '-c:a', 'aac',
             '-map', '0:v:0', '-map', '1:a:0',
             '-shortest', output_path],
            check=True, capture_output=True,
        )
        os.remove(tmp_path)

    print(f'Saved: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimSwap face swap inference')
    parser.add_argument('--source', required=True, help='Source image or video')
    parser.add_argument('--donor', required=True, help='Donor image')
    parser.add_argument('--output', default=None, help='Output file path (default: samples/sample_results/<source_filename>)')
    parser.add_argument('--onnx-dir', default=os.path.join(root, 'onnx'), help='Directory containing generator.onnx and arcface.onnx')
    parser.add_argument('--antelope-dir', default=os.path.join(root, 'insightface_func', 'models', 'antelope'),
                        help='Path to antelope face detection model directory')
    parser.add_argument('--crop-size', type=int, default=256, help='Face crop')
    parser.add_argument('--det-threshold', type=float, default=0.5, help='Face detection confidence threshold')
    args = parser.parse_args()

    ext = os.path.splitext(args.source)[1].lower()
    if ext in video_ext:
        mode = 'video'
    elif ext in image_ext:
        mode = 'image'
    else:
        parser.error(f'Unrecognised source extension "{ext}". Images: {image_ext}  Videos: {video_ext}')

    # Set default path if not provided
    if args.output is None:
        results_dir = os.path.join(root, 'samples', 'sample_results')
        args.output = os.path.join(results_dir, os.path.basename(args.source))

    print('Loading ONNX sessions')
    arcFace_session, gen_session = load_onnx(args.onnx_dir)

    det_model = os.path.join(args.antelope_dir, 'det_10g.onnx')
    print(f'Loading face detector from {det_model}')
    Face_detector = FaceDetector(det_model)
    Face_detector.prepare_detector(ctx_id=0, det_threshold=args.det_threshold, det_size=(640, 640))

    if mode == 'image':
        run_image(args.source, args.donor, args.output, arcFace_session, gen_session, Face_detector, args.crop_size)
    else:
        run_video(args.source, args.donor, args.output, arcFace_session, gen_session, Face_detector, args.crop_size)
