#!/usr/bin/env python3
"""
Training script for FaceSwap model.
This script handles training loop, logging, validation, and optional ONNX export.
Optional dataset update is supported during training with the "refresh_freq" config.
Author: Oliver (Chengyang) Yan 

Model and code adapted from: 
Chen, R., Chen, X., Ni, B. and Ge, Y., 2020, October. 
Simswap: An efficient framework for high fidelity face swapping. 
In Proceedings of the 28th ACM international conference on multimedia (pp. 2003-2011).
"""

import os
import csv
import time
import random
import argparse
import json
import subprocess
import numpy as np
import math
import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.backends import cudnn
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as T
import torchvision.utils as utils

from tqdm import tqdm

from models.fs_model import fsModel
from lfw_dataloader import GetLoader, refresh_splits
from evaluate import evaluate
from export_onnx import export_generator, export_arcface


# Load configs from config file
class TrainConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--config', type=str, default=None, required=True, help='path to json config file with training options')
        self.isTrain = True
        self.initialized = True

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        return config

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()

        args = self.parser.parse_args()

        config_path = args.config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'train_config.json')

        config = self.load_config(config_path)

        if isinstance(config.get('gpu_ids'), list):
            config['gpu_ids'] = ','.join(str(x) for x in config['gpu_ids'])

        self.opt = argparse.Namespace(**config)
        self.opt.isTrain = self.isTrain

        args_dict = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args_dict.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            os.makedirs(expr_dir, exist_ok=True)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt', encoding='utf-8') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args_dict.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')

        return self.opt

# ImageNet normalization for validation
val_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
val_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# The Validation function
def run_validation(model, val_dataset, opt, criterionFeat, criterionRec):
    from torch.utils import data as torch_data

    loader = torch_data.DataLoader(
        val_dataset, batch_size=opt.batchSize, shuffle=True,
        num_workers=0, drop_last=True,
    )
    acc = defaultdict(list)

    model.netG.eval()
    with torch.no_grad():
        for img_bg, img_id in loader:
            img_bg_n = img_bg.sub(val_mean).div(val_std).cuda()
            img_id_n = img_id.sub(val_mean).div(val_std).cuda()

            id_cross = F.normalize(model.netArc(F.interpolate(torch.roll(img_id_n, 1, 0), size=(112, 112), mode='bicubic')), p=2, dim=1)
            img_swap = model.netG(img_bg_n, id_cross)
            logits, feat_fake = model.netD(img_swap, None)
            id_sw = F.normalize(model.netArc(F.interpolate(img_swap, size=(112, 112), mode='bicubic')), p=2, dim=1)
            g_adv  = (-logits).mean()
            g_id   = (1 - model.cosin_metric(id_sw, id_cross)).mean()
            g_feat = criterionFeat(feat_fake["3"], model.netD.get_feature(img_bg_n)["3"])

            id_same  = F.normalize(model.netArc(F.interpolate(img_id_n, size=(112, 112), mode='bicubic')), p=2, dim=1)
            img_rec = model.netG(img_bg_n, id_same)
            g_rec   = criterionRec(img_rec, img_bg_n) * opt.lambda_rec

            # Computer total loss for val
            g_total = g_adv + g_id * opt.lambda_id + g_feat * opt.lambda_feat + g_rec

            acc['G_adv'].append(g_adv.item())
            acc['G_ID'].append(g_id.item())
            acc['G_feat'].append(g_feat.item())
            acc['G_rec'].append(g_rec.item())
            acc['G_total'].append(g_total.item())

    model.netG.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}

# Plot loss curves at the end of training for a visualisation
def plot_losses_curves(train_logs, val_logs, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots = [
        ('G_adv',   'G_adv',   'G_adv'),
        ('G_ID',    'G_ID',    'G_ID'),
        ('G_feat',  'G_feat',  'G_feat'),
        ('G_Rec',   'G_Rec',   'G_rec'),
        ('G_total', 'G_total', 'G_total'),
        ('D_fake',  'D_fake',  None),
        ('D_real',  'D_real',  None),
        ('D_loss',  'D_loss',  None),
    ]

    n_cols = 4
    n_rows = (len(plots) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    fig.suptitle('Loss History', fontsize=14)
    axes_flat = axes.flatten()

    train_steps = [r['step'] for r in train_logs]
    val_steps   = [r['step'] for r in val_logs]

    for i, (title, t_key, v_key) in enumerate(plots):
        ax = axes_flat[i]
        if train_steps:
            ax.plot(train_steps, [r.get(t_key) for r in train_logs], label='train', color='steelblue', alpha=0.8)
        if v_key and val_steps:
            ax.plot(val_steps, [r.get(v_key) for r in val_logs],
                    label='val', color='tomato', linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Step', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(len(plots), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    tqdm.write(f'Loss plot saved: {save_path}')

# Export the arcface and generator models to onnx for inference
def export_to_onnx(opt):
    if not getattr(opt, 'export_onnx', False):
        return

    out_dir    = getattr(opt, 'onnx_output_dir', os.path.join(opt.checkpoints_dir, opt.name, 'onnx'))
    g_ckpt     = getattr(opt, 'onnx_generator_ckpt', None)
    arc_ckpt   = getattr(opt, 'Arc_path', None)
    export_arc = getattr(opt, 'onnx_export_arcface', True)

    if g_ckpt is None:
        g_ckpt = os.path.join(opt.checkpoints_dir, opt.name, f'{opt.total_step}_net_G.pth')

    os.makedirs(out_dir, exist_ok=True)

    # Export generator
    if os.path.isfile(g_ckpt):
        export_generator(
            checkpoint_path=g_ckpt,
            output_path=os.path.join(out_dir, 'generator.onnx'),
            deep=getattr(opt, 'Gdeep', False),
        )
    else:
        print(f'[export] Generator checkpoint not found, skipping: {g_ckpt}')

    # Export ArcFace
    if export_arc:
        export_arcface(checkpoint_path=arc_ckpt, output_path=os.path.join(out_dir, 'arcface.onnx'))


if __name__ == '__main__':
    
    # Get configs and directories
    opt       = TrainConfig().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'summary')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Check if resume training from ckpts
    if opt.continue_train:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    print("GPU used : ", str(opt.gpu_ids))

    cudnn.benchmark = True

    # Load the face swap model
    model = fsModel()
    model.initialize(opt)

    # Setup Tensorboard
    if opt.use_tensorboard:
        tb_port = getattr(opt, 'tb_port', 6002)
        subprocess.Popen(
            ['tensorboard', '--logdir', log_path, '--port', str(tb_port), '--host', '0.0.0.0'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f'TensorBoard started at http://localhost:{tb_port}')
        logger = tensorboard.SummaryWriter(log_path)

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    # Setup loss functions
    criterionFeat = nn.L1Loss()
    criterionRec  = nn.L1Loss()

    # Get optimizers
    optimizer_G = model.optimizer_G
    optimizer_D = model.optimizer_D

    # Load datasets
    loss_avg = 0
    refresh_count = 0
    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    train_loader, val_dataset, test_dataset = GetLoader(
        opt.dataset, opt.batchSize, getattr(opt, 'num_workers', 4), 1234, val_split=getattr(opt, 'val_split', 0.1),
    )

    # If eval-only mode, skip training and run evaluation only (requires pre-trained ckpts)
    if getattr(opt, 'eval_only', False):
        eval_ckpt     = str(opt.total_step)
        load_eval_dir = opt.load_eval
        print(f'Eval-only mode — loading G from {load_eval_dir}/{eval_ckpt}_net_G.pth')
        model.load_network(model.netG, 'G', eval_ckpt, load_eval_dir)
        evaluate(model, test_dataset, opt, opt.total_step, logger=logger if opt.use_tensorboard else None,
                 save_dir=os.path.join(opt.checkpoints_dir, opt.name, 'eval_results', f'step_{opt.total_step}'))
        export_to_onnx(opt)
        raise SystemExit(0)

    rand_index = [i for i in range(opt.batchSize)]
    random.shuffle(rand_index)

    if not opt.continue_train:
        start = 0
    else:
        try:
            start = int(opt.which_epoch)
        except ValueError:
            start = 0
    total_step = opt.total_step

    val_freq       = getattr(opt, 'val_freq', 2000)
    refresh_freq   = getattr(opt, 'refresh_freq', 0)
    lr_decay_step  = getattr(opt, 'lr_decay_step', 0)
    lr_decay_gamma = getattr(opt, 'lr_decay_gamma', 0.5)
    best_val_loss  = float('inf')

    # Optional learning rate scheduler
    if lr_decay_step > 0:
        _last = (start - 1) if start > 0 else -1
        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=lr_decay_step, gamma=lr_decay_gamma, last_epoch=_last)
        scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=lr_decay_step, gamma=lr_decay_gamma, last_epoch=_last)
    else:
        scheduler_G = scheduler_D = None

    # Log training and validation losses for plotting
    train_logs = []
    val_logs   = []

    csv_path  = os.path.join(opt.checkpoints_dir, opt.name, 'loss_history.csv')
    cols = ['step', 'type', 'G_adv', 'G_ID', 'G_feat', 'G_rec', 'G_total', 'D_fake', 'D_real', 'D_loss']
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f: csv.writer(f).writerow(cols)

    model.netD.feature_network.requires_grad_(False)

    g_loss_rec = torch.tensor(0.0)

    # tqdm progress bar for training loop, visulise the progress and estimated time remaining
    pbar = tqdm(range(start, total_step), initial=start, total=total_step, unit='step', dynamic_ncols=True)

    # Main training loop
    for step in pbar:
        model.netG.train()

        # 2-phase training for generator and discriminator, phase 1 for discriminator and phase 0 for generator
        for phase in range(2):
            random.shuffle(rand_index)
            img_bg, img_id_src = train_loader.next()

            # Alternate between source image from the same identity (for reconstruction loss) and from a different identity (for swapping)
            # The same ID is used when even steps, and different ID is used when odd steps, following the implementation in SimSwap
            # When and ID only contains a single image, the same image is used but with augmentation
            img_id   = img_id_src if step % 2 == 0 else img_id_src[rand_index]
            id_embed = F.normalize(model.netArc(F.interpolate(img_id, size=(112, 112), mode='bicubic')), p=2, dim=1)

            # Discriminator update
            if phase:
                img_swapped = model.netG(img_bg, id_embed)

                logits_fake, _ = model.netD(img_swapped.detach(), None)
                logits_real, _ = model.netD(img_id_src, None)

                d_loss_fake = (F.relu(torch.ones_like(logits_fake) + logits_fake)).mean()
                d_loss_real = (F.relu(torch.ones_like(logits_real) - logits_real)).mean()
                d_loss      = d_loss_fake + d_loss_real

                optimizer_D.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.netD.parameters(), max_norm=1.0)
                optimizer_D.step()
            
            # Generator update
            else:
                img_swapped            = model.netG(img_bg, id_embed)
                logits_fake, feat_fake = model.netD(img_swapped, None)

                id_embed_swapped = F.normalize(
                    model.netArc(F.interpolate(img_swapped, size=(112, 112), mode='bicubic')), p=2, dim=1
                )

                feat_real   = model.netD.get_feature(img_bg)
                g_loss_adv  = (-logits_fake).mean()
                g_loss_id   = (1 - model.cosin_metric(id_embed_swapped, id_embed)).mean()
                g_loss_feat = criterionFeat(feat_fake["3"], feat_real["3"])
                g_loss      = g_loss_adv + g_loss_id * opt.lambda_id + g_loss_feat * opt.lambda_feat

                if step % 2 == 0:
                    g_loss_rec = criterionRec(img_swapped, img_bg) * opt.lambda_rec
                    g_loss     = g_loss + g_loss_rec

                optimizer_G.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.netG.parameters(), max_norm=1.0)
                optimizer_G.step()

        if scheduler_G is not None:
            scheduler_G.step()
            scheduler_D.step()
            if (step + 1) % lr_decay_step == 0:
                new_lr = optimizer_G.param_groups[0]['lr']
                tqdm.write(f'[LR] Decayed to {new_lr:.2e} at step {step + 1}')
                if opt.use_tensorboard:
                    logger.add_scalar('train/lr', new_lr, step)

        pbar.set_postfix({
            'G':   f'{g_loss_adv.item():.3f}',
            'D':   f'{d_loss.item():.3f}',
            'ID':  f'{g_loss_id.item():.3f}',
            'Rec': f'{g_loss_rec.item():.3f}',
        })

        g_total_train = g_loss.item()
        errors = {
            'G_adv':   g_loss_adv.item(),
            'G_ID':    g_loss_id.item(),
            'G_Rec':   g_loss_rec.item(),
            'G_feat':  g_loss_feat.item(),
            'G_total': g_total_train,
            'D_fake':  d_loss_fake.item(),
            'D_real':  d_loss_real.item(),
            'D_loss':  d_loss.item(),
        }

        # log to tensorboard and text file for visualisation
        if opt.use_tensorboard and (step + 1) % getattr(opt, 'tb_freq', 100) == 0:
            for tag, value in errors.items():
                logger.add_scalar(f'losses/{tag}', value, step)

        if (step + 1) % opt.log_frep == 0:
            message = 'step %d | ' % step + ' | '.join('%s: %.3f' % (k, v) for k, v in errors.items())
            tqdm.write(message)
            with open(log_name, 'a') as log_file:
                log_file.write('%s\n' % message)
            train_logs.append({'step': step, **errors})
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    step, 'train',
                    errors['G_adv'], errors['G_ID'], errors['G_feat'],
                    errors['G_Rec'], errors['G_total'],
                    errors['D_fake'], errors['D_real'], errors['D_loss'],
                ])

        # Run validations every val_freq steps when training
        if (step + 1) % val_freq == 0:
            tqdm.write(f'Running validation at step {step + 1}...')
            val_losses = run_validation(model, val_dataset, opt, criterionFeat, criterionRec)
            val_total  = val_losses['G_total']

            val_msg = 'VAL step %d | ' % step + ' | '.join('%s: %.4f' % (k, v) for k, v in val_losses.items())
            tqdm.write(val_msg)
            with open(log_name, 'a') as log_file:
                log_file.write('%s\n' % val_msg)

            val_logs.append({'step': step, **val_losses})
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    step, 'val',
                    val_losses['G_adv'], val_losses['G_ID'], val_losses['G_feat'],
                    val_losses['G_rec'], val_losses['G_total'],
                    '', '', '',
                ])

            if opt.use_tensorboard:
                for tag, value in val_losses.items():
                    logger.add_scalar(f'val/{tag}', value, step)

        # Logging for displaying the source, donor and swapped images
        if (step + 1) % opt.tb_freq == 0:
            model.netG.eval()
            with torch.no_grad():
                img_id_swapped  = torch.roll(img_id_src, 1, dims=0)
                id_embed_sample = F.normalize(
                    model.netArc(F.interpolate(img_id_swapped, size=(112, 112), mode='bicubic')), p=2, dim=1
                )
                swapped_batch = model.netG(img_bg, id_embed_sample).cpu()

                def denorm(t):
                    return (t * imagenet_std + imagenet_mean).clamp(0, 1)

                bg_vis = denorm(img_bg.cpu())
                id_vis  = denorm(img_id_swapped.cpu())
                swapped_vis = denorm(swapped_batch)

                if opt.use_tensorboard:
                    logger.add_image('samples/bg',       torchvision.utils.make_grid(bg_vis,      nrow=opt.batchSize), step)
                    logger.add_image('samples/identity', torchvision.utils.make_grid(id_vis,      nrow=opt.batchSize), step)
                    logger.add_image('samples/swapped',  torchvision.utils.make_grid(swapped_vis, nrow=opt.batchSize), step)
            model.netG.train()

        # Refresh dataset, optional for update training dataset during training, this config can also be set in the config file with "refresh_freq" key
        if refresh_freq > 0 and (step + 1) % refresh_freq == 0:
            tqdm.write(f'[dataset] Checking for new data at step {step + 1}...')
            changed, summary = refresh_splits(
                train_loader.loader.dataset, val_dataset, test_dataset,
                val_split=getattr(opt, 'val_split', 0.1),
                test_split=getattr(opt, 'test_split', 0.1),
            )
            if changed:
                tqdm.write(f'[dataset] Updated:\n{summary}')
                train_loader.rebuild()
            else:
                tqdm.write('[dataset] No new data found.')

        if (step + 1) % opt.model_freq == 0:
            tqdm.write('saving model at step %d' % (step + 1))
            model.save(step + 1)
            np.savetxt(iter_path, (step + 1, total_step), delimiter=',', fmt='%d')

    pbar.close()

    # Plot losses curves
    plot_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss_history.png')
    plot_losses_curves(train_logs, val_logs, plot_path)

    # Run evaluation and export to onnx
    tqdm.write('Training complete. Running eval...')
    evaluate(model, test_dataset, opt, total_step, logger=logger if opt.use_tensorboard else None, save_dir=os.path.join(opt.checkpoints_dir, opt.name, 'eval_results', f'step_{total_step}'))
    export_to_onnx(opt)
