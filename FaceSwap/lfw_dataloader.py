"""
Dataloader for the LFW dataset, include a feature for updating the dataset when training.
Author: Oliver (Chengyang) Yan
"""

import os
import glob
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
        self.num_images = len(loader)
        self.preload()


    def preload(self):
        try:
            self.src_image1, self.src_image2 = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.src_image1, self.src_image2 = next(self.dataiter)
        with torch.cuda.stream(self.stream):
            self.src_image1 = self.src_image1.cuda(non_blocking=True).sub_(self.mean).div_(self.std)
            self.src_image2 = self.src_image2.cuda(non_blocking=True).sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        src_image1 = self.src_image1
        src_image2 = self.src_image2
        self.preload()
        return src_image1, src_image2

    def rebuild(self):
        new_loader = data.DataLoader(
            self.loader.dataset,
            batch_size=self.loader.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.loader.num_workers,
            pin_memory=(self.loader.num_workers > 0),
        )
        self.loader = new_loader
        self.num_images = len(new_loader)
        self.dataiter = iter(new_loader)
        self.preload()

    def __len__(self):
        return self.num_images

# The dataset class for training and validation
class FaceSwapDataset(data.Dataset):
    def __init__(self, identity_groups, img_transform, image_dir=None, subffix='jpg'):
        self.dataset = identity_groups
        self.img_transform = img_transform
        self.image_dir = image_dir
        self.subffix = subffix
        self.known_dirs = {os.path.dirname(g[0]) for g in identity_groups}
        self.aug_transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        group = self.dataset[index]
        f1 = group[random.randint(0, len(group) - 1)]
        f2 = group[random.randint(0, len(group) - 1)]
        img1 = self.img_transform(Image.open(f1).convert('RGB'))
        img2 = (self.aug_transform if len(group) == 1 else self.img_transform)(
            Image.open(f2).convert('RGB')
        )
        return img1, img2

    def pop_new_groups(self):
        all_groups = scan_identities(self.image_dir, subffix=self.subffix)
        live_dirs = {os.path.dirname(g[0]) for g in all_groups}

        before = len(self.dataset)
        self.dataset = [g for g in self.dataset if os.path.dirname(g[0]) in live_dirs]
        n_pruned = before - len(self.dataset)

        new_groups = [g for g in all_groups if os.path.dirname(g[0]) not in self.known_dirs]
        for g in new_groups:
            self.known_dirs.add(os.path.dirname(g[0]))
        return new_groups, live_dirs, n_pruned

    def add_dir(self, new_dir, subffix=None):
        extra = scan_identities(new_dir, subffix=subffix or self.subffix)
        self.dataset = self.dataset + extra
        print(f'[dataset] Added {len(extra):,} identities from {new_dir} — '
              f'total: {len(self.dataset):,}')

# Special dataset for testing only
class FaceSwapTestDataset(data.Dataset):
    def __init__(self, identity_groups, transform):
        self.transform = transform
        self.identity_groups = identity_groups
        self.pairs = []
        for i, group in enumerate(identity_groups):
            next_group = identity_groups[(i + 1) % len(identity_groups)]
            self.pairs.append((group[0], next_group[0]))

    def add_id_groups(self, new_groups):
        self.identity_groups = self.identity_groups + new_groups
        self.rebuild_pairs()

    def prune(self, live_dirs):
        before = len(self.identity_groups)
        self.identity_groups = [g for g in self.identity_groups if os.path.dirname(g[0]) in live_dirs]
        n_pruned = before - len(self.identity_groups)
        if n_pruned:
            self.rebuild_pairs()
        return n_pruned

    def rebuild_pairs(self):
        n = len(self.identity_groups)
        self.pairs = [(self.identity_groups[i][0], self.identity_groups[(i + 1) % n][0]) for i in range(n)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        bg_path, id_path = self.pairs[idx]
        return (self.transform(Image.open(bg_path).convert('RGB')),
                self.transform(Image.open(id_path).convert('RGB')))

# Scan the dataset directory and group images by identity
def scan_identities(image_dir, subffix='jpg', random_seed=1234):
    folders = glob.glob(os.path.join(image_dir, '*/'))
    groups = []
    for folder in folders:
        images = glob.glob(os.path.join(folder, f'*.{subffix}'))
        if images:
            groups.append(sorted(images))
    random.seed(random_seed)
    random.shuffle(groups)
    return groups

# Print the number of images and ids for each set
def split_summary(train_ds, val_ds, test_ds):
    return (
        f'Train: {len(train_ds.dataset):,} identities: {sum(len(g) for g in train_ds.dataset):,} images\n'
        f'Val: {len(val_ds.dataset):,} identities: {sum(len(g) for g in val_ds.dataset):,} images\n'
        f'Test: {len(test_ds.identity_groups):,} identities: {sum(len(g) for g in test_ds.identity_groups):,} images'
    )

# Refresh the dataset split when updating the datast
def refresh_splits(train_dataset, val_dataset, test_dataset, val_split=0.1, test_split=0.1):
    new_groups, live_dirs, train_pruned = train_dataset.pop_new_groups()

    val_pruned = sum(1 for g in val_dataset.dataset if os.path.dirname(g[0]) not in live_dirs)
    val_dataset.dataset = [g for g in val_dataset.dataset if os.path.dirname(g[0]) in live_dirs]
    test_pruned = test_dataset.prune(live_dirs)
    n_pruned = train_pruned + val_pruned + test_pruned

    if not new_groups and not n_pruned:
        return False, None

    n_new = len(new_groups)
    n_test = int(n_new * test_split)
    n_val = int(n_new * val_split)

    if n_test == 0 or n_val == 0:
        train_dataset.dataset = train_dataset.dataset + new_groups
    else:
        test_new = new_groups[:n_test]
        val_new = new_groups[n_test:n_test + n_val]
        train_new = new_groups[n_test + n_val:]
        train_dataset.dataset = train_dataset.dataset + train_new
        val_dataset.dataset = val_dataset.dataset + val_new
        test_dataset.add_id_groups(test_new)

    summary = split_summary(train_dataset, val_dataset, test_dataset)
    if n_pruned:
        summary += f'\n({n_pruned:,} deleted {"identity" if n_pruned == 1 else "identities"} removed)'
    return True, summary

# Get the dataset loader
def GetLoader(dataset_roots, batch_size=16, dataloader_workers=8, random_seed=1234, test_split=0.1, val_split=0.1):

    data_root = dataset_roots or os.path.join('data', 'lfw-funneled')
    all_groups = scan_identities(data_root, random_seed=random_seed)

    n_test = max(1, int(len(all_groups) * test_split))
    n_val = max(1, int(len(all_groups) * val_split))
    test_groups = all_groups[:n_test]
    val_groups = all_groups[n_test:n_test + n_val]
    train_groups = all_groups[n_test + n_val:]

    print(f'Train: {len(train_groups):,} identities: {sum(len(g) for g in train_groups):,} images')
    print(f'Val: {len(val_groups):,} identities: {sum(len(g) for g in val_groups):,} images')
    print(f'Test: {len(test_groups):,} identities: {sum(len(g) for g in test_groups):,} images')

    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    train_dataset = FaceSwapDataset(train_groups, transform, image_dir=data_root)
    train_dataset.known_dirs |= {os.path.dirname(g[0]) for g in val_groups}
    train_dataset.known_dirs |= {os.path.dirname(g[0]) for g in test_groups}

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=dataloader_workers,
        pin_memory=(dataloader_workers > 0),
    )
    prefetcher = data_prefetcher(train_loader)
    val_dataset = FaceSwapDataset(val_groups, transform)
    test_dataset = FaceSwapTestDataset(test_groups, transform)

    return prefetcher, val_dataset, test_dataset


def denorm(x):
    return (x + 1).div(2).clamp_(0, 1)
