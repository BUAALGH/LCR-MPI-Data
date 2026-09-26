from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset


def resize_image(array, size=64):
    tensor = torch.from_numpy(array.astype(np.float32))[None, None]
    tensor = F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    return tensor[0, 0].numpy()


def minmax(array):
    minimum = array.min()
    maximum = array.max()
    return ((array - minimum) / (maximum - minimum + 1e-8)).astype(np.float32)


def prepare_pair(low, high):
    l3 = minmax(resize_image(-low[:, :, 0]))
    l5 = minmax(resize_image(low[:, :, 1]))
    h5 = minmax(resize_image(high[:, :, 1]))
    inputs = torch.from_numpy(np.stack([l3, l5], axis=0))
    target = torch.from_numpy(h5[None])
    return inputs, target


def numeric_phantom_dirs(root):
    return sorted(
        (path for path in Path(root).glob("phantom_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )


def make_folds(n_total=100, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    fold_size = n_total // n_folds
    folds = []
    for fold_index in range(n_folds):
        test = indices[fold_index * fold_size:(fold_index + 1) * fold_size]
        test_set = set(test)
        remaining = [index for index in indices if index not in test_set]
        rng.shuffle(remaining)
        folds.append({
            "train": remaining[:65],
            "val": remaining[65:],
            "test": test,
        })
    return folds


def load_pair(sample_dir):
    sample_dir = Path(sample_dir)
    with h5py.File(sample_dir / "L.h5", "r") as handle:
        low = handle["data/image"][()].astype(np.float32)
    with h5py.File(sample_dir / "H.h5", "r") as handle:
        high = handle["data/image"][()].astype(np.float32)
    return low, high


class IDDataset(Dataset):
    def __init__(self, sample_dirs):
        self.sample_dirs = [Path(path) for path in sample_dirs]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]
        low, high = load_pair(sample_dir)
        with h5py.File(sample_dir / "mask.h5", "r") as handle:
            mask = handle["data/mask"][()].astype(np.float32)
        inputs, target = prepare_pair(low, high)
        mask = torch.from_numpy(minmax(resize_image(mask))[None])
        return inputs, target, mask


class AugmentedIDDataset(Dataset):
    def __init__(self, root, phantom_names):
        allowed = set(phantom_names)
        self.files = [
            path for path in sorted(Path(root).glob("*.h5"))
            if path.stem.split("__", 1)[0] in allowed
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with h5py.File(self.files[index], "r") as handle:
            low = handle["data/L_image"][()].astype(np.float32)
            high = handle["data/H_image"][()].astype(np.float32)
            mask = handle["data/mask"][()].astype(np.float32)
        inputs, target = prepare_pair(low, high)
        mask = torch.from_numpy(minmax(resize_image(mask))[None])
        return inputs, target, mask


class GeneralizationDataset(Dataset):
    def __init__(self, root):
        self.sample_dirs = numeric_phantom_dirs(root)

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, index):
        low, high = load_pair(self.sample_dirs[index])
        inputs, target = prepare_pair(low, high)
        return inputs, target, self.sample_dirs[index].name


class InVivoDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).glob("*.h5"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with h5py.File(path, "r") as handle:
            image = handle["data/image/imag"][()].astype(np.float32)
        l3 = minmax(resize_image(-image[:, :, 0]))
        l5 = minmax(resize_image(image[:, :, 1]))
        return torch.from_numpy(np.stack([l3, l5], axis=0)), path.stem


def build_fold_datasets(data_root, fold_index, seed=42, use_augmented=True):
    data_root = Path(data_root)
    directories = numeric_phantom_dirs(data_root / "ID")
    if len(directories) != 100:
        raise RuntimeError(f"Expected 100 ID samples, found {len(directories)}")
    fold = make_folds(len(directories), seed=seed)[fold_index]
    train_dirs = [directories[index] for index in fold["train"]]
    val_dirs = [directories[index] for index in fold["val"]]
    test_dirs = [directories[index] for index in fold["test"]]
    train_dataset = IDDataset(train_dirs)
    if use_augmented:
        augmented = AugmentedIDDataset(
            data_root / "ID-Augmented", [path.name for path in train_dirs]
        )
        expected = len(train_dirs) * 14
        if len(augmented) != expected:
            raise RuntimeError(f"Expected {expected} augmented samples, found {len(augmented)}")
        train_dataset = ConcatDataset([train_dataset, augmented])
    return train_dataset, IDDataset(val_dirs), IDDataset(test_dirs)

