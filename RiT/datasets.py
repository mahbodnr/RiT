import os
import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment, AutoAugmentPolicy


class ImageNet64Dataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.transform = transform
        assert split in ["train", "val"], "Split must be either 'train' or 'val'"
        # Get list of all files in the directory
        all_files = [
            f
            for f in os.listdir(self.root)
            if os.path.isfile(os.path.join(self.root, f))
        ]
        self.file_list = [f for f in all_files if f.startswith(split)]

        # Load all data into memory
        self.data = []
        self.labels = []
        for file in self.file_list:
            npz_file = np.load(os.path.join(self.root, file), allow_pickle=True)
            self.data.append(npz_file["data"])
            self.labels.append(npz_file["labels"])

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0) - 1  # 1-indexed to 0-indexed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx].reshape(64, 64, 3))
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class TinyImageNetDataset(Dataset):
    # adopted and adjusted from: https://github.com/pranavphoenix/TinyImageNetLoader/blob/main/tinyimagenetloader.py

    def __init__(self, root, split="val", transform=None):
        self.root = root
        assert split in [
            "train",
            "val",
            "test",
        ], "Split must be either 'train', 'val' or 'test'"
        self.id_dict = {}
        for i, line in enumerate(open(os.path.join(self.root, "wnids.txt"), "r")):
            self.id_dict[line.replace("\n", "")] = i

        self.filenames = (
            glob.glob(os.path.join(self.root, f"train/*/*/*.JPEG"))
            if split == "train"
            else glob.glob(os.path.join(self.root, f"{split}/images/*.JPEG"))
        )
        self.transform = transform
        if split in ["val", "test"]:
            self.cls_dic = {}
            for i, line in enumerate(
                open(os.path.join(self.root, f"{split}/{split}_annotations.txt"), "r")
            ):
                a = line.split("\t")
                img, cls_id = a[0], a[1]
                self.cls_dic[img] = self.id_dict[cls_id]

        self.data = []
        self.labels = []
        for img_path in self.filenames:
            img = Image.open(img_path).convert("RGB")
            self.data.append(img.copy())
            img.close()
            if split == "train":
                self.labels.append(self.id_dict[img_path.split("/")[-3]])
            else:
                self.labels.append(self.cls_dic[img_path.split("/")[-1]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transform(args):
    train_transform = []
    val_transform = []
    input_size = args.dataset_image_shape[1:] if args.image_size is None else args.image_size

    if args.autoaugment:
        if args.dataset in ["cifar10", "imagenet", "svhn"]:
            train_transform.append(
                autoaugment.AutoAugment(
                    policy=getattr(AutoAugmentPolicy, args.dataset.upper())
                )
            )
        elif args.dataset == "cifar100":
            train_transform.append(
                autoaugment.AutoAugment(policy=getattr(AutoAugmentPolicy, "CIFAR10"))
            )
        elif args.dataset in ["imagenet64", "tiny-imagenet"]:
            train_transform.append(
                autoaugment.AutoAugment(policy=getattr(AutoAugmentPolicy, "IMAGENET"))
            )
        else:
            raise NotImplementedError(f"AutoAugment not implemented for {args.dataset}")

    train_transform.append(
        transforms.Resize(input_size)
    )
    if args.randaugment:
        train_transform.append(
            transforms.RandAugment(
                num_ops=2, magnitude=9 # TODO: add arguments
            )
        )
    train_transform.append(
        transforms.ToTensor(),
    )

    val_transform += [
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ]

    if args.mean is not None:
        train_transform.append(transforms.Normalize(mean=args.mean, std=args.std))
        val_transform.append(transforms.Normalize(mean=args.mean, std=args.std))



    if args.random_crop:
        assert args.random_crop_size is not None, "random_crop_size is required."
        train_transform.append(
            transforms.RandomCrop(
                size=args.random_crop_size, padding=args.random_crop_padding
            )
        )
        val_transform.append(transforms.CenterCrop(size=args.random_crop_size))

    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose(val_transform)

    return train_transform, val_transform


def get_dataloader(args):
    if args.dataset == "cifar10":
        args.in_c = 3
        args.num_classes = 10
        args.dataset_image_shape = (3, 32, 32)
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, val_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(
            args.data_root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        val_ds = torchvision.datasets.CIFAR10(
            args.data_root,
            train=False,
            transform=val_transform,
            download=args.download_data,
        )

    elif args.dataset == "cifar100":
        args.in_c = 3
        args.num_classes = 100
        args.dataset_image_shape = (3, 32, 32)
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, val_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(
            args.data_root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        val_ds = torchvision.datasets.CIFAR100(
            args.data_root,
            train=False,
            transform=val_transform,
            download=args.download_data,
        )

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes = 10
        args.dataset_image_shape = (3, 32, 32)
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, val_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(
            args.data_root,
            split="train",
            transform=train_transform,
            download=args.download_data,
        )
        val_ds = torchvision.datasets.SVHN(
            args.data_root,
            split="test",
            transform=val_transform,
            download=args.download_data,
        )

    elif args.dataset == "mnist":
        args.in_c = 1
        args.num_classes = 10
        args.size = (1, 28, 28)
        args.mean, args.std = [0.1307], [0.3081]
        train_transform, val_transform = get_transform(args)
        train_ds = torchvision.datasets.MNIST(
            args.data_root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        val_ds = torchvision.datasets.MNIST(
            args.data_root,
            train=False,
            transform=val_transform,
            download=args.download_data,
        )
    elif args.dataset == "fashionmnist":
        args.in_c = 1
        args.num_classes = 10
        args.dataset_image_shape = (1, 28, 28)
        args.mean, args.std = [0.2860], [0.3530]
        train_transform, val_transform = get_transform(args)
        train_ds = torchvision.datasets.FashionMNIST(
            args.data_root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        val_ds = torchvision.datasets.FashionMNIST(
            args.data_root,
            train=False,
            transform=val_transform,
            download=args.download_data,
        )

    elif args.dataset == "imagenet":
        args.in_c = 3
        args.num_classes = 1000
        args.dataset_image_shape = (3, 224, 224)
        args.mean, args.std = (
            (0.485, 0.456, 0.406),
           (0.229, 0.224, 0.225),
        )
        train_transform, val_transform = get_transform(args)
        train_ds = torchvision.datasets.ImageNet(
            args.data_root,
            split="train",
            transform=train_transform,
        )
        val_ds = torchvision.datasets.ImageNet(
            args.data_root,
            split="val",
            transform=val_transform,
        )

    elif args.dataset == "imagenet64":
        # Download from  https://www.image-net.org/download-images.php
        # and extract to data_root
        args.in_c = 3
        args.num_classes = 1000
        args.dataset_image_shape = (3, 64, 64)
        args.mean, args.std = (
            (0.485, 0.456, 0.406),
           (0.229, 0.224, 0.225),
        )
        train_transform, val_transform = get_transform(args)
        train_ds = ImageNet64Dataset(
            args.data_root,
            split="train",
            transform=train_transform,
        )
        val_ds = ImageNet64Dataset(
            args.data_root,
            split="val",
            transform=val_transform,
        )

    elif args.dataset == "tiny-imagenet":
        args.in_c = 3
        args.num_classes = 200
        args.dataset_image_shape = (3, 64, 64)
        args.mean, args.std = (
            (0.485, 0.456, 0.406),
           (0.229, 0.224, 0.225),
        )
        train_transform, val_transform = get_transform(args)
        train_ds = TinyImageNetDataset(
            args.data_root, split="train", transform=train_transform
        )
        val_ds = TinyImageNetDataset(
            args.data_root, split="val", transform=val_transform
        )

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return train_dl, val_dl
