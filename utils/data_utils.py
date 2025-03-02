"""
This module contains the LightningDataModule for loading datasets.
"""

# --------------------------------------------------------------------------- #
#                           Packages and Presets                              #
# --------------------------------------------------------------------------- #
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch
from typing import Tuple, Dict, Any, Optional
import os
import pytorch_lightning as pl


# --------------------------------------------------------------------------- #
#                               Data Module                                   #
# --------------------------------------------------------------------------- #
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "../../data",
        val_split: float = 0.2,
        train_batch_size: int = 256,
        val_batch_size: int = 512,
        test_batch_size: int = 512,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """Initialize the DataModule for loading datasets.

        Args:
            dataset_name (str): Name of the dataset ('cifar10', 'cifar100', or 'food101').
            data_dir (str, optional): Directory for data storage. Defaults to "../../data".
            val_split (float, optional): Validation split ratio. Defaults to 0.2.
            train_batch_size (int, optional): Batch size for training. Defaults to 256.
            val_batch_size (int, optional): Batch size for validation. Defaults to 512.
            test_batch_size (int, optional): Batch size for testing. Defaults to 512.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            pin_memory (bool, optional): Whether to pin memory for faster data transfer to GPU. Defaults to True.
        """
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.val_split = val_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        os.makedirs(data_dir, exist_ok=True)
        self._setup_transforms()

    def _setup_transforms(self):
        """Set up data transformations for different datasets."""
        if self.dataset_name == "cifar10":
            self.train_transforms = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                    ),
                ]
            )

            self.test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                    ),
                ]
            )

        elif self.dataset_name == "cifar100":
            self.train_transforms = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                    ),
                ]
            )

            self.test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                    ),
                ]
            )

        elif self.dataset_name == "food101":
            self.train_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.test_transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def prepare_data(self):
        """Download datasets if needed."""
        if self.dataset_name == "cifar10":
            datasets.CIFAR10(self.data_dir, train=True, download=True)
            datasets.CIFAR10(self.data_dir, train=False, download=True)
        elif self.dataset_name == "cifar100":
            datasets.CIFAR100(self.data_dir, train=True, download=True)
            datasets.CIFAR100(self.data_dir, train=False, download=True)
        elif self.dataset_name == "food101":
            datasets.Food101(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for various stages (train, val, test).

        Args:
            stage (Optional[str], optional): Stage ('fit', 'validate', 'test'). Defaults to None.
        """
        if self.dataset_name == "cifar10":
            if stage == "fit" or stage is None:
                dataset = datasets.CIFAR10(
                    self.data_dir, train=True, transform=self.train_transforms
                )

                train_size = int((1 - self.val_split) * len(dataset))
                val_size = len(dataset) - train_size

                self.train_dataset, self.val_dataset = random_split(
                    dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42),
                )

                # Apply test transforms to the validation dataset
                self.val_dataset.dataset.transform = self.test_transforms

            if stage == "test" or stage is None:
                self.test_dataset = datasets.CIFAR10(
                    self.data_dir, train=False, transform=self.test_transforms
                )

        elif self.dataset_name == "cifar100":
            if stage == "fit" or stage is None:
                dataset = datasets.CIFAR100(
                    self.data_dir, train=True, transform=self.train_transforms
                )

                train_size = int((1 - self.val_split) * len(dataset))
                val_size = len(dataset) - train_size

                self.train_dataset, self.val_dataset = random_split(
                    dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42),
                )

                # Apply test transforms to the validation dataset
                self.val_dataset.dataset.transform = self.test_transforms

            if stage == "test" or stage is None:
                self.test_dataset = datasets.CIFAR100(
                    self.data_dir, train=False, transform=self.test_transforms
                )

        elif self.dataset_name == "food101":
            if stage == "fit" or stage is None:
                full_dataset = datasets.Food101(
                    self.data_dir, transform=self.train_transforms
                )

                # Split into train and test (following Food-101's original 75k/25k split)
                train_size = 75750  # 75% of 101000
                test_size = len(full_dataset) - train_size

                train_temp_dataset, self.test_dataset = random_split(
                    full_dataset,
                    [train_size, test_size],
                    generator=torch.Generator().manual_seed(42),
                )

                # Further split train into train and validation
                train_len = int((1 - self.val_split) * len(train_temp_dataset))
                val_len = len(train_temp_dataset) - train_len

                self.train_dataset, self.val_dataset = random_split(
                    train_temp_dataset,
                    [train_len, val_len],
                    generator=torch.Generator().manual_seed(42),
                )

                # Apply test transforms to validation and test datasets
                self.val_dataset.dataset.transform = self.test_transforms
                self.test_dataset.dataset.transform = self.test_transforms

    def train_dataloader(self):
        """Create a DataLoader for training.

        Returns:
            DataLoader: Training data loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Create a DataLoader for validation.

        Returns:
            DataLoader: Validation data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Create a DataLoader for testing.

        Returns:
            DataLoader: Test data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self):
        """Get the number of classes in the dataset.

        Returns:
            int: Number of classes.
        """
        if self.dataset_name == "cifar10":
            return 10
        elif self.dataset_name == "cifar100":
            return 100
        elif self.dataset_name == "food101":
            return 101
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    @property
    def get_image_size(self):
        """Get the image size for the dataset.

        Returns:
            int: Image size.
        """
        if self.dataset_name == "cifar10" or self.dataset_name == "cifar100":
            return 32
        elif self.dataset_name == "food101":
            return 224
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    @property
    def get_recommended_patch_size(self):
        """Get the recommended patch size for the dataset.

        Returns:
            int: Recommended patch size.
        """
        if self.dataset_name == "cifar10" or self.dataset_name == "cifar100":
            return 4
        elif self.dataset_name == "food101":
            return 16
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
