from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from typing import Tuple


# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151 for mean and std
def prepare_cifar_100(
    val_perc_size: float,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare the CIFAR-100 dataset including train-validation split and data augmentation.

    Args:
        val_perc_size (float): Size of the validation set in percent.
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation.
        test_batch_size (int): Batch size for testing.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders.
    """

    train_transforms = transforms.Compose(
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

    # Validation and testing data transformations
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )
    dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transforms
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transforms
    )

    train_len = int((1 - val_perc_size) * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Apply test transforms to the validation dataset
    val_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=4)

    return train_loader, val_loader, test_loader


# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151 for mean and std
def prepare_cifar_10(
    val_perc_size: float,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare the CIFAR-10 dataset including train-validation split and data augmentation.

    Args:
        val_perc_size (float): Size of the validation set in percent.
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation.
        test_batch_size (int): Batch size for testing.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders.
    """

    train_transforms = transforms.Compose(
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

    # Validation and testing data transformations
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transforms
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transforms
    )

    train_len = int((1 - val_perc_size) * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Apply test transforms to the validation dataset
    val_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=4)

    return train_loader, val_loader, test_loader
