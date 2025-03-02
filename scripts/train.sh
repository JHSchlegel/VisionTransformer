#!/bin/bash


# Exit on error
set -e

cd "$(dirname "$0")/../train"



# Train ResNet on CIFAR-10
python train.py --model resnet --dataset cifar10 --use_wandb

# Train ResNet on CIFAR-100
python train.py --model resnet --dataset cifar100 --use_wandb

# Train ResNet on Food101
python train.py --model resnet --dataset food101 --use_wandb

# Train ViT on CIFAR-10
python train.py --model vit --dataset cifar10 --use_wandb

# Train ViT on CIFAR-100
python train.py --model vit --dataset cifar100 --use_wandb

# Train ViT on Food101
python train.py --model vit --dataset food101 --use_wandb