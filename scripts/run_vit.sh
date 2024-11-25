#!/bin/bash

source ~/miniconda3/bin/activate ml

export CUDA_VISIBLE_DEVICES=0

echo "Training ViT on CIFAR-10"
python ../train/train_vit.py "data.dataset='cifar10'"
echo "Training ViT on CIFAR-100"
python ../train/train_vit.py "data.dataset='cifar100'"
