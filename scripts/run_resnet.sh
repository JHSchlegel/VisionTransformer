#!/bin/bash

source ~/miniconda3/bin/activate ml

export CUDA_VISIBLE_DEVICES=0

echo "Training ResNet on CIFAR-10"
python ../train/train_resnet.py "data.dataset='cifar10'"
echo "Training ResNet on CIFAR-100"
python ../train/train_resnet.py "data.dataset='cifar100'"
