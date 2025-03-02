#!/bin/bash

# Exit on error
set -e

# Compare ViT and ResNet models
python ablation.py --model both --dataset cifar10 --epochs 30
python ablation.py --model both --dataset cifar100 --epochs 30
python ablation.py --model both --dataset food101 --epochs 30 --batch_size 256

# # Ablation studies for ViT
python ablation.py --model vit --param model.depth --values 2,4,6,8 --epochs 20
python ablation.py --model vit --param model.embed_dim --values 128,256,512,768 --epochs 20
python ablation.py --model vit --param model.patch_size --values 2,4,8 --epochs 20

# # Ablation studies for ResNet
python ablation.py --model resnet --param model.initial_filters --values 16,32,64,128 --epochs 20
python ablation.py --model resnet --param optimizer.lr --values 0.0001,0.0003,0.001,0.003 --epochs 20
