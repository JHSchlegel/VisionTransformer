# A Simple Vision Transformer Implementation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)![PyTorch](https://img.shields.io/badge/PyTorch-2.01%2B-orange)![Black](https://img.shields.io/badge/code%20style-black-000000.svg)![License](https://img.shields.io/badge/license-MIT-green)

A simple implementation of Vision Transformer (ViT) and ResNet architectures. Allows for easy training and comparison of the two models on CIFAR-10, CIFAR-100, and Food101 datasets, including ablation studies for various hyperparameter configurations.

## Quick Start

### Setup Environment

```bash
# Create conda environment
conda create -n vit python=3.10
conda activate vit

# Install dependencies
pip install -r requirements.txt
```

### Run Training

```bash
# Train with default config
python train.py

# Train with custom config
python train.py --config configs/vit_cifar100.yaml

# Train with command-line arguments
python train.py --model-type vit --dataset cifar100 --num-epochs 100

# Train ResNet and ViT on all datasets
bash train.sh
```

### Run Ablation Studies

```bash
# Run comprehensive comparison
python ablation.py --config configs/base_config.yaml --study all

# Compare specific aspects
python ablation.py --config configs/base_config.yaml --study vit --dataset cifar100

# Perform prepared ablations
bash ablation.sh
```

## Project Structure

```
.
├── config/                     # Configuration files
│   ├── resnet.yaml             # ResNet-specific config
│   └── vit.yaml                # ViT-specific config
├── models/
│   ├── lightning_module.py     # Pytorch Lightning Trainer
│   ├── resnet.py               # ResNet implementation
│   └── vit.py                  # Vision Transformer implementation
├── utils/                     
│   ├── data_utils.py           # Dataset processing and loading
│   └── plotting_utils.py       # Visualization utilities
├── scripts/                    
│   ├── train.sh                # Bash script for training 
│   └── plotting_utils.py       # Bash script for ablations
├── train.py                    # Unified training script
├── ablation.py                 # Ablation study framework
├── requirements.txt            # Package dependencies
└── README.md                   # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
