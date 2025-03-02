# 🔍 Vision Transformer vs ResNet Benchmark

![Python](https://img.shields.io/badge/python-3.10%2B-blue)![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)![Black](https://img.shields.io/badge/code%20style-black-000000.svg)![License](https://img.shields.io/badge/license-MIT-green)

A simple implementation of Vision Transformer (ViT) and ResNet architectures. Allows for easy training and comparison of the two models on CIFAR-10, CIFAR-100, and Food101 datasets, including ablation studies for various hyperparameters configurations.

## 🚀 Quick Start

### ⚙️ Setup Environment

```bash
# Create conda environment
conda create -n vit python=3.10
conda activate vit

# Install dependencies
pip install -r requirements.txt
```

### 🏃‍♂️ Run Training

```bash
# Train with default config
python train.py

# Train with custom config
python train.py --config configs/vit_cifar100.yaml

# Train with command-line arguments
python train.py --model-type vit --dataset cifar100 --num-epochs 100
```

### 📊 Run Ablation Studies

```bash
# Run comprehensive comparison
python ablation.py --config configs/base_config.yaml --study all

# Compare specific aspects
python ablation.py --config configs/base_config.yaml --study vit --dataset cifar100
```

## 📁 Project Structure

```
.
├── config/                     # Configuration files
│   ├── resnet.yaml             # ResNet-specific config
│   └── vit.yaml                # ViT-specific config
├── models/                     # Model implementations
│   ├── resnet.py               # ResNet implementation
│   └── vit.py                  # Vision Transformer implementation
├── utils/                      # Utility modules
│   ├── data_utils.py           # Dataset processing and loading
│   ├── plotting_utils.py       # Visualization utilities
├── train.py                    # Unified training script
├── ablation.py                 # Ablation study framework
├── requirements.txt            # Package dependencies
└── README.md                   # This file
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
