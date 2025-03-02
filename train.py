"""
Module implementing train loop for Vision Transformer and ResNet models.
"""

import os
import sys
import argparse
import yaml
import torch


import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lightning_module import LightningModel
from utils.data_utils import DataModule


# --------------------------------------------------------------------------- #
#                              Config and Arguments                           #
# --------------------------------------------------------------------------- #
def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a Vision Transformer or ResNet model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "resnet"],
        help="Model type (vit or resnet)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (defaults to 'config/{model}.yaml')",
    )

    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset name (overrides config)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )

    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )

    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )

    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )

    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory for dataset storage"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Directory for saving results",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--fast_dev_run", action="store_true", help="Run a single batch for debugging"
    )

    return parser.parse_args()


def load_config(args):
    """Load configuration from file and override with command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        dict: Configuration dictionary.
    """
    # Determine config file path
    if args.config is None:
        config_path = f"./config/{args.model}.yaml"
    else:
        config_path = args.config

    # Load the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with command line arguments
    if args.dataset is not None:
        config["data"]["dataset"] = args.dataset

    if args.batch_size is not None:
        config["data"]["train_batch_size"] = args.batch_size
        config["data"]["val_batch_size"] = args.batch_size * 2
        config["data"]["test_batch_size"] = args.batch_size * 2

    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs

    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr

    if args.use_wandb is not None:
        config["training"]["use_wandb"] = args.use_wandb

    return config


# --------------------------------------------------------------------------- #
#                              Main Train Loop                                #
# --------------------------------------------------------------------------- #


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    pl.seed_everything(args.seed)

    # Load configuration
    config = load_config(args)

    # Create save directory
    save_dir = os.path.join(args.save_dir, args.model, config["data"]["dataset"])
    os.makedirs(save_dir, exist_ok=True)

    # Initialize data module
    data_module = DataModule(
        dataset_name=config["data"]["dataset"],
        data_dir=args.data_dir,
        val_split=config["data"]["val_perc_size"],
        train_batch_size=config["data"]["train_batch_size"],
        val_batch_size=config["data"]["val_batch_size"],
        test_batch_size=config["data"]["test_batch_size"],
    )

    # Update image_size and patch_size for ViT if using Food101
    if args.model == "vit":
        config["model"]["image_size"] = data_module.get_image_size
        config["model"]["patch_size"] = data_module.get_recommended_patch_size

    # Setup the model
    model = LightningModel(
        model_type=args.model,
        model_config=config["model"],
        optimizer_config=config["optimizer"],
        scheduler_config=config["scheduler"],
        num_classes=data_module.num_classes,
        save_dir=save_dir,
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{save_dir}/checkpoints",
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=f"{save_dir}/checkpoints",
            filename="best-{val_acc:.4f}",
            save_top_k=1,
            monitor="val_acc",
            mode="max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Add early stopping if enabled
    if config["training"]["use_early_stopping"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=int(config["training"]["early_stopping_patience"]),
                min_delta=float(config["training"]["early_stopping_tol"]),
                mode="min",
            )
        )

    # Setup logger
    loggers = []
    if config["training"]["use_wandb"]:
        wandb_logger = WandbLogger(
            project=config["training"]["wandb_project"],
            name=f"{args.model}_{config['data']['dataset']}",
            save_dir=save_dir,
        )
        loggers.append(wandb_logger)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        callbacks=callbacks,
        logger=loggers if loggers else None,
        log_every_n_steps=10,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=save_dir,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    # Save the configuration
    with open(f"{save_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"Training completed. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
