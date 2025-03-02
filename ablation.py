"""
Ablation study for ViT and ResNet models.
"""

# --------------------------------------------------------------------------- #
#                           Packages and Presets                              #
# --------------------------------------------------------------------------- #
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import itertools
import copy
from tqdm import tqdm

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
from utils.plotting_utils import (
    plot_param_sensitivity,
    plot_model_comparison,
)


# --------------------------------------------------------------------------- #
#                              Config and Arguments                           #
# --------------------------------------------------------------------------- #
def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run ablation studies for ViT and ResNet models"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "resnet", "both"],
        help="Model type for ablation (vit, resnet, or both)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (defaults to './config/{model}.yaml')",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "food101"],
        help="Dataset for ablation",
    )

    parser.add_argument(
        "--param",
        type=str,
        default=None,
        help="Parameter to ablate (if not specified, compare models)",
    )

    parser.add_argument(
        "--values",
        type=str,
        default=None,
        help="Comma-separated values for the parameter",
    )

    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs for each ablation run"
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
        default="./ablation_results",
        help="Directory for saving results",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_config(model_type, config_path=None):
    """Load configuration from file.

    Args:
        model_type (str): Type of model.
        config_path (str, optional): Path to config file. Defaults to None.

    Returns:
        dict: Configuration dictionary.
    """
    # Determine config file path
    if config_path is None:
        config_path = f"./config/{model_type}.yaml"

    # Load the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# --------------------------------------------------------------------------- #
#                         ABLATION STUDY FUNCTIONS                            #
# --------------------------------------------------------------------------- #
def run_ablation_param(args, param_name, param_values):
    """Run ablation study for a specific parameter.

    Args:
        args (argparse.Namespace): Command line arguments.
        param_name (str): Name of the parameter to ablate.
        param_values (list): Values to try for the parameter.

    Returns:
        dict: Results of the ablation.
    """
    results = {}

    # Get the model type(s)
    model_types = ["vit", "resnet"] if args.model == "both" else [args.model]

    for model_type in model_types:
        # Load the base configuration
        config = load_config(model_type, args.config)

        # Override dataset
        config["data"]["dataset"] = args.dataset

        # Override epochs
        config["training"]["num_epochs"] = args.epochs

        # Disable wandb for individual runs unless specified
        if not args.use_wandb:
            config["training"]["use_wandb"] = False

        # Create save directory
        base_save_dir = os.path.join(
            args.save_dir, f"{model_type}_{args.dataset}_{param_name}"
        )
        os.makedirs(base_save_dir, exist_ok=True)

        # Save base configuration
        with open(f"{base_save_dir}/base_config.yaml", "w") as f:
            yaml.dump(config, f)

        model_results = {}

        # Run for each parameter value
        for value in tqdm(
            param_values, desc=f"Ablation for {model_type} - {param_name}"
        ):
            # Create a copy of the configuration
            run_config = copy.deepcopy(config)

            # Update the parameter value
            # Parse the parameter name (could be nested)
            param_parts = param_name.split(".")

            # Navigate to the correct part of the config
            config_section = run_config
            for part in param_parts[:-1]:
                config_section = config_section[part]

            # Set the parameter value
            config_section[param_parts[-1]] = value

            # Create run-specific save directory
            run_save_dir = os.path.join(base_save_dir, f"{param_name}_{value}")
            os.makedirs(run_save_dir, exist_ok=True)

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
            if model_type == "vit":
                config["model"]["image_size"] = data_module.get_image_size
                config["model"]["patch_size"] = data_module.get_recommended_patch_size

            # Setup the model
            model = LightningModel(
                model_type=model_type,
                model_config=run_config["model"],
                optimizer_config=run_config["optimizer"],
                scheduler_config=run_config["scheduler"],
                num_classes=data_module.num_classes,
                save_dir=run_save_dir,
            )

            # Setup callbacks
            callbacks = [
                ModelCheckpoint(
                    dirpath=f"{run_save_dir}/checkpoints",
                    filename="best-{val_acc:.4f}",
                    save_top_k=1,
                    monitor="val_acc",
                    mode="max",
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=int(config["training"]["early_stopping_patience"]),
                    min_delta=float(config["training"]["early_stopping_tol"]),
                    mode="min",
                ),
            ]

            # Setup logger
            loggers = []
            if run_config["training"]["use_wandb"]:
                wandb_logger = WandbLogger(
                    project=f"{run_config['training']['wandb_project']}_ablation",
                    name=f"{model_type}_{args.dataset}_{param_name}_{value}",
                    save_dir=run_save_dir,
                )
                loggers.append(wandb_logger)

            # Create trainer
            trainer = pl.Trainer(
                max_epochs=run_config["training"]["num_epochs"],
                callbacks=callbacks,
                logger=loggers if loggers else None,
                log_every_n_steps=10,
                deterministic=True,
                default_root_dir=run_save_dir,
            )

            # Train the model
            trainer.fit(model, data_module)

            # Get the metrics from the trainer
            results_dict = {
                "train_loss": trainer.callback_metrics.get(
                    "train_loss_epoch", 0
                ).item(),
                "val_loss": trainer.callback_metrics.get("val_loss", 0).item(),
                "train_acc": trainer.callback_metrics.get("train_acc_epoch", 0).item(),
                "val_acc": trainer.callback_metrics.get("val_acc", 0).item(),
            }

            # Save run-specific configuration
            with open(f"{run_save_dir}/config.yaml", "w") as f:
                yaml.dump(run_config, f)

            # Store the results
            model_results[str(value)] = results_dict

        # Store model results
        results[model_type] = model_results

        # Plot parameter sensitivity
        metrics = {
            "Train Accuracy": [
                model_results[str(v)]["train_acc"] for v in param_values
            ],
            "Validation Accuracy": [
                model_results[str(v)]["val_acc"] for v in param_values
            ],
            "Train Loss": [model_results[str(v)]["train_loss"] for v in param_values],
            "Validation Loss": [
                model_results[str(v)]["val_loss"] for v in param_values
            ],
        }

        plot_param_sensitivity(
            param_values,
            metrics,
            param_name,
            save_path=f"{base_save_dir}/{model_type}_{param_name}_sensitivity.png",
            title=f"{model_type.upper()} Sensitivity to {param_name}",
        )

    return results


def run_model_comparison(args):
    """Run model comparison between ViT and ResNet.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        dict: Results of the comparison.
    """
    results = {}

    # Load configurations
    vit_config = load_config("vit", args.config)
    resnet_config = load_config("resnet", args.config)

    # Override dataset and epochs
    vit_config["data"]["dataset"] = args.dataset
    resnet_config["data"]["dataset"] = args.dataset
    vit_config["training"]["num_epochs"] = args.epochs
    resnet_config["training"]["num_epochs"] = args.epochs

    # Disable wandb for individual runs unless specified
    if not args.use_wandb:
        vit_config["training"]["use_wandb"] = False
        resnet_config["training"]["use_wandb"] = False

    # Create save directory
    base_save_dir = os.path.join(args.save_dir, f"compare_{args.dataset}")
    os.makedirs(base_save_dir, exist_ok=True)

    # Run each model
    model_configs = {
        "vit": vit_config,
        "resnet": resnet_config,
    }

    for model_type, config in model_configs.items():
        # Create run-specific save directory
        run_save_dir = os.path.join(base_save_dir, model_type)
        os.makedirs(run_save_dir, exist_ok=True)

        # Initialize data module
        data_module = DataModule(
            dataset_name=config["data"]["dataset"],
            data_dir=args.data_dir,
            val_split=config["data"]["val_perc_size"],
            train_batch_size=config["data"]["train_batch_size"],
            val_batch_size=config["data"]["val_batch_size"],
            test_batch_size=config["data"]["test_batch_size"],
        )

        # Setup the model
        model = LightningModel(
            model_type=model_type,
            model_config=config["model"],
            optimizer_config=config["optimizer"],
            scheduler_config=config["scheduler"],
            num_classes=data_module.num_classes,
            save_dir=run_save_dir,
        )

        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=f"{run_save_dir}/checkpoints",
                filename="best-{val_acc:.4f}",
                save_top_k=1,
                monitor="val_acc",
                mode="max",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=int(config["training"]["early_stopping_patience"]),
                min_delta=float(config["training"]["early_stopping_tol"]),
                mode="min",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Setup logger
        loggers = []
        if config["training"]["use_wandb"]:
            wandb_logger = WandbLogger(
                project=f"{config['training']['wandb_project']}_comparison",
                name=f"{model_type}_{args.dataset}",
                save_dir=run_save_dir,
            )
            loggers.append(wandb_logger)

        # Create trainer with a callback to collect metrics
        metrics_dict = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        # Custom callback to collect metrics
        class MetricsCallback(pl.Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                metrics_dict["train_loss"].append(
                    trainer.callback_metrics.get("train_loss_epoch", 0).item()
                )
                metrics_dict["train_acc"].append(
                    trainer.callback_metrics.get("train_acc_epoch", 0).item()
                )

            def on_validation_epoch_end(self, trainer, pl_module):
                metrics_dict["val_loss"].append(
                    trainer.callback_metrics.get("val_loss", 0).item()
                )
                metrics_dict["val_acc"].append(
                    trainer.callback_metrics.get("val_acc", 0).item()
                )

        callbacks.append(MetricsCallback())

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=config["training"]["num_epochs"],
            callbacks=callbacks,
            logger=loggers if loggers else None,
            log_every_n_steps=10,
            deterministic=True,
            default_root_dir=run_save_dir,
        )

        # Train the model
        trainer.fit(model, data_module)

        # Test the model
        trainer.test(model, data_module)

        # Save configuration
        with open(f"{run_save_dir}/config.yaml", "w") as f:
            yaml.dump(config, f)

        # Store the results
        results[model_type] = metrics_dict

    # Plot model comparison
    plot_model_comparison(
        results,
        "val_acc",
        save_path=f"{base_save_dir}/model_comparison_val_acc.png",
        title=f"Model Comparison on {args.dataset} - Validation Accuracy",
    )

    plot_model_comparison(
        results,
        "val_loss",
        save_path=f"{base_save_dir}/model_comparison_val_loss.png",
        title=f"Model Comparison on {args.dataset} - Validation Loss",
    )

    return results


def main():
    """Main function for ablation studies."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    pl.seed_everything(args.seed)

    # Check if we're ablating a parameter or comparing models
    if args.param is not None:
        # Parse parameter values
        if args.values is not None:
            # Try to infer the type of values
            try:
                # Try as float
                param_values = [float(v) for v in args.values.split(",")]
                # If all values are integers, convert to int
                if all(v.is_integer() for v in param_values):
                    param_values = [int(v) for v in param_values]
            except ValueError:
                # If not numeric, keep as string
                param_values = args.values.split(",")
        else:
            # Default values based on common parameters
            if args.param == "model.depth" or args.param == "depth":
                param_values = [2, 4, 6, 8]
            elif args.param == "model.patch_size" or args.param == "patch_size":
                param_values = [2, 4, 8, 16]
            elif args.param == "model.embed_dim" or args.param == "embed_dim":
                param_values = [128, 256, 512, 768]
            elif args.param == "model.num_heads" or args.param == "num_heads":
                param_values = [4, 8, 12, 16]
            elif args.param == "model.dropout" or args.param == "dropout":
                param_values = [0.0, 0.1, 0.2, 0.3]
            elif args.param == "optimizer.lr" or args.param == "lr":
                param_values = [0.0001, 0.0003, 0.001, 0.003]
            elif (
                args.param == "model.initial_filters" or args.param == "initial_filters"
            ):
                param_values = [16, 32, 64, 128]
            else:
                # If no default, use a range of values
                param_values = [1, 2, 4, 8]
                print(f"No default values for {args.param}, using {param_values}")

        # Run ablation
        results = run_ablation_param(args, args.param, param_values)

        print(f"Ablation complete. Results saved to {args.save_dir}")

    else:
        # Run model comparison
        results = run_model_comparison(args)

        print(f"Model comparison complete. Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
