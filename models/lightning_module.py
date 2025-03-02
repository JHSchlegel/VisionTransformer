"""
Lightning Trainer for training and evaluating models.
"""

# --------------------------------------------------------------------------- #
#                           Packages and Presets                              #
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any, Union
import torchmetrics
import wandb
import os

from models.resnet import ResNet
from models.vit import VisionTransformer
from utils.plotting_utils import (
    plot_model_predictions,
    plot_attention_maps,
    plot_confusion_matrix,
    plot_class_accuracy,
    plot_learning_rate,
)


# --------------------------------------------------------------------------- #
#                               Trainer Module                                #
# --------------------------------------------------------------------------- #
class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model_type: str,
        model_config: Dict,
        optimizer_config: Dict,
        scheduler_config: Dict,
        num_classes: int,
        save_dir: str,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize the Lightning model.

        Args:
            model_type (str): Type of model ("resnet" or "vit").
            model_config (Dict): Configuration for the model.
            optimizer_config (Dict): Configuration for the optimizer.
            scheduler_config (Dict): Configuration for the scheduler.
            num_classes (int): Number of classes.
            save_dir (str): Directory to save results.
            class_names (Optional[List[str]], optional): Names of classes. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.class_names = (
            class_names if class_names else [str(i) for i in range(num_classes)]
        )

        # Initialize metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        # Initialize the model
        if self.model_type == "resnet":
            self.model = ResNet(
                input_channels=model_config.get("input_channels", 3),
                num_classes=num_classes,
                initial_filters=model_config.get("initial_filters", 16),
                block_configs=model_config.get("block_configs", None),
            )
        elif self.model_type == "vit":
            self.model = VisionTransformer(
                image_size=model_config.get("image_size", 32),
                patch_size=model_config.get("patch_size", 4),
                in_channels=model_config.get("in_channels", 3),
                num_classes=num_classes,
                embed_dim=model_config.get("embed_dim", 512),
                depth=model_config.get("depth", 4),
                num_heads=model_config.get("num_heads", 8),
                mlp_ratio=model_config.get("mlp_ratio", 4),
                dropout=model_config.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Store configs for optimizer and scheduler
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # Create save directories
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

        # Store predictions for evaluation
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizers and schedulers.

        Returns:
            Dict: Optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 0.001),
            weight_decay=self.optimizer_config.get("weight_decay", 0.0001),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.scheduler_config.get("T_0", 10),
            T_mult=self.scheduler_config.get("T_mult", 2),
        )

        plot_learning_rate(
            scheduler,
            self.trainer.max_epochs,  # Access max_epochs from the trainer
            f"{self.save_dir}/plots/lr_schedule.png",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        """Perform a training step.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)

        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        self.train_acc(preds, targets)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            Dict: Validation metrics.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)

        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        self.val_acc(preds, targets)

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)

        # Visualize predictions
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            self._visualize_predictions(inputs, targets)

        return {"val_loss": loss, "val_acc": self.val_acc}

    def test_step(self, batch, batch_idx):
        """Perform a test step.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            Dict: Test metrics.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)

        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        self.test_acc(preds, targets)

        # Store predictions for later evaluation
        self.test_preds.extend(preds.cpu().numpy())
        self.test_targets.extend(targets.cpu().numpy())

        # Log metrics
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "test_acc", self.test_acc, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return {"test_loss": loss, "test_acc": self.test_acc}

    def on_test_end(self):
        """Perform operations after test is complete."""
        # Plot confusion matrix
        plot_confusion_matrix(
            self.test_targets,
            self.test_preds,
            save_path=f"{self.save_dir}/plots/confusion_matrix.png",
            class_names=self.class_names,
            title=f"{self.model_type.upper()} Confusion Matrix",
        )

        # Plot per-class accuracy
        plot_class_accuracy(
            self.test_targets,
            self.test_preds,
            save_path=f"{self.save_dir}/plots/class_accuracy.png",
            class_names=self.class_names,
            title=f"{self.model_type.upper()} Per-Class Accuracy",
        )

        # Log images to wandb if available
        if wandb.run is not None:
            wandb.log(
                {
                    "confusion_matrix": wandb.Image(
                        f"{self.save_dir}/plots/confusion_matrix.png"
                    ),
                    "class_accuracy": wandb.Image(
                        f"{self.save_dir}/plots/class_accuracy.png"
                    ),
                }
            )

    def _visualize_predictions(self, inputs, targets):
        """Visualize model predictions.

        Args:
            inputs (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.
        """
        # Plot model predictions
        plot_model_predictions(
            self.model,
            [(inputs, targets)],
            self.device,
            save_path=f"{self.save_dir}/plots/epoch_{self.current_epoch}_predictions.png",
            class_names=self.class_names,
        )

        # Plot attention maps for ViT
        if self.model_type == "vit":
            plot_attention_maps(
                self.model,
                inputs[:5],
                targets[:5],
                save_path=f"{self.save_dir}/plots/epoch_{self.current_epoch}_attention.png",
                class_names=self.class_names,
            )

        # Log images to wandb if available
        if wandb.run is not None:
            wandb.log(
                {
                    f"predictions_epoch_{self.current_epoch}": wandb.Image(
                        f"{self.save_dir}/plots/epoch_{self.current_epoch}_predictions.png"
                    ),
                }
            )

            if self.model_type == "vit":
                wandb.log(
                    {
                        f"attention_epoch_{self.current_epoch}": wandb.Image(
                            f"{self.save_dir}/plots/epoch_{self.current_epoch}_attention.png"
                        ),
                    }
                )
