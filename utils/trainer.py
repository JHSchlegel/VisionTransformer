import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

import os
import wandb
from typing import Optional, Dict
import logging
import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from typing import Tuple
import matplotlib.pyplot as plt

import warnings


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: _LRScheduler,
        save_dir: str = "checkpoints",
        num_epochs: int = 100,
        use_early_stopping: bool = True,
        early_stopping_patience: int = 25,
        early_stopping_tol: float = 1e-7,
        save_model_every: int = 25,
        use_wandb: bool = True,
        wandb_logging_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_config_kwargs: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Initialize the trainer of the ViT and ResNet models.

        Args:
            model (nn.Module): Model to train.
            criterion (nn.Module): Loss function to use.
            optimizer (optim.Optimizer): Optimizer to use.
            scheduler (optim.lr_scheduler): Scheduler to use.
            save_dir (str, optional): Directory to which results are saved. Defaults to "results".
            num_epochs (int, optional): Number of epochs to train. Defaults to 100.
            use_early_stopping (bool, optional): Whether early stopping should be used. Defaults to True.
            early_stopping_patience (int, optional): Patience for early stopping. Defaults to 25.
            early_stopping_tol (float, optional): Tolerance for early stopping. Defaults to 1e-7.
            save_model_every (int, optional): Checkpoint frequency. Defaults to 25.
            use_wandb (bool, optional): Whether weights and biases should be used for training monitoring. Defaults to True.
            wandb_logging_dir (Optional[str], optional): Directory to which wandb logs to. Defaults to None.
            wandb_project (Optional[str], optional): Wandb project name. Defaults to None.
            wandb_config_kwargs (Optional[Dict], optional): Configs that should get logged to weights and biases. Defaults to None.
            device (torch.device, optional): Torch device that is used. Defaults to torch.device("cuda").
        """

        if num_epochs % save_model_every != 0:
            warnings.warn(
                "The number of epochs should be divisible by save_model_every to save the model correctly."
            )

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_model_every
        self.device = device

        self.num_epochs = num_epochs

        # check whether directories exist; if not create them
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(wandb_logging_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)

        # define if the best model should be saved
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.lowest_val_model = self.model

        self.early_stopping = use_early_stopping
        # patience for early stopping
        self.patience = early_stopping_patience
        # tolerance for early stopping
        self.tol = early_stopping_tol

        # initialize the wandb logger
        self.use_wandb: bool = use_wandb
        if self.use_wandb:
            os.makedirs(wandb_logging_dir, exist_ok=True)
            wandb.init(
                dir=wandb_logging_dir, project=wandb_project, config=wandb_config_kwargs
            )
            # log parameters and gradients of the model to spot any learning issues
            wandb.watch(self.model, log="all", log_freq=10)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"{save_dir}/training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.

        Returns:
            Tuple[float, float]: Loss and accuracy of the epoch.
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        num_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            num_train += targets.size(0)
            train_correct += torch.sum(preds == targets).item()

        train_accuracy = 100.0 * train_correct / num_train
        train_loss /= len(train_loader)

        if isinstance(self.scheduler, (CosineAnnealingLR, CosineAnnealingWarmRestarts)):
            self.scheduler.step()

        return train_loss, train_accuracy

    @torch.no_grad()
    def validate_one_epoch(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Validate the model for one epoch.

        Args:
            val_loader (torch.utils.data.DataLoader): Validation data loader.

        Returns:
            Tuple[float, float]: Loss and accuracy of the epoch.
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        num_val = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            num_val += targets.size(0)
            train_correct += torch.sum(preds == targets).item()
        val_accuracy = 100.0 * val_correct / num_val
        val_loss /= len(val_loader)

        return val_loss, val_accuracy

    def _check_early_stopping(self, val_loss: float, epoch_idx: int) -> bool:
        """Check if early stopping should be performed.

        Args:
            val_loss (float): Validation loss.
            epoch_idx (int): Current epoch index.

        Returns:
            bool: Whether early stopping should be performed.
        """
        if val_loss < self.best_val_loss - self.tol:
            self.best_val_loss = val_loss
            self.best_epoch = epoch_idx
            self.lowest_val_model = self.model
            self.logger.info(
                f"Validation loss decreased from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving model."
            )
            self._save_checkpoint(epoch_idx, best=True)
            return False
        elif epoch_idx - self.best_epoch > self.patience:
            self.logger.info(
                f"Validation loss did not improve for {self.patience} epochs. Early stopping."
            )
            return True
        return False

    def _save_checkpoint(self, epoch_idx: int, best: bool = False) -> None:
        """Save the model checkpoint.

        Args:
            epoch_idx (int): Current epoch index.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch_idx,
            },
            (
                f"{self.save_dir}/checkpoints/model_{epoch_idx}.pt"
                if not best
                else f"{self.save_dir}/checkpoints/lowest_val_model.pt"
            ),
        )

    def _log_metrics(
        self,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        epoch_idx: int,
    ) -> None:
        """Log the metrics to the console and wandb.

        Args:
            train_loss (float): Training loss.
            train_accuracy (float): Training accuracy.
            val_loss (float): Validation loss.
            val_accuracy (float): Validation accuracy.
            epoch_idx (int): Current epoch index.
        """
        self.logger.info(
            f"Epoch {epoch_idx + 1}/{self.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f} - "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}"
        )
        if self.use_wandb:
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Train Accuracy": train_accuracy,
                    "Val Loss": val_loss,
                    "Val Accuracy": val_accuracy,
                }
            )

    @torch.no_grad()
    def _visualize_samples(
        self,
        val_loader: torch.utils.data.DataLoader,
        num_samples: int,
        epoch_idx: int,
        top_k: int = 5,
    ) -> None:
        """Visualize samples from the validation set in a grid.

        Args:
            val_loader (torch.utils.data.DataLoader): Validation data loader.
            num_samples (int): Number of samples to visualize.
            epoch_idx (int): Current epoch index.
            top_k (int, optional): Number of top-k predictions to show. Defaults to 3.
        """
        self.model.eval()
        # check whether num_samples is less than the batch size
        assert (
            num_samples < val_loader.batch_size
        ), "num_samples should be less than the batch size of the validation loader."

        inputs, targets = next(iter(val_loader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        # preds = torch.argmax(self.model(inputs), dim=1)
        # # create a grid of images
        # grid = make_grid(inputs, nrow=5)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(grid.permute(1, 2, 0))
        # plt.axis("off")
        # plt.title(f"Epoch {epoch_idx} - Ground Truth: {targets} - Predictions: {preds}")
        # plt.savefig(f"{self.save_dir}/plots/epoch_{epoch_idx}.png")
        # plt.close()

        # Get model outputs and probabilities
        outputs = self.model(inputs)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, top_k, dim=1)

        plt.figure(figsize=(15, 10))
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)

            # Display image
            img = inputs[i].cpu().numpy().transpose(1, 2, 0)
            plt.imshow(img)
            plt.axis("off")

            # Annotation for ground truth and predictions
            true_class = targets[i].item()
            title = f"True: {true_class}\n"

            # Add top-k predictions with probabilities
            for j in range(top_k):
                pred_class = top_classes[i][j].item()
                pred_prob = top_probs[i][j].item() * 100
                is_correct = pred_class == true_class
                title += f"Top-{j+1}: {pred_class} ({pred_prob:.1f}%){'*' if is_correct else ''}\n"

            plt.title(title, fontsize=8)

        plt.tight_layout()
        plt.suptitle(f"Epoch {epoch_idx} - Model Predictions", fontsize=12)
        plt.savefig(f"{self.save_dir}/plots/epoch_{epoch_idx}_predictions.png")
        plt.close()

    def train_and_validate(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_samples: int = 10,
    ) -> None:
        """Train and validate the model.

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader): Validation data loader.
            num_samples (int, optional): Number of samples to visualize. Defaults to 10.
        """
        progress_bar = tqdm.tqdm(range(self.num_epochs), desc="Training and Validation")
        for epoch_idx in progress_bar:
            train_loss, train_accuracy = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate_one_epoch(val_loader)

            self._log_metrics(
                train_loss, train_accuracy, val_loss, val_accuracy, epoch_idx
            )

            if self.early_stopping and self._check_early_stopping(val_loss, epoch_idx):
                break

            if epoch_idx == 0 or (epoch_idx + 1) % self.num_epochs == 0:
                self._save_checkpoint(epoch_idx)
                self._visualize_samples(
                    val_loader, num_samples=num_samples, epoch_idx=epoch_idx
                )

            if not isinstance(
                self.scheduler, (CosineAnnealingLR, CosineAnnealingWarmRestarts)
            ):
                self.scheduler.step()

            self._visualize_samples(val_loader, num_samples=25, epoch_idx=epoch_idx)

            # update progress bar
            progress_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_accuracy=f"{train_accuracy:.2f}",
                val_loss=f"{val_loss:.4f}",
                val_accuracy=f"{val_accuracy:.2f}",
            )

        self.logger.info("Finished training and validation.")
        self.logger.info(
            f"Best model was saved at epoch {self.best_epoch} with a validation loss of {self.best_val_loss:.4f}."
        )
        self._save_checkpoint(epoch_idx, best=True)

        if self.use_wandb:
            wandb.finish()
