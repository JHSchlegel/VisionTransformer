"""
Utility functions for plotting training curves, confusion matrix, attention maps,
model predictions, learning rate schedule, parameter sensitivity, 
class accuracy, and model comparison.
"""

# --------------------------------------------------------------------------- #
#                             Packages and Presets                            #
# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import pandas as pd
from torchvision.utils import make_grid
import torch.nn.functional as F
import cv2
import matplotlib as mpl


# --------------------------------------------------------------------------- #
#                             Plotting Functions                              #
# --------------------------------------------------------------------------- #
def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (12, 6),
    dark_mode: bool = False,
):
    """Plot training curves for various metrics.

    Args:
        metrics (Dict[str, List[float]]): Dictionary of metrics with their values over epochs.
        save_path (str): Path to save the plot.
        title (str, optional): Title of the plot. Defaults to "Training Curves".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        dark_mode (bool, optional): Use dark background. Defaults to False.
    """
    plt.figure(figsize=figsize)

    if dark_mode:
        plt.style.use("dark_background")

    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: str,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
):
    """Plot confusion matrix.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        save_path (str): Path to save the plot.
        class_names (Optional[List[str]], optional): Names of classes. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
        normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to True.
    """
    plt.figure(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap="Blues", values_format=".2f" if normalize else "d")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_attention_maps(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    save_path: str,
    top_k: int = 5,
    figsize: Tuple[int, int] = (20, 14),
    num_heads: int = 4,
    class_names: Optional[List[str]] = None,
):
    """Plot attention maps for Vision Transformer with individual head heatmaps and aggregated attention overlay.

    Args:
        model: The Vision Transformer model.
        inputs (torch.Tensor): Input tensor.
        targets (torch.Tensor): Target tensor.
        save_path (str): Path to save the plot.
        top_k (int, optional): Number of top predictions to show. Defaults to 5.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (20, 14).
        num_heads (int, optional): Number of attention heads to visualize. Defaults to 4.
        class_names (Optional[List[str]], optional): Names of classes. Defaults to None.
    """

    model.eval()

    with torch.no_grad():
        outputs, attn_weights = model(inputs, return_attention_map=True)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, min(top_k, probs.size(1)), dim=1)

    # Plot images with attention maps
    num_samples = min(5, inputs.size(0))

    # Create the figure with a layout matching the reference
    fig, axes = plt.subplots(num_samples, num_heads + 3, figsize=figsize)

    # Make axes indexable even for a single sample
    if num_samples == 1:
        axes = np.array([axes])

    # Create a consistent colormap for attention visualization
    cmap = plt.cm.viridis

    for idx in range(num_samples):
        # Display the original image
        ax = axes[idx, 0] if num_samples > 1 else axes[0]

        img = inputs[idx].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        img_uint8 = (img * 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_axis_off()

        true_class = targets[idx].item()
        class_label = class_names[true_class] if class_names else str(true_class)
        ax.set_title(f"True: {class_label}")

        # Display top predictions
        ax = axes[idx, 1] if num_samples > 1 else axes[1]
        ax.axis("off")

        for k in range(min(top_k, len(top_classes[idx]))):
            pred_class = top_classes[idx][k].item()
            pred_prob = top_probs[idx][k].item() * 100
            pred_label = class_names[pred_class] if class_names else str(pred_class)
            is_correct = pred_class == true_class
            marker = "✓" if is_correct else ""

            ax.text(
                0.1,
                0.1 + 0.12 * k,
                f"Top-{k+1}: {pred_label}\n{pred_prob:.1f}%{marker}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )

        # Process attention weights for individual heads
        # Use only the last layer attention
        last_layer_attn = attn_weights[-1][idx]  # [num_heads, seq_len, seq_len]

        # Properly handle attention weights for each head
        for head in range(min(num_heads, last_layer_attn.shape[0])):
            head_idx = head + 2  # Column index for this head
            ax = axes[idx, head_idx] if num_samples > 1 else axes[head_idx]

            # Get attention weights for the current head
            head_attn = last_layer_attn[head].cpu()  # [seq_len, seq_len]

            # Add residual connection and re-normalize
            seq_len = head_attn.shape[0]
            residual_attn = torch.eye(seq_len)
            aug_attn = head_attn + residual_attn
            aug_attn = aug_attn / aug_attn.sum(dim=-1).unsqueeze(-1)

            # Get attention from CLS token (index 0) to patches (indices 1:)
            cls_attn = aug_attn[0, 1:]

            # Calculate grid size for reshaping
            grid_size = int(np.sqrt(cls_attn.shape[0]))
            attn_map = cls_attn.reshape(grid_size, grid_size).detach().numpy()

            # Visualize the attention map as a heatmap
            im = ax.imshow(attn_map, cmap=cmap, interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(f"Head {head+1}")

        # Now add the aggregated attention visualization
        ax = axes[idx, num_heads + 2] if num_samples > 1 else axes[num_heads + 2]

        # Stack attention matrices from all layers
        att_mats = []
        for layer_attn in attn_weights:
            # Average attention across heads for this layer and sample
            layer_attn_sample = layer_attn[idx].mean(dim=0).cpu()  # [seq_len, seq_len]
            att_mats.append(layer_attn_sample)

        # Stack to get [num_layers, seq_len, seq_len]
        att_mat = torch.stack(att_mats)

        # Add identity for residual connections and re-normalize
        residual_att = torch.eye(att_mat.size(1), device=att_mat.device)
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices to get attention flow
        joint_attentions = torch.zeros_like(aug_att_mat)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        # Get attention from CLS token (first token) to image patches
        v = joint_attentions[-1]  # Use the last layer for final attention

        # Reshape the attention map to a grid
        grid_size = int(np.sqrt(aug_att_mat.size(-1) - 1))  # -1 for CLS token
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

        # Resize mask to match image dimensions
        h, w = img.shape[:2]
        mask_resized = cv2.resize(mask / mask.max(), (w, h))

        # Create a colored heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)

        # Convert BGR to RGB (OpenCV uses BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay the heatmap on the original image
        overlay = cv2.addWeighted(
            img_uint8,
            0.6,  # Original image with 60% weight
            heatmap,
            0.4,  # Heatmap with 40% weight
            0,  # Gamma correction
        )

        # Display the overlay
        ax.imshow(overlay)
        ax.set_axis_off()
        ax.set_title("Aggregated Attention")

    # Add a global colorbar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention Weight")

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    plt.suptitle("Vision Transformer Attention Visualization", fontsize=16, y=0.98)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_predictions(
    model,
    dataloader,
    device,
    save_path: str,
    top_k: int = 5,
    figsize: Tuple[int, int] = (15, 10),
    num_samples: int = 5,
    class_names: Optional[List[str]] = None,
):
    """Plot model predictions.

    Args:
        model: The model.
        dataloader: DataLoader to sample from.
        device: Device to run inference on.
        save_path (str): Path to save the plot.
        top_k (int, optional): Number of top predictions to show. Defaults to 5.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (15, 10).
        num_samples (int, optional): Number of samples to visualize. Defaults to 5.
        class_names (Optional[List[str]], optional): Names of classes. Defaults to None.
    """
    model.eval()

    # Get a batch of data
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs[:num_samples].to(device), targets[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, min(top_k, probs.size(1)), dim=1)

    # Plot images with predictions
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)

    for idx in range(num_samples):
        if num_samples > 1:
            ax = axes[idx]
        else:
            ax = axes

        img = inputs[idx].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.set_axis_off()

        true_class = targets[idx].item()
        class_label = class_names[true_class] if class_names else str(true_class)
        title = f"True: {class_label}\n"

        for k in range(min(top_k, len(top_classes[idx]))):
            pred_class = top_classes[idx][k].item()
            pred_prob = top_probs[idx][k].item() * 100
            pred_label = class_names[pred_class] if class_names else str(pred_class)
            is_correct = pred_class == true_class
            title += f"Top-{k+1}: {pred_label} ({pred_prob:.1f}%){' ✓' if is_correct else ''}\n"

        ax.set_title(title, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_learning_rate(
    lr_scheduler,
    num_epochs: int,
    save_path: str,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Learning Rate Schedule",
):
    """Plot learning rate schedule.

    Args:
        lr_scheduler: Learning rate scheduler.
        num_epochs (int): Number of epochs.
        save_path (str): Path to save the plot.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
        title (str, optional): Title of the plot. Defaults to "Learning Rate Schedule".
    """
    plt.figure(figsize=figsize)

    # Store original state to restore later
    original_optimizer = lr_scheduler.optimizer
    original_last_epoch = lr_scheduler.last_epoch

    # Create a clone of the scheduler for visualization
    optimizer_clone = type(original_optimizer)(
        original_optimizer.param_groups,
        lr=original_optimizer.param_groups[0]["lr"],
    )

    if hasattr(lr_scheduler, "T_0"):
        # CosineAnnealingWarmRestarts
        scheduler_clone = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_clone,
            T_0=lr_scheduler.T_0,
            T_mult=lr_scheduler.T_mult,
        )
    else:
        # Default to CosineAnnealingLR
        scheduler_clone = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_clone, T_max=num_epochs
        )

    lrs = []
    for _ in range(num_epochs):
        lrs.append(scheduler_clone.get_last_lr()[0])
        scheduler_clone.step()

    plt.plot(lrs)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_param_sensitivity(
    param_values: List[float],
    metrics: Dict[str, List[float]],
    param_name: str,
    save_path: str,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Parameter Sensitivity",
):
    """Plot parameter sensitivity analysis.

    Args:
        param_values (List[float]): Values of the parameter.
        metrics (Dict[str, List[float]]): Metrics for each parameter value.
        param_name (str): Name of the parameter.
        save_path (str): Path to save the plot.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
        title (str, optional): Title of the plot. Defaults to "Parameter Sensitivity".
    """
    plt.figure(figsize=figsize)

    for metric_name, values in metrics.items():
        plt.plot(param_values, values, "o-", label=metric_name)

    plt.title(f"{title} - {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_accuracy(
    y_true: List[int],
    y_pred: List[int],
    save_path: str,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Per-Class Accuracy",
):
    """Plot per-class accuracy.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        save_path (str): Path to save the plot.
        class_names (Optional[List[str]], optional): Names of classes. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        title (str, optional): Title of the plot. Defaults to "Per-Class Accuracy".
    """
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    # Get class names or indices
    if class_names is None:
        class_names = [str(i) for i in range(len(class_accuracy))]

    # Create DataFrame for better visualization
    df = pd.DataFrame({"Class": class_names, "Accuracy": class_accuracy})

    # Sort by accuracy for better visualization
    df = df.sort_values("Accuracy", ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(x="Class", y="Accuracy", data=df)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict[str, List[float]]],
    metric: str,
    save_path: str,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Model Comparison",
):
    """Plot comparison of multiple models.

    Args:
        results (Dict[str, Dict[str, List[float]]]): Results for different models.
        metric (str): Metric to plot.
        save_path (str): Path to save the plot.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        title (str, optional): Title of the plot. Defaults to "Model Comparison".
    """
    plt.figure(figsize=figsize)

    for model_name, metrics in results.items():
        if metric in metrics:
            # For metrics like accuracy where higher is better
            if "acc" in metric.lower():
                best_value = max(metrics[metric])
                plt.plot(
                    metrics[metric], label=f"{model_name} (Best: {best_value:.4f})"
                )
            # For metrics like loss where lower is better
            else:
                best_value = min(metrics[metric])
                plt.plot(
                    metrics[metric], label=f"{model_name} (Best: {best_value:.4f})"
                )

    plt.title(f"{title} - {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
