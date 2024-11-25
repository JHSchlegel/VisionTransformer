import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.append("../")
from models.vit import VisionTransformer
from utils.data_utils import prepare_cifar_100, prepare_cifar_10
from utils.trainer import Trainer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="../config", config_name="vit")
def main(config: DictConfig):
    # save dir is folder to which hydra saves the config file and the results
    save_dir = f"{os.getcwd()}"

    # prepare the data
    if config.data.dataset == "cifar10":
        train_loader, val_loader, test_loader = prepare_cifar_10(
            config.data.val_perc_size,
            config.data.train_batch_size,
            config.data.val_batch_size,
            config.data.test_batch_size,
        )
    elif config.data.dataset == "cifar100":
        train_loader, val_loader, test_loader = prepare_cifar_100(
            config.data.val_perc_size,
            config.data.train_batch_size,
            config.data.val_batch_size,
            config.data.test_batch_size,
        )
    else:
        raise ValueError("Invalid dataset name")

    ViT = VisionTransformer(
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        num_classes=10 if config.data.dataset == "cifar10" else 100,
        embed_dim=config.model.embed_dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    )

    ViT.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        ViT.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.scheduler.T_0, T_mult=config.scheduler.T_mult
    )

    ViT_trainer = Trainer(
        model=ViT,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        num_epochs=config.training.num_epochs,
        use_early_stopping=config.training.use_early_stopping,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_tol=config.training.early_stopping_tol,
        save_model_every=config.training.save_model_every,
        use_wandb=config.training.use_wandb,
        wandb_project=f"{config.training.wandb_project}_{config.data.dataset}",
        wandb_logging_dir=f"{save_dir}/wandb",
        wandb_config_kwargs=OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        ),  # pass config in a wandb compatible format
        device=DEVICE,
    )

    ViT_trainer.train_and_validate(
        train_loader=train_loader,
        val_loader=val_loader,
        num_samples=config.training.num_samples,
        top_k=config.training.top_k,
    )


if __name__ == "__main__":
    main()
