import torch
import torch.nn as nn
import torch.optim as optim

import os
from typing import Optional, Dict


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        save_dir: str = "checkpoints",
        save_model_every: int = 25,
        use_wandb: bool = True,
        wandb_logging_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_config_kwargs: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        # check whether directories exist; if not create them
        os.makedirs
