from datetime import datetime

import numpy as np
import pandas as pd
import wandb
import os


class WanDBWriter:
    def __init__(self, wandb_project, wandb_api_key):
        self.writer = None
        self.selected_module = ""

        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.login()

        wandb.init(project=wandb_project)
        self.wandb = wandb

        self.step = 0
        self.mode = ""

    def set_step(self, step, mode=""):
        self.mode = mode
        self.step = step

    def _scalar_name(self, scalar_name):
        return f"{scalar_name}_{self.mode}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self._scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in
               scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)